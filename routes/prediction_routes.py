import logging

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from modules.context import load_context
from modules.normalization import compute_weighted_scores
from modules.rule_engine import compute_risk_score, apply_rules
from modules.cnn_lstm import predict_risk
from modules.fusion import fuse_risk
from modules.risk_adjustment import apply_rainfall_adjustment
from modules.feedback import store_data
from modules.weather_api import get_weather
from modules.database import (
    get_barangay_centroid,
    get_barangay_name,
    get_barangay_hazard_profile,
    get_barangay_features,
    get_recent_weather,
    list_barangay_profiles,
)

router = APIRouter(tags=["Predictions"])
logger = logging.getLogger(__name__)

_RISK_ORDER = {
    "VERY HIGH": 0, "HIGH": 1,
    "MODERATE": 2, "LOW": 3, "VERY LOW": 4,
}

_RISK_EMOJI = {
    "VERY HIGH": "🔴", "HIGH": "🟠",
    "MODERATE": "🟡", "LOW": "🟢", "VERY LOW": "⚪",
}


class PredictRequest(BaseModel):
    barangay_id: int
    hazard_type: str = "Flood"


# ── Soil computation ──────────────────────────────────────────────────────────

def compute_soil_saturation(humidity: float, rainfall: float, season: str) -> float:
    base          = (humidity / 100.0) * 2.0
    rain_contrib  = min(1.0, rainfall / 20.0)
    season_mod    = 1.1 if season == "Wet Season" else (0.9 if season == "Dry Season" else 1.0)
    return round(min(3.0, (base + rain_contrib) * season_mod), 4)


def get_season_from_month(month: int) -> str:
    if month in [6, 7, 8, 9, 10, 11]:
        return "Wet Season"
    elif month in [3, 4, 5]:
        return "Summer"
    return "Dry Season"


# ── Auto-announce builder ─────────────────────────────────────────────────────

def _build_live_announcement(result: dict) -> dict | None:
    """
    Returns a ready-to-post news payload if live prediction hits
    HIGH or VERY HIGH. Returns None if threshold not met.
    Frontend shows a modal — user confirms before posting.
    """
    risk_level = result.get("risk_level", "")
    if risk_level not in ("HIGH", "VERY HIGH"):
        return None

    barangay_name = result.get("barangay_name", "Unknown")
    emoji         = _RISK_EMOJI.get(risk_level, "⚠️")
    rainfall      = result.get("rainfall", 0)
    humidity      = result.get("humidity", 0)
    wind_speed    = result.get("wind_speed", 0)
    temperature   = result.get("temperature", 0)
    final_risk    = result.get("final_risk", 0)

    title = (
        f"{emoji} LIVE ALERT: {risk_level} Flood Risk — {barangay_name}"
    )

    message = (
        f"Live weather monitoring has detected a {risk_level} flood risk "
        f"in {barangay_name}.\n\n"
        f"Current Conditions:\n"
        f"  • Rainfall:    {rainfall:.1f} mm/h\n"
        f"  • Humidity:    {humidity:.1f}%\n"
        f"  • Wind Speed:  {wind_speed:.1f} km/h\n"
        f"  • Temperature: {temperature:.1f}°C\n\n"
        f"Risk Score: {final_risk:.4f} — Level: {risk_level}\n\n"
        f"Flood Hazard: {result.get('flood_hazard_level')} | "
        f"Storm Surge Score: {result.get('storm_surge_score', 0):.2f}\n\n"
        f"This is an automated live weather alert. "
        f"Please verify conditions before issuing evacuation orders."
    )

    return {
        "title":    title,
        "category": "Weather",
        "priority": "High",
        "date":     datetime.now(timezone.utc).isoformat(),
        "audience": "All Residents",
        "pinned":   "No",
        "message":  message,
    }


# ── Core prediction logic ─────────────────────────────────────────────────────

def _run_prediction_for_barangay(barangay_id: int, hazard_type: str = "Flood") -> dict:
    lat, lon       = get_barangay_centroid(barangay_id)
    barangay_name  = get_barangay_name(barangay_id)
    hazard_profile = get_barangay_hazard_profile(barangay_id)

    HR = {
        "type": hazard_type, "location": barangay_name,
        "barangay_id": barangay_id, "lat": lat, "lon": lon,
    }
    E = {"osm_lat": lat, "osm_lon": lon}

    weather = get_weather(lat, lon)
    E.update(weather)

    if E.get("rainfall") is None:
        db_features   = get_barangay_features(barangay_id)
        E["rainfall"] = db_features.get("rainfall", 0.0) or 0.0
        logger.info("Barangay %d | DB rainfall fallback: %.2f mm", barangay_id, E["rainfall"])
    elif float(E.get("rainfall", 0.0)) < 1.0:
        E["rainfall"] = 0.0

    current_month = datetime.utcnow().month
    season        = get_season_from_month(current_month)
    humidity      = float(E.get("humidity", 0.0))
    rainfall      = float(E.get("rainfall", 0.0))
    soil          = compute_soil_saturation(humidity, rainfall, season)

    E["soil"]        = soil
    E["season"]      = season
    E["flood"]       = float(hazard_profile.get("flood_hazard_score", 0.0))
    E["storm_surge"] = float(hazard_profile.get("storm_surge_score",  0.0))

    logger.info(
        "Barangay %d (%s) | rainfall=%.2f humidity=%.2f soil=%.4f "
        "flood=%.2f surge=%.2f season=%s",
        barangay_id, barangay_name, rainfall, humidity,
        soil, E["flood"], E["storm_surge"], season
    )

    _, _, indicators, weights, rules = load_context(HR, hazard_profile)
    weighted_scores  = compute_weighted_scores(HR, E, hazard_profile, weights, barangay_id)
    rule_score_total = compute_risk_score(weighted_scores)
    recent_history   = list(reversed(get_recent_weather(barangay_id, limit=90)))
    predicted        = predict_risk(barangay_id, E, {"history": recent_history})
    final_risk       = fuse_risk(predicted, rule_score_total,
                                 barangay_id=barangay_id, hazard_profile=hazard_profile)
    final_risk       = apply_rainfall_adjustment(final_risk, rainfall, barangay_id)
    risk_level       = apply_rules(rules, final_risk)

    HR.update({
        "rainfall": rainfall, "humidity": humidity, "soil": soil,
        "flood": E["flood"], "storm_surge": E["storm_surge"], "season": season,
    })
    store_data(HR, E, rule_score_total, predicted, final_risk, risk_level)

    return {
        "barangay_id":        barangay_id,
        "barangay_name":      barangay_name,
        "lat":                lat,
        "lon":                lon,
        "rainfall":           rainfall,
        "humidity":           humidity,
        "temperature":        E.get("temperature"),
        "wind_speed":         E.get("wind_speed"),
        "season":             season,
        "soil":               soil,
        "flood_hazard_score": E["flood"],
        "storm_surge_score":  E["storm_surge"],
        "flood_hazard_level": hazard_profile.get("flood_hazard_level"),
        "max_ssa_level":      hazard_profile.get("max_ssa_level"),
        "overall_hazard":     hazard_profile.get("overall_hazard"),
        "weights_used":       weights,
        "rule_score":         round(rule_score_total, 4),
        "predicted":          round(predicted,        4),
        "final_risk":         round(final_risk,       4),
        "risk_level":         risk_level,
    }


# =========================================================
# ROUTES
# =========================================================

@router.post("/predict-risk")
def predict(req: PredictRequest):
    """
    Run live weather prediction for a single barangay.
    Returns suggest_announcement if risk level is HIGH or VERY HIGH.
    Frontend shows a confirmation modal before posting.
    """
    try:
        get_barangay_centroid(req.barangay_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Barangay {req.barangay_id} not found.")

    try:
        result = _run_prediction_for_barangay(req.barangay_id, req.hazard_type)

        # ── Auto-announce suggestion ──────────────────────────────────────────
        result["suggest_announcement"] = _build_live_announcement(result)

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("predict-risk failed for barangay_id=%d: %s", req.barangay_id, exc)
        raise HTTPException(status_code=500, detail="Risk assessment failed.")


@router.post("/predict-risk/all")
def predict_all(hazard_type: str = "Flood"):
    """
    Run live weather prediction for ALL 15 barangays in parallel.
    Returns suggest_announcement if any barangay hits HIGH or VERY HIGH.
    """
    try:
        all_profiles = list_barangay_profiles()
        barangay_ids = [p["barangay_id"] for p in all_profiles]
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to load barangay list.")

    results  = []
    failures = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(_run_prediction_for_barangay, bid, hazard_type): bid
            for bid in barangay_ids
        }
        for future in as_completed(future_map):
            bid = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error("Prediction failed for barangay %d: %s", bid, exc)
                failures.append({"barangay_id": bid, "error": str(exc)})

    results.sort(key=lambda r: (
        _RISK_ORDER.get(r["risk_level"], 99), -(r["final_risk"] or 0)
    ))

    summary = {level: 0 for level in _RISK_ORDER}
    for r in results:
        if r.get("risk_level") in summary:
            summary[r["risk_level"]] += 1

    # ── Auto-announce: suggest if any barangay is HIGH or VERY HIGH ───────────
    high_barangays = [
        r for r in results if r.get("risk_level") in ("HIGH", "VERY HIGH")
    ]
    suggest_announcement = None
    if high_barangays:
        top   = sorted(high_barangays, key=lambda r: _RISK_ORDER.get(r["risk_level"], 99))[0]
        emoji = _RISK_EMOJI.get(top["risk_level"], "⚠️")
        names = ", ".join(r["barangay_name"] for r in high_barangays)
        suggest_announcement = {
            "title":    f"{emoji} LIVE ALERT: {top['risk_level']} Flood Risk Detected",
            "category": "Weather",
            "priority": "High",
            "date":     datetime.now(timezone.utc).isoformat(),
            "audience": "All Residents",
            "pinned":   "No",
            "message": (
                f"Live weather monitoring has detected elevated flood risk "
                f"in the following barangays: {names}.\n\n"
                f"Risk Summary:\n"
                f"  🔴 Very High: {summary.get('VERY HIGH', 0)}\n"
                f"  🟠 High:      {summary.get('HIGH', 0)}\n"
                f"  🟡 Moderate:  {summary.get('MODERATE', 0)}\n"
                f"  🟢 Low:       {summary.get('LOW', 0)}\n\n"
                f"Please verify conditions before issuing evacuation orders."
            ),
        }

    return {
        "hazard_type": hazard_type,
        "total":       len(results),
        "failures":    len(failures),
        "summary": {
            "very_high": summary.get("VERY HIGH", 0),
            "high":      summary.get("HIGH",      0),
            "moderate":  summary.get("MODERATE",  0),
            "low":       summary.get("LOW",       0),
            "very_low":  summary.get("VERY LOW",  0),
        },
        "barangays":          results,
        "failed_barangays":   failures,
        "suggest_announcement": suggest_announcement,
    }