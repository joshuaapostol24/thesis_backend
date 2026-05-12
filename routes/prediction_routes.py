import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from modules.context import load_context
from modules.normalization import compute_weighted_scores
from modules.rule_engine import compute_risk_score, apply_rules
from modules.cnn_lstm import predict_risk
from modules.fusion import fuse_risk
from modules.feedback import store_data
from modules.weather_api import get_weather
from modules.database import (
    get_barangay_features,
    get_barangay_centroid,
    get_barangay_name,
    get_barangay_hazard_profile,
    get_recent_weather,
)

router = APIRouter(tags=["Predictions"])
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    barangay_id: int
    hazard_type: str = "Flood"


def compute_soil_saturation(humidity: float, rainfall: float, season: str) -> float:
    """
    Estimates soil saturation from humidity and rainfall.
    Scale: 0.0 - 3.0 (matches INDICATOR_RANGES soil in normalization.py)

    This matches the compute_soil_saturation() used in
    generate_barangay_training_data.py — keeping training and
    inference consistent.
    """
    base        = (humidity / 100.0) * 2.0
    rain_contrib = min(1.0, rainfall / 20.0)
    season_mod  = 1.1 if season == "Wet Season" else (0.9 if season == "Dry Season" else 1.0)
    return round(min(3.0, (base + rain_contrib) * season_mod), 4)


def get_season_from_month(month: int) -> str:
    """
    Returns the season based on month number.
    Matches the season classification in weather_final.csv.
    """
    if month in [6, 7, 8, 9, 10, 11]:
        return "Wet Season"
    elif month in [3, 4, 5]:
        return "Summer"
    else:
        return "Dry Season"


@router.post("/predict-risk")
def predict(req: PredictRequest):
    # ── 1. Resolve barangay ───────────────────────────────────────────────────
    try:
        lat, lon = get_barangay_centroid(req.barangay_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Barangay {req.barangay_id} not found."
        )

    try:
        barangay_name  = get_barangay_name(req.barangay_id)
        hazard_profile = get_barangay_hazard_profile(req.barangay_id)
        B              = hazard_profile

        HR = {
            "type":        req.hazard_type,
            "location":    barangay_name,
            "barangay_id": req.barangay_id,
            "lat":         lat,
            "lon":         lon,
        }

        E = {"osm_lat": lat, "osm_lon": lon}

        # ── 2. Weather ────────────────────────────────────────────────────────
        weather = get_weather(lat, lon)
        E.update(weather)

        # Fall back to DB weather if live API returned zeros
        if E.get("rainfall", 0) == 0:
            db_features = get_barangay_features(req.barangay_id)
            if db_features["rainfall"] > 0:
                E["rainfall"] = db_features["rainfall"]
                logger.info(
                    "Barangay %d — using DB rainfall fallback: %.2f mm",
                    req.barangay_id, E["rainfall"]
                )

        # ── FIX: Compute soil saturation from humidity + rainfall ─────────────
        # Previously soil was never set → NULL in risk_assessments table.
        # This matches compute_soil_saturation() in training data generator
        # so training and inference use identical soil values.
        from datetime import datetime
        current_month = datetime.utcnow().month
        season        = E.get("season") or get_season_from_month(current_month)
        humidity      = float(E.get("humidity", 0))
        rainfall      = float(E.get("rainfall", 0))
        E["soil"]     = compute_soil_saturation(humidity, rainfall, season)
        E["season"]   = season

        # Structural hazard scores — fixed per barangay, not dynamic
        E["flood"]       = hazard_profile["flood_hazard_score"]
        E["storm_surge"] = hazard_profile["storm_surge_score"]

        logger.info(
            "Barangay %d (%s) | rainfall=%.2f soil=%.4f flood=%.2f surge=%.2f season=%s",
            req.barangay_id, barangay_name,
            rainfall, E["soil"], E["flood"], E["storm_surge"], season
        )

        # ── 3. Rule engine ────────────────────────────────────────────────────
        hazard_type, barangay, indicators, weights, rules = load_context(HR, B)

        logger.info(
            "Barangay %d | overall=%s weights=%s",
            req.barangay_id, hazard_profile.get("overall_hazard"), weights
        )

        weighted_scores  = compute_weighted_scores(HR, E, B, weights, req.barangay_id)
        rule_score_total = compute_risk_score(weighted_scores)

        # ── 4. ML prediction ──────────────────────────────────────────────────
        recent_history = list(reversed(get_recent_weather(req.barangay_id, limit=90)))
        predicted      = predict_risk(req.barangay_id, E, {"history": recent_history})

        # ── 5. Fusion + classification ────────────────────────────────────────
        final_risk = fuse_risk(
            predicted,
            rule_score_total,
            barangay_id=req.barangay_id,
            hazard_profile=hazard_profile
        )
        risk_level = apply_rules(rules, final_risk)

        # ── 6. Persist assessment ─────────────────────────────────────────────
        # FIX: pass soil explicitly so it is never NULL in risk_assessments
        HR["rainfall"]    = rainfall
        HR["humidity"]    = humidity
        HR["soil"]        = E["soil"]
        HR["flood"]       = E["flood"]
        HR["storm_surge"] = E["storm_surge"]
        HR["season"]      = season

        store_data(
            HR,
            E,
            rule_score_total,
            predicted,
            final_risk,
            risk_level
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "predict-risk failed for barangay_id=%d: %s",
            req.barangay_id, e
        )
        raise HTTPException(
            status_code=500,
            detail="Risk assessment failed. Please try again later."
        )

    return {
        "barangay_id":        req.barangay_id,
        "barangay_name":      barangay_name,
        "lat":                lat,
        "lon":                lon,
        # ── Weather indicators ─────────────────────────────────────────────
        "rainfall":           rainfall,
        "humidity":           humidity,
        "temperature":        E.get("temperature"),
        "wind_speed":         E.get("wind_speed"),
        "season":             season,
        # ── Computed indicators ────────────────────────────────────────────
        "soil":               E["soil"],
        "flood_hazard_score": E["flood"],
        "storm_surge_score":  E["storm_surge"],
        # ── Per-barangay context ───────────────────────────────────────────
        "flood_hazard_level": hazard_profile["flood_hazard_level"],
        "max_ssa_level":      hazard_profile["max_ssa_level"],
        "overall_hazard":     hazard_profile["overall_hazard"],
        "weights_used":       weights,
        # ── Risk scores ────────────────────────────────────────────────────
        "rule_score":         round(rule_score_total, 4),
        "predicted":          round(predicted, 4),
        "final_risk":         round(final_risk, 4),
        "risk_level":         risk_level,
    }