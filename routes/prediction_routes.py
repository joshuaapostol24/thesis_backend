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
        barangay_name   = get_barangay_name(req.barangay_id)
        hazard_profile  = get_barangay_hazard_profile(req.barangay_id)
        B               = hazard_profile

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

        # Override flood/storm_surge with structural hazard scores
        E["flood"]       = hazard_profile["flood_hazard_score"]
        E["storm_surge"] = hazard_profile["storm_surge_score"]

        # Fall back to DB weather if API returned zeros
        if E.get("rainfall", 0) == 0:
            db_features = get_barangay_features(req.barangay_id)
            if db_features["rainfall"] > 0:
                E["rainfall"] = db_features["rainfall"]

        # ── 3. Rule engine ────────────────────────────────────────────────────
        hazard_type, barangay, indicators, weights, rules = load_context(HR, B)

        weighted_scores  = compute_weighted_scores(HR, E, B, weights, req.barangay_id)
        rule_score_total = compute_risk_score(weighted_scores)

        # ── 4. ML prediction ──────────────────────────────────────────────────
        # OPTIMIZED: Get per-barangay weather sequence (now 90 days instead of 10)
        recent_history = list(reversed(get_recent_weather(req.barangay_id, limit=90)))
        predicted      = predict_risk(req.barangay_id, E, {"history": recent_history})

        # ── 5. Fusion + classification ────────────────────────────────────────
        # OPTIMIZED: Use adaptive fusion weights per barangay hazard context
        final_risk = fuse_risk(
            predicted,
            rule_score_total,
            barangay_id=req.barangay_id,
            hazard_profile=hazard_profile
        )
        risk_level = apply_rules(rules, final_risk)

        # ── 6. Persist assessment ─────────────────────────────────────────────
        store_data(HR, rule_score_total, predicted, final_risk, risk_level)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("predict-risk failed for barangay_id=%d: %s", req.barangay_id, e)
        raise HTTPException(
            status_code=500,
            detail="Risk assessment failed. Please try again later."
        )

    return {
        "barangay_id":   req.barangay_id,
        "barangay_name": barangay_name,
        "final_risk":    round(final_risk, 4),
        "risk_level":    risk_level,
    }