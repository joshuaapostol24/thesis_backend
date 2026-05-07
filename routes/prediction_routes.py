import logging

from fastapi import APIRouter
from pydantic import BaseModel

from ml_service_python.modules.context import load_context
from ml_service_python.modules.normalization import compute_weighted_scores
from ml_service_python.modules.rule_engine import (
    compute_risk_score,
    apply_rules
)

from ml_service_python.modules.cnn_lstm import predict_risk
from ml_service_python.modules.fusion import fuse_risk
from ml_service_python.modules.feedback import store_data
from ml_service_python.modules.weather_api import get_weather

from ml_service_python.modules.database import (
    get_barangay_features,
    get_barangay_centroid,
    get_barangay_name,
    get_barangay_hazard_profile,
    get_recent_weather,
)

router = APIRouter(
    tags=["Predictions"]
)

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    barangay_id: int
    hazard_type: str = "Flood"


@router.post("/predict-risk")
def predict(req: PredictRequest):

    lat, lon = get_barangay_centroid(
        req.barangay_id
    )

    barangay_name = get_barangay_name(
        req.barangay_id
    )

    hazard_profile = get_barangay_hazard_profile(
        req.barangay_id
    )

    B = hazard_profile

    HR = {
        "type": req.hazard_type,
        "location": barangay_name,
        "barangay_id": req.barangay_id,
        "lat": lat,
        "lon": lon,
    }

    E = {
        "osm_lat": lat,
        "osm_lon": lon,
    }

    weather = get_weather(lat, lon)
    E.update(weather)

    E["flood"] = (
        hazard_profile["flood_hazard_score"]
    )

    E["storm_surge"] = (
        hazard_profile["storm_surge_score"]
    )

    db_features = get_barangay_features(
        req.barangay_id
    )

    if (
        E.get("rainfall", 0) == 0
        and db_features["rainfall"] > 0
    ):
        E["rainfall"] = db_features["rainfall"]

    hazard_type, barangay, indicators, weights, rules = (
        load_context(HR, B)
    )

    rule_score = compute_weighted_scores(
        HR,
        E,
        B,
        weights,
        req.barangay_id
    )

    rule_score_total = compute_risk_score(
        rule_score
    )

    recent_history = list(
        reversed(
            get_recent_weather(
                req.barangay_id,
                limit=10
            )
        )
    )

    predicted = predict_risk(
        req.barangay_id,
        E,
        {"history": recent_history}
    )

    final_risk = fuse_risk(
        predicted,
        rule_score_total
    )

    risk_level = apply_rules(
        rules,
        final_risk
    )

    store_data(
        HR,
        rule_score_total,
        predicted,
        final_risk,
        risk_level
    )

    return {
        "barangay_id": req.barangay_id,
        "barangay_name": barangay_name,
        "final_risk": round(final_risk, 4),
        "risk_level": risk_level,
    }