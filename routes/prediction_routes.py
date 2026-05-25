import logging

from datetime import datetime

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import BaseModel

from modules.context import load_context
from modules.normalization import compute_weighted_scores
from modules.rule_engine import (
    compute_risk_score,
    apply_rules,
)
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
)

router = APIRouter(
    tags=["Predictions"]
)

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):

    barangay_id: int
    hazard_type: str = "Flood"


# ── Shared soil logic ─────────────────────────────────────────────────────────

def compute_soil_saturation(
    humidity: float,
    rainfall: float,
    season: str
) -> float:
    """
    MUST match:
        - weather_service.py
        - training generator
        - CNN-LSTM training data
    """

    base = (
        (humidity / 100.0) *
        2.0
    )

    rain_contrib = min(
        1.0,
        rainfall / 20.0
    )

    season_modifier = (
        1.1 if season == "Wet Season"
        else (
            0.9 if season == "Dry Season"
            else 1.0
        )
    )

    soil = (
        base +
        rain_contrib
    ) * season_modifier

    return round(
        min(3.0, soil),
        4
    )


def get_season_from_month(
    month: int
) -> str:

    if month in [6, 7, 8, 9, 10, 11]:
        return "Wet Season"

    elif month in [3, 4, 5]:
        return "Summer"

    return "Dry Season"


# ── Main prediction route ────────────────────────────────────────────────────

@router.post("/predict-risk")
def predict(req: PredictRequest):

    # ── Resolve barangay ────────────────────────────────────────────────

    try:

        lat, lon = get_barangay_centroid(
            req.barangay_id
        )

    except ValueError:

        raise HTTPException(
            status_code=404,
            detail=f"Barangay {req.barangay_id} not found."
        )

    try:

        barangay_name = get_barangay_name(
            req.barangay_id
        )

        hazard_profile = (
            get_barangay_hazard_profile(
                req.barangay_id
            )
        )

        HR = {
            "type": req.hazard_type,
            "location": barangay_name,
            "barangay_id": req.barangay_id,
            "lat": lat,
            "lon": lon,
        }

        # ── Weather collection ────────────────────────────────────────

        E = {
            "osm_lat": lat,
            "osm_lon": lon,
        }

        weather = get_weather(
            lat,
            lon
        )

        E.update(weather)

        # ── DB rainfall fallback ──────────────────────────────────────

        if float(E.get("rainfall", 0.0)) <= 0:

            db_features = get_barangay_features(
                req.barangay_id
            )

            if db_features.get("rainfall", 0) > 0:

                E["rainfall"] = (
                    db_features["rainfall"]
                )

                logger.info(
                    "Barangay %d | "
                    "Using DB rainfall fallback: %.2f mm",
                    req.barangay_id,
                    E["rainfall"]
                )

        # ── Season + soil computation ────────────────────────────────

        current_month = datetime.utcnow().month

        season = get_season_from_month(
            current_month
        )

        humidity = float(
            E.get("humidity", 0.0)
        )

        rainfall = float(
            E.get("rainfall", 0.0)
        )

        soil = compute_soil_saturation(
            humidity,
            rainfall,
            season
        )

        E["soil"] = soil
        E["season"] = season

        # ── GIS structural hazards ───────────────────────────────────

        E["flood"] = float(
            hazard_profile.get(
                "flood_hazard_score",
                0.0
            )
        )

        E["storm_surge"] = float(
            hazard_profile.get(
                "storm_surge_score",
                0.0
            )
        )

        logger.info(
            "Barangay %d (%s) | "
            "rainfall=%.2f humidity=%.2f "
            "soil=%.4f flood=%.2f "
            "surge=%.2f season=%s",
            req.barangay_id,
            barangay_name,
            rainfall,
            humidity,
            soil,
            E["flood"],
            E["storm_surge"],
            season
        )

        # ── Context + adaptive weights ───────────────────────────────

        (
            hazard_type,
            barangay,
            indicators,
            weights,
            rules,
        ) = load_context(
            HR,
            hazard_profile
        )

        logger.info(
            "Barangay %d | "
            "overall=%s | weights=%s",
            req.barangay_id,
            hazard_profile.get("overall_hazard"),
            weights
        )

        # ── Rule engine ──────────────────────────────────────────────

        weighted_scores = compute_weighted_scores(
            HR,
            E,
            hazard_profile,
            weights,
            req.barangay_id
        )

        rule_score_total = compute_risk_score(
            weighted_scores
        )

        # ── Historical weather ───────────────────────────────────────

        recent_history = list(reversed(
            get_recent_weather(
                req.barangay_id,
                limit=90
            )
        ))

        # ── CNN + LSTM prediction ────────────────────────────────────

        predicted = predict_risk(
            req.barangay_id,
            E,
            {"history": recent_history}
        )

        # ── Fusion layer ─────────────────────────────────────────────

        final_risk = fuse_risk(
            predicted,
            rule_score_total,
            barangay_id=req.barangay_id,
            hazard_profile=hazard_profile
        )

        # ── Rainfall gating ──────────────────────────────────────────
        # Suppress dry-weather false alerts and preserve severe rainfall.

        final_risk = apply_rainfall_adjustment(
            final_risk,
            rainfall,
            req.barangay_id
        )

        # ── Final classification ────────────────────────────────────

        risk_level = apply_rules(
            rules,
            final_risk
        )

        # ── Persist assessment ──────────────────────────────────────

        HR.update({

            "rainfall": rainfall,
            "humidity": humidity,
            "soil": soil,
            "flood": E["flood"],
            "storm_surge": E["storm_surge"],
            "season": season,

        })

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

    except Exception as exc:

        logger.exception(
            "predict-risk failed "
            "for barangay_id=%d: %s",
            req.barangay_id,
            exc
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk assessment failed. "
                "Please try again later."
            )
        )

    # ── API response ────────────────────────────────────────────────

    return {

        "barangay_id": req.barangay_id,
        "barangay_name": barangay_name,

        "lat": lat,
        "lon": lon,

        # Weather
        "rainfall": rainfall,
        "humidity": humidity,
        "temperature": E.get("temperature"),
        "wind_speed": E.get("wind_speed"),
        "season": season,

        # Computed indicators
        "soil": soil,
        "flood_hazard_score": E["flood"],
        "storm_surge_score": E["storm_surge"],

        # Hazard context
        "flood_hazard_level": (
            hazard_profile.get(
                "flood_hazard_level"
            )
        ),

        "max_ssa_level": (
            hazard_profile.get(
                "max_ssa_level"
            )
        ),

        "overall_hazard": (
            hazard_profile.get(
                "overall_hazard"
            )
        ),

        "weights_used": weights,

        # Scores
        "rule_score": round(
            rule_score_total,
            4
        ),

        "predicted": round(
            predicted,
            4
        ),

        "final_risk": round(
            final_risk,
            4
        ),

        "risk_level": risk_level,
    }
