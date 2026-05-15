import json
import logging
import math
import os

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

from sqlalchemy import text

from modules.database import engine

logger = logging.getLogger(__name__)

# ── Local audit log ──────────────────────────────────────────────────────────

_LOG_PATH = Path(
    os.environ.get(
        "ASSESSMENT_LOG_PATH",
        "assessment_log.jsonl"
    )
)


# ── Numeric sanitizer ────────────────────────────────────────────────────────

def _safe_float(
    value,
    default: float = 0.0
) -> float:
    """
    Prevent NaN/inf corruption in DB and logs.
    """

    try:

        value = float(value)

        if (
            math.isnan(value) or
            math.isinf(value)
        ):
            return default

        return value

    except Exception:
        return default


# ── Main persistence function ────────────────────────────────────────────────

def store_data(
    HR: dict,
    E: dict,
    rule_score: float,
    predicted: float,
    final_risk: float,
    risk_level: str,
) -> None:
    """
    Stores finalized risk assessment.

    Data is persisted to:
        1. Supabase/PostgreSQL
        2. Local JSONL audit log

    Used for:
        - auditing
        - analytics
        - future retraining
        - temporal history generation
    """

    # ── Indicator extraction ─────────────────────────────────────────

    rainfall = _safe_float(
        E.get("rainfall")
    )

    humidity = _safe_float(
        E.get("humidity")
    )

    temperature = _safe_float(
        E.get("temperature")
    )

    wind_speed = _safe_float(
        E.get("wind_speed")
    )

    soil = _safe_float(
        E.get("soil")
    )

    flood = _safe_float(
        E.get("flood")
    )

    storm_surge = _safe_float(
        E.get("storm_surge")
    )

    # ── Risk scores ──────────────────────────────────────────────────

    rule_score = round(
        _safe_float(rule_score),
        4
    )

    predicted = round(
        _safe_float(predicted),
        4
    )

    final_risk = round(
        _safe_float(final_risk),
        4
    )

    # ── Construct assessment record ──────────────────────────────────

    record = {

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "barangay_id":
            HR.get("barangay_id"),

        "hazard":
            HR.get("type"),

        "location":
            HR.get("location"),

        # Weather indicators
        "rainfall":
            rainfall,

        "humidity":
            humidity,

        "temperature":
            temperature,

        "wind_speed":
            wind_speed,

        "season":
            (
                E.get("season") or
                HR.get("season") or
                "Unknown"
            ),

        # Computed indicators
        "soil":
            round(soil, 4),

        "flood":
            flood,

        "storm_surge":
            storm_surge,

        # Scores
        "rule_score":
            rule_score,

        "predicted":
            predicted,

        "final_risk":
            final_risk,

        "risk_level":
            risk_level,

        # Metadata
        "osm_is_fallback":
            HR.get("osm_is_fallback"),
    }

    logger.info(
        "Barangay %s | "
        "risk=%s | rainfall=%.2f "
        "humidity=%.2f soil=%.4f "
        "flood=%.2f surge=%.2f | "
        "rule=%.4f predicted=%.4f "
        "final=%.4f",
        record["barangay_id"],
        risk_level,
        rainfall,
        humidity,
        soil,
        flood,
        storm_surge,
        rule_score,
        predicted,
        final_risk,
    )

    # ── PostgreSQL persistence ───────────────────────────────────────

    try:

        with engine.begin() as conn:

            conn.execute(text("""
                INSERT INTO risk_assessments (

                    timestamp,
                    barangay_id,
                    hazard,
                    location,

                    rainfall,
                    humidity,
                    temperature,
                    wind_speed,
                    season,

                    soil,
                    flood,
                    storm_surge,

                    rule_score,
                    predicted,
                    final_risk,

                    risk_level,
                    osm_is_fallback

                ) VALUES (

                    :timestamp,
                    :barangay_id,
                    :hazard,
                    :location,

                    :rainfall,
                    :humidity,
                    :temperature,
                    :wind_speed,
                    :season,

                    :soil,
                    :flood,
                    :storm_surge,

                    :rule_score,
                    :predicted,
                    :final_risk,

                    :risk_level,
                    :osm_is_fallback
                )
            """), record)

        logger.debug(
            "Assessment stored in DB."
        )

    except Exception as exc:

        logger.error(
            "DB insert failed: %s | record=%s",
            exc,
            record
        )

    # ── Local JSONL audit trail ──────────────────────────────────────

    try:

        with _LOG_PATH.open(
            "a",
            encoding="utf-8"
        ) as fh:

            fh.write(
                json.dumps(record) + "\n"
            )

        logger.debug(
            "Assessment appended to %s",
            _LOG_PATH
        )

    except OSError as exc:

        logger.error(
            "Failed to append audit log: %s",
            exc
        )


# ── Future adaptive feedback system ──────────────────────────────────────────

def update_system(
    feedback: bool,
    weight_set: dict
) -> None:
    """
    Placeholder for future adaptive learning.

    Future capabilities:
        - online learning
        - adaptive fuzzy weights
        - feedback-based calibration
        - automatic CNN retraining
    """

    raise NotImplementedError(
        "Adaptive feedback system "
        "not yet implemented."
    )