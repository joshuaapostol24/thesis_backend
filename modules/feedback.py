import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text
from modules.database import engine

logger = logging.getLogger(__name__)

_LOG_PATH = Path(os.environ.get("ASSESSMENT_LOG_PATH", "assessment_log.jsonl"))


def store_data(
    HR: dict,
    E: dict,
    rule_score: float,
    predicted: float,
    final_risk: float,
    risk_level: str,
) -> None:
    """
    Persists an assessment record for future model retraining and auditing.

    Reads all indicator values from E (environmental data) and HR (hazard report).
    E must contain: rainfall, humidity, soil, flood, storm_surge
    HR must contain: barangay_id, type, location, season

    Records are inserted into Supabase risk_assessments table AND
    appended to a local JSONL log file for audit trail.
    Failure to write is logged but never raises — assessment results
    must not be lost because logging failed.
    """

    # ── Safely extract all indicator values ──────────────────────────────────
    # soil is explicitly guarded — this was previously NULL in risk_assessments
    # because prediction_routes.py never set E["soil"] before calling store_data.
    # Fixed in prediction_routes.py — E["soil"] is now always computed before
    # this function is called.
    soil_value = E.get("soil")
    if soil_value is None:
        logger.warning(
            "soil is None for barangay_id=%d — defaulting to 0.0. "
            "Check that E['soil'] is set in prediction_routes.py before store_data().",
            HR.get("barangay_id", 0)
        )
        soil_value = 0.0

    record = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "barangay_id":     HR.get("barangay_id"),
        "hazard":          HR.get("type"),
        "location":        HR.get("location"),
        # ── Weather indicators ─────────────────────────────────────────────
        "rainfall":        float(E.get("rainfall") or 0.0),
        "humidity":        float(E.get("humidity") or 0.0),
        "temperature":     float(E.get("temperature") or 0.0),
        "wind_speed":      float(E.get("wind_speed") or 0.0),
        "season":          E.get("season") or HR.get("season") or "Unknown",
        # ── Computed indicators ────────────────────────────────────────────
        "soil":            round(float(soil_value), 4),
        "flood":           float(E.get("flood") or 0.0),
        "storm_surge":     float(E.get("storm_surge") or 0.0),
        # ── Risk scores ────────────────────────────────────────────────────
        "rule_score":      round(float(rule_score), 4),
        "predicted":       round(float(predicted), 4),
        "final_risk":      round(float(final_risk), 4),
        "risk_level":      risk_level,
        # ── Metadata ──────────────────────────────────────────────────────
        "osm_is_fallback": HR.get("osm_is_fallback"),
    }

    logger.info(
        "Barangay %d | risk=%s | rainfall=%.2f soil=%.4f flood=%.2f surge=%.2f "
        "rule=%.4f predicted=%.4f final=%.4f",
        record["barangay_id"] or 0,
        risk_level,
        record["rainfall"],
        record["soil"],
        record["flood"],
        record["storm_surge"],
        record["rule_score"],
        record["predicted"],
        record["final_risk"],
    )

    # ── Insert into Supabase risk_assessments ─────────────────────────────────
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO risk_assessments (
                    timestamp, barangay_id, hazard, location,
                    rainfall, humidity, temperature, wind_speed, season,
                    soil, flood, storm_surge,
                    rule_score, predicted, final_risk,
                    risk_level, osm_is_fallback
                ) VALUES (
                    :timestamp, :barangay_id, :hazard, :location,
                    :rainfall, :humidity, :temperature, :wind_speed, :season,
                    :soil, :flood, :storm_surge,
                    :rule_score, :predicted, :final_risk,
                    :risk_level, :osm_is_fallback
                )
            """), record)
        logger.debug("Record inserted into risk_assessments.")
    except Exception as exc:
        logger.error(
            "Failed to insert assessment record into risk_assessments: %s. "
            "Record: %s",
            exc, record
        )

    # ── Append to local JSONL audit log ───────────────────────────────────────
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Record appended to '%s'.", _LOG_PATH)
    except OSError as exc:
        logger.error(
            "Failed to write assessment record to '%s': %s. Record: %s",
            _LOG_PATH, exc, record,
        )


def update_system(feedback: bool, weight_set: dict) -> None:
    """
    Adjusts weights/rules based on ground-truth feedback.

    NOT YET IMPLEMENTED. Raises NotImplementedError to prevent silent
    no-ops if called before the adaptive logic is in place.

    Once implemented, this should:
    1. Read accumulated records from assessment_log.jsonl
    2. Compare predicted risk_level against confirmed ground-truth outcomes
    3. Apply gradient-free optimisation or Bayesian update to weights
    4. Call cnn_lstm.retrain_barangay() once enough labelled records accumulate
    """
    raise NotImplementedError(
        "update_system() is not yet implemented. "
        "Feedback signal received (feedback=%s, weights=%s) but no update was applied. "
        "Implement adaptive weight logic before calling this function."
        % (feedback, weight_set)
    )