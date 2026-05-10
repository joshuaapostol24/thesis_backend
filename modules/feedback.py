import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_PATH = Path(os.environ.get("ASSESSMENT_LOG_PATH", "assessment_log.jsonl"))


def store_data(
    HR: dict,
    rule_score: float,
    predicted: float,
    final_risk: float,
    risk_level: str,
) -> None:
    """
    Persists an assessment record for future model retraining and auditing.
    Records are appended in JSONL format (one JSON object per line).
    Failure to write is logged but never raises.
    """
    record = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "barangay_id":    HR.get("barangay_id"),
        "hazard":         HR.get("type"),
        "location":       HR.get("location"),
        "rainfall":       HR.get("rainfall"),
        "humidity":       HR.get("humidity"),
        "soil":           HR.get("soil"),
        "flood":          HR.get("flood"),
        "storm_surge":    HR.get("storm_surge"),
        "rule_score":     round(rule_score, 4),
        "predicted":      round(predicted, 4),
        "final_risk":     round(final_risk, 4),
        "risk_level":     risk_level,
        "osm_is_fallback": HR.get("osm_is_fallback"),
    }

    logger.info("Assessment record: %s", record)

    try:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Record appended to '%s'.", _LOG_PATH)
    except OSError as exc:
        logger.error(
            "Failed to write assessment record to '%s': %s. Record was: %s",
            _LOG_PATH, exc, record,
        )


def update_system(feedback: bool, weight_set: dict) -> None:
    """
    Adjusts weights/rules based on ground-truth feedback.

    NOT YET IMPLEMENTED. Raises NotImplementedError to prevent silent
    no-ops if called before the adaptive logic is in place.

    Once implemented, this should perform gradient-free optimisation over
    the JSONL log or a Bayesian update rule, then call cnn_lstm.retrain()
    once enough new labelled records have accumulated.
    """
    raise NotImplementedError(
        "update_system() is not yet implemented. "
        "Feedback signal received (feedback=%s, weights=%s) but no update was applied. "
        "Implement adaptive weight logic before calling this function." % (feedback, weight_set)
    )