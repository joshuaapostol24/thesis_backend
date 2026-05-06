import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Path for the assessment log.  Each line is a JSON object (JSONL format).
# Override via environment variable before the process starts.
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

    Records are appended in JSONL format (one JSON object per line) to
    ASSESSMENT_LOG_PATH (default: assessment_log.jsonl in the working
    directory).  Each record includes a UTC ISO-8601 timestamp so that
    temporal ordering is preserved without relying on filesystem metadata.

    The accumulated log is the corpus used for future CNN+LSTM retraining
    once ground-truth outcome labels (e.g. "flood occurred: yes/no") are
    added.  Until then, the file serves as an audit trail.

    Failure to write (permission error, disk full, etc.) is logged as an
    error but never raises — assessment results must not be lost because
    logging failed.
    """
    record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "barangay_id": HR.get("barangay_id"),

    "hazard": HR.get("type"),
    "location": HR.get("location"),

    # raw features (IMPORTANT for retraining CNN + LSTM)
    "rainfall": HR.get("rainfall"),
    "humidity": HR.get("humidity"),
    "soil": HR.get("soil"),
    "flood": HR.get("flood"),
    "storm_surge": HR.get("storm_surge"),

    # model outputs
    "rule_score": round(rule_score, 4),
    "predicted": round(predicted, 4),
    "final_risk": round(final_risk, 4),
    "risk_level": risk_level,

    # system metadata
    "osm_is_fallback": HR.get("osm_is_fallback"),
    }

    logger.info("Assessment record: %s", record)

    try:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Record appended to '%s'.", _LOG_PATH)
    except OSError as exc:
        logger.error(
            "Failed to write assessment record to '%s': %s. "
            "Record was: %s",
            _LOG_PATH, exc, record,
        )


def update_system(feedback: bool, weight_set: dict) -> None:
    """
    Adjusts weights/rules based on ground-truth feedback.

    Args:
        feedback   : True if the assessment outcome was confirmed correct
                     by a human reviewer or post-event observation.
        weight_set : The weight dict that was active during the assessment
                     (a copy of DEFAULT_WEIGHTS from context.py).

    TODO: Implement actual adaptive weight update logic — e.g. gradient-free
    optimisation over the JSONL log, or a Bayesian update rule — and call
    cnn_lstm.retrain() once enough new labelled records have accumulated.

    Current behaviour: logs the weight snapshot and feedback signal so that
    the information is preserved for future use even before the adaptive
    logic is implemented.
    """
    if feedback:
        logger.info(
            "Positive feedback received. Assessment confirmed correct. "
            "Active weights at time of assessment: %s. "
            "Weight update not yet implemented — record preserved in log.",
            weight_set,
        )
    else:
        logger.info(
            "Negative feedback received. Assessment outcome did not match "
            "ground truth. Active weights: %s. "
            "Weight adjustment not yet implemented — record preserved in log.",
            weight_set,
        )
