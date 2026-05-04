import logging

logger = logging.getLogger(__name__)

DEFAULT_RULES = [
    {"priority": 1, "threshold": 2.5, "action": "HIGH"},
    {"priority": 2, "threshold": 1.5, "action": "MODERATE"},
    {"priority": 3, "threshold": 0.0, "action": "LOW"},
]

DEFAULT_WEIGHTS = {
    "rainfall": 0.45,
    "soil":     0.30,
    "flood":    0.25,
}

INDICATOR_SET = ["rainfall", "soil", "flood"]


def load_context(HR: dict) -> tuple:
    hazard_type = HR.get("type", "Unknown")
    barangay    = HR.get("location", "Unknown")

    sorted_rules = sorted(DEFAULT_RULES, key=lambda r: r["threshold"], reverse=True)

    logger.debug("Loaded context for hazard '%s' in '%s'", hazard_type, barangay)

    return hazard_type, barangay, INDICATOR_SET, DEFAULT_WEIGHTS, sorted_rules