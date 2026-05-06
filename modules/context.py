import logging

logger = logging.getLogger(__name__)

DEFAULT_RULES = [
    {"priority": 1, "threshold": 2.5, "action": "HIGH"},
    {"priority": 2, "threshold": 1.5, "action": "MODERATE"},
    {"priority": 3, "threshold": 0.0, "action": "LOW"},
]

DEFAULT_WEIGHTS = {
    "rainfall":    0.35,
    "humidity":    0.10,
    "soil":        0.20,
    "flood":       0.20,
    "storm_surge": 0.15,
}

INDICATOR_SET = ["rainfall", "humidity", "soil", "flood", "storm_surge"]


def load_context(HR: dict, B: dict | None = None) -> tuple:
    hazard_type = HR.get("type", "Unknown")
    barangay    = HR.get("location", "Unknown")

    sorted_rules = sorted(DEFAULT_RULES, key=lambda r: r["threshold"], reverse=True)
    weights = DEFAULT_WEIGHTS.copy()

    if B:
        overall = str(B.get("overall_hazard", "")).upper()
        if overall == "HIGH":
            weights.update({
                "rainfall": 0.35,
                "humidity": 0.08,
                "soil": 0.17,
                "flood": 0.22,
                "storm_surge": 0.18,
            })
        elif overall == "MODERATE":
            weights.update({
                "rainfall": 0.35,
                "humidity": 0.10,
                "soil": 0.18,
                "flood": 0.22,
                "storm_surge": 0.15,
            })

    logger.debug("Loaded context for hazard '%s' in '%s'", hazard_type, barangay)

    return hazard_type, barangay, INDICATOR_SET, weights, sorted_rules
