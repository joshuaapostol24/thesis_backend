import logging

logger = logging.getLogger(__name__)

# Value ranges per indicator (min, max)
INDICATOR_RANGES = {
    "rainfall": (0, 3),
    "soil":     (0, 3),
    "flood":    (0, 1),
    "report":   (0, 1),
}


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalization. Returns 0.0 if range is zero (avoids division by zero)."""
    if max_val == min_val:
        logger.warning("normalize() called with equal min/max (%s). Returning 0.", min_val)
        return 0.0
    return (value - min_val) / (max_val - min_val)


def compute_weighted_scores(HR: dict, E: dict, B: dict, weight_set: dict) -> list:
    """
    Normalizes each indicator and applies its weight.
    Returns a list of weighted scores (one per indicator).
    """
    raw_values = {
        "rainfall": E.get("rainfall", 0),
        "soil":     E.get("soil", 0),
        "flood":    E.get("flood", 0),
        "report":   HR.get("verified", 0),
    }

    weighted_scores = []
    for key, val in raw_values.items():
        min_val, max_val = INDICATOR_RANGES.get(key, (0, 1))
        norm    = normalize(val, min_val, max_val)
        weight  = weight_set.get(key, 0)
        score   = norm * weight
        logger.debug("Indicator '%s': raw=%.2f, norm=%.4f, weight=%.2f, score=%.4f",
                     key, val, norm, weight, score)
        weighted_scores.append(score)

    return weighted_scores