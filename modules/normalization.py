import logging

logger = logging.getLogger(__name__)

# ── Global fallback ranges ───────────────────────────────────────────────────
INDICATOR_RANGES = {
    "rainfall":    (0.0, 40.0),
    "soil":        (0.0, 3.0),
    "flood":       (0.0, 1.0),
    "humidity":    (40.0, 95.0),
    "storm_surge": (0.0, 1.0),
}

# ── Unified barangay hazard profiles ─────────────────────────────────────────
# Synced with GIS + weather collector + context.py
BARANGAY_PROFILES = {
    1:  {"name": "Balansay",    "overall": "LOW",      "flood": 0.20, "storm_surge": 0.00},
    2:  {"name": "Fatima",      "overall": "LOW",      "flood": 0.20, "storm_surge": 0.00},
    3:  {"name": "Payompon",    "overall": "MODERATE", "flood": 1.00, "storm_surge": 0.00},
    4:  {"name": "San Luis",    "overall": "LOW",      "flood": 0.20, "storm_surge": 0.00},
    5:  {"name": "Talabaan",    "overall": "LOW",      "flood": 0.60, "storm_surge": 0.00},
    6:  {"name": "Tangkalan",   "overall": "LOW",      "flood": 0.60, "storm_surge": 0.00},
    7:  {"name": "Tayamaan",    "overall": "MODERATE", "flood": 1.00, "storm_surge": 0.00},
    8:  {"name": "Poblacion 1", "overall": "LOW",      "flood": 0.20, "storm_surge": 0.00},
    9:  {"name": "Poblacion 2", "overall": "HIGH",     "flood": 0.60, "storm_surge": 1.00},
    10: {"name": "Poblacion 3", "overall": "LOW",      "flood": 0.20, "storm_surge": 0.00},
    11: {"name": "Poblacion 4", "overall": "LOW",      "flood": 0.20, "storm_surge": 0.00},
    12: {"name": "Poblacion 5", "overall": "HIGH",     "flood": 0.60, "storm_surge": 1.00},
    13: {"name": "Poblacion 6", "overall": "MODERATE", "flood": 1.00, "storm_surge": 0.00},
    14: {"name": "Poblacion 7", "overall": "MODERATE", "flood": 1.00, "storm_surge": 0.00},
    15: {"name": "Poblacion 8", "overall": "LOW",      "flood": 0.20, "storm_surge": 0.00},
}

# ── Rainfall normalization bounds ────────────────────────────────────────────
# Lower max value = more rainfall sensitivity
BARANGAY_RAINFALL_BOUNDS = {
    1:  (0.0, 40.0),
    2:  (0.0, 40.0),
    3:  (0.0, 35.0),
    4:  (0.0, 40.0),
    5:  (0.0, 38.0),
    6:  (0.0, 38.0),
    7:  (0.0, 35.0),
    8:  (0.0, 40.0),
    9:  (0.0, 30.0),
    10: (0.0, 40.0),
    11: (0.0, 40.0),
    12: (0.0, 30.0),
    13: (0.0, 35.0),
    14: (0.0, 35.0),
    15: (0.0, 40.0),
}

# ── Flood normalization bounds ──────────────────────────────────────────────
BARANGAY_FLOOD_BOUNDS = {
    barangay_id: INDICATOR_RANGES["flood"]
    for barangay_id in BARANGAY_PROFILES.keys()
}

# ── Humidity normalization bounds ───────────────────────────────────────────
BARANGAY_HUMIDITY_BOUNDS = {
    barangay_id: (40.0, 95.0)
    for barangay_id in BARANGAY_PROFILES.keys()
}

# ── Storm surge normalization bounds ────────────────────────────────────────
BARANGAY_STORM_SURGE_BOUNDS = {
    barangay_id: INDICATOR_RANGES["storm_surge"]
    for barangay_id in BARANGAY_PROFILES.keys()
}


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Safe min-max normalization.
    Returns clamped value between 0.0 and 1.0.
    """

    if max_val <= min_val:

        logger.warning(
            "Invalid normalization bounds "
            "(min=%.2f max=%.2f)",
            min_val,
            max_val
        )

        return 0.0

    normalized = (
        (value - min_val) /
        (max_val - min_val)
    )

    return max(0.0, min(1.0, normalized))


def get_indicator_bounds(
    indicator: str,
    barangay_id: int
) -> tuple:
    """
    Retrieves correct normalization bounds
    for a specific indicator and barangay.
    """

    if indicator == "rainfall":
        return BARANGAY_RAINFALL_BOUNDS.get(
            barangay_id,
            INDICATOR_RANGES["rainfall"]
        )

    if indicator == "flood":
        return BARANGAY_FLOOD_BOUNDS.get(
            barangay_id,
            INDICATOR_RANGES["flood"]
        )

    if indicator == "humidity":
        return BARANGAY_HUMIDITY_BOUNDS.get(
            barangay_id,
            INDICATOR_RANGES["humidity"]
        )

    if indicator == "storm_surge":
        return BARANGAY_STORM_SURGE_BOUNDS.get(
            barangay_id,
            INDICATOR_RANGES["storm_surge"]
        )

    return INDICATOR_RANGES.get(
        indicator,
        (0.0, 1.0)
    )


def compute_weighted_scores(
    HR: dict,
    E: dict,
    B: dict,
    weight_set: dict,
    barangay_id: int = 0
) -> list:
    """
    Computes normalized weighted indicator scores.

    Parameters
    ----------
    HR : dict
        Hazard report data.

    E : dict
        Environmental/weather data.

    B : dict
        Barangay hazard profile.

    weight_set : dict
        Adaptive weights from context.py

    barangay_id : int
        Barangay identifier for adaptive normalization.
    """

    raw_values = {
        "rainfall":    float(E.get("rainfall", 0.0)),
        "soil":        float(E.get("soil", 0.0)),
        "flood":       float(E.get("flood", 0.0)),
        "humidity":    float(E.get("humidity", 0.0)),
        "storm_surge": float(E.get("storm_surge", 0.0)),
    }

    weighted_scores = []

    for indicator, raw_value in raw_values.items():

        min_val, max_val = get_indicator_bounds(
            indicator,
            barangay_id
        )

        normalized = normalize(
            raw_value,
            min_val,
            max_val
        )

        weight = weight_set.get(
            indicator,
            0.0
        )

        score = normalized * weight

        logger.debug(
            "Barangay %d | %s | raw=%.4f "
            "bounds=(%.2f, %.2f) "
            "normalized=%.4f weight=%.4f "
            "score=%.4f",
            barangay_id,
            indicator,
            raw_value,
            min_val,
            max_val,
            normalized,
            weight,
            score
        )

        weighted_scores.append(score)

    return weighted_scores
