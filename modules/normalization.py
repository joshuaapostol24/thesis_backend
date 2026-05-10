import logging

logger = logging.getLogger(__name__)

# ── Global indicator ranges (fallback) ────────────────────────────────────────
INDICATOR_RANGES = {
    "rainfall":    (0.0, 40.0),
    "soil":        (0.0, 3.0),
    "flood":       (0.0, 1.0),
    "humidity":    (0.0, 100.0),      # NEW: % RH
    "storm_surge": (0.0, 1.0),        # NEW: normalized hazard score
}

# ── Per-barangay rainfall bounds ──────────────────────────────────────────────
# HIGH risk barangays (Tayamaan, Poblacion 8) use lower max = more sensitive
# LOW risk barangays (San Luis, Tangkalan) use higher max = less sensitive
BARANGAY_RAINFALL_BOUNDS = {
    1:  (0.0, 35.0),   # Balansay    — coastal SSA3, MODERATE
    2:  (0.0, 35.0),   # Fatima      — coastal SSA3, MODERATE
    3:  (0.0, 35.0),   # Payompon    — coastal SSA3, MODERATE
    4:  (0.0, 40.0),   # San Luis    — inland, LOW
    5:  (0.0, 35.0),   # Talabaan    — coastal SSA3, MODERATE
    6:  (0.0, 40.0),   # Tangkalan   — no storm surge, LOW
    7:  (0.0, 30.0),   # Tayamaan    — HIGH, most sensitive
    8:  (0.0, 35.0),   # Poblacion 1 — coastal SSA3, MODERATE
    9:  (0.0, 35.0),   # Poblacion 2 — coastal SSA3, MODERATE
    10: (0.0, 35.0),   # Poblacion 3 — coastal SSA3, MODERATE
    11: (0.0, 35.0),   # Poblacion 4 — coastal SSA3, MODERATE
    12: (0.0, 35.0),   # Poblacion 5 — coastal SSA3, MODERATE
    13: (0.0, 35.0),   # Poblacion 6 — coastal SSA3, MODERATE
    14: (0.0, 35.0),   # Poblacion 7 — coastal SSA3, MODERATE
    15: (0.0, 30.0),   # Poblacion 8 — HIGH, most sensitive
}

# ── Per-barangay flood bounds ─────────────────────────────────────────────────
# Matches flood_hazard_score from barangay_hazard_profile table
BARANGAY_FLOOD_BOUNDS = {
    1:  (0.0, 0.20),   # Balansay    — Low flood
    2:  (0.0, 0.20),   # Fatima      — Low flood
    3:  (0.0, 0.20),   # Payompon    — Low flood
    4:  (0.0, 0.20),   # San Luis    — Low flood
    5:  (0.0, 0.20),   # Talabaan    — Low flood
    6:  (0.0, 0.60),   # Tangkalan   — Medium flood
    7:  (0.0, 0.60),   # Tayamaan    — Medium flood + SSA3
    8:  (0.0, 0.20),   # Poblacion 1 — Low flood
    9:  (0.0, 0.20),   # Poblacion 2 — Low flood
    10: (0.0, 0.20),   # Poblacion 3 — Low flood
    11: (0.0, 0.20),   # Poblacion 4 — Low flood
    12: (0.0, 0.20),   # Poblacion 5 — Low flood
    13: (0.0, 0.20),   # Poblacion 6 — Low flood
    14: (0.0, 0.20),   # Poblacion 7 — Low flood
    15: (0.0, 0.60),   # Poblacion 8 — Medium flood + SSA3
}

# ── Per-barangay humidity bounds ──────────────────────────────────────────────
# All barangays: 40–95% RH (tropical climate)
# > 85% = saturation (high risk)
BARANGAY_HUMIDITY_BOUNDS = {
    i: (40.0, 95.0) for i in range(1, 16)
}

# ── Per-barangay storm surge bounds ──────────────────────────────────────────
# Coastal (SSA3): 0–1.0 | Inland: 0–0.2
BARANGAY_STORM_SURGE_BOUNDS = {
    1:  (0.0, 1.0),   # Balansay — SSA3
    2:  (0.0, 1.0),   # Fatima — SSA3
    3:  (0.0, 1.0),   # Payompon — SSA3
    4:  (0.0, 0.2),   # San Luis — inland
    5:  (0.0, 1.0),   # Talabaan — SSA3
    6:  (0.0, 0.2),   # Tangkalan — inland
    7:  (0.0, 1.0),   # Tayamaan — SSA3
    8:  (0.0, 1.0),   # Poblacion 1 — SSA3
    9:  (0.0, 1.0),   # Poblacion 2 — SSA3
    10: (0.0, 1.0),   # Poblacion 3 — SSA3
    11: (0.0, 1.0),   # Poblacion 4 — SSA3
    12: (0.0, 0.2),   # Poblacion 5 — inland
    13: (0.0, 1.0),   # Poblacion 6 — SSA3
    14: (0.0, 0.2),   # Poblacion 7 — inland
    15: (0.0, 1.0),   # Poblacion 8 — SSA3
}


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Min-max normalization per Algorithm 3 of the paper.
    Returns value clamped to [0.0, 1.0].
    """
    if max_val == min_val:
        logger.warning(
            "normalize() called with equal min/max (%.2f). Returning 0.", min_val
        )
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def get_indicator_bounds(indicator: str, barangay_id: int) -> tuple:
    """
    Returns correct (min, max) bounds for a given indicator and barangay.
    Per-barangay bounds for rainfall, flood, humidity, and storm_surge.
    Global bounds for soil (universal scale).
    """
    if indicator == "rainfall" and barangay_id in BARANGAY_RAINFALL_BOUNDS:
        return BARANGAY_RAINFALL_BOUNDS[barangay_id]

    if indicator == "flood" and barangay_id in BARANGAY_FLOOD_BOUNDS:
        return BARANGAY_FLOOD_BOUNDS[barangay_id]

    if indicator == "humidity" and barangay_id in BARANGAY_HUMIDITY_BOUNDS:
        return BARANGAY_HUMIDITY_BOUNDS[barangay_id]

    if indicator == "storm_surge" and barangay_id in BARANGAY_STORM_SURGE_BOUNDS:
        return BARANGAY_STORM_SURGE_BOUNDS[barangay_id]

    return INDICATOR_RANGES.get(indicator, (0.0, 1.0))


def compute_weighted_scores(
    HR: dict,
    E: dict,
    B: dict,
    weight_set: dict,
    barangay_id: int = 0
) -> list:
    """
    Normalizes each indicator using per-barangay bounds
    and applies its weight. Implements Algorithm 3.

    Parameters
    ----------
    HR          : Hazard Report data
    E           : Environmental data (weather, flood score)
    B           : Barangay context data (hazard profile)
    weight_set  : Per-barangay weights from context.py
    barangay_id : Selects correct normalization bounds per barangay
    """
    raw_values = {
        "rainfall":    float(E.get("rainfall", 0)),
        "soil":        float(E.get("soil", 0)),
        "flood":       float(E.get("flood", 0)),
        "humidity":    float(E.get("humidity", 0)),     # NEW
        "storm_surge": float(E.get("storm_surge", 0)),  # NEW
    }

    weighted_scores = []
    for key, val in raw_values.items():
        min_val, max_val = get_indicator_bounds(key, barangay_id)
        norm   = normalize(val, min_val, max_val)
        weight = weight_set.get(key, 0.0)
        score  = norm * weight

        logger.debug(
            "Barangay %d | %s: raw=%.4f bounds=(%.2f,%.2f) "
            "norm=%.4f weight=%.4f score=%.4f",
            barangay_id, key, val, min_val, max_val, norm, weight, score
        )
        weighted_scores.append(score)

    return weighted_scores