import logging

logger = logging.getLogger(__name__)

# ── Rules (same for all barangays) ────────────────────────────────────────────
DEFAULT_RULES = [
    {"priority": 1, "threshold": 2.5, "action": "HIGH"},
    {"priority": 2, "threshold": 1.5, "action": "MODERATE"},
    {"priority": 3, "threshold": 0.0, "action": "LOW"},
]

# ── Updated base weights (now 5 indicators) ──────────────────────────────────
# Includes new indicators: humidity, storm_surge
BASE_WEIGHTS = {
    "rainfall":    0.35,   # Reduced from 0.40 (now includes humidity signal)
    "soil":        0.25,   # Reduced from 0.30
    "flood":       0.20,
    "humidity":    0.10,   # NEW: Saturation indicator (> 85% RH = warning)
    "storm_surge": 0.10,   # NEW: Coastal hazard (coastal barangays)
}

INDICATOR_SET = ["rainfall", "soil", "flood", "humidity", "storm_surge"]

# ── Per-barangay hazard profiles (from barangay_hazard_profile.csv) ───────────
# These are the actual values from your shapefiles — not survey means
BARANGAY_PROFILES = {
    1:  {"name": "Balansay",    "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    2:  {"name": "Fatima",      "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    3:  {"name": "Payompon",    "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    4:  {"name": "San Luis",    "overall": "LOW",      "ssa_level": 0, "flood_score": 0.20},
    5:  {"name": "Talabaan",    "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    6:  {"name": "Tangkalan",   "overall": "LOW",      "ssa_level": 0, "flood_score": 0.60},
    7:  {"name": "Tayamaan",    "overall": "HIGH",     "ssa_level": 3, "flood_score": 0.60},
    8:  {"name": "Poblacion 1", "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    9:  {"name": "Poblacion 2", "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    10: {"name": "Poblacion 3", "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    11: {"name": "Poblacion 4", "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    12: {"name": "Poblacion 5", "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    13: {"name": "Poblacion 6", "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    14: {"name": "Poblacion 7", "overall": "MODERATE", "ssa_level": 3, "flood_score": 0.20},
    15: {"name": "Poblacion 8", "overall": "HIGH",     "ssa_level": 3, "flood_score": 0.60},
}


def compute_barangay_weights(barangay_id: int, hazard_profile: dict = None) -> dict:
    """
    Dynamically computes indicator weights based on the barangay's
    hazard profile — this is the B (Barangay Context) parameter
    described in Algorithm 3 of the paper.

    Updated to include humidity and storm_surge indicators.

    Weight logic per overall hazard level:

    HIGH   (Tayamaan, Poblacion 8)
           → Flood-prone weight increased to 0.35
           → Storm surge significant (SSA3) → 0.22
           → Rainfall reduced since flood exposure is already structural
           → These barangays have BOTH medium flood AND SSA3 storm surge

    LOW    (San Luis, Tangkalan)
           → Rainfall weight increased to 0.40
           → Humidity elevated (pre-saturation signal) → 0.15
           → Flood reduced since structural exposure is minimal
           → Rainfall is the primary early warning signal here

    MODERATE (all Poblacion 1-7, Balansay, Fatima, Payompon, Talabaan)
           → Coastal moderate: elevated storm surge (0.18)
           → Inland moderate: use base Table 4 weights
           → Balanced exposure — no dominant single factor
    """
    # Get profile from built-in dict or from passed hazard_profile
    profile = hazard_profile or BARANGAY_PROFILES.get(barangay_id, {})
    overall   = profile.get("overall", "MODERATE")
    ssa_level = profile.get("ssa_level", 0)

    if overall == "HIGH":
        if ssa_level == 3:
            # HIGH flood + SSA3 storm surge — both are dominant structural risks
            weights = {
                "rainfall":    0.15,   # Reduced: already exposed
                "soil":        0.20,
                "flood":       0.35,   # Primary risk factor
                "humidity":    0.08,   # Low: not primary indicator
                "storm_surge": 0.22,   # Significant coastal risk
            }
        else:
            # HIGH flood only (unlikely in study area)
            weights = {
                "rainfall":    0.20,
                "soil":        0.25,
                "flood":       0.35,
                "humidity":    0.10,
                "storm_surge": 0.10,
            }

    elif overall == "LOW":
        # Minimal structural hazard — rainfall + humidity are early warnings
        weights = {
            "rainfall":    0.40,   # Primary indicator
            "soil":        0.25,
            "flood":       0.10,   # Minimal exposure
            "humidity":    0.15,   # Pre-saturation indicator
            "storm_surge": 0.10,
        }

    else:  # MODERATE
        # Differentiate coastal vs inland MODERATE barangays
        if ssa_level == 3:
            # Coastal moderate: elevated storm surge weighting
            weights = {
                "rainfall":    0.32,
                "soil":        0.23,
                "flood":       0.18,
                "humidity":    0.09,
                "storm_surge": 0.18,   # Elevated for coastal exposure
            }
        else:
            # Inland moderate: use base weights (shouldn't occur in study area)
            weights = BASE_WEIGHTS.copy()

    # Always normalize so weights sum exactly to 1.0
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    logger.info(
        "Barangay %d (%s) | overall=%s ssa=%d → weights=%s",
        barangay_id,
        profile.get("name", "Unknown"),
        overall,
        ssa_level,
        weights,
    )
    return weights


def load_context(HR: dict, hazard_profile: dict = None) -> tuple:
    """
    Loads context with per-barangay weights.

    Parameters
    ----------
    HR             : Hazard Report dict (must contain barangay_id)
    hazard_profile : B parameter from Algorithm 3 — barangay context data.
                     If None, falls back to BARANGAY_PROFILES lookup by id.
    """
    hazard_type  = HR.get("type", "Unknown")
    barangay     = HR.get("location", "Unknown")
    barangay_id  = HR.get("barangay_id", 0)

    weights = compute_barangay_weights(barangay_id, hazard_profile)

    sorted_rules = sorted(DEFAULT_RULES, key=lambda r: r["threshold"], reverse=True)

    logger.debug(
        "Context loaded | barangay_id=%d name=%s hazard=%s weights=%s",
        barangay_id, barangay, hazard_type, weights
    )

    return hazard_type, barangay, INDICATOR_SET, weights, sorted_rules