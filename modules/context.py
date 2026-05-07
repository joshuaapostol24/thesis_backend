import logging

logger = logging.getLogger(__name__)

# ── Rules (same for all barangays) ────────────────────────────────────────────
DEFAULT_RULES = [
    {"priority": 1, "threshold": 2.5, "action": "HIGH"},
    {"priority": 2, "threshold": 1.5, "action": "MODERATE"},
    {"priority": 3, "threshold": 0.0, "action": "LOW"},
]

# ── Base weights from Table 4 of your paper ───────────────────────────────────
BASE_WEIGHTS = {
    "rainfall": 0.40,
    "soil":     0.30,
    "flood":    0.20,
    "report":   0.10,
}

INDICATOR_SET = ["rainfall", "soil", "flood", "report"]

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

    Weight logic per overall hazard level:

    HIGH   (Tayamaan, Poblacion 8)
           → Flood-prone weight increased to 0.40
           → Rainfall reduced since flood exposure is already structural
           → These barangays have BOTH medium flood AND SSA3 storm surge

    LOW    (San Luis, Tangkalan)
           → Rainfall weight increased to 0.50
           → Flood reduced since structural exposure is minimal
           → Rainfall is the primary early warning signal here

    MODERATE (all Poblacion 1-7, Balansay, Fatima, Payompon, Talabaan)
           → Use base Table 4 weights
           → Balanced exposure — no dominant single factor
    """
    # Get profile from built-in dict or from passed hazard_profile
    profile = hazard_profile or BARANGAY_PROFILES.get(barangay_id, {})
    overall   = profile.get("overall", "MODERATE")
    ssa_level = profile.get("ssa_level", 0)

    if overall == "HIGH":
        if ssa_level == 3:
            # HIGH flood + SSA3 storm surge — flood is dominant structural risk
            weights = {
                "rainfall": 0.22,
                "soil":     0.28,
                "flood":    0.40,
                "report":   0.10,
            }
        else:
            # HIGH flood only — no storm surge
            weights = {
                "rainfall": 0.25,
                "soil":     0.30,
                "flood":    0.35,
                "report":   0.10,
            }

    elif overall == "LOW":
        # Minimal structural hazard — rainfall is the early warning signal
        weights = {
            "rainfall": 0.50,
            "soil":     0.28,
            "flood":    0.12,
            "report":   0.10,
        }

    else:
        # MODERATE — use Table 4 base weights
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