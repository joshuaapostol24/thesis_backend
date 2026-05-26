import logging

logger = logging.getLogger(__name__)

# ── Risk interpretation rules ────────────────────────────────────────────────
DEFAULT_RULES = [
    {"priority": 1, "threshold": 2.7, "action": "VERY HIGH"},
    {"priority": 2, "threshold": 2.1, "action": "HIGH"},
    {"priority": 3, "threshold": 1.2, "action": "MODERATE"},
    {"priority": 4, "threshold": 0.5, "action": "LOW"},
    {"priority": 5, "threshold": 0.0, "action": "VERY LOW"},
]

# ── Base indicator weights ───────────────────────────────────────────────────
# Used when barangay has no special flood/storm-surge exposure.
BASE_WEIGHTS = {
    "rainfall":    0.35,
    "soil":        0.25,
    "flood":       0.20,
    "humidity":    0.10,
    "storm_surge": 0.10,
}

INDICATOR_SET = [
    "rainfall",
    "soil",
    "flood",
    "humidity",
    "storm_surge",
]

# ── Barangay hazard profiles ────────────────────────────────────────────────
# Synced with gis_sync.py and weather collection module.
BARANGAY_PROFILES = {
    1:  {"name": "Balansay",    "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    2:  {"name": "Fatima",      "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    3:  {"name": "Payompon",    "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    4:  {"name": "San Luis",    "overall": "LOW",      "flood": 0.20, "storm_surge": 0.00},
    5:  {"name": "Talabaan",    "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    6:  {"name": "Tangkalan",   "overall": "LOW",      "flood": 0.60, "storm_surge": 0.00},
    7:  {"name": "Tayamaan",    "overall": "HIGH",     "flood": 0.60, "storm_surge": 1.00},
    8:  {"name": "Poblacion 1", "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    9:  {"name": "Poblacion 2", "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    10: {"name": "Poblacion 3", "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    11: {"name": "Poblacion 4", "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    12: {"name": "Poblacion 5", "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    13: {"name": "Poblacion 6", "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    14: {"name": "Poblacion 7", "overall": "MODERATE", "flood": 0.20, "storm_surge": 1.00},
    15: {"name": "Poblacion 8", "overall": "HIGH",     "flood": 0.60, "storm_surge": 1.00},
}


def compute_barangay_weights(
    barangay_id: int,
    hazard_profile: dict = None
) -> dict:
    """
    Computes adaptive indicator weights per barangay.

    HIGH:
        Storm surge + flood sensitive coastal barangays.
    MODERATE:
        High flood exposure inland barangays.
    LOW:
        Rainfall-driven early warning areas.
    """

    profile  = hazard_profile or BARANGAY_PROFILES.get(barangay_id, {})
    overall  = profile.get("overall_hazard", profile.get("overall", "MODERATE"))
    flood    = profile.get("flood_hazard_score", profile.get("flood", 0.20))
    surge    = profile.get("storm_surge_score", profile.get("storm_surge", 0.00))

    # ── HIGH RISK COASTAL ────────────────────────────────────────────────
    if overall == "HIGH":

        weights = {
            "rainfall":    0.20,
            "soil":        0.20,
            "flood":       0.25,
            "humidity":    0.10,
            "storm_surge": 0.25,
        }

    # ── MODERATE FLOOD-SENSITIVE ────────────────────────────────────────
    elif overall == "MODERATE":

        if flood >= 1.00:

            weights = {
                "rainfall":    0.25,
                "soil":        0.25,
                "flood":       0.35,
                "humidity":    0.10,
                "storm_surge": 0.05,
            }

        else:
            weights = BASE_WEIGHTS.copy()

    # ── LOW RISK ────────────────────────────────────────────────────────
    else:

        weights = {
            "rainfall":    0.40,
            "soil":        0.25,
            "flood":       0.10,
            "humidity":    0.15,
            "storm_surge": 0.10,
        }

    # Normalize weights
    total = sum(weights.values())

    normalized = {
        key: round(value / total, 4)
        for key, value in weights.items()
    }

    logger.info(
        "Barangay %d (%s) | overall=%s flood=%.2f surge=%.2f | weights=%s",
        barangay_id,
        profile.get("name", "Unknown"),
        overall,
        flood,
        surge,
        normalized
    )

    return normalized


def load_context(
    HR: dict,
    hazard_profile: dict = None
) -> tuple:
    """
    Loads fuzzy inference context.

    Parameters
    ----------
    HR : dict
        Hazard report dictionary.

    hazard_profile : dict
        Optional barangay hazard profile override.
    """

    hazard_type = HR.get("type", "Unknown")
    barangay    = HR.get("location", "Unknown")
    barangay_id = HR.get("barangay_id", 0)

    weights = compute_barangay_weights(
        barangay_id,
        hazard_profile
    )

    sorted_rules = sorted(
        DEFAULT_RULES,
        key=lambda r: r["threshold"],
        reverse=True
    )

    logger.debug(
        "Context loaded | barangay_id=%d barangay=%s "
        "hazard=%s weights=%s",
        barangay_id,
        barangay,
        hazard_type,
        weights
    )

    return (
        hazard_type,
        barangay,
        INDICATOR_SET,
        weights,
        sorted_rules,
    )
