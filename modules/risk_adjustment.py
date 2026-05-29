import logging

logger = logging.getLogger(__name__)

DRY_RAINFALL_MM = 2.0
YELLOW_RAINFALL_MM = 7.5
ORANGE_RAINFALL_MM = 15.0
RED_RAINFALL_MM = 30.0
MODERATE_RAINFALL_SCORE_FLOOR = 0.9
HIGH_RAINFALL_SCORE_FLOOR = 1.5
VERY_HIGH_RAINFALL_SCORE_FLOOR = 1.8
MAX_FINAL_RISK = 3.0
MAX_ML_SCORE = 3.0
MAX_RULE_SCORE = 1.0


def _normalize_score(
    value: float,
    maximum: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value) / float(maximum)
        )
    )


def apply_rainfall_adjustment(
    final_risk: float,
    rainfall: float,
    barangay_id: int,
    ml_score: float = None,
    rule_score: float = None,
    hazard_profile: dict = None,
) -> float:
    """
    Applies PAGASA rainfall guardrails after ML/rule/GIS fusion.

    Purpose: enforce a *minimum* floor so that objectively dangerous
    rainfall levels are never under-classified — but never override a
    correctly fused score that is already higher.

    The old _compute_combined_floor() recomputed a score from ml_score
    and rule_score and used it as a floor, which replaced the fused
    per-barangay score with a weather-only number.  That caused every
    barangay to show the same final_score when given the same weather
    inputs.  It has been removed — fuse_risk() already produced the
    correct blended score; this function only applies PAGASA floors.

    PAGASA color-coded rainfall floors:
        Dry   < 2.0 mm             → suppress false alerts (cap at 0.49)
        Yellow  7.5–15.0 mm/hr     → floor at MODERATE (1.2)
        Orange 15.0–30.0 mm/hr     → floor at HIGH     (2.1)
        Red    > 30.0 mm/hr        → floor at VERY HIGH (2.7)
    """

    final_risk = float(final_risk)
    rainfall   = max(0.0, float(rainfall))

    # ── Per-barangay hazard boost ─────────────────────────────────────
    # Scale PAGASA floors up based on the barangay structural hazard so
    # high-flood-zone barangays score higher than low-hazard ones even
    # when both are in the same PAGASA rainfall band.
    # Boost range: 0.0 (LOW) → 0.15 (MODERATE) → 0.30 (HIGH)

    _profile = hazard_profile or {}
    _overall = _profile.get("overall_hazard", "MODERATE")
    _hazard_boost = {"HIGH": 0.30, "MODERATE": 0.15, "LOW": 0.0}.get(_overall, 0.15)

    _vhigh_floor  = VERY_HIGH_RAINFALL_SCORE_FLOOR + _hazard_boost
    _high_floor   = HIGH_RAINFALL_SCORE_FLOOR      + _hazard_boost
    _mod_floor    = MODERATE_RAINFALL_SCORE_FLOOR  + _hazard_boost

    # ── Dry: suppress false alerts ───────────────────────────────────
    # No meaningful rainfall → cap score so dry-weather noise cannot
    # trigger a moderate/high alert.

    if rainfall < DRY_RAINFALL_MM:
        DRY_SCORE_CEILING = 0.49
        adjusted = min(final_risk * 0.4, DRY_SCORE_CEILING)
        logger.info(
            "Barangay %s | dry rainfall (%.2f mm) → cap %.4f → %.4f",
            barangay_id, rainfall, final_risk, adjusted,
        )
        return adjusted

    # ── Red: > 30 mm/hr — enforce VERY HIGH floor ────────────────────

    if rainfall > RED_RAINFALL_MM:
        adjusted = max(final_risk, _vhigh_floor)
        logger.info(
            "Barangay %s | red rainfall %.2f mm | %.4f → %.4f",
            barangay_id, rainfall, final_risk, adjusted,
        )
        return min(MAX_FINAL_RISK, adjusted)

    # ── Orange: 15–30 mm/hr — enforce HIGH floor ─────────────────────

    if rainfall >= ORANGE_RAINFALL_MM:
        adjusted = max(final_risk, _high_floor)
        logger.info(
            "Barangay %s | orange rainfall %.2f mm | %.4f → %.4f",
            barangay_id, rainfall, final_risk, adjusted,
        )
        return min(MAX_FINAL_RISK, adjusted)

    # ── Yellow: 7.5–15 mm/hr — enforce MODERATE floor ────────────────

    if rainfall >= YELLOW_RAINFALL_MM:
        adjusted = max(final_risk, _mod_floor)
        logger.info(
            "Barangay %s | yellow rainfall %.2f mm | %.4f → %.4f",
            barangay_id, rainfall, final_risk, adjusted,
        )
        return min(MAX_FINAL_RISK, adjusted)

    # ── Between dry and yellow: pass through unchanged ────────────────

    return final_risk