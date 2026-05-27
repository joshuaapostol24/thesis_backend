import logging

logger = logging.getLogger(__name__)

DRY_RAINFALL_MM = 2.0
YELLOW_RAINFALL_MM = 7.5
ORANGE_RAINFALL_MM = 15.0
RED_RAINFALL_MM = 30.0
MODERATE_RAINFALL_SCORE_FLOOR = 1.2
HIGH_RAINFALL_SCORE_FLOOR = 2.1
VERY_HIGH_RAINFALL_SCORE_FLOOR = 2.7
MAX_FINAL_RISK = 3.0


def apply_rainfall_adjustment(
    final_risk: float,
    rainfall: float,
    barangay_id: int,
) -> float:
    """
    Applies rainfall-specific guardrails after ML/rule fusion.

    PAGASA color-coded rainfall advisories are used as score floors:
        Yellow: 7.5-15 mm in 1 hour -> MODERATE
        Orange: 15-30 mm in 1 hour  -> HIGH
        Red: >30 mm in 1 hour       -> VERY HIGH

    The rule engine normalizes rainfall to 0-1, so heavy rainfall can
    otherwise be softened by ML/rule fusion before classification.
    """

    final_risk = float(final_risk)
    rainfall = max(0.0, float(rainfall))
    DRY_RAINFALL_MM = 2.0
    DRY_SCORE_CEILING = 0.49  

    if rainfall < DRY_RAINFALL_MM:
        adjusted = min(final_risk * 0.4, DRY_SCORE_CEILING)
        logger.info(
        "Barangay %s | dry rainfall adjustment %.2f -> %.4f",
        barangay_id,
        rainfall,
        adjusted,
    )
        return adjusted

    if rainfall > RED_RAINFALL_MM:
        adjusted = max(
            final_risk,
            VERY_HIGH_RAINFALL_SCORE_FLOOR,
        )
        logger.info(
            "Barangay %s | red rainfall adjustment %.2f mm "
            "| %.4f -> %.4f",
            barangay_id,
            rainfall,
            final_risk,
            adjusted,
        )
        return min(MAX_FINAL_RISK, adjusted)

    if rainfall >= ORANGE_RAINFALL_MM:
        adjusted = max(
            final_risk,
            HIGH_RAINFALL_SCORE_FLOOR,
        )
        logger.info(
            "Barangay %s | orange rainfall adjustment %.2f mm "
            "| %.4f -> %.4f",
            barangay_id,
            rainfall,
            final_risk,
            adjusted,
        )
        return min(MAX_FINAL_RISK, adjusted)

    if rainfall >= YELLOW_RAINFALL_MM:
        adjusted = max(
            final_risk,
            MODERATE_RAINFALL_SCORE_FLOOR,
        )
        logger.info(
            "Barangay %s | yellow rainfall adjustment %.2f mm "
            "| %.4f -> %.4f",
            barangay_id,
            rainfall,
            final_risk,
            adjusted,
        )
        return min(MAX_FINAL_RISK, adjusted)

    return final_risk
