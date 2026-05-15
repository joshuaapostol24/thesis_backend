import logging

logger = logging.getLogger(__name__)

# ── Required report fields ───────────────────────────────────────────────────

REQUIRED_KEYS = [
    "type",
    "location",
]

VALID_HAZARD_TYPES = {
    "Flood",
    "Storm Surge",
    "Rainfall",
    "Typhoon",
}


def validate_report(
    HR: dict
) -> str:
    """
    Validates incoming hazard report data.

    Returns
    -------
    "Valid"
        if report passes validation.

    error string
        if validation fails.
    """

    # ── Basic type validation ────────────────────────────────────────

    if not isinstance(HR, dict):

        logger.error(
            "Hazard report must be a dictionary."
        )

        return "Invalid Report: bad format"

    # ── Completion validation ────────────────────────────────────────

    if not HR.get("isComplete", True):

        logger.warning(
            "Incomplete report: %s",
            HR
        )

        return "Invalid Report: incomplete"

    # ── Required field validation ────────────────────────────────────

    for key in REQUIRED_KEYS:

        if key not in HR:

            logger.error(
                "Missing required key: %s",
                key
            )

            return (
                f"Invalid Report: "
                f"missing '{key}'"
            )

    # ── Hazard type validation ───────────────────────────────────────

    hazard_type = str(
        HR.get("type", "")
    ).strip()

    if (
        hazard_type and
        hazard_type not in VALID_HAZARD_TYPES
    ):

        logger.warning(
            "Unknown hazard type: %s",
            hazard_type
        )

    # ── Optional verification logic ─────────────────────────────────
    # Verification is now optional because:
    #   - system relies mainly on weather APIs
    #   - GIS hazard layers
    #   - ML predictions
    #
    # Community reports are supplementary only.

    verified_score = HR.get("verified")

    if verified_score is not None:

        if not isinstance(
            verified_score,
            (int, float)
        ):

            logger.error(
                "'verified' must be numeric."
            )

            return (
                "Invalid Report: "
                "'verified' must be numeric"
            )

        if verified_score < 0:

            logger.error(
                "'verified' cannot be negative."
            )

            return (
                "Invalid Report: "
                "'verified' cannot be negative"
            )

    logger.debug(
        "Hazard report validated successfully."
    )

    return "Valid"