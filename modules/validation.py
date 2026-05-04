import logging

logger = logging.getLogger(__name__)


def validate_report(HR: dict) -> str:
    """
    Validates a hazard report before processing.
    Returns 'Valid' or a descriptive error string.

    Checks (in order):
      1. HR must be a dict.
      2. Report must be marked complete (isComplete=True).
      3. Report must be marked verified (isVerified=True).
      4. Required keys must be present: type, location, verified.
      5. Consistency: if isVerified=True, the numeric 'verified' field
         must be > 0.  A verified flag with a zero credibility score is
         contradictory and would silently produce misleading assessments.
    """
    if not isinstance(HR, dict):
        logger.error("HR must be a dictionary.")
        return "Invalid Report: bad format"

    if not HR.get("isComplete", False):
        logger.warning("Report is incomplete: %s", HR)
        return "Invalid Report: incomplete"

    if not HR.get("isVerified", False):
        logger.warning("Report is unverified: %s", HR)
        return "Unverified Report"

    required_keys = ["type", "location", "verified"]
    for key in required_keys:
        if key not in HR:
            logger.error("Missing required key: %s", key)
            return f"Invalid Report: missing '{key}'"

    # ── Consistency check ─────────────────────────────────────────────────────
    # isVerified=True declares that a human or trusted source confirmed this
    # report.  The numeric 'verified' field carries that credibility into the
    # weighted scoring pipeline.  A value of 0 means "no credibility" — which
    # contradicts isVerified=True and would cause the report indicator to
    # contribute zero score regardless of how trustworthy the source is.
    verified_value = HR.get("verified")
    if not isinstance(verified_value, (int, float)):
        logger.error(
            "'verified' must be a number, got %s (%s).",
            verified_value, type(verified_value).__name__,
        )
        return "Invalid Report: 'verified' must be a number"

    if verified_value <= 0:
        logger.error(
            "Contradictory report: isVerified=True but verified=%s. "
            "'verified' must be > 0 when the report is marked as verified.",
            verified_value,
        )
        return "Invalid Report: 'verified' must be > 0 when isVerified is True"

    return "Valid"