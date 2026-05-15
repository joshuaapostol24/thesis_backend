import logging

logger = logging.getLogger(__name__)

# ── Risk score limits ────────────────────────────────────────────────────────

MIN_RISK_SCORE = 0.0
MAX_RULE_SCORE = 1.0
MAX_FINAL_RISK = 3.0


def compute_risk_score(
    weighted_scores: list
) -> float:
    """
    Computes final rule-engine score by summing all
    normalized weighted indicators.

    Expected range:
        0.0 → 1.0

    Notes
    -----
    Scores above 1.0 indicate:
        - invalid indicator normalization
        - weights not summing to 1.0
        - upstream logic inconsistency
    """

    score = float(sum(weighted_scores))

    if score < MIN_RISK_SCORE:

        logger.warning(
            "Rule score below %.1f: %.4f",
            MIN_RISK_SCORE,
            score
        )

        score = MIN_RISK_SCORE

    if score > MAX_RULE_SCORE:

        logger.warning(
            "Rule score exceeded %.1f: %.4f. "
            "Check normalization or weights.",
            MAX_RULE_SCORE,
            score
        )

    logger.debug(
        "Computed rule-engine score: %.4f",
        score
    )

    return score


def apply_rules(
    rule_set: list,
    final_risk: float
) -> str:
    """
    Applies ordered fuzzy-rule thresholds
    to classify final fused risk.

    Parameters
    ----------
    rule_set : list
        Ordered threshold rules from context.py

    final_risk : float
        Final fused risk score (0–3)

    Returns
    -------
    str
        Risk classification label.
    """

    # ── Safety clamp ────────────────────────────────────────────────

    final_risk = max(
        MIN_RISK_SCORE,
        min(
            MAX_FINAL_RISK,
            final_risk
        )
    )

    # ── Rule matching ───────────────────────────────────────────────

    for rule in rule_set:

        threshold = float(
            rule.get("threshold", 0.0)
        )

        action = rule.get(
            "action",
            "UNKNOWN"
        )

        if final_risk >= threshold:

            logger.info(
                "Rule matched | "
                "score=%.4f >= %.2f → %s",
                final_risk,
                threshold,
                action
            )

            return action

    # ── Should never happen ─────────────────────────────────────────

    logger.error(
        "Rule engine exhausted all rules | "
        "final_risk=%.4f | rules=%s",
        final_risk,
        rule_set
    )

    raise AssertionError(
        "Rule engine failed to classify risk score."
    )