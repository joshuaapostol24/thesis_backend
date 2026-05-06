import logging

logger = logging.getLogger(__name__)


def fuse_risk(predicted: float, rule_score: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    """
    Fuses ML-predicted risk and rule-based score using a weighted sum.

    Args:
        predicted:   CNN+LSTM model output (range 0–3)
        rule_score:  Weighted rule-based score
        alpha:       Weight for ML prediction (default 0.5)
        beta:        Weight for rule-based score (default 0.5)

    Note: alpha + beta should equal 1.0 for interpretable results.
    """
    if abs((alpha + beta) - 1.0) > 1e-6:
        logger.warning("alpha + beta = %.2f (expected 1.0)", alpha + beta)

    predicted_norm = max(0.0, min(1.0, predicted / 3.0))
    rule_norm      = max(0.0, min(1.0, rule_score))

    fused_norm = (alpha * predicted_norm) + (beta * rule_norm)

    fused = fused_norm * 3.0

    logger.debug(
        "Fusion | predicted=%.4f norm=%.4f | rule=%.4f | fused=%.4f",
        predicted, predicted_norm, rule_score, fused
    )

    return fused