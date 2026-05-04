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
        logger.warning("alpha + beta = %.2f (expected 1.0). Results may not be normalized.", alpha + beta)

    fused = (alpha * predicted) + (beta * rule_score)
    logger.debug("Fused risk: alpha=%.2f * predicted=%.4f + beta=%.2f * rule=%.4f = %.4f",
                 alpha, predicted, beta, rule_score, fused)
    return fused