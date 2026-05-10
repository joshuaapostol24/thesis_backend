import logging

logger = logging.getLogger(__name__)


def fuse_risk(
    predicted: float,
    rule_score: float,
    barangay_id: int = None,
    hazard_profile: dict = None,
    alpha: float = None,
    beta: float = None,
) -> float:
    """
    Fuses ML prediction (0–3) and rule-based score (0–1) with adaptive per-barangay weights.

    Both inputs are normalized to [0,1] using their known ranges, then fused and rescaled to [0,3].

    Args:
        predicted:      CNN-LSTM output (0–3 range)
        rule_score:     Weighted sum of rule engine indicators (0–1 range, but can exceed)
        barangay_id:    Optional for adaptive weighting lookup
        hazard_profile: Optional barangay context for weight computation
        alpha:          ML prediction weight (default computed from hazard_profile)
        beta:           Rule engine weight (default computed from hazard_profile)

    Returns:
        Final fused risk score (0–3 range)
    """
    # ── Determine adaptive weights if not explicitly provided ────────────────
    if alpha is None or beta is None:
        alpha, beta = _compute_fusion_weights(barangay_id, hazard_profile)

    # Validate weights
    if abs((alpha + beta) - 1.0) > 1e-6:
        logger.warning("alpha + beta = %.2f (expected 1.0). Normalizing.", alpha + beta)
        total = alpha + beta
        alpha, beta = alpha / total, beta / total

    # ── Normalize both inputs to [0,1] using their known ranges ──────────────
    # ML output is guaranteed 0–3 (clamped in CNN-LSTM forward)
    predicted_norm = max(0.0, min(1.0, predicted / 3.0))

    # Rule score is bounded by sum of weights (which should sum to 1.0)
    # If weights are correctly normalized, rule_score ≤ 1.0
    # If they exceed 1.0 due to bugs, we clip and log warning
    rule_norm = max(0.0, min(1.0, rule_score / 1.0))
    if rule_score > 1.0:
        logger.warning(
            "Rule score exceeds 1.0: %.4f (barangay_id=%s). "
            "This indicates indicator weights don't sum to 1.0. "
            "Check compute_barangay_weights() and indicator normalization.",
            rule_score, barangay_id
        )

    # ── Fuse on normalized scale ──────────────────────────────────────────────
    fused_norm = (alpha * predicted_norm) + (beta * rule_norm)

    # ── Rescale to output range [0,3] ─────────────────────────────────────────
    fused = fused_norm * 3.0

    logger.debug(
        "Fusion | barangay_id=%s | predicted=%.4f (norm=%.4f) | "
        "rule=%.4f (norm=%.4f) | alpha=%.2f beta=%.2f | fused=%.4f",
        barangay_id, predicted, predicted_norm, rule_score, rule_norm, alpha, beta, fused
    )

    return fused


def _compute_fusion_weights(barangay_id: int = None, hazard_profile: dict = None) -> tuple:
    """
    Compute per-barangay fusion weights (alpha, beta).

    HIGH hazard zones (accurate GIS data) → trust rule engine more (beta=0.6)
    LOW hazard zones (uncertain exposure) → balance both sources (beta=0.5)
    MODERATE zones → slight rule engine bias (beta=0.55)

    Returns:
        (alpha, beta) where alpha + beta = 1.0
    """
    if hazard_profile is None and barangay_id is not None:
        try:
            from modules.database import get_barangay_hazard_profile
            hazard_profile = get_barangay_hazard_profile(barangay_id)
        except Exception as e:
            logger.warning("Could not load hazard profile for barangay %d: %s", barangay_id, e)
            hazard_profile = {}

    overall = hazard_profile.get("overall_hazard", "MODERATE") if hazard_profile else "MODERATE"

    if overall == "HIGH":
        # GIS data is highly certain (shapefiles, site surveys)
        # Trust rule engine more
        alpha, beta = 0.40, 0.60
    elif overall == "LOW":
        # Minimal structural hazard, weather is primary indicator
        # More balanced, slight ML advantage (captures temporal anomalies)
        alpha, beta = 0.55, 0.45
    else:  # MODERATE
        # Default: balanced with slight rule engine bias
        alpha, beta = 0.45, 0.55

    logger.debug(
        "Fusion weights for barangay_id=%s | overall=%s → alpha=%.2f beta=%.2f",
        barangay_id, overall, alpha, beta
    )

    return alpha, beta