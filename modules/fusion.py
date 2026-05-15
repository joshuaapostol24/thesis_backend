import logging

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_ML_SCORE = 3.0
MAX_RULE_SCORE = 1.0


def fuse_risk(
    predicted: float,
    rule_score: float,
    barangay_id: int = None,
    hazard_profile: dict = None,
    alpha: float = None,
    beta: float = None,
) -> float:
    """
    Adaptive fusion of:
        - CNN + LSTM prediction
        - Rule-engine score

    Workflow:
        1. Normalize both signals to [0,1]
        2. Apply adaptive fusion weights
        3. Rescale fused output to [0,3]

    Parameters
    ----------
    predicted : float
        CNN-LSTM prediction (0–3)

    rule_score : float
        Rule engine score (0–1)

    barangay_id : int
        Optional barangay identifier.

    hazard_profile : dict
        GIS-derived barangay hazard profile.

    alpha : float
        ML weight override.

    beta : float
        Rule-engine weight override.
    """

    # ── Compute adaptive weights ─────────────────────────────────────

    if alpha is None or beta is None:

        alpha, beta = _compute_fusion_weights(
            barangay_id,
            hazard_profile
        )

    # ── Weight validation ────────────────────────────────────────────

    total_weight = alpha + beta

    if abs(total_weight - 1.0) > 1e-6:

        logger.warning(
            "Fusion weights invalid "
            "(alpha=%.3f beta=%.3f). "
            "Normalizing automatically.",
            alpha,
            beta
        )

        alpha /= total_weight
        beta  /= total_weight

    # ── Normalize ML prediction ──────────────────────────────────────

    predicted_norm = max(
        0.0,
        min(
            1.0,
            predicted / MAX_ML_SCORE
        )
    )

    # ── Normalize rule score ─────────────────────────────────────────

    if rule_score > MAX_RULE_SCORE:

        logger.warning(
            "Rule score exceeded %.1f "
            "(score=%.4f barangay_id=%s). "
            "Indicator weights may be invalid.",
            MAX_RULE_SCORE,
            rule_score,
            barangay_id
        )

    rule_norm = max(
        0.0,
        min(
            1.0,
            rule_score / MAX_RULE_SCORE
        )
    )

    # ── Adaptive fusion ──────────────────────────────────────────────

    fused_norm = (
        (alpha * predicted_norm) +
        (beta  * rule_norm)
    )

    # ── Rescale to 0–3 ───────────────────────────────────────────────

    fused_score = (
        fused_norm *
        MAX_ML_SCORE
    )

    # Final clamp for safety
    fused_score = max(
        0.0,
        min(
            MAX_ML_SCORE,
            fused_score
        )
    )

    logger.debug(
        "Fusion | barangay=%s | "
        "pred=%.4f norm=%.4f | "
        "rule=%.4f norm=%.4f | "
        "alpha=%.2f beta=%.2f | "
        "fused=%.4f",
        barangay_id,
        predicted,
        predicted_norm,
        rule_score,
        rule_norm,
        alpha,
        beta,
        fused_score
    )

    return fused_score


def _compute_fusion_weights(
    barangay_id: int = None,
    hazard_profile: dict = None
) -> tuple:
    """
    Computes adaptive fusion weights.

    Strategy
    --------
    HIGH hazard:
        Trust GIS + rule-engine more.

    MODERATE hazard:
        Balanced hybrid fusion.

    LOW hazard:
        Slight ML preference because
        weather anomalies dominate.

    Returns
    -------
    (alpha, beta)
        alpha = ML weight
        beta  = rule-engine weight
    """

    # ── Lazy-load hazard profile if missing ──────────────────────────

    if (
        hazard_profile is None and
        barangay_id is not None
    ):

        try:

            from modules.database import (
                get_barangay_hazard_profile
            )

            hazard_profile = (
                get_barangay_hazard_profile(
                    barangay_id
                )
            )

        except Exception as exc:

            logger.warning(
                "Could not load hazard profile "
                "for barangay %s: %s",
                barangay_id,
                exc
            )

            hazard_profile = {}

    overall = (
        hazard_profile.get(
            "overall_hazard",
            "MODERATE"
        )
        if hazard_profile
        else "MODERATE"
    )

    # ── HIGH hazard barangays ────────────────────────────────────────
    # Structural GIS hazard is highly reliable

    if overall == "HIGH":

        alpha = 0.40
        beta  = 0.60

    # ── LOW hazard barangays ─────────────────────────────────────────
    # Weather anomalies matter more

    elif overall == "LOW":

        alpha = 0.55
        beta  = 0.45

    # ── MODERATE hazard barangays ────────────────────────────────────

    else:

        alpha = 0.45
        beta  = 0.55

    logger.debug(
        "Fusion weights | barangay=%s "
        "overall=%s | alpha=%.2f "
        "beta=%.2f",
        barangay_id,
        overall,
        alpha,
        beta
    )

    return alpha, beta