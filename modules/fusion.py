import logging

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_ML_SCORE   = 3.0
MAX_RULE_SCORE = 1.0

# GIS hazard score ceilings (from gis_sync.py / FLOOD_LEVEL_SCORE)
MAX_FLOOD_SCORE = 4.0   # High=4.0, Moderate=2.0, Low=1.0
MAX_SURGE_SCORE = 4.0   # SSA4=4.0 … SSA1=1.0

# Practical ceiling of predict_risk() output.
# The model is trained on historical data and rarely exceeds 1.5
# until retrained with real GIS data. Normalizing against 3.0
# compresses the ML signal to ~28% of scale, weakening its contribution.
# Use the practical ceiling so ml_norm reflects actual model confidence.
ML_PRACTICAL_CEIL = 1.5


def fuse_risk(
    predicted: float,
    rule_score: float,
    barangay_id: int = None,
    hazard_profile: dict = None,
    alpha: float = None,
    beta: float = None,
) -> float:
    """
    Adaptive fusion of three independent signals:
        1. CNN-LSTM prediction   (weather-driven, same across barangays)
        2. Rule-engine score     (weighted indicator sum, varies per barangay)
        3. GIS structural hazard (flood + surge scores, unique per barangay)

    Without signal 3, final_score was identical for every barangay when
    weather inputs are the same, because ml_score dominates and rule_score
    only shifts slightly. Adding the GIS term makes each barangay's
    structural risk directly visible in the final score so high-flood-zone
    barangays always rank higher than low-flood-zone ones.

    Workflow:
        1. Normalize all three signals to [0, 1]
        2. Compute adaptive three-way weights (alpha + beta + gamma = 1.0)
        3. Fuse and rescale output to [0, 3]

    Parameters
    ----------
    predicted     : CNN-LSTM prediction (0–3)
    rule_score    : Rule engine score (0–1)
    barangay_id   : Optional barangay identifier.
    hazard_profile: GIS-derived barangay hazard profile dict.
    alpha         : ML weight override.
    beta          : Rule-engine weight override.
    """

    # ── Compute adaptive three-way weights ──────────────────────────

    if alpha is None or beta is None:
        alpha, beta, gamma = _compute_fusion_weights(
            barangay_id,
            hazard_profile,
        )
    else:
        # Legacy two-weight override path — derive gamma from remainder
        total = alpha + beta
        gamma = max(0.0, 1.0 - total)
        if total > 1.0:
            # Caller passed weights that sum > 1; scale all three down
            alpha /= (total + gamma)
            beta  /= (total + gamma)
            gamma  = 0.0

    # ── Weight validation ────────────────────────────────────────────

    total_weight = alpha + beta + gamma

    if abs(total_weight - 1.0) > 1e-6:
        logger.warning(
            "Fusion weights do not sum to 1.0 "
            "(alpha=%.3f beta=%.3f gamma=%.3f). "
            "Normalizing automatically.",
            alpha, beta, gamma,
        )
        alpha /= total_weight
        beta  /= total_weight
        gamma /= total_weight

    # ── Normalize ML prediction → [0, 1] ────────────────────────────

    # Normalize against practical ceiling (not theoretical 3.0) so the
    # ML signal isn't compressed to ~28% while rule/GIS signals are at 90%+.
    # Once models are retrained on real GIS data, raise ML_PRACTICAL_CEIL
    # back toward MAX_ML_SCORE.
    _ml_ceil       = min(predicted, MAX_ML_SCORE)   # clamp first
    predicted_norm = max(0.0, min(1.0, _ml_ceil / ML_PRACTICAL_CEIL))

    # ── Normalize rule score → [0, 1] ───────────────────────────────

    if rule_score > MAX_RULE_SCORE:
        logger.warning(
            "Rule score exceeded %.1f "
            "(score=%.4f barangay_id=%s). "
            "Indicator weights may be invalid.",
            MAX_RULE_SCORE, rule_score, barangay_id,
        )

    rule_norm = max(0.0, min(1.0, rule_score / MAX_RULE_SCORE))

    # ── Normalize GIS structural hazard → [0, 1] ────────────────────
    # Combine flood and surge into a single structural hazard index.
    # Each is normalized against its own ceiling then averaged so
    # neither dominates unfairly (e.g. a surge-free inland barangay
    # is not penalized for having surge_score = 0).

    flood_score = float(
        (hazard_profile or {}).get("flood_hazard_score", 0.0)
    )
    surge_score = float(
        (hazard_profile or {}).get("storm_surge_score") or (hazard_profile or {}).get("surge_hazard_score") or 0.0
    )

    flood_norm = max(0.0, min(1.0, flood_score / MAX_FLOOD_SCORE))
    surge_norm = max(0.0, min(1.0, surge_score / MAX_SURGE_SCORE))

    # Average the two GIS components; if surge is zero (inland barangay)
    # the average still reflects the flood risk rather than collapsing to 0.
    hazard_norm = (flood_norm + surge_norm) / 2.0

    # ── Three-way adaptive fusion ────────────────────────────────────

    fused_norm = (
        (alpha * predicted_norm) +
        (beta  * rule_norm) +
        (gamma * hazard_norm)
    )

    # ── Rescale to 0–3 ───────────────────────────────────────────────

    fused_score = fused_norm * MAX_ML_SCORE

    # Final clamp for safety
    fused_score = max(0.0, min(MAX_ML_SCORE, fused_score))

    logger.debug(
        "Fusion | barangay=%s | "
        "pred=%.4f norm=%.4f | "
        "rule=%.4f norm=%.4f | "
        "flood=%.4f surge=%.4f hazard_norm=%.4f | "
        "alpha=%.2f beta=%.2f gamma=%.2f | "
        "fused=%.4f",
        barangay_id,
        predicted, predicted_norm,
        rule_score, rule_norm,
        flood_score, surge_score, hazard_norm,
        alpha, beta, gamma,
        fused_score,
    )

    return fused_score


def _compute_fusion_weights(
    barangay_id: int = None,
    hazard_profile: dict = None,
) -> tuple:
    """
    Computes adaptive three-way fusion weights (alpha, beta, gamma).

    Three signals
    -------------
    alpha : ML weight   — CNN-LSTM prediction
    beta  : Rule weight — rule-engine weighted indicators
    gamma : GIS weight  — structural flood + surge hazard

    Strategy
    --------
    HIGH hazard barangay:
        Structural GIS data is highly reliable → boost gamma.
        alpha=0.35  beta=0.45  gamma=0.20

    MODERATE hazard barangay:
        Balanced hybrid, GIS still meaningful.
        alpha=0.40  beta=0.45  gamma=0.15

    LOW hazard barangay:
        Weather anomalies dominate → trust ML more, reduce GIS term.
        alpha=0.50  beta=0.40  gamma=0.10

    Returns
    -------
    (alpha, beta, gamma) — always sums to 1.0
    """

    # ── Lazy-load hazard profile if missing ──────────────────────────

    if hazard_profile is None and barangay_id is not None:
        try:
            from modules.database import get_barangay_hazard_profile
            hazard_profile = get_barangay_hazard_profile(barangay_id)
        except Exception as exc:
            logger.warning(
                "Could not load hazard profile "
                "for barangay %s: %s",
                barangay_id, exc,
            )
            hazard_profile = {}

    overall = (
        hazard_profile.get("overall_hazard", "MODERATE")
        if hazard_profile
        else "MODERATE"
    )

    if overall == "HIGH":
        # Structural hazard is dominant — trust GIS + rules most
        alpha, beta, gamma = 0.35, 0.45, 0.20

    elif overall == "LOW":
        # Weather anomalies drive risk — trust ML more, downweight GIS
        alpha, beta, gamma = 0.50, 0.40, 0.10

    else:
        # MODERATE — balanced, GIS still contributes meaningfully
        alpha, beta, gamma = 0.40, 0.45, 0.15

    logger.debug(
        "Fusion weights | barangay=%s overall=%s | "
        "alpha=%.2f beta=%.2f gamma=%.2f",
        barangay_id, overall, alpha, beta, gamma,
    )

    return alpha, beta, gamma