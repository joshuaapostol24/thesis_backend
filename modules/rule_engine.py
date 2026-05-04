import logging

logger = logging.getLogger(__name__)


def compute_risk_score(weighted_scores: list) -> float:
    """Sums all weighted indicator scores into a single rule-based score."""
    score = sum(weighted_scores)
    logger.debug("Rule-based risk score: %.4f", score)
    return score


def apply_rules(rule_set: list, final_risk: float) -> str:
    """
    Matches the final risk score against the rule set and returns the
    corresponding action string (e.g. 'LOW', 'MODERATE', 'HIGH').

    Rules must have 'threshold' and 'action' keys and must be sorted by
    threshold descending (load_context() guarantees this).  The function
    iterates from highest to lowest threshold and returns the action for
    the first rule whose threshold is met.

    Exhaustion invariant
    ────────────────────
    The rule set loaded from context.py always includes a catch-all rule
    with threshold=0.0.  Because final_risk is a sum of non-negative
    weighted normalised values — and therefore always >= 0.0 — this rule
    will always match before the loop can exhaust.  The loop should never
    reach the end.

    The assertion below makes this invariant explicit and loud.  If it
    ever fires it means either:
      (a) the rule set was modified and the 0.0 catch-all was removed, or
      (b) final_risk is somehow negative (a bug upstream in normalization
          or fusion that should be surfaced immediately, not silently
          defaulted away).

    Both cases are programming errors, not runtime conditions, so an
    AssertionError is the right response — a silent default would mask
    the root cause.
    """
    for rule in rule_set:
        if final_risk >= rule["threshold"]:
            logger.info(
                "Rule matched: score=%.4f >= threshold=%.2f → %s",
                final_risk, rule["threshold"], rule["action"],
            )
            return rule["action"]

    # This line is unreachable given a well-formed rule set and non-negative
    # final_risk.  If it is ever reached, something has gone wrong upstream.
    raise AssertionError(
        f"apply_rules() exhausted all rules for final_risk={final_risk:.4f}. "
        f"Rule set is missing a 0.0 catch-all, or final_risk is negative. "
        f"Rule set: {rule_set}"
    )