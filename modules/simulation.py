"""
modules/simulation.py
─────────────────────
Runs the full hazard risk assessment pipeline against user-supplied
weather inputs (rainfall, humidity, wind_speed, temperature) without
writing anything to the database or sending any alerts.

Used by the website's simulation feature so admins / researchers can
answer "what would the system predict if conditions were X?"

Pipeline (mirrors main_api.py exactly):
    inputs
      ↓
    build E dict  (weather inputs + barangay static profile)
      ↓
    context.load_context()         → weights, rules
      ↓
    normalization.compute_weighted_scores()  → rule-based score
      ↓
    cnn_lstm.predict_risk()        → ML score
      ↓
    fusion.fuse_risk()             → final score
      ↓
    rule_engine.apply_rules()      → risk level (LOW / MODERATE / HIGH)
      ↓
    return SimulationResult per barangay

Scale notes
───────────
    The E dict passed to the rule engine uses:
        rainfall  — mm/h  (raw user input)
        soil      — index from static barangay profile (0–3)
        flood     — flood_hazard_score 0–1  (from barangay_hazard_profile DB)

    The E dict passed to cnn_lstm uses raw DB scale:
        flood       — 1.8 (Low) / 3.2 (Medium)
        storm_surge — 1.0 (no SSA) / 4.2 (SSA3)
    These come from the same static _BARANGAY_PROFILES used in weather_api.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Static per-barangay profiles ──────────────────────────────────────────────
# Mirrors weather_api._BARANGAY_PROFILES exactly.
# flood / storm_surge are on the raw CNN+LSTM DB scale (0–4 / 0–5).
_BARANGAY_PROFILES: Dict[int, dict] = {
    1:  {"name": "Balansay",    "soil": 2.07, "flood": 1.8, "storm_surge": 4.2},
    2:  {"name": "Fatima",      "soil": 2.03, "flood": 1.8, "storm_surge": 4.2},
    3:  {"name": "Payompon",    "soil": 2.10, "flood": 1.8, "storm_surge": 4.2},
    4:  {"name": "San Luis",    "soil": 2.22, "flood": 1.8, "storm_surge": 1.0},
    5:  {"name": "Talabaan",    "soil": 2.20, "flood": 1.8, "storm_surge": 4.2},
    6:  {"name": "Tangkalan",   "soil": 2.18, "flood": 3.2, "storm_surge": 1.0},
    7:  {"name": "Tayamaan",    "soil": 2.17, "flood": 3.2, "storm_surge": 4.2},
    8:  {"name": "Poblacion 1", "soil": 2.14, "flood": 1.8, "storm_surge": 4.2},
    9:  {"name": "Poblacion 2", "soil": 2.12, "flood": 1.8, "storm_surge": 4.2},
    10: {"name": "Poblacion 3", "soil": 2.09, "flood": 1.8, "storm_surge": 4.2},
    11: {"name": "Poblacion 4", "soil": 2.72, "flood": 1.8, "storm_surge": 4.2},
    12: {"name": "Poblacion 5", "soil": 1.96, "flood": 1.8, "storm_surge": 1.0},
    13: {"name": "Poblacion 6", "soil": 2.01, "flood": 1.8, "storm_surge": 4.2},
    14: {"name": "Poblacion 7", "soil": 2.47, "flood": 3.2, "storm_surge": 1.0},
    15: {"name": "Poblacion 8", "soil": 2.67, "flood": 3.2, "storm_surge": 4.2},
}

# flood_hazard_score on the 0–1 rule-engine scale
# matches barangay_hazard_profile.flood_hazard_score
_FLOOD_HAZARD_SCORE: Dict[int, float] = {
    1:  0.20, 2:  0.20, 3:  0.20, 4:  0.20,
    5:  0.20, 6:  0.60, 7:  0.60, 8:  0.20,
    9:  0.20, 10: 0.20, 11: 0.20, 12: 0.20,
    13: 0.20, 14: 0.20, 15: 0.60,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BarangaySimResult:
    barangay_id:   int
    barangay_name: str
    risk_level:    str          # LOW | MODERATE | HIGH
    final_score:   float        # fused score 0–3
    rule_score:    float        # rule engine score 0–1
    ml_score:      float        # CNN+LSTM score 0–3
    weights:       dict         # per-barangay indicator weights
    breakdown:     dict         # per-indicator normalized scores


@dataclass
class SimulationResult:
    # Inputs echoed back so the frontend can display them
    rainfall:    float
    humidity:    float
    wind_speed:  float
    temperature: float

    # Per-barangay results, sorted HIGH → MODERATE → LOW then by score desc
    barangays: List[BarangaySimResult] = field(default_factory=list)

    # Convenience summary
    @property
    def high_risk_barangays(self) -> List[BarangaySimResult]:
        return [b for b in self.barangays if b.risk_level == "HIGH"]

    @property
    def moderate_risk_barangays(self) -> List[BarangaySimResult]:
        return [b for b in self.barangays if b.risk_level == "MODERATE"]

    @property
    def low_risk_barangays(self) -> List[BarangaySimResult]:
        return [b for b in self.barangays if b.risk_level == "LOW"]

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON API responses."""
        _order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
        sorted_b = sorted(
            self.barangays,
            key=lambda b: (_order.get(b.risk_level, 9), -b.final_score),
        )
        return {
            "inputs": {
                "rainfall":    self.rainfall,
                "humidity":    self.humidity,
                "wind_speed":  self.wind_speed,
                "temperature": self.temperature,
            },
            "summary": {
                "high":     len(self.high_risk_barangays),
                "moderate": len(self.moderate_risk_barangays),
                "low":      len(self.low_risk_barangays),
            },
            "barangays": [
                {
                    "barangay_id":   b.barangay_id,
                    "barangay_name": b.barangay_name,
                    "risk_level":    b.risk_level,
                    "final_score":   round(b.final_score, 4),
                    "rule_score":    round(b.rule_score,  4),
                    "ml_score":      round(b.ml_score,    4),
                    "weights":       b.weights,
                    "breakdown":     {k: round(v, 4) for k, v in b.breakdown.items()},
                }
                for b in sorted_b
            ],
        }


# ── Core simulation logic ─────────────────────────────────────────────────────

def _simulate_barangay(
    barangay_id:    int,
    rainfall:       float,
    humidity:       float,
    wind_speed:     float,
    temperature:    float,
    hazard_profile: dict,
) -> BarangaySimResult:
    """
    Run the full pipeline for a single barangay with simulated inputs.
    hazard_profile must be the dict returned by
    database.get_barangay_hazard_profile(barangay_id).
    """
    from modules.context      import load_context
    from modules.normalization import compute_weighted_scores, get_indicator_bounds, normalize
    from modules.rule_engine  import compute_risk_score, apply_rules
    from modules.cnn_lstm     import predict_risk
    from modules.fusion       import fuse_risk

    profile = _BARANGAY_PROFILES.get(barangay_id, {})
    name    = profile.get("name", f"Barangay {barangay_id}")

    # ── Build HR (minimal — only fields the pipeline needs) ───────────────────
    HR = {
        "type":        "Simulation",
        "location":    name,
        "barangay_id": barangay_id,
        "rainfall":    rainfall,
        "humidity":    humidity,
        "isComplete":  True,
        "isVerified":  True,
        "verified":    1.0,
    }

    # ── Build E for rule engine (rule-engine flood scale 0–1) ─────────────────
    E_rule = {
        "rainfall":    rainfall,
        "humidity":    humidity,     # NEW
        "soil":        profile.get("soil", 2.0),
        "flood":       _FLOOD_HAZARD_SCORE.get(barangay_id, 0.20),
        "storm_surge": hazard_profile.get("storm_surge_score", 0.0) if hazard_profile else 0.0,  # NEW
    }

    # ── Build E for CNN+LSTM (raw DB scale) ───────────────────────────────────
    E_ml = {
        "rainfall":    rainfall,
        "humidity":    humidity,
        "soil":        profile.get("soil",        2.0),
        "flood":       profile.get("flood",        1.8),
        "storm_surge": profile.get("storm_surge",  1.0),
    }

    # ── Context → weights & rules ─────────────────────────────────────────────
    _, _, indicator_set, weights, rules = load_context(HR, hazard_profile)

    # ── Rule engine ───────────────────────────────────────────────────────────
    weighted_scores = compute_weighted_scores(
        HR, E_rule, hazard_profile, weights, barangay_id
    )
    rule_score = compute_risk_score(weighted_scores)

    # ── Per-indicator breakdown (for frontend display) ─────────────────────────
    breakdown = {}
    for indicator, raw_score in zip(indicator_set, weighted_scores):
        min_v, max_v = get_indicator_bounds(indicator, barangay_id)
        raw_val = E_rule.get(indicator, 0.0)
        norm    = normalize(raw_val, min_v, max_v)
        breakdown[indicator] = {
            "raw":    raw_val,
            "norm":   round(norm, 4),
            "weight": weights.get(indicator, 0.0),
            "score":  round(raw_score, 4),
        }

    # ── CNN+LSTM prediction ───────────────────────────────────────────────────
    # History is empty for simulation — the model falls back to zero-padded seq
    ml_score = predict_risk(barangay_id, E_ml, {})

    # ── Fusion ────────────────────────────────────────────────────────────────
    # OPTIMIZED: Use adaptive fusion weights per barangay hazard context
    final_score = fuse_risk(
        ml_score,
        rule_score,
        barangay_id=barangay_id,
        hazard_profile=hazard_profile
    )

    # ── Risk level ────────────────────────────────────────────────────────────
    risk_level = apply_rules(rules, final_score)

    logger.info(
        "Simulation | Barangay %02d %-12s | rule=%.4f ml=%.4f "
        "final=%.4f → %s",
        barangay_id, name, rule_score, ml_score, final_score, risk_level,
    )

    return BarangaySimResult(
        barangay_id   = barangay_id,
        barangay_name = name,
        risk_level    = risk_level,
        final_score   = final_score,
        rule_score    = rule_score,
        ml_score      = ml_score,
        weights       = weights,
        breakdown     = breakdown,
    )


def run_simulation(
    rainfall:    float,
    humidity:    float,
    wind_speed:  float,
    temperature: float,
    barangay_ids: Optional[List[int]] = None,
) -> SimulationResult:
    """
    Run a full hazard simulation across all (or selected) barangays.

    Parameters
    ----------
    rainfall     : mm/h   — e.g. 25.0
    humidity     : %      — e.g. 85.0
    wind_speed   : km/h   — informational, used for display + future features
    temperature  : °C     — informational, used for display + future features
    barangay_ids : list of barangay IDs to simulate (default: all 15)

    Returns
    -------
    SimulationResult — call .to_dict() for the JSON-serialisable version.

    Notes
    -----
    wind_speed and temperature are accepted and echoed back in the result
    but the current pipeline does not use them as scoring indicators.
    They are included in the input signature so the API contract is stable
    when the model is extended to use them in future iterations.
    """
    from modules.database import get_barangay_hazard_profile

    # Validate inputs
    rainfall    = max(0.0, float(rainfall))
    humidity    = max(0.0, min(100.0, float(humidity)))
    wind_speed  = max(0.0, float(wind_speed))
    temperature = float(temperature)

    target_ids = barangay_ids if barangay_ids else list(_BARANGAY_PROFILES.keys())

    results: List[BarangaySimResult] = []

    for bid in target_ids:
        try:
            hazard_profile = get_barangay_hazard_profile(bid)
            result = _simulate_barangay(
                barangay_id    = bid,
                rainfall       = rainfall,
                humidity       = humidity,
                wind_speed     = wind_speed,
                temperature    = temperature,
                hazard_profile = hazard_profile,
            )
            results.append(result)
        except Exception as e:
            logger.error("Simulation failed for barangay %d: %s", bid, e)

    return SimulationResult(
        rainfall    = rainfall,
        humidity    = humidity,
        wind_speed  = wind_speed,
        temperature = temperature,
        barangays   = results,
    )