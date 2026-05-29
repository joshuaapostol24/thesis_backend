"""
modules/simulation.py
─────────────────────
Runs the full hazard risk assessment pipeline against user-supplied
weather inputs without writing to the database.

Fully synchronized with:
    - prediction_routes.py
    - normalization.py
    - cnn_lstm.py
    - fusion.py
    - context.py
"""

from __future__ import annotations

import logging

from datetime import datetime

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    List,
    Optional,
)

logger = logging.getLogger(__name__)

# ── Shared soil logic ────────────────────────────────────────────────────────

from routes.prediction_routes import (
    compute_soil_saturation,
    get_season_from_month,
)

# ── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class BarangaySimResult:

    barangay_id: int
    barangay_name: str

    risk_level: str

    final_score: float
    rule_score: float
    ml_score: float

    weights: dict
    breakdown: dict


@dataclass
class SimulationResult:

    rainfall: float
    humidity: float
    wind_speed: float
    temperature: float

    barangays: List[BarangaySimResult] = field(
        default_factory=list
    )

    @property
    def very_high_risk_barangays(
        self
    ) -> List[BarangaySimResult]:

        return [
            b for b in self.barangays
            if b.risk_level == "VERY HIGH"
        ]

    @property
    def high_risk_barangays(
        self
    ) -> List[BarangaySimResult]:

        return [
            b for b in self.barangays
            if b.risk_level == "HIGH"
        ]

    @property
    def moderate_risk_barangays(
        self
    ) -> List[BarangaySimResult]:

        return [
            b for b in self.barangays
            if b.risk_level == "MODERATE"
        ]

    @property
    def low_risk_barangays(
        self
    ) -> List[BarangaySimResult]:

        return [
            b for b in self.barangays
            if b.risk_level == "LOW"
        ]

    @property
    def very_low_risk_barangays(
        self
    ) -> List[BarangaySimResult]:

        return [
            b for b in self.barangays
            if b.risk_level == "VERY LOW"
        ]

    def to_dict(self) -> dict:

        ordering = {
            "VERY HIGH": 0,
            "HIGH": 1,
            "MODERATE": 2,
            "LOW": 3,
            "VERY LOW": 4,
        }

        sorted_barangays = sorted(
            self.barangays,
            key=lambda b: (
                ordering.get(
                    b.risk_level,
                    99
                ),
                -b.final_score
            )
        )

        return {

            "inputs": {

                "rainfall":
                    self.rainfall,

                "humidity":
                    self.humidity,

                "wind_speed":
                    self.wind_speed,

                "temperature":
                    self.temperature,
            },

            "summary": {

                "very_high":
                    len(self.very_high_risk_barangays),

                "high":
                    len(self.high_risk_barangays),

                "moderate":
                    len(self.moderate_risk_barangays),

                "low":
                    len(self.low_risk_barangays),

                "very_low":
                    len(self.very_low_risk_barangays),
            },

            "barangays": [

                {

                    "barangay_id":
                        b.barangay_id,

                    "barangay_name":
                        b.barangay_name,

                    "risk_level":
                        b.risk_level,

                    "final_score":
                        round(
                            b.final_score,
                            4
                        ),

                    "rule_score":
                        round(
                            b.rule_score,
                            4
                        ),

                    "ml_score":
                        round(
                            b.ml_score,
                            4
                        ),

                    "weights":
                        b.weights,

                    "breakdown":
                        b.breakdown,
                }

                for b in sorted_barangays
            ],
        }


# ── Core simulation pipeline ────────────────────────────────────────────────

def _simulate_barangay(

    barangay_id: int,

    rainfall: float,
    humidity: float,

    wind_speed: float,
    temperature: float,

    hazard_profile: dict,

) -> BarangaySimResult:

    from modules.context import load_context

    from modules.normalization import (
        compute_weighted_scores,
        get_indicator_bounds,
        normalize,
    )

    from modules.rule_engine import (
        compute_risk_score,
        apply_rules,
    )

    from modules.cnn_lstm import (
        predict_risk
    )

    from modules.fusion import (
        fuse_risk
    )

    from modules.risk_adjustment import (
        apply_rainfall_adjustment
    )

    # ── Resolve barangay name ───────────────────────────────────────

    name = (
        hazard_profile.get("name")
        or f"Barangay {barangay_id}"
    )

    # ── Patch hazard_profile with fallback GIS scores ───────────────
    # Temporary: until gis_sync.py populates real values, inject a
    # per-barangay spread so context.py, fusion.py, and rule_score
    # all see different flood/surge values per barangay.
    # Remove this block once barangay_hazard_profile has real GIS data.

    # Temporary fallback until gis_sync.py populates real GIS values.
    # Flood scores: Low=1.0, Moderate=2.0, High=4.0  (FLOOD_LEVEL_SCORE scale)
    # Surge scores: None=0.0, SSA1=1.0, SSA2=2.0, SSA3=3.0, SSA4=4.0
    _FLOOD_FALLBACK = {
        1:  1.0,   # Balansay    - Low flood
        2:  1.0,   # Fatima      - Low flood
        3:  1.0,   # Payompon    - Low flood
        4:  1.0,   # San Luis    - Low flood
        5:  1.0,   # Talabaan    - Low flood
        6:  2.0,   # Tangkalan   - Moderate flood
        7:  2.0,   # Tayamaan    - Moderate flood  (HIGH overall)
        8:  1.0,   # Poblacion 1 - Low flood
        9:  1.0,   # Poblacion 2 - Low flood
        10: 1.0,   # Poblacion 3 - Low flood
        11: 1.0,   # Poblacion 4 - Low flood
        12: 1.0,   # Poblacion 5 - Low flood
        13: 1.0,   # Poblacion 6 - Low flood
        14: 1.0,   # Poblacion 7 - Low flood
        15: 2.0,   # Poblacion 8 - Moderate flood  (HIGH overall)
    }
    _SURGE_FALLBACK = {
        1:  1.0,   # Balansay    - SSA1
        2:  1.0,   # Fatima      - SSA1
        3:  1.0,   # Payompon    - SSA1
        4:  0.0,   # San Luis    - No surge
        5:  1.0,   # Talabaan    - SSA1
        6:  0.0,   # Tangkalan   - No surge
        7:  1.0,   # Tayamaan    - SSA1  (HIGH overall)
        8:  1.0,   # Poblacion 1 - SSA1
        9:  1.0,   # Poblacion 2 - SSA1
        10: 1.0,   # Poblacion 3 - SSA1
        11: 1.0,   # Poblacion 4 - SSA1
        12: 1.0,   # Poblacion 5 - SSA1
        13: 1.0,   # Poblacion 6 - SSA1
        14: 1.0,   # Poblacion 7 - SSA1
        15: 1.0,   # Poblacion 8 - SSA1  (HIGH overall)
    }
    _OVERALL_FALLBACK = {
        1:  "MODERATE",  # Balansay
        2:  "MODERATE",  # Fatima
        3:  "MODERATE",  # Payompon
        4:  "LOW",       # San Luis
        5:  "MODERATE",  # Talabaan
        6:  "LOW",       # Tangkalan
        7:  "HIGH",      # Tayamaan
        8:  "MODERATE",  # Poblacion 1
        9:  "MODERATE",  # Poblacion 2
        10: "MODERATE",  # Poblacion 3
        11: "MODERATE",  # Poblacion 4
        12: "MODERATE",  # Poblacion 5
        13: "MODERATE",  # Poblacion 6
        14: "MODERATE",  # Poblacion 7
        15: "HIGH",      # Poblacion 8
    }

    _raw_flood = hazard_profile.get("flood_hazard_score")
    _raw_surge = hazard_profile.get("storm_surge_score")

    if _raw_flood in (None, 0.0, 0.20):
        hazard_profile = {
            **hazard_profile,
            "flood_hazard_score": _FLOOD_FALLBACK.get(barangay_id, 1.0),
            "storm_surge_score":  _SURGE_FALLBACK.get(barangay_id, 0.0),
            "overall_hazard":     _OVERALL_FALLBACK.get(barangay_id, "MODERATE"),
        }

    # ── Build HR context ────────────────────────────────────────────

    HR = {

        "type":
            "Simulation",

        "location":
            name,

        "barangay_id":
            barangay_id,

        "isComplete":
            True,
    }

    # ── Dynamic soil computation ────────────────────────────────────

    season = get_season_from_month(
        datetime.utcnow().month
    )

    soil = compute_soil_saturation(
        humidity,
        rainfall,
        season
    )

    # ── Unified environmental context ───────────────────────────────

    E = {

        "rainfall":
            rainfall,

        "humidity":
            humidity,

        "soil":
            soil,

        "flood":
            float(hazard_profile.get("flood_hazard_score", 1.0)),

        "storm_surge":
            float(hazard_profile.get("storm_surge_score", 0.0)),

        "season":
            season,
    }

    # ── Context loading ─────────────────────────────────────────────
    # Pass rainfall + wind_speed so load_context computes the same
    # adaptive weights as prediction_routes.py — without these, every
    # barangay gets identical default weights regardless of conditions.

    (
        _,
        _,
        indicator_set,
        weights,
        rules,
    ) = load_context(
        HR,
        hazard_profile,
        rainfall=rainfall,
        wind_speed=wind_speed,
    )



    # ── Rule engine ─────────────────────────────────────────────────

    weighted_scores = compute_weighted_scores(

        HR,
        E,
        hazard_profile,
        weights,
        barangay_id,
    )

    rule_score = compute_risk_score(
        weighted_scores
    )

    # ── Indicator breakdown ─────────────────────────────────────────

    breakdown = {}

    for indicator, raw_score in zip(
        indicator_set,
        weighted_scores
    ):

        min_val, max_val = (
            get_indicator_bounds(
                indicator,
                barangay_id
            )
        )

        raw_value = E.get(
            indicator,
            0.0
        )

        normalized = normalize(
            raw_value,
            min_val,
            max_val
        )

        breakdown[indicator] = {

            "raw":
                raw_value,

            "normalized":
                round(
                    normalized,
                    4
                ),

            "weight":
                weights.get(
                    indicator,
                    0.0
                ),

            "score":
                round(
                    raw_score,
                    4
                ),
        }

    # ── CNN + LSTM prediction ──────────────────────────────────────

    ml_score = predict_risk(
        barangay_id,
        E,
        {}
    )

    # ── Hazard profile boost on ml_score ────────────────────────────
    # The model was trained on flat GIS data (flood≈0.20, surge≈0.00)
    # so it underestimates risk for high-hazard barangays.
    # We correct this by scaling ml_score up based on the barangay's
    # structural hazard, capped at MAX_ML_SCORE (3.0).
    #
    # Boost formula:
    #   flood_ratio  = flood_score  / MAX_FLOOD  (0.0 – 1.0)
    #   surge_ratio  = surge_score  / MAX_SURGE  (0.0 – 1.0)
    #   hazard_index = (flood_ratio + surge_ratio) / 2
    #
    # overall_hazard multiplier:
    #   HIGH     → up to +35% boost
    #   MODERATE → up to +20% boost
    #   LOW      → up to +10% boost
    #
    # Remove this block once models are retrained on real GIS data.

    _MAX_FLOOD  = 4.0
    _MAX_SURGE  = 4.0
    _MAX_ML     = 3.0

    _flood_ratio  = float(hazard_profile.get("flood_hazard_score", 0.0)) / _MAX_FLOOD
    _surge_ratio  = float(hazard_profile.get("storm_surge_score",  0.0)) / _MAX_SURGE
    _hazard_index = (_flood_ratio + _surge_ratio) / 2.0

    _boost_ceiling = {
        "HIGH":     0.35,
        "MODERATE": 0.20,
        "LOW":      0.10,
    }.get(hazard_profile.get("overall_hazard", "MODERATE"), 0.20)

    _ml_boost  = _hazard_index * _boost_ceiling
    ml_score   = min(_MAX_ML, ml_score * (1.0 + _ml_boost))

    logger.info(
        "Simulation | Barangay %02d %s | "
        "hazard_index=%.3f boost=+%.1f%% ml_score=%.4f",
        barangay_id, name,
        _hazard_index,
        _ml_boost * 100,
        ml_score,
    )

    # ── Fusion ──────────────────────────────────────────────────────

    final_score = fuse_risk(

        ml_score,
        rule_score,

        barangay_id=barangay_id,
        hazard_profile=hazard_profile,
    )

    final_score = apply_rainfall_adjustment(
        final_score,
        rainfall,
        barangay_id,
        ml_score=ml_score,
        rule_score=rule_score,
        hazard_profile=hazard_profile,
    )

    # ── Risk classification ────────────────────────────────────────

    risk_level = apply_rules(
        rules,
        final_score
    )

    logger.info(
        "Simulation | Barangay %02d "
        "%-15s | rule=%.4f "
        "ml=%.4f final=%.4f "
        "→ %s",
        barangay_id,
        name,
        rule_score,
        ml_score,
        final_score,
        risk_level,
    )

    return BarangaySimResult(

        barangay_id=barangay_id,

        barangay_name=name,

        risk_level=risk_level,

        final_score=final_score,

        rule_score=rule_score,

        ml_score=ml_score,

        weights=weights,

        breakdown=breakdown,
    )


# ── Public simulation API ───────────────────────────────────────────────────

def run_simulation(

    rainfall: float,
    humidity: float,

    wind_speed: float,
    temperature: float,

    barangay_ids: Optional[List[int]] = None,

) -> SimulationResult:

    from modules.database import (
        get_barangay_hazard_profile,
        list_barangay_profiles,
    )

    # ── Input validation ────────────────────────────────────────────

    rainfall = max(
        0.0,
        float(rainfall)
    )

    humidity = max(
        0.0,
        min(
            100.0,
            float(humidity)
        )
    )

    wind_speed = max(
        0.0,
        float(wind_speed)
    )

    temperature = float(
        temperature
    )

    # ── Resolve barangays ───────────────────────────────────────────

    if barangay_ids:

        target_ids = barangay_ids

    else:

        target_ids = [

            p["barangay_id"]

            for p in list_barangay_profiles()
        ]

    results = []

    # ── Run simulation ──────────────────────────────────────────────

    for barangay_id in target_ids:

        try:

            hazard_profile = (
                get_barangay_hazard_profile(
                    barangay_id
                )
            )

            result = _simulate_barangay(

                barangay_id=barangay_id,

                rainfall=rainfall,
                humidity=humidity,

                wind_speed=wind_speed,
                temperature=temperature,

                hazard_profile=hazard_profile,
            )

            results.append(result)

        except Exception as exc:

            logger.error(
                "Simulation failed for "
                "barangay %d: %s",
                barangay_id,
                exc
            )

    return SimulationResult(

        rainfall=rainfall,

        humidity=humidity,

        wind_speed=wind_speed,

        temperature=temperature,

        barangays=results,
    )
