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

                "high":
                    len(self.high_risk_barangays),

                "moderate":
                    len(self.moderate_risk_barangays),

                "low":
                    len(self.low_risk_barangays),
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

    # ── Resolve barangay name ───────────────────────────────────────

    name = (
        hazard_profile.get("name")
        or f"Barangay {barangay_id}"
    )

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

    season = get_season_from_month(1)

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
            float(
                hazard_profile.get(
                    "flood_hazard_score",
                    0.20
                )
            ),

        "storm_surge":
            float(
                hazard_profile.get(
                    "storm_surge_score",
                    0.0
                )
            ),
    }

    # ── Context loading ─────────────────────────────────────────────

    (
        _,
        _,
        indicator_set,
        weights,
        rules,
    ) = load_context(
        HR,
        hazard_profile
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

    # ── Fusion ──────────────────────────────────────────────────────

    final_score = fuse_risk(

        ml_score,
        rule_score,

        barangay_id=barangay_id,
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