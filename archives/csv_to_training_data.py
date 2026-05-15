from __future__ import annotations

import pandas as pd

from sqlalchemy import text

from modules.database import engine
from modules.context import load_context
from modules.normalization import compute_weighted_scores
from modules.rule_engine import compute_risk_score
from modules.fusion import fuse_risk
from routes.prediction_routes import (
    compute_soil_saturation,
)

# ── CSV path ────────────────────────────────────────────────────────────────

CSV_PATH = r"C:\Users\josua\OneDrive\Desktop\migrating data\weather_barangay_training.csv"


# ── Risk label normalization ────────────────────────────────────────────────

RISK_MAP = {

    "LOW": 1.0,

    "MODERATE": 2.0,

    "HIGH": 3.0,

    "VERY HIGH": 3.0,
}


def main():

    print("\nLoading CSV...\n")

    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} rows")

    # ── Clean data ──────────────────────────────────────────────────

    df["rainfall"] = (
        pd.to_numeric(
            df["rainfall"],
            errors="coerce"
        )
        .fillna(0.0)
    )

    df["humidity"] = (
        pd.to_numeric(
            df["humidity"],
            errors="coerce"
        )
        .fillna(0.0)
    )

    # ── Load GIS hazard profiles ────────────────────────────────────

    with engine.connect() as conn:

        rows = conn.execute(text("""

            SELECT
                barangay_id,
                flood_hazard_score,
                storm_surge_score,
                overall_hazard

            FROM barangay_hazard_profile

        """)).fetchall()

    profiles = {

        row[0]: {

            "flood_hazard_score":
                float(row[1]),

            "storm_surge_score":
                float(row[2]),

            "overall_hazard":
                row[3],
        }

        for row in rows
    }

    inserts = []

    print("\nGenerating synchronized training rows...\n")

    for _, row in df.iterrows():

        barangay_id = int(
            row["barangay_id"]
        )

        profile = profiles.get(
            barangay_id
        )

        if not profile:
            continue

        rainfall = float(
            row["rainfall"]
        )

        humidity = float(
            row["humidity"]
        )

        season = str(
            row.get(
                "season",
                "Wet Season"
            )
        )

        # ── Dynamic soil computation ───────────────────────

        soil = compute_soil_saturation(

            humidity,
            rainfall,
            season
        )

        # ── Environmental context ──────────────────────────

        E = {

            "rainfall":
                rainfall,

            "humidity":
                humidity,

            "soil":
                soil,

            "flood":
                float(
                    profile[
                        "flood_hazard_score"
                    ]
                ),

            "storm_surge":
                float(
                    profile[
                        "storm_surge_score"
                    ]
                ),
        }

        # ── Hazard report ──────────────────────────────────

        HR = {

            "barangay_id":
                barangay_id,

            "location":
                row["barangay_name"],

            "type":
                "Flood",
        }

        # ── Load adaptive weights ──────────────────────────

        (
            _,
            _,
            _,
            weights,
            _,
        ) = load_context(
            HR,
            profile
        )

        # ── Rule engine ────────────────────────────────────

        weighted_scores = (
            compute_weighted_scores(

                HR,
                E,
                profile,
                weights,
                barangay_id,
            )
        )

        rule_score = compute_risk_score(
            weighted_scores
        )

        # ── ML approximation ───────────────────────────────

        base_ml = RISK_MAP.get(
            str(
                row.get(
                    "risk_level",
                    "LOW"
                )
            ).upper(),
            1.0
        )

        # ── Fusion ─────────────────────────────────────────

        final_risk = fuse_risk(

            predicted=base_ml,

            rule_score=rule_score,

            barangay_id=barangay_id,

            hazard_profile=profile,
        )

        risk_label = round(
            min(
                3.0,
                max(
                    0.0,
                    final_risk
                )
            ),
            4
        )

        inserts.append({

            "barangay_id":
                barangay_id,

            "timestamp":
                row["timestamp"],

            "rainfall":
                rainfall,

            "humidity":
                humidity,

            "soil":
                soil,

            "flood":
                E["flood"],

            "storm_surge":
                E["storm_surge"],

            "risk_label":
                risk_label,
        })

    print(
        f"\nPrepared {len(inserts)} rows"
    )

    # ── Clear old inconsistent rows ────────────────────────────────

    with engine.begin() as conn:

        conn.execute(text("""
            TRUNCATE TABLE
            barangay_training_data
        """))

        conn.execute(text("""

            INSERT INTO barangay_training_data (

                barangay_id,
                timestamp,

                rainfall,
                humidity,
                soil,

                flood,
                storm_surge,

                risk_label

            ) VALUES (

                :barangay_id,
                :timestamp,

                :rainfall,
                :humidity,
                :soil,

                :flood,
                :storm_surge,

                :risk_label
            )

        """), inserts)

    print("\nDONE.\n")
    print(
        "Training dataset synchronized "
        "with new architecture."
    )


if __name__ == "__main__":
    main()