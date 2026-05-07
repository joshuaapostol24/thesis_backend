"""
generate_barangay_training_data.py
───────────────────────────────────
Generates per-barangay training data from weather CSV files.
Flood and storm surge scores are pulled directly from PostgreSQL
(barangay_hazard_profile table — derived from shapefiles).
"""

import csv
from io import StringIO
import time

import requests
import psycopg2
from supabase import create_client

from ml_service_python.modules.config import (
    get_database_url,
    get_supabase_key,
    get_supabase_url,
)

# ── Supabase config ───────────────────────────────────────────────────────────
SUPABASE_URL = get_supabase_url()
SUPABASE_KEY = get_supabase_key()

# ── Local PostgreSQL ──────────────────────────────────────────────────────────
DATABASE_URL = get_database_url()

# ── Weather CSV files ─────────────────────────────────────────────────────────
WEATHER_FILES = [
    r"C:\Users\josua\OneDrive\Desktop\DATA_SET\weather_part_1.csv",
    r"C:\Users\josua\OneDrive\Desktop\DATA_SET\weather_part_2.csv",
    r"C:\Users\josua\OneDrive\Desktop\DATA_SET\weather_part_3.csv",
]

# ── Weights — MUST match context.py DEFAULT_WEIGHTS ──────────────────────────
WEIGHTS = {
    "rainfall":    0.35,
    "humidity":    0.10,
    "soil":        0.20,
    "flood":       0.20,
    "storm_surge": 0.15,
}

# ── PDF Validation weighted means (from expert validation document) ───────────
PDF_WEIGHTED_MEANS = {
    1:  {"rainfall": 1.52, "soil": 1.21, "flood": 1.18},
    2:  {"rainfall": 1.48, "soil": 1.16, "flood": 1.05},
    3:  {"rainfall": 1.61, "soil": 1.34, "flood": 1.42},
    4:  {"rainfall": 1.43, "soil": 1.08, "flood": 0.92},
    5:  {"rainfall": 1.50, "soil": 1.19, "flood": 1.10},
    6:  {"rainfall": 1.46, "soil": 1.12, "flood": 0.98},
    7:  {"rainfall": 1.54, "soil": 1.25, "flood": 1.22},
    8:  {"rainfall": 1.66, "soil": 1.49, "flood": 1.78},
    9:  {"rainfall": 1.65, "soil": 1.47, "flood": 1.74},
    10: {"rainfall": 1.63, "soil": 1.44, "flood": 1.69},
    11: {"rainfall": 1.62, "soil": 1.43, "flood": 1.66},
    12: {"rainfall": 1.60, "soil": 1.39, "flood": 1.58},
    13: {"rainfall": 1.58, "soil": 1.35, "flood": 1.49},
    14: {"rainfall": 1.57, "soil": 1.33, "flood": 1.44},
    15: {"rainfall": 1.56, "soil": 1.31, "flood": 1.39},
}

# ── Per-barangay rainfall sensitivity bounds ──────────────────────────────────
BARANGAY_RAINFALL_BOUNDS = {
    1: 40.0,  2: 40.0,  3: 40.0,  4: 40.0,  5: 40.0,
    6: 40.0,  7: 30.0,  8: 40.0,  9: 40.0, 10: 40.0,
   11: 40.0, 12: 40.0, 13: 40.0, 14: 40.0, 15: 30.0,
}

# ── Season and rainfall multipliers ──────────────────────────────────────────
SEASON_MULTIPLIER   = {"Wet Season": 1.20, "Summer": 1.00, "Dry Season": 0.85}
RAINFALL_MULTIPLIER = {"Heavy Rain": 1.30, "Moderate Rain": 1.10,
                        "Light Rain": 1.00, "No Rain": 0.90}


def load_hazard_profiles_from_db() -> dict:
    """
    Load flood_hazard_score and storm_surge_score directly from
    barangay_hazard_profile table — derived from shapefiles.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            SELECT barangay_id, flood_hazard_score, storm_surge_score
            FROM barangay_hazard_profile
            ORDER BY barangay_id
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"Direct Postgres failed ({exc}). Loading hazard profiles via Supabase API...")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        response = (
            supabase.table("barangay_hazard_profile")
            .select("barangay_id,flood_hazard_score,storm_surge_score")
            .order("barangay_id")
            .execute()
        )
        rows = [
            (
                row["barangay_id"],
                row.get("flood_hazard_score", 0),
                row.get("storm_surge_score", 0),
            )
            for row in response.data
        ]

    profiles = {}
    for bid, flood_score, storm_surge_score in rows:
        profiles[bid] = {
            "flood_score":        float(flood_score),
            "storm_surge_score":  float(storm_surge_score),
        }

    print("Loaded hazard profiles from DB:")
    for bid, p in profiles.items():
        print(f"  Barangay {bid:2d} | flood={p['flood_score']:.2f} | "
              f"storm_surge={p['storm_surge_score']:.2f}")
    return profiles


def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def pdf_sensitivity(barangay_id: int, indicator: str) -> float:
    """Convert PDF weighted mean to sensitivity multiplier (0.85-1.15)."""
    raw = PDF_WEIGHTED_MEANS.get(barangay_id, {}).get(indicator, 1.5)
    return 0.85 + ((raw - 1.0) / 1.0) * 0.30


def compute_soil_saturation(humidity: float, rainfall: float,
                             season: str, barangay_id: int) -> float:
    """Estimates soil saturation (0.0-3.0) per barangay."""
    base        = (humidity / 100.0) * 2.0
    rain_contrib = min(1.0, rainfall / 20.0)
    season_mod  = 1.1 if season == "Wet Season" else 0.9
    sens        = pdf_sensitivity(barangay_id, "soil")
    return round(min(3.0, (base + rain_contrib) * season_mod * sens), 4)


def compute_risk_label(rainfall, humidity, soil, flood_score, storm_surge_score,
                        season, rainfall_category, barangay_id) -> float:
    """
    Computes risk label [0.0-3.0] using:
    - Weights matching context.py
    - Flood and storm surge scores from shapefiles (via DB)
    - PDF sensitivity multipliers (per barangay unique)
    - Storm surge as a weighted feature
    """
    rainfall_max = BARANGAY_RAINFALL_BOUNDS.get(barangay_id, 40.0)

    r_norm = normalize(rainfall, 0.0, rainfall_max) * pdf_sensitivity(barangay_id, "rainfall")
    h_norm = normalize(humidity, 0.0, 100.0)
    s_norm = normalize(soil, 0.0, 3.0)              * pdf_sensitivity(barangay_id, "soil")
    f_norm = normalize(flood_score, 0.0, 1.0)       * pdf_sensitivity(barangay_id, "flood")
    ss_norm = normalize(storm_surge_score, 0.0, 1.0)

    base_score = (
        r_norm * WEIGHTS["rainfall"] +
        h_norm * WEIGHTS["humidity"] +
        s_norm * WEIGHTS["soil"] +
        f_norm * WEIGHTS["flood"] +
        ss_norm * WEIGHTS["storm_surge"]
    )

    season_mult     = SEASON_MULTIPLIER.get(season, 1.0)
    rainfall_mult   = RAINFALL_MULTIPLIER.get(rainfall_category, 1.0)

    label = base_score * season_mult * rainfall_mult * 3.0
    return round(min(3.0, max(0.0, label)), 4)


def load_weather_data() -> list:
    all_rows = []
    for path in WEATHER_FILES:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows.extend(list(reader))
    print(f"\nLoaded {len(all_rows):,} weather rows")
    return all_rows


def load_weather_data_from_urls(urls: list[str]) -> list:
    all_rows = []
    for url in urls:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        reader = csv.DictReader(StringIO(response.text))
        rows = list(reader)
        all_rows.extend(rows)
        print(f"Loaded {len(rows):,} weather rows from {url}")
    print(f"\nLoaded {len(all_rows):,} weather rows from URL source(s)")
    return all_rows


def load_weather_data_from_supabase_storage(bucket: str, paths: list[str]) -> list:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    all_rows = []
    for path in paths:
        data = supabase.storage.from_(bucket).download(path)
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        reader = csv.DictReader(StringIO(text))
        rows = list(reader)
        all_rows.extend(rows)
        print(f"Loaded {len(rows):,} weather rows from storage:{bucket}/{path}")
    print(f"\nLoaded {len(all_rows):,} weather rows from Supabase Storage")
    return all_rows


def load_weather_data_from_supabase_table(
    table: str = "weather_data",
    limit: int | None = None,
) -> list:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    page_size = 1000
    all_rows = []

    while True:
        start = len(all_rows)
        end = start + page_size - 1
        if limit is not None:
            remaining = limit - len(all_rows)
            if remaining <= 0:
                break
            end = start + min(page_size, remaining) - 1

        response = (
            supabase.table(table)
            .select("timestamp,rainfall,humidity,season,rainfall_category")
            .order("timestamp", desc=True)
            .range(start, end)
            .execute()
        )
        rows = response.data or []
        if not rows:
            break
        all_rows.extend(rows)
        print(f"Loaded {len(all_rows):,} weather rows from table:{table}")
        if len(rows) < page_size:
            break

    all_rows = list(reversed(all_rows))
    print(f"\nLoaded {len(all_rows):,} weather rows from Supabase table '{table}'")
    return all_rows


def generate_training_records(weather_rows: list, hazard_profiles: dict) -> list:
    records = []
    for row in weather_rows:
        try:
            rainfall  = float(row["rainfall"])
            humidity  = float(row["humidity"])
            season    = row["season"]
            rain_cat  = row["rainfall_category"]
            timestamp = row["timestamp"]

            for bid in range(1, 16):
                profile           = hazard_profiles[bid]
                flood_score       = profile["flood_score"]
                storm_surge_score = profile["storm_surge_score"]

                soil = compute_soil_saturation(humidity, rainfall, season, bid)

                risk_label = compute_risk_label(
                    rainfall, humidity, soil, flood_score, storm_surge_score,
                    season, rain_cat, bid
                )

                records.append({
                    "barangay_id":  bid,
                    "timestamp":    timestamp,
                    "rainfall":     round(rainfall, 4),
                    "humidity":     round(humidity, 4),
                    "soil":         round(soil, 4),
                    "flood":        flood_score,
                    "storm_surge":  storm_surge_score,
                    "risk_label":   risk_label,
                })
        except Exception:
            continue

    print(f"Generated {len(records):,} training records "
          f"({len(weather_rows):,} rows × 15 barangays)")
    return records


def balance_training_records(
    records: list,
    low_threshold: float = 1.0,
    max_low_per_barangay: int = 500,
) -> list:
    """
    Keep all moderate/high rows and downsample low-risk rows per barangay.

    Low rows are selected evenly through time instead of randomly so the LSTM
    still sees calm-weather history across the dataset span.
    """
    balanced = []
    by_barangay = {}
    for record in records:
        by_barangay.setdefault(record["barangay_id"], []).append(record)

    for barangay_id, barangay_records in sorted(by_barangay.items()):
        low_rows = []
        risk_rows = []
        for record in barangay_records:
            if float(record["risk_label"]) < low_threshold:
                low_rows.append(record)
            else:
                risk_rows.append(record)

        low_rows = sorted(low_rows, key=lambda row: row.get("timestamp") or "")
        sampled_low = _evenly_sample(low_rows, max_low_per_barangay)

        barangay_balanced = sorted(
            [*sampled_low, *risk_rows],
            key=lambda row: row.get("timestamp") or "",
        )
        balanced.extend(barangay_balanced)

        print(
            f"  Barangay {barangay_id:2d} | kept {len(sampled_low):,}/{len(low_rows):,} "
            f"LOW + {len(risk_rows):,} MOD/HIGH = {len(barangay_balanced):,}"
        )

    print(f"\nBalanced records: {len(records):,} -> {len(balanced):,}")
    return balanced


def _evenly_sample(rows: list, limit: int) -> list:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]

    last_index = len(rows) - 1
    return [
        rows[round(i * last_index / (limit - 1))]
        for i in range(limit)
    ]


def print_risk_distribution(records: list) -> None:
    labels = [float(r["risk_label"]) for r in records]
    if not labels:
        print("No records to summarize.")
        return

    low  = sum(1 for label in labels if label < 1.0)
    mod  = sum(1 for label in labels if 1.0 <= label < 2.0)
    high = sum(1 for label in labels if label >= 2.0)

    print(f"\nRisk distribution:")
    print(f"  LOW  (<1.0):    {low:,} ({low/len(labels)*100:.1f}%)")
    print(f"  MOD  (1.0-2.0): {mod:,} ({mod/len(labels)*100:.1f}%)")
    print(f"  HIGH (>=2.0):   {high:,} ({high/len(labels)*100:.1f}%)")
    print(f"  Min={min(labels):.4f} Max={max(labels):.4f} "
          f"Mean={sum(labels)/len(labels):.4f}")


def upload_to_supabase(records: list) -> None:
    supabase   = create_client(SUPABASE_URL, SUPABASE_KEY)
    BATCH_SIZE = 500
    total      = len(records)
    uploaded   = 0
    failed     = 0

    print(f"\nUploading {total:,} records in batches of {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            supabase.table("barangay_training_data").insert(batch).execute()
            uploaded += len(batch)
            print(f"  {uploaded:,} / {total:,} uploaded")
        except Exception as e:
            failed += len(batch)
            print(f"  Batch {i}-{i+BATCH_SIZE} failed: {e}")
        time.sleep(0.1)

    print(f"\nDone! Uploaded: {uploaded:,} | Failed: {failed:,}")


def clear_training_data() -> None:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase.table("barangay_training_data").delete().neq("id", 0).execute()
    print("Cleared existing barangay_training_data rows.")


if __name__ == "__main__":
    print("=" * 60)
    print("BARANGAY TRAINING DATA GENERATOR")
    print("Flood + Storm Surge from PostgreSQL shapefiles")
    print("=" * 60)

    print("\nSTEP 1: Run this SQL in Supabase SQL Editor first:")
    print("""
    DROP TABLE IF EXISTS barangay_training_data;

    CREATE TABLE barangay_training_data (
        id           SERIAL PRIMARY KEY,
        barangay_id  INTEGER NOT NULL,
        timestamp    TIMESTAMP,
        rainfall     FLOAT,
        humidity     FLOAT,
        soil         FLOAT,
        flood        FLOAT,
        storm_surge  FLOAT,
        risk_label   FLOAT
    );

    CREATE INDEX idx_training_barangay
    ON barangay_training_data(barangay_id);
    """)

    input("Press Enter once the table is created...")

    # Load hazard profiles from DB (flood + storm surge from shapefiles)
    hazard_profiles = load_hazard_profiles_from_db()

    # Load weather data
    weather_rows = load_weather_data()

    # Generate records
    records = generate_training_records(weather_rows, hazard_profiles)

    # Show distribution
    labels = [r['risk_label'] for r in records]
    low  = sum(1 for l in labels if l < 1.0)
    mod  = sum(1 for l in labels if 1.0 <= l < 2.0)
    high = sum(1 for l in labels if l >= 2.0)

    print(f"\nSample (first 3 barangays):")
    for r in records[:3]:
        print(f"  Barangay {r['barangay_id']:2d} | rainfall={r['rainfall']} | "
              f"soil={r['soil']:.2f} | flood={r['flood']} | "
              f"storm_surge={r['storm_surge']} | risk={r['risk_label']}")

    print(f"\nRisk distribution:")
    print(f"  LOW  (<1.0):    {low:,} ({low/len(labels)*100:.1f}%)")
    print(f"  MOD  (1.0-2.0): {mod:,} ({mod/len(labels)*100:.1f}%)")
    print(f"  HIGH (>=2.0):   {high:,} ({high/len(labels)*100:.1f}%)")
    print(f"  Min={min(labels):.4f} Max={max(labels):.4f} "
          f"Mean={sum(labels)/len(labels):.4f}")

    confirm = input("\nUpload to Supabase? (y/n): ")
    if confirm.lower() == "y":
        upload_to_supabase(records)
    else:
        print("Cancelled.")
