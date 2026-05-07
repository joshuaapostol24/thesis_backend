from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archives.generate_barangay_training_data import (
    balance_training_records,
    clear_training_data,
    generate_training_records,
    load_hazard_profiles_from_db,
    load_weather_data,
    load_weather_data_from_supabase_storage,
    load_weather_data_from_supabase_table,
    load_weather_data_from_urls,
    print_risk_distribution,
    upload_to_supabase,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a smaller balanced CNN+LSTM training dataset and upload it to Supabase."
    )
    parser.add_argument(
        "--weather-rows",
        type=int,
        default=10000,
        help="Use only the latest N weather rows before expanding to 15 barangays.",
    )
    parser.add_argument(
        "--max-low-per-barangay",
        type=int,
        default=500,
        help="Maximum LOW/no-risk rows to keep for each barangay.",
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Delete existing barangay_training_data rows before uploading.",
    )
    parser.add_argument(
        "--weather-url",
        action="append",
        default=[],
        help="Direct CSV URL. Pass multiple times for multiple weather CSVs.",
    )
    parser.add_argument(
        "--weather-bucket",
        help="Supabase Storage bucket containing the weather CSV files.",
    )
    parser.add_argument(
        "--weather-file",
        action="append",
        default=[],
        help="Path/name of a CSV inside --weather-bucket. Pass multiple times.",
    )
    parser.add_argument(
        "--weather-table",
        help="Supabase table containing weather rows, for example weather_data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hazard_profiles = load_hazard_profiles_from_db()
    if args.weather_table:
        weather_rows = load_weather_data_from_supabase_table(
            args.weather_table,
            limit=args.weather_rows if args.weather_rows > 0 else None,
        )
    elif args.weather_url:
        weather_rows = load_weather_data_from_urls(args.weather_url)
    elif args.weather_bucket and args.weather_file:
        weather_rows = load_weather_data_from_supabase_storage(
            args.weather_bucket,
            args.weather_file,
        )
    else:
        weather_rows = load_weather_data()
    if args.weather_rows > 0 and not args.weather_table:
        weather_rows = weather_rows[-args.weather_rows:]
        print(f"Using latest {len(weather_rows):,} weather rows.")

    records = generate_training_records(weather_rows, hazard_profiles)
    print_risk_distribution(records)

    balanced = balance_training_records(
        records,
        max_low_per_barangay=args.max_low_per_barangay,
    )
    print_risk_distribution(balanced)

    if args.clear_first:
        clear_training_data()

    upload_to_supabase(balanced)


if __name__ == "__main__":
    main()
