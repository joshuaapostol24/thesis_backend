from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_service_python.modules.cnn_lstm import BARANGAY_IDS, retrain_barangay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN+LSTM barangay risk models from Supabase Postgres data."
    )
    parser.add_argument(
        "--barangay-id",
        type=int,
        choices=BARANGAY_IDS,
        help="Train only one barangay model. Defaults to all barangays.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    barangay_ids = [args.barangay_id] if args.barangay_id else BARANGAY_IDS
    for barangay_id in barangay_ids:
        retrain_barangay(barangay_id, force=True)


if __name__ == "__main__":
    main()
