from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.cnn_lstm import BARANGAY_IDS, _upload_weights_to_supabase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local CNN+LSTM weight files to Supabase Storage."
    )
    parser.add_argument(
        "--barangay-id",
        type=int,
        choices=BARANGAY_IDS,
        help="Upload only one barangay weight file. Defaults to all barangays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    barangay_ids = [args.barangay_id] if args.barangay_id else BARANGAY_IDS
    for barangay_id in barangay_ids:
        _upload_weights_to_supabase(barangay_id)


if __name__ == "__main__":
    main()
