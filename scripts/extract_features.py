"""Thin CLI: extract per-block GAP features for trained/untrained/raw.

    python scripts/extract_features.py [--config CONFIG] [--recompute]

Prints the alignment assertion result and the cache shapes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.probing.extract import run_extraction
from src.utils.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "day3_probing.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract GAP features for probing.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--recompute", action="store_true", help="rebuild even if cache exists")
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = run_extraction(cfg, recompute=args.recompute)
    print("\n=== extraction meta ===")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
