"""Thin CLI: build the mel-spectrogram cache for the spectrogram 2D CNN.

    python scripts/build_spectrograms.py [--config CONFIG] [--recompute]

Idempotent — skips if a cache built with the same mel params already exists.
Defaults to configs/week2_architectures.yaml (which carries the spectrogram block).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.spectrogram import build_spec_cache
from src.utils.config import load_config

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "week2_architectures.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the mel-spectrogram cache.")
    ap.add_argument("--config", default=str(DEFAULT))
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_spec_cache(cfg, recompute=args.recompute)


if __name__ == "__main__":
    main()
