"""Thin CLI: build the preprocessed waveform cache.

    python scripts/preprocess.py [--config CONFIG] [--recompute]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import main

if __name__ == "__main__":
    main()
