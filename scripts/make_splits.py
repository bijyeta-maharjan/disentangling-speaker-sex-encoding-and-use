"""Thin CLI: build & version all three split regimes to splits/.

    python scripts/make_splits.py [--config CONFIG] [--seed SEED]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits.make_splits import main

if __name__ == "__main__":
    main()
