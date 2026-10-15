"""Render the EDA figures into figures/output/.

Full set: run AFTER ``scripts/download_data.py`` and
``scripts/compute_acoustic_features.py`` (needs the AudioMNIST metadata).

    python figures/plot_eda.py

F0-by-sex only (paper figure f0_by_sex.png), from the saved per-utterance
acoustic features in results/eda/ -- no dataset needed:

    python figures/plot_eda.py --f0-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.index import build_manifest  # noqa: E402
from src.data.metadata import load_metadata  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.viz import eda  # noqa: E402

FEATURES_PATH = resolve("data/processed/acoustic_features.parquet")
SAVED_FEATURES_CSV = PROJECT_ROOT / "results" / "eda" / "acoustic_features.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f0-only", action="store_true",
                    help="only draw the F0-by-sex figure from results/eda/acoustic_features.csv")
    args = ap.parse_args()
    set_seed(42)
    if args.f0_only:
        feats = pd.read_csv(SAVED_FEATURES_CSV)
        eda.plot_f0_by_sex(feats)
        print(f"Wrote {PROJECT_ROOT / 'figures' / 'output' / 'eda_f0_by_sex.png'}")
        return 0
    manifest = build_manifest()
    speakers = load_metadata()
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Missing {FEATURES_PATH}. Run scripts/compute_acoustic_features.py first."
        )
    feats = pd.read_parquet(FEATURES_PATH)

    eda.plot_digit_balance(manifest)
    eda.plot_sex_distribution_speakers(speakers)
    eda.plot_accent_distribution_speakers(speakers)
    eda.plot_age_distribution_speakers(speakers)
    eda.plot_sex_accent_heatmap_speakers(speakers)
    eda.plot_accent_origin_heatmap_speakers(speakers)
    eda.plot_sex_agebin_heatmap_speakers(speakers)
    eda.plot_f0_by_sex(feats)
    eda.plot_centroid_by_sex(feats)
    eda.plot_duration_by_sex(feats)
    eda.plot_centroid_by_accent(feats)
    eda.plot_recordingdate_by_sex(speakers)

    out_dir = PROJECT_ROOT / "figures" / "output"
    written = sorted(out_dir.glob("eda_*.png"))
    print(f"Wrote {len(written)} figures to {out_dir}:")
    for p in written:
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
