"""Paper figures for the data section (IEEE style), plus the full exploratory set.

    python figures/plot_eda.py              # sex_accent_heatmap (needs the AudioMNIST metadata)
                                            #   + f0_by_sex (from results/eda/acoustic_features.csv)
    python figures/plot_eda.py --f0-only    # f0_by_sex only -- no dataset needed
    python figures/plot_eda.py --all-eda    # every exploratory figure into figures/output/
                                            #   (needs scripts/download_data.py and
                                            #    scripts/compute_acoustic_features.py first)

Paper figures are written as PDF + 600-dpi PNG into figures/ and paper/figures/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figures import ieee_style as S  # noqa: E402
from src.utils.config import resolve  # noqa: E402

FEATURES_PATH = resolve("data/processed/acoustic_features.parquet")
SAVED_FEATURES_CSV = PROJECT_ROOT / "results" / "eda" / "acoustic_features.csv"


def sex_accent_heatmap() -> None:
    """Speakers per sex x accent, from the AudioMNIST metadata (60 speakers)."""
    from src.data.metadata import load_metadata

    speakers = load_metadata()
    ct = pd.crosstab(speakers["accent"], speakers["gender"]).reindex(columns=["female", "male"], fill_value=0)
    ct = ct.sort_values(["male", "female"], ascending=False, kind="stable")
    ct.index = [a.replace("_", " ") for a in ct.index]

    fig, ax = S.figure(2.0, 2.0)
    vals = ct.to_numpy()
    ax.imshow(np.log1p(vals), cmap="Blues", aspect="auto", vmin=0, vmax=np.log1p(vals.max()) * 1.15)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, str(vals[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if vals[i, j] >= 8 else "black")
    ax.set_xticks([0, 1], ["Female", "Male"])
    ax.set_yticks(range(len(ct)), ct.index)
    ax.tick_params(length=0)
    ax.set_xlabel("Sex")
    ax.set_ylabel("Accent")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    S.save(fig, "sex_accent_heatmap")


def f0_by_sex(feats: pd.DataFrame) -> None:
    """Density of median F0 per utterance, by speaker sex."""
    feats = feats.dropna(subset=["f0_median_hz"])
    fig, ax = S.figure(3.1, 2.0)
    grid = np.linspace(50, 350, 400)
    for sex in ("male", "female"):
        f0 = feats.loc[feats["gender"] == sex, "f0_median_hz"].to_numpy()
        dens = gaussian_kde(f0)(grid)
        ax.fill_between(grid, dens, color=S.SEX_COLOR[sex], alpha=0.25, lw=0)
        ax.plot(grid, dens, color=S.SEX_COLOR[sex], label=S.SEX_LABEL[sex])
    ax.set_xlim(50, 350)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Median F0 per utterance (Hz)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    S.light_grid(ax, "x")
    S.save(fig, "f0_by_sex")


def all_eda() -> None:
    from src.data.index import build_manifest
    from src.data.metadata import load_metadata
    from src.utils.seeding import set_seed
    from src.viz import eda

    set_seed(42)
    manifest = build_manifest()
    speakers = load_metadata()
    if not FEATURES_PATH.exists():
        raise SystemExit(f"Missing {FEATURES_PATH}. Run scripts/compute_acoustic_features.py first.")
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
    print(f"Wrote {len(sorted(out_dir.glob('eda_*.png')))} exploratory figures to {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f0-only", action="store_true", help="only f0_by_sex, from results/eda/acoustic_features.csv")
    ap.add_argument("--all-eda", action="store_true", help="every exploratory figure into figures/output/")
    args = ap.parse_args()
    if args.all_eda:
        all_eda()
        return 0
    f0_by_sex(pd.read_csv(SAVED_FEATURES_CSV))
    if not args.f0_only:
        sex_accent_heatmap()
    return 0


if __name__ == "__main__":
    sys.exit(main())
