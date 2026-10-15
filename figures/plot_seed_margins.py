"""Trained-minus-untrained penultimate probe margins across seeds 42/43/44.

Redraws paper figure seed_margins.png (appendix, panel a) from the per-seed margin
tables written by scripts/week2_probe.py, and recomputes the cross-seed summary used
for Table II's margin columns. If the summary file already exists, the recomputed
values are checked against it (to 1e-12) instead of overwriting it:

  in : results/probing_penultimate/seed{42,43,44}/margins.csv
  out: results/probing_penultimate/multiseed_margin_summary.csv
       figures/output/seed_margins.png

    python figures/plot_seed_margins.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "results" / "probing_penultimate"
SUMMARY_CSV = PROBE_DIR / "multiseed_margin_summary.csv"
OUT_PNG = ROOT / "figures" / "output" / "seed_margins.png"

SEEDS = [42, 43, 44]
ARCH_ORDER = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
ATTRS = ["sex", "speaker_id"]
COLORS = {"sex": "#7f8c8d", "speaker_id": "#c0392b"}


def main() -> None:
    frames = []
    for seed in SEEDS:
        df = pd.read_csv(PROBE_DIR / f"seed{seed}" / "margins.csv")
        frames.append(df[df["probe"] == "linear"].assign(seed=seed))
    margins = pd.concat(frames, ignore_index=True)

    rows = []
    for attr in ATTRS:
        for arch in ARCH_ORDER:
            m = margins[(margins["attribute"] == attr) & (margins["arch"] == arch)]["margin"]
            assert len(m) == len(SEEDS), f"{attr}/{arch}: expected {len(SEEDS)} seeds, got {len(m)}"
            mean = float(m.mean())
            rows.append({"attribute": attr, "arch": arch, "mean": mean, "min": float(m.min()),
                         "max": float(m.max()), "verdict": "induced" if mean > 0 else "inherited"})
    summary = pd.DataFrame(rows)
    if SUMMARY_CSV.exists():
        saved = pd.read_csv(SUMMARY_CSV)
        assert saved[["attribute", "arch", "verdict"]].equals(summary[["attribute", "arch", "verdict"]])
        assert np.allclose(saved[["mean", "min", "max"]], summary[["mean", "min", "max"]], rtol=0, atol=1e-12)
        print(f"recomputed summary matches {SUMMARY_CSV}")
    else:
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"wrote {SUMMARY_CSV}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, attr in zip(axes, ATTRS):
        s = summary[summary["attribute"] == attr].set_index("arch").loc[ARCH_ORDER]
        x = np.arange(len(ARCH_ORDER))
        ax.bar(x, s["mean"], color=COLORS[attr], width=0.8,
               yerr=[s["mean"] - s["min"], s["max"] - s["mean"]], capsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(ARCH_ORDER, rotation=20)
        ax.set_ylabel("trained − untrained probe acc")
        ax.set_title(f"{attr} margin (mean ± seed-range, n={len(SEEDS)} seeds)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
