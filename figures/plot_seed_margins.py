"""Trained-minus-untrained penultimate probe margins across seeds 42/43/44.

Redraws paper figure seed_margins.png (appendix, panel a) from the per-seed margin
tables written by scripts/week2_probe.py, and recomputes the cross-seed summary used
for Table II's margin columns. If the summary file already exists, the recomputed
values are checked against it (to 1e-12) instead of overwriting it:

  in : results/probing_penultimate/seed{42,43,44}/margins.csv
  out: results/probing_penultimate/multiseed_margin_summary.csv
       seed_margins.{pdf,png} in figures/ and paper/figures/ (IEEE style; bars = mean,
       error bars = seed range, markers = individual seeds)

    python figures/plot_seed_margins.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from figures import ieee_style as S  # noqa: E402

PROBE_DIR = ROOT / "results" / "probing_penultimate"
SUMMARY_CSV = PROBE_DIR / "multiseed_margin_summary.csv"

SEEDS = [42, 43, 44]
ARCH_ORDER = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
ATTRS = ["sex", "speaker_id"]
TITLES = {"sex": "Sex", "speaker_id": "Speaker identity"}


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

    fig, axes = S.figure(6.3, 2.0, 1, 2)
    x = np.arange(len(ARCH_ORDER))
    for ax, attr in zip(axes, ATTRS):
        s = summary[summary["attribute"] == attr].set_index("arch").loc[ARCH_ORDER]
        ax.bar(x, s["mean"], width=0.6, color=[S.MODEL_COLOR[a] for a in ARCH_ORDER], alpha=0.85,
               yerr=[s["mean"] - s["min"], s["max"] - s["mean"]], error_kw=dict(ecolor="black", **S.ERRORBAR))
        for i, arch in enumerate(ARCH_ORDER):
            seeds = margins[(margins["attribute"] == attr) & (margins["arch"] == arch)]["margin"]
            ax.plot(np.full(len(seeds), i + 0.38), seeds, ls="none", marker=S.MODEL_MARKER[arch],
                    ms=2.5, color=S.MODEL_COLOR[arch], mec="black", mew=0.3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, [S.MODEL_NAME[a].replace(" ", "\n", 1) if a != "crnn_gru" else "GRU-\nCRNN"
                          for a in ARCH_ORDER])
        ax.set_title(TITLES[attr])
        S.light_grid(ax)
    axes[0].set_ylabel("Trained \u2212 untrained accuracy")
    S.save(fig, "seed_margins")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
