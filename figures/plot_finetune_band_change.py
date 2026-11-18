"""Relative change in cnn1d DFT-LRP band relevance after fine-tuning, female vs. male control.

Redraws paper figure finetune_band_change.png from the saved per-run band relevances
written by scripts/female_finetune_band_shift.py:

  in : results/female_finetune_band_shift/band_relevance_shift.csv
  out: finetune_band_change.{pdf,png} in figures/ and paper/figures/ (IEEE style)

Per (condition, seed, fold, band) the relative change is (after - before) / before, in
percent. Points are the mean over the 12 seed x fold runs of each condition; error bars
are their min/max (never pooled into a confidence interval).

    python figures/plot_finetune_band_change.py
"""
from __future__ import annotations

from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from figures import ieee_style as S  # noqa: E402

SHIFT_CSV = ROOT / "results" / "female_finetune_band_shift" / "band_relevance_shift.csv"
CONDITIONS = {"female": ("Women", S.FEMALE, "o", -25), "male_control": ("Male control", S.MALE, "s", 25)}


def main() -> None:
    df = pd.read_csv(SHIFT_CSV)
    assert df["faithfulness_gate_pass"].all(), "a run failed the faithfulness gate; do not interpret"
    df["rel_change_pct"] = 100.0 * (df["relevance_after"] - df["relevance_before"]) / df["relevance_before"]
    df["center"] = (df["band_lo"] + df["band_hi"]) / 2
    agg = (df.groupby(["condition", "center"])["rel_change_pct"]
             .agg(["mean", "min", "max", "count"]).reset_index())

    fig, ax = S.figure(3.1, 1.9)
    S.shade_speech_regions(ax)
    for cond, (label, color, marker, dx) in CONDITIONS.items():
        a = agg[agg["condition"] == cond].sort_values("center")
        ax.errorbar(a["center"] + dx, a["mean"], yerr=[a["mean"] - a["min"], a["max"] - a["mean"]],
                    color=color, marker=marker, ms=S.MARKERSIZE, label=label, **S.ERRORBAR)
        print(f"{cond}: band means {a['mean'].min():+.1f}% .. {a['mean'].max():+.1f}% "
              f"(runs per band: {int(a['count'].iloc[0])})")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(0, 4000)
    ax.set_ylim(-46, 40)
    ax.set_xlabel("Frequency band centre (Hz)")
    ax.set_ylabel("Relevance change (%)")
    ax.legend(loc="upper right", ncol=2)
    S.save(fig, "finetune_band_change")


if __name__ == "__main__":
    main()
