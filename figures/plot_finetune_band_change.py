"""Relative change in cnn1d DFT-LRP band relevance after fine-tuning, female vs. male control.

Redraws paper figure finetune_band_change.png from the saved per-run band relevances
written by scripts/female_finetune_band_shift.py:

  in : results/female_finetune_band_shift/band_relevance_shift.csv
  out: figures/output/finetune_band_change.png

Per (condition, seed, fold, band) the relative change is (after - before) / before, in
percent. Points are the mean over the 12 seed x fold runs of each condition; error bars
are their min/max (never pooled into a confidence interval).

    python figures/plot_finetune_band_change.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHIFT_CSV = ROOT / "results" / "female_finetune_band_shift" / "band_relevance_shift.csv"
OUT_PNG = ROOT / "figures" / "output" / "finetune_band_change.png"

PITCH_BAND_HZ = (0, 250)
FORMANT_BAND_HZ = (250, 750)
STYLE = {"female": dict(color="tab:red", marker="o"), "male_control": dict(color="tab:blue", marker="s")}


def main() -> None:
    df = pd.read_csv(SHIFT_CSV)
    assert df["faithfulness_gate_pass"].all(), "a run failed the faithfulness gate; do not interpret"
    df["rel_change_pct"] = 100.0 * (df["relevance_after"] - df["relevance_before"]) / df["relevance_before"]
    df["center"] = (df["band_lo"] + df["band_hi"]) / 2
    agg = (df.groupby(["condition", "center"])["rel_change_pct"]
             .agg(["mean", "min", "max", "count"]).reset_index())

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.axvspan(*PITCH_BAND_HZ, color="0.85", alpha=0.4, lw=0)
    ax.axvspan(*FORMANT_BAND_HZ, color="tab:blue", alpha=0.08, lw=0)
    for cond in ["female", "male_control"]:
        a = agg[agg["condition"] == cond].sort_values("center")
        ax.errorbar(a["center"], a["mean"], yerr=[a["mean"] - a["min"], a["max"] - a["mean"]],
                    capsize=3, label=cond, **STYLE[cond])
        print(f"{cond}: band means {a['mean'].min():+.1f}% .. {a['mean'].max():+.1f}% "
              f"(runs per band: {int(a['count'].iloc[0])})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("frequency band center (Hz)")
    ax.set_ylabel("relative change in DFT-LRP relevance (%)")
    ax.set_title("cnn1d: band relevance change after fine-tuning\n(mean, min/max over seeds x folds)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
