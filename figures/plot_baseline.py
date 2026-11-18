"""Baseline cnn1d (seed 42): leakage gap and speaker-level accuracy by sex (IEEE style).

Redraws paper figures leakage_gap and accuracy_by_sex from the saved evaluation
summaries written by scripts/evaluate.py:

  in : results/baseline/speaker_independent/{eval_metrics.json, per_speaker_accuracy.csv}
       results/baseline/speaker_dependent/eval_metrics.json
  out: leakage_gap.{pdf,png}, accuracy_by_sex.{pdf,png} in figures/ and paper/figures/

    python figures/plot_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from figures import ieee_style as S  # noqa: E402

BASELINE = ROOT / "results" / "baseline"


def leakage_gap(acc_si: float, acc_sd: float) -> None:
    fig, ax = S.figure(1.7, 1.8)
    x = np.arange(2)
    ax.bar(x, [acc_si, acc_sd], width=0.6, color=[S.NEUTRAL_DARK, S.NEUTRAL_LIGHT])
    for xi, v in zip(x, [acc_si, acc_sd]):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, ["Unseen\nspeakers", "Familiar\nspeakers"])
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Digit accuracy")
    S.light_grid(ax)
    S.save(fig, "leakage_gap")


def accuracy_by_sex(by_sex: dict, per_speaker: pd.DataFrame) -> None:
    fig, ax = S.figure(1.7, 1.8)
    rng = np.random.default_rng(0)
    for i, sex in enumerate(("female", "male")):
        b = by_sex[sex]
        acc = per_speaker.loc[per_speaker["sex"] == sex, "accuracy"].to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(acc)), acc, s=6, lw=0, alpha=0.45,
                   color=S.SEX_COLOR[sex], zorder=2)
        ax.errorbar(i + 0.28, b["point"], yerr=[[b["point"] - b["lo"]], [b["hi"] - b["point"]]],
                    fmt="o", ms=S.MARKERSIZE, color=S.SEX_COLOR[sex], zorder=3, **S.ERRORBAR)
    n = {s: by_sex[s]["n_speakers"] for s in ("female", "male")}
    ax.set_xticks([0.1, 1.1], [f"Female\n(n = {n['female']})", f"Male\n(n = {n['male']})"])
    ax.set_xlim(-0.45, 1.6)
    ax.set_ylim(0.7, 1.005)
    ax.set_ylabel("Speaker-level accuracy")
    S.light_grid(ax)
    S.save(fig, "accuracy_by_sex")


def main() -> None:
    si = json.loads((BASELINE / "speaker_independent" / "eval_metrics.json").read_text())
    sd = json.loads((BASELINE / "speaker_dependent" / "eval_metrics.json").read_text())
    per_spk = pd.read_csv(BASELINE / "speaker_independent" / "per_speaker_accuracy.csv")
    leakage_gap(si["overall_accuracy"], sd["overall_accuracy"])
    accuracy_by_sex(si["accuracy_by_sex"], per_spk)


if __name__ == "__main__":
    main()
