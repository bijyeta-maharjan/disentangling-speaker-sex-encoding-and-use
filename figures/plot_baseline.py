"""Baseline cnn1d (seed 42): leakage gap and speaker-level accuracy by sex.

Redraws paper figures leakage_gap.png and accuracy_by_sex.png from the saved
evaluation summaries written by scripts/evaluate.py, using the same plotting
functions (src/fairness/plots.py):

  in : results/baseline/speaker_independent/eval_metrics.json
       results/baseline/speaker_dependent/eval_metrics.json
  out: figures/output/leakage_gap.png
       figures/output/accuracy_by_sex.png

    python figures/plot_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fairness.plots import plot_accuracy_by_sex, plot_leakage_gap  # noqa: E402

BASELINE = ROOT / "results" / "baseline"
OUT_DIR = ROOT / "figures" / "output"


def main() -> None:
    si = json.loads((BASELINE / "speaker_independent" / "eval_metrics.json").read_text())
    sd = json.loads((BASELINE / "speaker_dependent" / "eval_metrics.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_leakage_gap(si["overall_accuracy"], sd["overall_accuracy"], OUT_DIR / "leakage_gap.png")
    plot_accuracy_by_sex(si["accuracy_by_sex"], OUT_DIR / "accuracy_by_sex.png",
                         "Accuracy by sex (speaker-level) — speaker_independent")

    f, m = si["accuracy_by_sex"]["female"], si["accuracy_by_sex"]["male"]
    print(f"SI acc {si['overall_accuracy']:.4f} | SD acc {sd['overall_accuracy']:.4f} | "
          f"gap {sd['overall_accuracy'] - si['overall_accuracy']:+.4f}")
    print(f"female {f['point']:.3f} [{f['lo']:.3f}, {f['hi']:.3f}] (n={f['n_speakers']}) | "
          f"male {m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}] (n={m['n_speakers']})")
    print(f"wrote {OUT_DIR / 'leakage_gap.png'} and {OUT_DIR / 'accuracy_by_sex.png'}")


if __name__ == "__main__":
    main()
