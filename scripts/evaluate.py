"""Thin CLI: evaluate a trained run on its test set.

    python scripts/evaluate.py [--config CONFIG] [--regime REGIME] [--run-dir DIR]

Defaults to the latest run for the regime. Prints a concise summary including
per-sex accuracy with speaker-level bootstrap CIs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fairness.evaluate import evaluate_run
from src.utils.config import load_config


def _fmt_by_sex(by_sex: dict) -> str:
    parts = []
    for sex, b in sorted(by_sex.items()):
        parts.append(f"{sex}: {b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] (n={b['n_speakers']})")
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained run.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    regime = args.regime or cfg["split_regime"]

    s = evaluate_run(cfg, regime, args.run_dir)
    print(f"\n===== EVAL: {regime} =====")
    print(f"run: {s['run_dir']}")
    print(f"test: {s['n_test_utterances']} utterances / {s['n_test_speakers']} speakers")
    print(f"overall accuracy: {s['overall_accuracy']:.4f}   macro-F1: {s['macro_f1']:.4f}")
    print(f"accuracy by sex (per-speaker mean, 95% CI): {_fmt_by_sex(s['accuracy_by_sex'])}")
    print(f"worst speaker: id={s['worst_speaker']['speaker_id']} "
          f"acc={s['worst_speaker']['accuracy']:.3f} sex={s['worst_speaker']['sex']}")
    print(f"worst group: {s['worst_group']['sex']} acc={s['worst_group']['point']:.3f}")
    print(f"ECE overall: {s['ece_overall']:.4f}  by group: "
          + ", ".join(f"{k}={v:.4f}" for k, v in s['ece_by_group'].items() if k != 'overall'))


if __name__ == "__main__":
    main()
