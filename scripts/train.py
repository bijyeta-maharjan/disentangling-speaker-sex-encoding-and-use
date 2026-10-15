"""Thin CLI: train one architecture for a given split regime + seed.

    python scripts/train.py [--config CONFIG] [--arch ARCH] [--seed SEED] \
                            [--split_regime REGIME] [--regime REGIME]

Week-2: ARCHITECTURE is the independent variable. ``--arch`` selects an entry
from the config's ``architectures:`` block (cnn1d | resnet1d | crnn_gru |
spec_cnn2d | tcn | conv_attn); everything else — preprocessing, the REUSED
Week-1 SI/SD split files, and the training protocol — is held fixed so runs are
comparable. ``--seed`` (42/43/44) and ``--split_regime`` override the config.

Back-compat: with no ``--arch`` (e.g. the Week-1 day2 config), behaves exactly
as before. ``--regime`` is kept as an alias for ``--split_regime``.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.trainer import train
from src.utils.config import load_config, resolve_arch_cfg

DEFAULT_WEEK2 = Path(__file__).resolve().parents[1] / "configs" / "week2_architectures.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one architecture under the shared protocol.")
    ap.add_argument("--config", default=None, help="config YAML (defaults to week2 when --arch is given)")
    ap.add_argument("--arch", default=None, help="architecture key from the config's architectures: block")
    ap.add_argument("--seed", type=int, default=None, help="override seed (42/43/44)")
    ap.add_argument("--split_regime", default=None, help="speaker_independent | speaker_dependent | sex_balanced")
    ap.add_argument("--regime", default=None, help="alias for --split_regime")
    args = ap.parse_args()

    config_path = args.config or (str(DEFAULT_WEEK2) if args.arch else None)
    cfg = load_config(config_path) if config_path else load_config()
    cfg = resolve_arch_cfg(cfg, arch=args.arch, seed=args.seed)

    regime = args.split_regime or args.regime or cfg["split_regime"]
    train(cfg, regime)


if __name__ == "__main__":
    main()
