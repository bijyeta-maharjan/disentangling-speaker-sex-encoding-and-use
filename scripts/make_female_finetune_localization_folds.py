"""Build the k-fold splits for the female fine-tune localization round.

Question this split serves: starting from the male-only model
(splits/cross_sex_male_train_female_test.json -- trained on all 48 male
speakers, ZERO female exposure), fine-tune it on a subset of women and ask
what changed. This file builds the FEMALE fine-tune/eval folds; the matched
MALE control folds are built by make_male_control_finetune_localization_folds.py.

Design: k=4 folds over the 12 female speakers. Each fold holds out 3 women
for evaluation (never fine-tuned on, that fold) and fine-tunes on the other
9. Every woman appears in the eval role in exactly one fold (a clean 4x3
partition of the 12, not resampling-with-replacement). Fine-tune/eval sets
are disjoint within every fold by construction.

n_flag: fine-tune female n<=9, eval female n=3 per fold -- within-sample,
not a population claim. See
results/cross_sex_generalization/run_meta.json for why all 12 women were
already established as the usable pool (none were seen by the male-only
model in train or val).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits.make_splits import speaker_sex_table
from src.utils.config import load_config, paths

SPLIT_SEED = 42
K_FOLDS = 4
VERSION = 1


def build() -> dict:
    with open(Path(__file__).resolve().parents[1] / "splits" / "cross_sex_male_train_female_test.json") as fh:
        cross_sex_spec = json.load(fh)
    seen_by_male_only_model = set(cross_sex_spec["roles"]["train"]) | set(cross_sex_spec["roles"]["val"])

    tbl = speaker_sex_table()
    all_females = sorted(tbl.loc[tbl["sex"] == "female", "speaker"].tolist())
    assert len(all_females) == 12, f"expected 12 female speakers, got {len(all_females)}"
    leaked = set(all_females) & seen_by_male_only_model
    assert not leaked, f"female speaker(s) seen by male-only model train/val: {sorted(leaked)}"

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(all_females).tolist()

    eval_group_size = len(all_females) // K_FOLDS
    assert eval_group_size * K_FOLDS == len(all_females), (
        f"{len(all_females)} females does not divide evenly into {K_FOLDS} folds of equal size"
    )

    folds = []
    for i in range(K_FOLDS):
        eval_spk = sorted(shuffled[i * eval_group_size:(i + 1) * eval_group_size])
        finetune_spk = sorted(set(all_females) - set(eval_spk))
        assert set(eval_spk).isdisjoint(finetune_spk)
        folds.append({
            "fold": i,
            "finetune": finetune_spk,
            "eval": eval_spk,
            "n_finetune": len(finetune_spk),
            "n_eval": len(eval_spk),
        })

    # every woman appears in eval exactly once across folds
    all_eval_concat = sorted(s for f in folds for s in f["eval"])
    assert all_eval_concat == all_females, "eval roles do not partition all 12 women exactly once"

    return {
        "regime": "female_finetune_localization_folds",
        "version": VERSION,
        "split_seed": SPLIT_SEED,
        "unit": "speaker",
        "k_folds": K_FOLDS,
        "base_model_split": "splits/cross_sex_male_train_female_test.json",
        "description": (
            "K-fold fine-tune/eval split over the 12 female speakers, for fine-tuning "
            "the male-only-trained model (zero prior female exposure, confirmed against "
            "sex metadata) on women and localizing what changed. Each fold: 9 women "
            "fine-tune (never eval'd that fold), 3 women eval (never fine-tuned on that "
            "fold, and untouched by any gradient update in that fold's run). Every woman "
            "is in the eval role in exactly one fold. NOT a population claim -- "
            "fine-tune female n<=9, eval female n=3 per fold, within-sample."
        ),
        "n_female_total": len(all_females),
        "n_flag": "fine-tune female n<=9, eval female n=3 per fold -- within-sample, not a population claim",
        "folds": folds,
    }


def main() -> None:
    cfg = load_config()
    out_path = paths(cfg).splits_dir / "female_finetune_localization_folds_v1.json"
    spec = build()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"[make_female_finetune_localization_folds] wrote {out_path}")
    for f in spec["folds"]:
        print(f"  fold {f['fold']}: finetune(n={f['n_finetune']})={f['finetune']}  eval(n={f['n_eval']})={f['eval']}")


if __name__ == "__main__":
    main()
