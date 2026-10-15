"""Build the matched MALE control folds for the female fine-tune localization round.

Companion to make_female_finetune_localization_folds.py. The control's job:
determine whether any frequency/layer shift seen under female fine-tuning is
female-SPECIFIC, by fine-tuning the SAME male-only checkpoint on held-out men
with the same fold structure, then differencing.

IMPORTANT PREMISE CHECK (read before trusting this file): the male-only model
(splits/cross_sex_male_train_female_test.json) was trained on ALL 48 male
speakers -- 38 in train (gradient updates), 10 in val (used only for
early-stopping/checkpoint selection, i.e. no gradient ever computed on their
audio). Zero male speakers have TRUE zero exposure to this checkpoint -- the
"largest matched design" the round's pre-registration anticipated for this
exact case draws from the 10 val-only males instead (weak exposure: model
SELECTION only, not weight updates). This is NOT the same guarantee as the
female pool (true zero exposure) and every artifact from the male-control
condition must say so.

Design: k=4 folds (matching the female k for side-by-side reporting) over the
10 val-only male speakers, partitioned as evenly as possible (eval sizes
3,3,2,2 -- 10 does not divide evenly by 4). Every one of the 10 appears in
eval exactly once. Fine-tune = the other val-only males in that fold
(sizes 7,7,8,8) -- smaller than the female fine-tune set (9) because only 10
candidates exist at all, not 12.
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
EVAL_GROUP_SIZES = [3, 3, 2, 2]  # sums to 10, matches k=4
VERSION = 1


def build() -> dict:
    cross_sex_path = Path(__file__).resolve().parents[1] / "splits" / "cross_sex_male_train_female_test.json"
    with open(cross_sex_path) as fh:
        cross_sex_spec = json.load(fh)
    train_males = set(cross_sex_spec["roles"]["train"])
    val_males = set(cross_sex_spec["roles"]["val"])

    tbl = speaker_sex_table()
    all_males = set(tbl.loc[tbl["sex"] == "male", "speaker"].tolist())
    assert len(all_males) == 48
    assert train_males | val_males == all_males, (
        "male-only model's train+val does not cover all 48 males -- "
        "re-check whether a truly-unseen male pool exists after all"
    )
    unseen_true = all_males - train_males - val_males
    assert len(unseen_true) == 0, (
        f"found {len(unseen_true)} truly unseen males ({sorted(unseen_true)}) -- "
        "premise assumed 0; re-run with the truly-unseen pool instead of val-only males"
    )

    val_only_males = sorted(val_males)
    assert sum(EVAL_GROUP_SIZES) == len(val_only_males), (
        f"eval group sizes {EVAL_GROUP_SIZES} sum to {sum(EVAL_GROUP_SIZES)}, "
        f"expected {len(val_only_males)} val-only males"
    )
    assert len(EVAL_GROUP_SIZES) == K_FOLDS

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(val_only_males).tolist()

    folds = []
    cursor = 0
    for i, size in enumerate(EVAL_GROUP_SIZES):
        eval_spk = sorted(shuffled[cursor:cursor + size])
        cursor += size
        finetune_spk = sorted(set(val_only_males) - set(eval_spk))
        assert set(eval_spk).isdisjoint(finetune_spk)
        folds.append({
            "fold": i,
            "finetune": finetune_spk,
            "eval": eval_spk,
            "n_finetune": len(finetune_spk),
            "n_eval": len(eval_spk),
        })

    all_eval_concat = sorted(s for f in folds for s in f["eval"])
    assert all_eval_concat == val_only_males, "eval roles do not partition all 10 val-only males exactly once"

    return {
        "regime": "male_control_finetune_localization_folds",
        "version": VERSION,
        "split_seed": SPLIT_SEED,
        "unit": "speaker",
        "k_folds": K_FOLDS,
        "base_model_split": "splits/cross_sex_male_train_female_test.json",
        "candidate_pool": "val_only_males_weak_exposure",
        "description": (
            "K-fold fine-tune/eval split over the 10 male speakers the male-only model "
            "used ONLY for early-stopping selection (val role: no gradient update on "
            "their audio, but their accuracy did influence which epoch's checkpoint was "
            "kept). This is the largest available matched pool -- the male-only model's "
            "train+val already covers all 48 male speakers, so ZERO males have true "
            "zero exposure (unlike the 12 women, all of whom are true zero-exposure). "
            "Eval sizes [3,3,2,2] (10 does not divide evenly by 4); fine-tune sizes "
            "[7,7,8,8] accordingly -- both smaller than the female design's clean 3/9 "
            "because the male candidate pool itself is smaller (10, not 12)."
        ),
        "n_male_total": len(all_males),
        "n_val_only_males": len(val_only_males),
        "n_truly_unseen_males": 0,
        "n_flag": (
            "male control uses WEAK-exposure (val-selection only) speakers, not true "
            "zero-exposure speakers like the female pool -- interpret any male-control "
            "shift as a conservative (likely underestimated) floor, not a clean analogue "
            "of the female condition"
        ),
        "folds": folds,
    }


def main() -> None:
    cfg = load_config()
    out_path = paths(cfg).splits_dir / "male_control_finetune_localization_folds_v1.json"
    spec = build()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"[make_male_control_finetune_localization_folds] wrote {out_path}")
    for f in spec["folds"]:
        print(f"  fold {f['fold']}: finetune(n={f['n_finetune']})={f['finetune']}  eval(n={f['n_eval']})={f['eval']}")


if __name__ == "__main__":
    main()
