"""Build the train=male / test=female split for the cross-sex generalization check
(behavioural follow-up to the "encoded vs used" analyses).

Regime: cross_sex_male_train_female_test
  test  = ALL 12 female speakers (dataset-wide, not just the SI test fold's 3).
  train/val = the 48 male speakers, split 38/10 (~80/20, mirroring the ~20% val
    fraction speaker_independent.json uses: 12/60). Carved by a fixed, seeded
    (seed=42) random sample -- this split construction itself is NOT re-drawn
    per training seed, matching this repo's existing convention of reusing one
    split across seeds 42/43/44.

Speaker-disjointness (train vs val vs test) is trivially satisfied since the
partition is by sex first (no speaker can be both male and female) and then by
a disjoint sample within the male pool.

Output: splits/cross_sex_male_train_female_test.json, same schema as
speaker_independent.json (unit="speaker", roles={train,val,test}) so it loads
through the existing src/splits/make_splits.py::load_split_indices unchanged --
no changes to train.py / evaluate.py / trainer.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits.make_splits import speaker_sex_table
from src.utils.config import load_config, paths

VAL_FRACTION_OF_MALE_POOL = 10  # ~20% of 48 male speakers, mirrors SI's 12/60 val fraction
SPLIT_SEED = 42


def build() -> dict:
    tbl = speaker_sex_table()  # columns: speaker, sex
    females = sorted(tbl.loc[tbl["sex"] == "female", "speaker"].tolist())
    males = sorted(tbl.loc[tbl["sex"] == "male", "speaker"].tolist())

    rng = np.random.default_rng(SPLIT_SEED)
    males_shuffled = rng.permutation(males).tolist()
    val_males = sorted(males_shuffled[:VAL_FRACTION_OF_MALE_POOL])
    train_males = sorted(males_shuffled[VAL_FRACTION_OF_MALE_POOL:])

    assert set(val_males).isdisjoint(train_males)
    assert set(train_males) | set(val_males) == set(males)
    assert set(females).isdisjoint(males)

    spec = {
        "regime": "cross_sex_male_train_female_test",
        "split_seed": SPLIT_SEED,
        "unit": "speaker",
        "stratify_by": "sex",
        "description": (
            "train/val = male speakers only; test = ALL 12 female speakers "
            "(dataset-wide). Deliberately NOT the SI regime's 3-female test "
            "fold -- see results/cross_sex_generalization/run_meta.json for "
            "why (maximizes female test n)."
        ),
        "n_female_total": len(females),
        "n_male_total": len(males),
        "roles": {
            "train": train_males,
            "val": val_males,
            "test": females,
        },
        "excluded_from_experiment": {
            "train_female_test_male": (
                "NOT RUN. <=12 female training speakers makes any accuracy "
                "drop a sample-size confound, not a sex effect -- excluded "
                "per pre-registration."
            ),
            "balanced_50_50_dataset": (
                "NOT BUILT. 12 female speakers cap any balanced design; "
                "rebuilding an elaborate design on an unfixable sample was "
                "excluded per pre-registration."
            ),
        },
    }
    return spec


def main() -> None:
    cfg = load_config()  # day2_baseline.yaml has no `architectures:` block; only used for paths
    out_path = paths(cfg).splits_dir / "cross_sex_male_train_female_test.json"
    spec = build()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"[make_cross_sex_split] wrote {out_path}")
    print(f"  train (male):  n={len(spec['roles']['train'])}  {spec['roles']['train']}")
    print(f"  val   (male):  n={len(spec['roles']['val'])}  {spec['roles']['val']}")
    print(f"  test  (female): n={len(spec['roles']['test'])}  {spec['roles']['test']}")


if __name__ == "__main__":
    main()
