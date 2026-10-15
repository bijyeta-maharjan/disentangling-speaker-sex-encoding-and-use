"""Build the same-size male-subset control split (used by scripts/cross_sex_controls.py).

Question this split serves: is the cross-sex behavioral gap about SEX, or just
about training on fewer/different speakers? Well-powered on males (n=9 held
out, vs. n=3 for the cross-sex female comparison).

Regime: male_subset_control_male_train_male_test
  test  = the 9 SI-regime test-fold MALE speakers (01,03,16,17,18,23,38,44,55)
    -- genuinely held out from BOTH this new model AND the pre-existing SI
    baseline (the SI baseline never trained on its own test fold, by
    definition), so this comparison needs NO memorization-confound fix, unlike
    the cross-sex female comparison.
  train = 38 speakers sampled from the 39 remaining (non-SI-test) male
    speakers -- SAME SIZE as the cross-sex split's male training set (38), so
    the only thing that changes vs. the cross-sex run is the SEX of the test
    set, not the training-set size.
  val   = the single leftover male speaker (39 - 38 = 1). Val only feeds early
    stopping bookkeeping, not the scientific comparison, so this thinness is a
    disclosed simplification, not hidden.

Fixed split (seed=42 for the train/val draw itself), reused across training
seeds 42/43/44 -- same convention as scripts/make_cross_sex_split.py.
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
N_TRAIN = 38  # matches splits/cross_sex_male_train_female_test.json's male train-set size


def build() -> dict:
    with open(Path(__file__).resolve().parents[1] / "splits" / "speaker_independent.json") as fh:
        si_spec = json.load(fh)
    si_test_males = sorted(
        s for s in si_spec["roles"]["test"] if si_spec["speaker_sex"][s] == "male"
    )
    assert len(si_test_males) == 9, f"expected 9 SI-test males, got {len(si_test_males)}"

    tbl = speaker_sex_table()
    all_males = sorted(tbl.loc[tbl["sex"] == "male", "speaker"].tolist())
    assert len(all_males) == 48
    remaining = sorted(set(all_males) - set(si_test_males))
    assert len(remaining) == 39

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(remaining).tolist()
    train = sorted(shuffled[:N_TRAIN])
    val = sorted(shuffled[N_TRAIN:])
    assert len(train) == N_TRAIN
    assert len(val) == 39 - N_TRAIN

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(si_test_males)
    assert set(val).isdisjoint(si_test_males)

    return {
        "regime": "male_subset_control_male_train_male_test",
        "split_seed": SPLIT_SEED,
        "unit": "speaker",
        "stratify_by": "none (all-male subset by construction)",
        "description": (
            "Same-size male->male control for the cross-sex generalization check. "
            "test = the 9 SI-regime test-fold male speakers (genuinely held "
            "out from both this model and the pre-existing SI baseline -- no "
            "memorization-confound fix needed here, unlike the cross-sex "
            "female comparison). train size (38) matches "
            "cross_sex_male_train_female_test.json's male training set "
            "exactly, so training-set SIZE is held fixed relative to the "
            "cross-sex run and only the test set's SEX differs."
        ),
        "n_male_total": 48,
        "n_si_test_males_held_out": len(si_test_males),
        "roles": {"train": train, "val": val, "test": si_test_males},
    }


def main() -> None:
    cfg = load_config()
    out_path = paths(cfg).splits_dir / "male_subset_control_male_train_male_test.json"
    spec = build()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"[make_male_subset_control_split] wrote {out_path}")
    print(f"  train (male): n={len(spec['roles']['train'])}  {spec['roles']['train']}")
    print(f"  val   (male): n={len(spec['roles']['val'])}  {spec['roles']['val']}")
    print(f"  test  (male, SI-held-out): n={len(spec['roles']['test'])}  {spec['roles']['test']}")


if __name__ == "__main__":
    main()
