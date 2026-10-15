"""Split-regime invariants. The speaker-disjointness check is a HARD STOP.

A fairness result is meaningless if speakers leak across train/val/test or if the
partition can't be reproduced from the seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits.make_splits import (
    build_sex_balanced,
    build_speaker_dependent,
    build_speaker_independent,
    speaker_sex_table,
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# (a) HARD STOP: speaker_independent train/val/test speaker sets pairwise DISJOINT
# --------------------------------------------------------------------------- #
def test_speaker_independent_pairwise_disjoint(cfg):
    spec = build_speaker_independent(cfg, cfg["seed"])
    train = set(spec["roles"]["train"])
    val = set(spec["roles"]["val"])
    test = set(spec["roles"]["test"])

    assert train and val and test, "every role must have speakers"
    leak_tv = train & val
    leak_tt = train & test
    leak_vt = val & test
    assert not leak_tv, f"SPEAKER LEAKAGE train∩val: {sorted(leak_tv)}"
    assert not leak_tt, f"SPEAKER LEAKAGE train∩test: {sorted(leak_tt)}"
    assert not leak_vt, f"SPEAKER LEAKAGE val∩test: {sorted(leak_vt)}"

    # the three roles must exactly cover all 60 speakers, no overlap
    all_speakers = set(speaker_sex_table()["speaker"])
    assert train | val | test == all_speakers
    assert len(train) + len(val) + len(test) == len(all_speakers)


def test_sex_balanced_pairwise_disjoint(cfg):
    spec = build_sex_balanced(cfg, cfg["seed"])
    train = set(spec["roles"]["train"])
    val = set(spec["roles"]["val"])
    test = set(spec["roles"]["test"])
    assert not (train & val), "leakage train∩val (sex_balanced)"
    assert not (train & test), "leakage train∩test (sex_balanced)"
    assert not (val & test), "leakage val∩test (sex_balanced)"
    # subset is exactly 12 female + N male
    assert spec["n_female"] == 12
    assert len(spec["subset_speakers"]) == 12 + spec["n_male_sampled"]


# --------------------------------------------------------------------------- #
# (b) every female speaker is assigned to a fold; each test fold gets some women
# --------------------------------------------------------------------------- #
def test_every_female_assigned_to_fold(cfg):
    spec = build_speaker_independent(cfg, cfg["seed"])
    tbl = speaker_sex_table()
    females = set(tbl.loc[tbl["sex"] == "female", "speaker"])
    assigned = set(spec["speaker_fold"].keys())
    assert females <= assigned, f"unassigned females: {sorted(females - assigned)}"
    # each of the 12 women lands in exactly one fold
    for f in females:
        assert isinstance(spec["speaker_fold"][f], int)


def test_women_spread_across_folds(cfg):
    """Stratified-by-sex => every fold (hence every possible test fold) has women."""
    spec = build_speaker_independent(cfg, cfg["seed"])
    counts = spec["fold_sex_counts"]
    assert len(counts) == spec["n_folds"]
    for fold, c in counts.items():
        assert c["female"] >= 1, f"fold {fold} has no women: {c}"
    # the chosen test fold contains women
    assert sum(1 for s in spec["roles"]["test"] if spec["speaker_sex"][s] == "female") >= 1


# --------------------------------------------------------------------------- #
# (c) reproducible from the seed
# --------------------------------------------------------------------------- #
def test_speaker_independent_reproducible(cfg):
    a = build_speaker_independent(cfg, cfg["seed"])
    b = build_speaker_independent(cfg, cfg["seed"])
    assert a["roles"] == b["roles"]
    assert a["speaker_fold"] == b["speaker_fold"]


def test_speaker_dependent_reproducible_and_disjoint_indices(cfg):
    a = build_speaker_dependent(cfg, cfg["seed"])
    b = build_speaker_dependent(cfg, cfg["seed"])
    assert a["train_idx"] == b["train_idx"]
    assert a["test_idx"] == b["test_idx"]
    # utterance indices are disjoint and cover the corpus
    tr, va, te = set(a["train_idx"]), set(a["val_idx"]), set(a["test_idx"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) + len(va) + len(te) == 30000


def test_speaker_dependent_has_shared_speakers(cfg):
    """Sanity: the diagnostic regime SHOULD share speakers across splits."""
    import numpy as np
    from src.data.index import build_manifest
    from src.utils.config import paths

    spec = build_speaker_dependent(cfg, cfg["seed"])
    manifest = build_manifest(paths(cfg).raw_root).reset_index(drop=True)
    spk = manifest["speaker"].to_numpy()
    train_spk = set(spk[np.array(spec["train_idx"])])
    test_spk = set(spk[np.array(spec["test_idx"])])
    assert train_spk & test_spk, "speaker_dependent must share speakers (it's the leaky diagnostic)"
