"""Probe train/test row splits, enforcing discipline C.

GENERALIZABLE attrs (sex/accent/age) -> SPEAKER-DISJOINT split, reusing the
Day-2 speaker_independent partition:
    probe-train speakers = SI train + val speakers
    probe-test  speakers = SI test speakers
A probe trained this way must learn the CONCEPT (it never sees a test speaker),
not memorize individuals.

SPEAKER-ID (identity) -> per-speaker UTTERANCE split: the SAME speakers appear in
train and test (that is the point — it measures re-identifiability). This is the
labeled exception to the disjointness rule.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import train_test_split

from ..utils.config import resolve


def load_si_spec(cfg: dict) -> dict:
    regime = cfg["probe_split_regime"]
    path = resolve(cfg["paths"]["splits_dir"]) / f"{regime}.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generalizable_split(labels_df, attr_col: str, si_spec: dict):
    """Speaker-disjoint row positions for a generalizable attribute.

    Rows with a missing label (e.g. age dropped for speaker 45) are excluded
    from BOTH sides. Returns (train_pos, test_pos) as int arrays of row indices.
    """
    train_spk = set(si_spec["roles"]["train"]) | set(si_spec["roles"]["val"])
    test_spk = set(si_spec["roles"]["test"])
    valid = labels_df[attr_col].notna()
    spk = labels_df["speaker"].astype(str)
    train_pos = labels_df.index[valid & spk.isin(train_spk)].to_numpy()
    test_pos = labels_df.index[valid & spk.isin(test_spk)].to_numpy()
    # disjointness guard (speaker sets must not overlap)
    assert not (train_spk & test_spk), "SI train/val and test speakers overlap"
    return train_pos.astype(np.int64), test_pos.astype(np.int64)


def identity_split(labels_df, test_size: float, seed: int):
    """Per-speaker utterance split (stratified by speaker). All speakers on both
    sides — this measures re-identifiability, so it is NOT speaker-disjoint."""
    pos = labels_df.index.to_numpy()
    spk = labels_df["speaker_id"].to_numpy()
    train_pos, test_pos = train_test_split(
        pos, test_size=test_size, stratify=spk, random_state=seed
    )
    return np.sort(train_pos).astype(np.int64), np.sort(test_pos).astype(np.int64)
