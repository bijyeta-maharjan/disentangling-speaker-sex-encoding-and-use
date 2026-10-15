"""Three split regimes, versioned to disk under splits/.

SPEAKER is the unit of analysis. The two speaker-level regimes partition by
*speaker* so train/val/test speaker sets are pairwise disjoint (no leakage). The
speaker_dependent regime deliberately violates that (utterance-level split) as a
memorization-inflated diagnostic for the leakage gap.

Regimes
-------
speaker_independent : StratifiedGroupKFold(groups=speaker, stratify=sex, 5 folds).
    One fold = TEST, a *different* fold = VAL (carved from the remaining training
    speakers, still speaker-disjoint & sex-stratified), the rest = TRAIN. The 12
    female speakers are spread across folds, so every test fold contains women.
speaker_dependent : random UTTERANCE split (60/20/20, stratified by digit) with
    the same proportions as speaker_independent; speakers appear on both sides.
sex_balanced : subset to all 12 female + N randomly-sampled male speakers, then a
    speaker_independent split on that subset.

Each regime is written to splits/<regime>.json and committed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from ..data.index import build_manifest
from ..data.metadata import load_metadata
from ..utils.config import load_config, paths


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def speaker_sex_table() -> pd.DataFrame:
    """One row per speaker: speaker, sex. Sorted by speaker."""
    meta = load_metadata()[["speaker", "gender"]].rename(columns={"gender": "sex"})
    return meta.sort_values("speaker").reset_index(drop=True)


def _fold_of_speaker(speakers: list[str], sex: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Assign each speaker to one fold, stratified by sex (each speaker its own group)."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X = np.arange(len(speakers)).reshape(-1, 1)
    fold_of: dict[str, int] = {}
    for fold_idx, (_, test_pos) in enumerate(sgkf.split(X, sex, groups=speakers)):
        for pos in test_pos:
            fold_of[speakers[pos]] = fold_idx
    return fold_of


def _roles_from_folds(fold_of: dict[str, int], test_fold: int, val_fold: int) -> dict[str, list[str]]:
    roles = {"train": [], "val": [], "test": []}
    for spk, fold in sorted(fold_of.items()):
        if fold == test_fold:
            roles["test"].append(spk)
        elif fold == val_fold:
            roles["val"].append(spk)
        else:
            roles["train"].append(spk)
    return roles


# --------------------------------------------------------------------------- #
# regimes
# --------------------------------------------------------------------------- #
def build_speaker_independent(cfg: dict, seed: int) -> dict:
    s = cfg["splits"]
    tbl = speaker_sex_table()
    speakers, sex = tbl["speaker"].tolist(), tbl["sex"].tolist()
    fold_of = _fold_of_speaker(speakers, sex, s["n_folds"], seed)
    roles = _roles_from_folds(fold_of, s["test_fold"], s["val_fold"])
    sex_of = dict(zip(speakers, sex))
    return {
        "regime": "speaker_independent",
        "seed": seed,
        "unit": "speaker",
        "stratify_by": "sex",
        "n_folds": s["n_folds"],
        "test_fold": s["test_fold"],
        "val_fold": s["val_fold"],
        "speaker_fold": {k: int(v) for k, v in sorted(fold_of.items())},
        "speaker_sex": sex_of,
        "roles": roles,
        "fold_sex_counts": _fold_sex_counts(fold_of, sex_of, s["n_folds"]),
    }


def _fold_sex_counts(fold_of, sex_of, n_folds) -> dict:
    out = {f: {"male": 0, "female": 0} for f in range(n_folds)}
    for spk, fold in fold_of.items():
        out[fold][sex_of[spk]] += 1
    return {str(k): v for k, v in out.items()}


def build_speaker_dependent(cfg: dict, seed: int) -> dict:
    """Random utterance split, 60/20/20, stratified by digit. Speakers on both sides."""
    s = cfg["splits"]
    manifest = build_manifest(paths(cfg).raw_root).reset_index(drop=True)
    idx = np.arange(len(manifest))
    digit = manifest["digit"].to_numpy()

    test_size = s["sd_test_size"]
    val_size = s["sd_val_size"]
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=digit, random_state=seed
    )
    # val_size is a fraction of the WHOLE set; convert to a fraction of trainval
    val_frac = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_frac, stratify=digit[trainval_idx], random_state=seed
    )
    return {
        "regime": "speaker_dependent",
        "seed": seed,
        "unit": "utterance",
        "stratify_by": s["sd_stratify"],
        "proportions": {
            "train": round(len(train_idx) / len(idx), 4),
            "val": round(len(val_idx) / len(idx), 4),
            "test": round(len(test_idx) / len(idx), 4),
        },
        "train_idx": sorted(int(i) for i in train_idx),
        "val_idx": sorted(int(i) for i in val_idx),
        "test_idx": sorted(int(i) for i in test_idx),
    }


def build_sex_balanced(cfg: dict, seed: int) -> dict:
    """12 female + N random male speakers, then a speaker_independent split on the subset."""
    s = cfg["splits"]
    tbl = speaker_sex_table()
    females = tbl.loc[tbl["sex"] == "female", "speaker"].tolist()
    males = tbl.loc[tbl["sex"] == "male", "speaker"].tolist()
    rng = np.random.default_rng(seed)
    males_sampled = sorted(rng.choice(males, size=s["sb_n_male"], replace=False).tolist())
    subset = sorted(females + males_sampled)
    sex_of = dict(zip(tbl["speaker"], tbl["sex"]))
    sub_sex = [sex_of[spk] for spk in subset]

    fold_of = _fold_of_speaker(subset, sub_sex, s["n_folds"], seed)
    roles = _roles_from_folds(fold_of, s["test_fold"], s["val_fold"])
    return {
        "regime": "sex_balanced",
        "seed": seed,
        "unit": "speaker",
        "stratify_by": "sex",
        "n_folds": s["n_folds"],
        "test_fold": s["test_fold"],
        "val_fold": s["val_fold"],
        "subset_speakers": subset,
        "n_female": len(females),
        "n_male_sampled": len(males_sampled),
        "speaker_fold": {k: int(v) for k, v in sorted(fold_of.items())},
        "speaker_sex": {spk: sex_of[spk] for spk in subset},
        "roles": roles,
        "fold_sex_counts": _fold_sex_counts(fold_of, sex_of, s["n_folds"]),
    }


BUILDERS = {
    "speaker_independent": build_speaker_independent,
    "speaker_dependent": build_speaker_dependent,
    "sex_balanced": build_sex_balanced,
}


def build_all(cfg: dict, seed: int | None = None) -> dict[str, Path]:
    seed = cfg["seed"] if seed is None else seed
    out_dir = paths(cfg).splits_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for regime, builder in BUILDERS.items():
        spec = builder(cfg, seed)
        path = out_dir / f"{regime}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2)
        written[regime] = path
        print(f"[splits] wrote {regime} -> {path}")
    return written


# --------------------------------------------------------------------------- #
# loading committed splits -> manifest-row indices for the Dataset
# --------------------------------------------------------------------------- #
def load_split_spec(cfg: dict, regime: str) -> dict:
    path = paths(cfg).splits_dir / f"{regime}.json"
    if not path.exists():
        raise FileNotFoundError(f"split file missing: {path}. Run scripts/make_splits.py")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_split_indices(cfg: dict, regime: str, cache: dict) -> dict[str, np.ndarray]:
    """Return {'train','val','test'} as manifest-row index arrays for `regime`.

    Speaker-level regimes map speaker roles -> rows via the cache's aligned
    speaker_str array. The utterance-level regime stores indices directly.
    """
    spec = load_split_spec(cfg, regime)
    if spec["unit"] == "utterance":
        return {
            "train": np.array(spec["train_idx"], dtype=np.int64),
            "val": np.array(spec["val_idx"], dtype=np.int64),
            "test": np.array(spec["test_idx"], dtype=np.int64),
        }
    speaker_str = np.asarray(cache["speaker_str"])
    out = {}
    for role, spk_list in spec["roles"].items():
        out[role] = np.where(np.isin(speaker_str, np.array(spk_list, dtype=object)))[0].astype(np.int64)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build & version all split regimes.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    build_all(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
