"""Pin the male-only base checkpoint per (arch, seed) for the female fine-tune
localization round, and verify it is the EXACT model behind the existing
cross-sex behavioral result (results/cross_sex_generalization).

Multiple timestamped run dirs exist on disk for regime
cross_sex_male_train_female_test per (arch, seed) -- scripts/cross_sex_generalization.py
retrains from scratch every time it's invoked, and training is deterministic
(fixed seed, CPU), so re-runs produced numerically identical checkpoints. This
script (1) hashes every candidate checkpoint's state_dict to confirm that
determinism claim rather than assume it, (2) picks one canonical dir per
(arch, seed), and (3) re-runs inference with the pinned checkpoint on its own
female test set + male val set and compares the resulting accuracy to the
point estimate already recorded in results/cross_sex_generalization/results_overall.csv
-- the artifact that reports the cross-sex behavioral finding this whole round
is trying to localize. A mismatch means we would be fine-tuning from the wrong
model and must stop.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_cache
from src.fairness.bootstrap import bootstrap_speaker_ci
from src.models.factory import build_model, input_kind
from src.utils.config import load_config, resolve, resolve_arch_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
REGIME = "cross_sex_male_train_female_test"
ARCHS = ["cnn1d", "spec_cnn2d"]
SEEDS = [42, 43, 44]
CONFIG_PATH = "configs/week2_architectures.yaml"
OUT_PATH = resolve("results/female_finetune/pinned_checkpoints.json")


def state_dict_hash(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().cpu().numpy()
        h.update(k.encode())
        h.update(t.tobytes())
    return h.hexdigest()


def find_candidate_dirs(arch: str, seed: int) -> list[Path]:
    pattern = f"{REGIME}_seed{seed}_*"
    dirs = []
    for d in sorted((resolve("experiments")).glob(pattern)):
        summary_path = d / "train_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("arch") == arch and summary.get("seed") == seed:
            dirs.append(d)
    return dirs


def load_cache_for(cfg: dict) -> dict:
    cache = load_cache(cfg)
    if input_kind(cfg) == "spectrogram":
        from src.data.spectrogram import load_spec_cache

        cache["spec"] = load_spec_cache(cfg)["spec"]
    return cache


@torch.no_grad()
def infer_accuracy_by_speaker(model, cache, speakers, feature, batch_size, device):
    from src.data.dataset import AudioMNISTCacheDataset

    speaker_str = np.asarray(cache["speaker_str"])
    idx = np.where(np.isin(speaker_str, np.array(speakers, dtype=object)))[0].astype(np.int64)
    ds = AudioMNISTCacheDataset(cache, idx, feature=feature)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, ys, spk = [], [], []
    for i, (x, y, _spk_id) in enumerate(loader):
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    spk_str = speaker_str[idx]
    df = pd.DataFrame({"speaker": spk_str, "correct": (y_true == y_pred).astype(float)})
    per_spk = df.groupby("speaker")["correct"].mean().to_numpy()
    return bootstrap_speaker_ci(per_spk, seed=0)


def main() -> None:
    cross_sex_results = pd.read_csv(resolve("results/cross_sex_generalization/results_overall.csv"))
    cross_sex_results["seed_str"] = cross_sex_results["seed"].astype(str)

    pinned = {}
    mismatches = []

    for arch in ARCHS:
        cfg = resolve_arch_cfg(load_config(CONFIG_PATH), arch=arch)
        feature = input_kind(cfg)
        cache = load_cache_for(cfg)
        device = torch.device("cpu")

        for seed in SEEDS:
            candidates = find_candidate_dirs(arch, seed)
            if not candidates:
                raise FileNotFoundError(f"no candidate checkpoints found for {arch} seed {seed}")

            hashes = {}
            for d in candidates:
                ckpt = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
                hashes[str(d)] = state_dict_hash(ckpt["model_state"])
            unique_hashes = set(hashes.values())
            determinism_ok = len(unique_hashes) == 1
            print(f"[pin] {arch} seed{seed}: {len(candidates)} candidate dirs, "
                  f"{len(unique_hashes)} unique state_dict hash(es) -- "
                  f"{'DETERMINISTIC' if determinism_ok else 'MISMATCH -- NOT deterministic!'}")

            canonical_dir = sorted(candidates)[0]
            ckpt = torch.load(canonical_dir / "best.pt", map_location="cpu", weights_only=False)
            model = build_model(resolve_arch_cfg(load_config(CONFIG_PATH), arch=arch, seed=seed)).to(device)
            model.load_state_dict(ckpt["model_state"])

            with open(REPO_ROOT / "splits" / f"{REGIME}.json") as fh:
                split_spec = json.load(fh)
            female_test = split_spec["roles"]["test"]
            male_val = split_spec["roles"]["val"]

            female_ci = infer_accuracy_by_speaker(model, cache, female_test, feature, cfg["train"]["batch_size"], device)
            male_val_ci = infer_accuracy_by_speaker(model, cache, male_val, feature, cfg["train"]["batch_size"], device)

            ref_row_f = cross_sex_results[
                (cross_sex_results["arch"] == arch) & (cross_sex_results["seed_str"] == str(seed))
                & (cross_sex_results["condition"] == "male_only_on_female")
            ]
            ref_row_m = cross_sex_results[
                (cross_sex_results["arch"] == arch) & (cross_sex_results["seed_str"] == str(seed))
                & (cross_sex_results["condition"] == "male_only_on_male_val_ref")
            ]
            ref_f = float(ref_row_f["accuracy"].iloc[0])
            ref_m = float(ref_row_m["accuracy"].iloc[0])

            match_f = abs(female_ci["point"] - ref_f) < 1e-6
            match_m = abs(male_val_ci["point"] - ref_m) < 1e-6
            print(f"       female-eval: pinned={female_ci['point']:.6f} vs recorded={ref_f:.6f} "
                  f"{'OK' if match_f else 'MISMATCH'}")
            print(f"       male-val-ref: pinned={male_val_ci['point']:.6f} vs recorded={ref_m:.6f} "
                  f"{'OK' if match_m else 'MISMATCH'}")

            if not (determinism_ok and match_f and match_m):
                mismatches.append((arch, seed))

            pinned[f"{arch}_{seed}"] = {
                "arch": arch,
                "seed": seed,
                "canonical_checkpoint_dir": str(Path("experiments") / canonical_dir.relative_to(resolve("experiments"))),
                "state_dict_sha256": hashes[str(canonical_dir)],
                "n_candidate_dirs": len(candidates),
                "n_unique_hashes": len(unique_hashes),
                "determinism_verified": determinism_ok,
                "female_eval_pinned": female_ci,
                "female_eval_recorded_in_cross_sex_generalization": ref_f,
                "female_eval_match": match_f,
                "male_val_ref_pinned": male_val_ci,
                "male_val_ref_recorded_in_cross_sex_generalization": ref_m,
                "male_val_ref_match": match_m,
            }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(pinned, fh, indent=2, default=str)
    print(f"\n[pin] wrote {OUT_PATH}")

    if mismatches:
        print(f"\n[pin] *** MISMATCHES FOUND for {mismatches} -- DO NOT PROCEED to fine-tuning ***")
        sys.exit(1)
    print("\n[pin] all (arch, seed) pins verified deterministic and matching the cross-sex behavioral result.")


if __name__ == "__main__":
    main()
