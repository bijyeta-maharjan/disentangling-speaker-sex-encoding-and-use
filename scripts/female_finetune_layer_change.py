"""Layer localization: per-layer activation change under fine-tuning.

Control-independent question this answers directly: does fine-tuning the
male-only model to read a new group retune LOW-level layers (early conv
blocks -- acoustic/frequency-level retuning) or HIGH-level layers (late
conv blocks / penultimate -- decision-level retuning)? This does not require
the male control to be valid, because it is a property of the female
fine-tuned model alone (before vs. after, same eval speech).

Reporting frame: the female profile is
reported as descriptive localization -- "what the model changed to read
women" -- explicitly NOT as "female-specific". The male-control profile is
computed and reported as a secondary, confounded reference (see
female_finetune.py docstring for the two structural confounds:
novel-vs-familiar exposure, and ceiling/no-headroom). The female-minus-male
double-difference is computed and reported, but framed as uninterpretable
for specificity in EITHER direction: a difference could be the confounds,
not sex; agreement could be masked by the male pool's ceiling, not genuine
convergence. AudioMNIST has no clean alternative control (only 12 female
speakers total, all already used), so specificity is not resolvable on this
dataset -- stated explicitly in run-meta, not smoothed over.

Metric: per layer, a single relative L2 change over the eval set --
    relative_change = sqrt(sum_i ||act_after_i - act_before_i||^2)
                       / sqrt(sum_i ||act_before_i||^2)
computed in streaming fashion per batch (never materializing all per-sample
activations at once). Layers = each arch's declared intermediate_names (conv
blocks, in depth order) + "penultimate", normalized to depth in [0, 1].
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.female_finetune import (
    ARCHS, FEMALE_FOLDS_PATH, FT_BATCH_SIZE, FT_LR, FT_WEIGHT_DECAY,
    MALE_FOLDS_PATH, SEEDS, build_loader, finetune_, indices_for_speakers,
    load_cache_for, load_pinned_model,
)
from src.models.factory import input_kind
from src.utils.config import load_config, resolve, resolve_arch_cfg
from src.utils.experiment import git_sha
from src.utils.seeding import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = resolve("results/female_finetune")
FT_EPOCHS = 2  # locked by the fine-tuning pilot (pilot_epoch_gate.json)


def layer_names_for(arch: str) -> list[str]:
    cfg = resolve_arch_cfg(load_config("configs/week2_architectures.yaml"), arch=arch)
    from src.models.factory import build_model

    model = build_model(cfg)
    names = list(model.intermediate_names) + ["penultimate"]
    return names


@torch.no_grad()
def streaming_layer_relative_change(model_before, model_after, cache, speakers, feature, batch_size, device,
                                     layer_names) -> dict[str, float]:
    idx = indices_for_speakers(cache, speakers)
    loader = build_loader(cache, idx, batch_size, False, feature, seed=0)
    diff_sq = {ln: 0.0 for ln in layer_names}
    before_sq = {ln: 0.0 for ln in layer_names}
    model_before.eval()
    model_after.eval()
    for x, _y, _spk in loader:
        x = x.to(device)
        _, feats_before = model_before(x, return_features=True)
        _, feats_after = model_after(x, return_features=True)
        for ln in layer_names:
            fb = feats_before[ln].flatten(1).double()
            fa = feats_after[ln].flatten(1).double()
            diff_sq[ln] += (fa - fb).pow(2).sum().item()
            before_sq[ln] += fb.pow(2).sum().item()
    return {ln: float(np.sqrt(diff_sq[ln]) / (np.sqrt(before_sq[ln]) + 1e-12)) for ln in layer_names}


def run_one(arch: str, seed: int, fold: dict, condition: str, device, layer_names) -> dict:
    set_seed(seed)
    model_before, cfg, pinned_entry = load_pinned_model(arch, seed, device)
    model_after, _, _ = load_pinned_model(arch, seed, device)  # fresh copy of the SAME pinned weights
    feature = input_kind(cfg)
    cache = load_cache_for(cfg)

    finetune_spk = fold["finetune"]
    eval_spk = fold["eval"]

    finetune_(model_after, cache, finetune_spk, feature, FT_EPOCHS, FT_LR, FT_BATCH_SIZE, FT_WEIGHT_DECAY,
              seed=seed, device=device)

    rel_change = streaming_layer_relative_change(model_before, model_after, cache, eval_spk, feature,
                                                  FT_BATCH_SIZE, device, layer_names)
    return {
        "arch": arch, "seed": seed, "fold": fold["fold"], "condition": condition,
        "eval_speakers": eval_spk, "n_finetune": len(finetune_spk), "n_eval": len(eval_spk),
        "relative_change_by_layer": rel_change,
    }


def main() -> None:
    with open(FEMALE_FOLDS_PATH) as fh:
        female_spec = json.load(fh)
    with open(MALE_FOLDS_PATH) as fh:
        male_spec = json.load(fh)
    device = torch.device("cpu")

    all_results = []
    for arch in ARCHS:
        layer_names = layer_names_for(arch)
        print(f"\n[layer-change] {arch} layers (depth order): {layer_names}")
        for seed in SEEDS:
            for fold in female_spec["folds"]:
                print(f"[layer-change] {arch} seed{seed} fold{fold['fold']} FEMALE")
                res = run_one(arch, seed, fold, "female", device, layer_names)
                all_results.append(res)
                print(f"  {res['relative_change_by_layer']}")
            for fold in male_spec["folds"]:
                print(f"[layer-change] {arch} seed{seed} fold{fold['fold']} MALE_CONTROL")
                res = run_one(arch, seed, fold, "male_control", device, layer_names)
                all_results.append(res)
                print(f"  {res['relative_change_by_layer']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "layer_change_raw_results.json", "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n[layer-change] wrote {OUT_DIR / 'layer_change_raw_results.json'}")


if __name__ == "__main__":
    main()
