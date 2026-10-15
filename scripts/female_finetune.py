"""Fine-tuning recovery: fine-tune the pinned male-only checkpoint on held-out
speakers (female condition) or a matched male-control pool, fixed-epoch, no
early stopping, and measure eval-speaker accuracy before vs. after.

Bounded claim (state on every artifact): fine-tuning shows what the model CAN
RECRUIT to help with a new group, not what the original model was using.

Male-control asymmetry (state on every artifact): the male-control pool is
NOT a clean analogue of the female pool. The male-only model already trained
on all 48 male speakers (38 train + 10 val); the 10 val-only males used here
had no gradient computed on their audio but DID influence checkpoint
selection (early-stop val accuracy) -- weak exposure, not true zero exposure.
This means:
  - If female and male-control signatures MATCH -> strong evidence for
    "generic new-speaker adaptation" (the match happens DESPITE the male
    pool's residual exposure biasing toward a DIFFERENCE, i.e. against this
    conclusion, which makes agreement more convincing, not less).
  - If female and male-control signatures DIFFER -> CANNOT cleanly claim
    "female-specific". The difference is confounded with familiar-vs-novel
    exposure, not just sex. Report as suggestive, control-confounded --
    never as a clean female-specific finding.

Usage:
  python scripts/female_finetune.py --mode pilot --epochs 2 3
  python scripts/female_finetune.py --mode full --epochs_locked 2
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import AudioMNISTCacheDataset
from src.data.preprocess import load_cache
from src.fairness.bootstrap import bootstrap_speaker_ci
from src.models.factory import build_model, input_kind
from src.utils.config import load_config, resolve, resolve_arch_cfg
from src.utils.experiment import git_sha
from src.utils.seeding import seed_worker, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/week2_architectures.yaml"
ARCHS = ["cnn1d", "spec_cnn2d"]
SEEDS = [42, 43, 44]

FEMALE_FOLDS_PATH = REPO_ROOT / "splits" / "female_finetune_localization_folds_v1.json"
MALE_FOLDS_PATH = REPO_ROOT / "splits" / "male_control_finetune_localization_folds_v1.json"
PINNED_PATH = resolve("results/female_finetune/pinned_checkpoints.json")
OUT_DIR = resolve("results/female_finetune")

# Fine-tune hyperparameters -- IDENTICAL across both conditions, all folds,
# archs and seeds (locked after the pilot). FT_EPOCHS is set by the pilot
# gate below before the full grid is run.
FT_LR = 1e-4
FT_WEIGHT_DECAY = 0.0
FT_BATCH_SIZE = 128
FT_EPOCHS_DEFAULT = 3  # overwritten by --epochs in pilot mode; locked value goes in run_meta

N_FLAG_FEMALE = "fine-tune female n<=9, eval female n=3 per fold -- within-sample, not a population claim"
N_FLAG_MALE = (
    "male-control fine-tune n<=8, eval n<=3 per fold, drawn from val-only "
    "(weak-exposure) males -- within-sample, control-confounded, not a "
    "clean zero-exposure analogue of the female pool"
)


def load_cache_for(cfg: dict) -> dict:
    cache = load_cache(cfg)
    if input_kind(cfg) == "spectrogram":
        from src.data.spectrogram import load_spec_cache

        cache["spec"] = load_spec_cache(cfg)["spec"]
    return cache


def indices_for_speakers(cache: dict, speakers: list[str]) -> np.ndarray:
    speaker_str = np.asarray(cache["speaker_str"])
    return np.where(np.isin(speaker_str, np.array(speakers, dtype=object)))[0].astype(np.int64)


def build_loader(cache, indices, batch_size, shuffle, feature, seed):
    ds = AudioMNISTCacheDataset(cache, indices, feature=feature)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                       worker_init_fn=None, generator=g, drop_last=False)


@torch.no_grad()
def eval_on_speakers(model, cache, speakers, feature, batch_size, device, seed=0) -> dict:
    idx = indices_for_speakers(cache, speakers)
    loader = build_loader(cache, idx, batch_size, False, feature, seed)
    model.eval()
    preds, ys = [], []
    for x, y, _spk in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    speaker_str = np.asarray(cache["speaker_str"])[idx]
    df = pd.DataFrame({"speaker": speaker_str, "correct": (y_true == y_pred).astype(float)})
    per_spk = df.groupby("speaker")["correct"].mean()
    ci = bootstrap_speaker_ci(per_spk.to_numpy(), seed=seed)
    return {"overall": ci, "per_speaker": per_spk.to_dict()}


def finetune_(model, cache, speakers, feature, epochs, lr, batch_size, weight_decay, seed, device) -> None:
    """Fine-tune `model` in place for a fixed number of epochs. No early
    stopping (locked pre-registration decision: fixed epochs for
    cross-condition comparability)."""
    idx = indices_for_speakers(cache, speakers)
    loader = build_loader(cache, idx, batch_size, True, feature, seed)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        correct = total = 0
        loss_sum = 0.0
        for x, y, _spk in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        print(f"    [finetune] epoch {epoch}/{epochs} train_loss={loss_sum/total:.4f} train_acc={correct/total:.4f}")


def load_pinned_model(arch: str, seed: int, device):
    with open(PINNED_PATH) as fh:
        pinned = json.load(fh)
    entry = pinned[f"{arch}_{seed}"]
    ckpt_dir = resolve(entry["canonical_checkpoint_dir"])
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = resolve_arch_cfg(load_config(CONFIG_PATH), arch=arch, seed=seed)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg, entry


def run_one(arch: str, seed: int, fold: dict, condition: str, epochs: int, device, ft_seed_offset=0) -> dict:
    """condition: 'female' or 'male_control'. Returns before/after eval dicts."""
    set_seed(seed + ft_seed_offset)
    model, cfg, pinned_entry = load_pinned_model(arch, seed, device)
    feature = input_kind(cfg)
    cache = load_cache_for(cfg)

    finetune_spk = fold["finetune"]
    eval_spk = fold["eval"]

    before = eval_on_speakers(model, cache, eval_spk, feature, FT_BATCH_SIZE, device, seed=seed)
    finetune_(model, cache, finetune_spk, feature, epochs, FT_LR, FT_BATCH_SIZE, FT_WEIGHT_DECAY,
              seed=seed + ft_seed_offset, device=device)
    after = eval_on_speakers(model, cache, eval_spk, feature, FT_BATCH_SIZE, device, seed=seed)

    delta_point = after["overall"]["point"] - before["overall"]["point"]
    return {
        "arch": arch, "seed": seed, "fold": fold["fold"], "condition": condition,
        "finetune_speakers": finetune_spk, "eval_speakers": eval_spk,
        "epochs": epochs, "lr": FT_LR, "batch_size": FT_BATCH_SIZE,
        "before": before["overall"], "after": after["overall"],
        "delta_point": delta_point,
        "base_checkpoint": pinned_entry["canonical_checkpoint_dir"],
    }


def pilot(epoch_options: list[int]) -> None:
    with open(FEMALE_FOLDS_PATH) as fh:
        female_spec = json.load(fh)
    fold0 = female_spec["folds"][0]
    device = torch.device("cpu")

    print(f"[pilot] fold0 female eval={fold0['eval']} finetune(n={fold0['n_finetune']})")
    results = []
    for epochs in epoch_options:
        print(f"\n[pilot] === cnn1d seed42 fold0 female, epochs={epochs} ===")
        res = run_one("cnn1d", 42, fold0, "female", epochs, device)
        print(f"[pilot] before={res['before']['point']:.4f} [{res['before']['lo']:.4f},{res['before']['hi']:.4f}]  "
              f"after={res['after']['point']:.4f} [{res['after']['lo']:.4f},{res['after']['hi']:.4f}]  "
              f"delta={res['delta_point']:+.4f}")
        results.append(res)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "pilot_epoch_gate.json", "w", encoding="utf-8") as fh:
        json.dump({"git_sha": git_sha(), "results": results}, fh, indent=2, default=str)
    print(f"\n[pilot] wrote {OUT_DIR / 'pilot_epoch_gate.json'}")


def run_full_grid(epochs: int) -> None:
    with open(FEMALE_FOLDS_PATH) as fh:
        female_spec = json.load(fh)
    with open(MALE_FOLDS_PATH) as fh:
        male_spec = json.load(fh)
    device = torch.device("cpu")

    all_results = []
    for arch in ARCHS:
        for seed in SEEDS:
            for fold in female_spec["folds"]:
                print(f"\n[finetune] {arch} seed{seed} fold{fold['fold']} FEMALE")
                res = run_one(arch, seed, fold, "female", epochs, device)
                all_results.append(res)
                print(f"  before={res['before']['point']:.4f} after={res['after']['point']:.4f} delta={res['delta_point']:+.4f}")
            for fold in male_spec["folds"]:
                print(f"\n[finetune] {arch} seed{seed} fold{fold['fold']} MALE_CONTROL")
                res = run_one(arch, seed, fold, "male_control", epochs, device)
                all_results.append(res)
                print(f"  before={res['before']['point']:.4f} after={res['after']['point']:.4f} delta={res['delta_point']:+.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "recovery_raw_results.json", "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n[finetune] wrote {OUT_DIR / 'recovery_raw_results.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--epochs", nargs="+", type=int, default=[2, 3], help="pilot: epoch options to try")
    ap.add_argument("--epochs_locked", type=int, default=FT_EPOCHS_DEFAULT, help="full: locked epoch count (the published run used 2, chosen by the pilot)")
    args = ap.parse_args()

    if args.mode == "pilot":
        pilot(args.epochs)
    else:
        run_full_grid(args.epochs_locked)


if __name__ == "__main__":
    main()
