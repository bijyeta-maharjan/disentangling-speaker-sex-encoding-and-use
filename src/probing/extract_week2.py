"""Week-2 per-architecture feature extraction for cross-architecture probing.

Reuses the Day-3 disciplines wholesale (does NOT rebuild the probe harness):
  * ordered, paired extraction: shuffle=False over indices=arange(N); the label
    table is built in the SAME cache order; digit/speaker are re-read FROM the
    loader and asserted row-for-row equal to the label table before caching.
  * the shared `penultimate` embedding (pooled vector into the classifier head)
    is the apples-to-apples cross-architecture probing point — extracted for
    EVERY architecture.
  * conv-family archs (cnn1d, resnet1d, spec_cnn2d) additionally GAP-pool each
    intermediate stage over space/time -> [N, C] for the funnel comparison.

For each architecture we extract features from TWO models through the identical
pipeline:
  * trained  : the seed-42 SI checkpoint (from the training sweep).
  * untrained: the same architecture, random init under seed 42, frozen, eval.
The untrained control is what isolates *training's* effect per architecture.

Idempotent: caches to data/processed/activations_week2/ and skips on re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import AudioMNISTCacheDataset
from ..data.preprocess import load_cache
from ..models.factory import build_model, input_kind
from ..utils.config import REPO_ROOT, resolve, resolve_arch_cfg
from ..utils.experiment import git_sha
from ..utils.seeding import set_seed
from .labels import build_label_table

TRAINED, UNTRAINED = "trained", "untrained"
PENULTIMATE = "penultimate"
# architectures with a clear depth sequence -> intermediates extracted for funnel
CONV_FAMILY = {"cnn1d", "resnet1d", "spec_cnn2d"}


# --------------------------------------------------------------------------- #
# checkpoint discovery (authoritative: the training sweep's results log)
# --------------------------------------------------------------------------- #
def si_seed42_checkpoints(cfg: dict) -> dict[str, Path]:
    """Map arch -> seed-42 SI best.pt, read from experiments/week2_sweep/results.jsonl."""
    results = resolve(cfg["sweep_results"])
    out: dict[str, Path] = {}
    with open(results, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["regime"] == "speaker_independent" and r["seed"] == 42:
                out[r["arch"]] = resolve(r["run_dir"]) / "best.pt"
    return out


# --------------------------------------------------------------------------- #
# cache layout / idempotency
# --------------------------------------------------------------------------- #
def activations_dir(cfg: dict) -> Path:
    d = resolve(cfg["paths"]["activations_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def feature_path(cfg: dict, arch: str, model: str, key: str) -> Path:
    return activations_dir(cfg) / f"{arch}__{model}__{key}.npy"


def labels_path(cfg: dict) -> Path:
    return activations_dir(cfg) / "labels.parquet"


def meta_path(cfg: dict) -> Path:
    return activations_dir(cfg) / "extract_meta.json"


def feature_keys_for(arch: str) -> list[str]:
    """penultimate (always) + intermediate stage names (conv-family only)."""
    keys = [PENULTIMATE]
    return keys  # intermediates appended at extraction time from the model itself


def arch_complete(cfg: dict, arch: str, intermediate_names: list[str]) -> bool:
    keys = [PENULTIMATE] + (intermediate_names if arch in CONV_FAMILY else [])
    for model in (TRAINED, UNTRAINED):
        for k in keys:
            if not feature_path(cfg, arch, model, k).exists():
                return False
    return True


# --------------------------------------------------------------------------- #
# model builders
# --------------------------------------------------------------------------- #
def build_trained(cfg: dict, arch: str, ckpt_path: Path, device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config") or resolve_arch_cfg(cfg, arch=arch)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def build_untrained(cfg: dict, arch: str, device) -> torch.nn.Module:
    """Same architecture, random init under seed 42, frozen (isolates training)."""
    set_seed(cfg["seed"])
    model = build_model(resolve_arch_cfg(cfg, arch=arch))
    model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def _gap(t: torch.Tensor) -> torch.Tensor:
    """Global-average-pool a feature map to [B, C]: 1D over time, 2D over (H,W)."""
    if t.ndim == 3:  # [B, C, T]
        return t.mean(dim=2)
    if t.ndim == 4:  # [B, C, H, W]
        return t.mean(dim=(2, 3))
    if t.ndim == 2:  # already pooled [B, D]
        return t
    raise ValueError(f"unexpected feature ndim {t.ndim}")


@torch.no_grad()
def extract_features(model, loader, device, arch: str, intermediate_names: list[str]):
    """Forward the full dataset once; collect penultimate + GAP'd intermediates.

    Returns ({key: [N, C] float32}, digit[N], speaker_id[N]) with digit/speaker
    read FROM the loader for the alignment assertion.
    """
    want_inter = arch in CONV_FAMILY
    keys = [PENULTIMATE] + (intermediate_names if want_inter else [])
    feats: dict[str, list] = {k: [] for k in keys}
    digits, spk_ids = [], []
    for x, y, spk in tqdm(loader, desc=f"{arch} forward", leave=False):
        x = x.to(device)
        _, fmaps = model(x, return_features=True)
        feats[PENULTIMATE].append(_gap(fmaps[PENULTIMATE]).cpu().numpy().astype(np.float32))
        if want_inter:
            for k in intermediate_names:
                feats[k].append(_gap(fmaps[k]).cpu().numpy().astype(np.float32))
        digits.append(np.asarray(y, dtype=np.int64))
        spk_ids.append(np.asarray(spk, dtype=np.int64))
    feats = {k: np.concatenate(v, axis=0) for k, v in feats.items()}
    return feats, np.concatenate(digits), np.concatenate(spk_ids)


def _loader(cfg: dict, cache: dict, feature: str) -> DataLoader:
    n = len(cache["digit"])
    indices = np.arange(n, dtype=np.int64)  # DISCIPLINE A: ordered, never shuffle
    ds = AudioMNISTCacheDataset(cache, indices, feature=feature)
    return DataLoader(ds, batch_size=cfg["extract"]["batch_size"], shuffle=False,
                      num_workers=cfg["extract"].get("num_workers", 0), drop_last=False)


def assert_alignment(labels_df, digit_loader, spk_loader) -> dict:
    n_lab, n_act = len(labels_df), len(digit_loader)
    assert n_lab == n_act, f"N mismatch: {n_lab} labels vs {n_act} activations"
    assert np.array_equal(labels_df["digit"].to_numpy(), digit_loader), "digit desync"
    assert np.array_equal(labels_df["speaker_id"].to_numpy(), spk_loader), "speaker desync"
    return {"n": int(n_lab), "digit_aligned": True, "speaker_aligned": True}


def run_extraction(cfg: dict, recompute: bool = False) -> dict:
    device = torch.device(cfg["extract"]["device"])
    ckpts = si_seed42_checkpoints(cfg)
    archs = cfg["architectures"]

    base_cache = load_cache(cfg)  # waveform cache; labels built from this (shared order)
    labels_df = build_label_table(base_cache, cfg)
    labels_df.to_parquet(labels_path(cfg))

    shapes, align_ref = {}, None
    for arch in archs:
        # need the model's intermediate names to know completeness + what to pool
        probe_cfg = resolve_arch_cfg(cfg, arch=arch)
        feature = input_kind(probe_cfg)
        # build trained early so we can read intermediate_names even when skipping
        if arch not in ckpts:
            raise FileNotFoundError(f"no SI seed-42 checkpoint for {arch} in {cfg['sweep_results']}")
        if not recompute:
            # peek intermediate names cheaply from an untrained instance
            tmp = build_model(probe_cfg)
            inter_names = list(getattr(tmp, "intermediate_names", []))
            if arch_complete(cfg, arch, inter_names):
                print(f"[extract] {arch}: cache complete, skipping")
                continue

        cache = dict(base_cache)
        if feature == "spectrogram":
            from ..data.spectrogram import load_spec_cache

            cache["spec"] = load_spec_cache(cfg)["spec"]

        for model_kind in (TRAINED, UNTRAINED):
            if model_kind == TRAINED:
                model = build_trained(cfg, arch, ckpts[arch], device)
            else:
                model = build_untrained(cfg, arch, device)
            inter_names = list(getattr(model, "intermediate_names", []))
            loader = _loader(cfg, cache, feature)
            feats, dl, sl = extract_features(model, loader, device, arch, inter_names)

            align = assert_alignment(labels_df, dl, sl)
            if align_ref is None:
                align_ref = align
            for k, arr in feats.items():
                np.save(feature_path(cfg, arch, model_kind, k), arr)
                shapes[f"{arch}__{model_kind}__{k}"] = list(arr.shape)
            print(f"[extract] {arch}/{model_kind}: penultimate={feats[PENULTIMATE].shape} "
                  f"keys={list(feats)}")

    meta = {
        "n_utterances": int(len(labels_df)),
        "architectures": list(archs),
        "conv_family": sorted(CONV_FAMILY & set(archs)),
        "shapes": shapes,
        "alignment": align_ref,
        "checkpoints": {a: str(p) for a, p in ckpts.items()},
        "seed": cfg["seed"],
        "git_sha": git_sha(),
    }
    with open(meta_path(cfg), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[extract] done -> {activations_dir(cfg)}")
    return meta


# --------------------------------------------------------------------------- #
# loading cached features for the probing stage
# --------------------------------------------------------------------------- #
def load_features(cfg: dict, arch: str, model: str, key: str = PENULTIMATE) -> np.ndarray:
    return np.load(feature_path(cfg, arch, model, key))


def load_labels(cfg: dict):
    import pandas as pd

    return pd.read_parquet(labels_path(cfg))
