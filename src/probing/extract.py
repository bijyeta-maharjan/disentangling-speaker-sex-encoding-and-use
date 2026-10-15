"""Extract per-block GAP features for the trained SI model, an untrained control,
and a raw-input baseline — through ONE pipeline so the baselines are identical.

Disciplines enforced here:
  A. Ordered, paired extraction. shuffle=False over indices=arange(N); the label
     table is built in the SAME cache order; we re-read (digit, speaker_id) FROM
     the loader and assert they match the label table row-for-row before caching.
  B. Global average pooling over TIME per block -> feature length = channel count
     (32,64,128,128,256). We never flatten (no dimensionality artifact).
  D. Two controls in the same pass:
       - untrained: identically-architected model, random init (seed 42), frozen.
       - raw: log-mel spectrogram of the input, GAP over time -> [N, n_mels].
     All from frozen models in eval() under torch.no_grad(), cached to disk.

Idempotent: skips work if the cache is complete unless ``recompute=True``.
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
from ..models.factory import build_model
from ..utils.config import resolve
from ..utils.experiment import git_sha
from ..utils.seeding import set_seed
from .labels import build_label_table

BLOCK_NAMES = ["conv1", "conv2", "conv3", "conv4", "conv5"]
TRAINED, UNTRAINED, RAW = "trained", "untrained", "raw"


# --------------------------------------------------------------------------- #
# cache layout / idempotency
# --------------------------------------------------------------------------- #
def activations_dir(cfg: dict) -> Path:
    d = resolve(cfg["paths"]["activations_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def feature_path(cfg: dict, model: str, block: str | None = None) -> Path:
    d = activations_dir(cfg)
    if model == RAW:
        return d / "raw_input.npy"
    return d / f"{model}_{block}.npy"


def labels_path(cfg: dict) -> Path:
    return activations_dir(cfg) / "labels.parquet"


def meta_path(cfg: dict) -> Path:
    return activations_dir(cfg) / "extract_meta.json"


def expected_files(cfg: dict) -> list[Path]:
    files = [labels_path(cfg), meta_path(cfg), feature_path(cfg, RAW)]
    for model in (TRAINED, UNTRAINED):
        files += [feature_path(cfg, model, b) for b in BLOCK_NAMES]
    return files


def cache_complete(cfg: dict) -> bool:
    return all(p.exists() for p in expected_files(cfg))


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _model_cfg_from_checkpoint(ckpt: dict, cfg: dict) -> dict:
    """Prefer the architecture stored in the checkpoint; fall back to day3 cfg."""
    stored = ckpt.get("config")
    if stored and "model" in stored:
        return stored
    return {"model": cfg["model"]}


def load_trained_model(cfg: dict, device) -> torch.nn.Module:
    ckpt_path = resolve(cfg["checkpoint"]["path"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = _model_cfg_from_checkpoint(ckpt, cfg)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def build_untrained_model(cfg: dict, device) -> torch.nn.Module:
    """Identically-architected model, random init under the SAME seed, frozen."""
    set_seed(cfg["seed"])  # same seed discipline as Day-2 training init
    model = build_model({"model": cfg["model"]})
    model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# GAP feature extraction through a CNN
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_gap_features(model, loader, device):
    """Forward the full dataset; GAP each conv block over time.

    Returns ({block: [N, C] float32}, digit[N], speaker_id[N]) where digit and
    speaker_id are read FROM THE LOADER (for the alignment assertion).
    """
    feats = {b: [] for b in BLOCK_NAMES}
    digits, spk_ids = [], []
    for x, y, spk in tqdm(loader, desc="forward", leave=False):
        x = x.to(device)
        _, fmaps = model(x, return_features=True)
        for b in BLOCK_NAMES:
            # fmaps[b]: (B, C, T) -> GAP over time -> (B, C)
            feats[b].append(fmaps[b].mean(dim=2).cpu().numpy().astype(np.float32))
        digits.append(np.asarray(y, dtype=np.int64))
        spk_ids.append(np.asarray(spk, dtype=np.int64))
    feats = {b: np.concatenate(v, axis=0) for b, v in feats.items()}
    return feats, np.concatenate(digits), np.concatenate(spk_ids)


# --------------------------------------------------------------------------- #
# raw-input baseline: log-mel spectrogram, GAP over time
# --------------------------------------------------------------------------- #
def extract_raw_features(cache, indices, cfg) -> np.ndarray:
    """GAP(log-mel spectrogram) per utterance -> [N, n_mels]. 'Before any net'."""
    import librosa

    rb = cfg["extract"]["raw_baseline"]
    sr = cfg["preprocess"]["target_sr"]
    n_mels, n_fft, hop = rb["n_mels"], rb["n_fft"], rb["hop_length"]
    out = np.empty((len(indices), n_mels), dtype=np.float32)
    for row, idx in enumerate(tqdm(indices, desc="raw-logmel", leave=False)):
        wav = np.asarray(cache["wav"][int(idx)], dtype=np.float32)
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        out[row] = logmel.mean(axis=1)  # GAP over time -> [n_mels]
    return out


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _loader(cache, indices, cfg):
    ds = AudioMNISTCacheDataset(cache, indices)
    return DataLoader(
        ds,
        batch_size=cfg["extract"]["batch_size"],
        shuffle=False,  # DISCIPLINE A: never shuffle during extraction
        num_workers=cfg["extract"]["num_workers"],
        drop_last=False,
    )


def assert_alignment(labels_df, digit_from_loader, spk_from_loader) -> dict:
    """Row-for-row check that activations line up with the label table."""
    n_lab = len(labels_df)
    n_act = len(digit_from_loader)
    assert n_lab == n_act, f"N mismatch: {n_lab} labels vs {n_act} activations"
    digit_ok = bool(np.array_equal(labels_df["digit"].to_numpy(), digit_from_loader))
    spk_ok = bool(np.array_equal(labels_df["speaker_id"].to_numpy(), spk_from_loader))
    assert digit_ok, "digit labels desynced from activation order"
    assert spk_ok, "speaker_id labels desynced from activation order"
    return {"n": int(n_lab), "digit_aligned": digit_ok, "speaker_aligned": spk_ok}


def run_extraction(cfg: dict, recompute: bool = False) -> dict:
    if cache_complete(cfg) and not recompute:
        print(f"[extract] cache complete, skipping (use --recompute): {activations_dir(cfg)}")
        with open(meta_path(cfg), "r", encoding="utf-8") as fh:
            return json.load(fh)

    device = torch.device(cfg["extract"]["device"])
    cache = load_cache(cfg)
    n = len(cache["digit"])
    indices = np.arange(n, dtype=np.int64)  # full dataset, manifest order

    labels_df = build_label_table(cache, cfg)

    # --- trained SI model ---
    print("[extract] trained SI model ...")
    trained = load_trained_model(cfg, device)
    loader = _loader(cache, indices, cfg)
    feats_tr, digit_loader, spk_loader = extract_gap_features(trained, loader, device)

    # DISCIPLINE A: assert alignment using labels read from the loader itself.
    align = assert_alignment(labels_df, digit_loader, spk_loader)
    print(f"[extract] ALIGNMENT OK: {align}")

    # --- untrained control ---
    print("[extract] untrained control model ...")
    untrained = build_untrained_model(cfg, device)
    loader = _loader(cache, indices, cfg)
    feats_un, d2, s2 = extract_gap_features(untrained, loader, device)
    assert np.array_equal(d2, digit_loader) and np.array_equal(s2, spk_loader), (
        "untrained pass produced a different row order than the trained pass"
    )

    # --- raw-input baseline ---
    print("[extract] raw-input baseline (log-mel + GAP) ...")
    feats_raw = extract_raw_features(cache, indices, cfg)

    # --- persist ---
    labels_df.to_parquet(labels_path(cfg))
    shapes = {}
    for b in BLOCK_NAMES:
        np.save(feature_path(cfg, TRAINED, b), feats_tr[b])
        np.save(feature_path(cfg, UNTRAINED, b), feats_un[b])
        shapes[f"{TRAINED}_{b}"] = list(feats_tr[b].shape)
        shapes[f"{UNTRAINED}_{b}"] = list(feats_un[b].shape)
    np.save(feature_path(cfg, RAW), feats_raw)
    shapes[RAW] = list(feats_raw.shape)

    meta = {
        "n_utterances": int(n),
        "block_names": BLOCK_NAMES,
        "shapes": shapes,
        "alignment": align,
        "checkpoint": str(resolve(cfg["checkpoint"]["path"])),
        "checkpoint_regime": cfg["checkpoint"]["regime"],
        "untrained_seed": cfg["seed"],
        "raw_baseline": cfg["extract"]["raw_baseline"],
        "git_sha": git_sha(),
    }
    with open(meta_path(cfg), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[extract] wrote features + labels -> {activations_dir(cfg)}")
    return meta


# --------------------------------------------------------------------------- #
# loading cached features for the probing stage
# --------------------------------------------------------------------------- #
def load_features(cfg: dict, model: str, block: str | None = None) -> np.ndarray:
    return np.load(feature_path(cfg, model, block))


def load_labels(cfg: dict):
    import pandas as pd

    return pd.read_parquet(labels_path(cfg))
