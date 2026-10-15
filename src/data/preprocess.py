"""Preprocess AudioMNIST waveforms into a memory-mapped cache.

Pipeline (per utterance):
  1. decode wav, resample orig_sr -> target_sr (48 kHz -> 8 kHz)
  2. pad/trim to a fixed length (8192 samples ~ 1.024 s)
  3. per-UTTERANCE peak normalization (x / max|x|)  -- NOT per-speaker, and no
     statistic is shared across utterances, so there is zero test leakage here.

The cache is aligned ROW-FOR-ROW to the Day-1 manifest order
(``src.data.index.build_manifest`` -> sorted by speaker, digit, repetition), so
manifest index i indexes cache row i. We persist alongside it:
  <cache>.npy            float32 [N, target_length]   the waveforms
  <cache>_meta.npz       digit[int64 N], speaker_id[int64 N], speaker_str, id->str map
  <cache>_manifest.parquet   the exact manifest used (provenance / re-alignment)

Caching means audio is decoded ONCE, not every epoch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.index import build_manifest
from ..data.metadata import load_metadata
from ..utils.config import load_config, paths

_EPS = 1e-8


def _normalize(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "peak":
        peak = np.max(np.abs(x))
        return x / peak if peak > _EPS else x
    if mode == "rms":
        rms = np.sqrt(np.mean(x**2))
        return x / rms if rms > _EPS else x
    raise ValueError(f"unknown normalize mode: {mode!r}")


def _fix_length(x: np.ndarray, length: int) -> np.ndarray:
    """Center-pad or trim to exactly `length` samples."""
    if len(x) == length:
        return x
    if len(x) > length:
        start = (len(x) - length) // 2
        return x[start : start + length]
    pad = length - len(x)
    left = pad // 2
    return np.pad(x, (left, pad - left))


def cache_paths(cfg: dict) -> dict[str, Path]:
    p = paths(cfg)
    base = p.processed_dir / cfg["preprocess"]["cache_name"]
    return {
        "wav": base.with_suffix(".npy"),
        "meta": Path(str(base) + "_meta.npz"),
        "manifest": Path(str(base) + "_manifest.parquet"),
    }


def cache_exists(cfg: dict) -> bool:
    cp = cache_paths(cfg)
    return all(path.exists() for path in cp.values())


def build_cache(cfg: dict, recompute: bool = False) -> dict[str, Path]:
    """Decode/resample/normalize every utterance and write the memmap cache."""
    cp = cache_paths(cfg)
    if cache_exists(cfg) and not recompute:
        print(f"[preprocess] cache exists, skipping (use --recompute to rebuild): {cp['wav']}")
        return cp

    pp = cfg["preprocess"]
    target_sr, length, norm = pp["target_sr"], pp["target_length"], pp["normalize"]

    manifest = build_manifest(paths(cfg).raw_root)
    meta = load_metadata().set_index("speaker")
    n = len(manifest)

    # contiguous speaker id mapping 0..K-1 over sorted unique speakers
    speakers_sorted = sorted(manifest["speaker"].unique())
    spk_to_id = {s: i for i, s in enumerate(speakers_sorted)}

    cp["wav"].parent.mkdir(parents=True, exist_ok=True)
    wav = np.lib.format.open_memmap(
        cp["wav"], mode="w+", dtype=np.float32, shape=(n, length)
    )
    digit = np.empty(n, dtype=np.int64)
    speaker_id = np.empty(n, dtype=np.int64)
    sex = np.empty(n, dtype=object)

    for i, row in enumerate(tqdm(manifest.itertuples(index=False), total=n, desc="preprocess")):
        x, _ = librosa.load(row.filepath, sr=target_sr, mono=True)
        x = _fix_length(np.asarray(x, dtype=np.float32), length)
        x = _normalize(x, norm)
        wav[i] = x
        digit[i] = row.digit
        speaker_id[i] = spk_to_id[row.speaker]
        sex[i] = meta.loc[row.speaker, "gender"]

    wav.flush()
    np.savez(
        cp["meta"],
        digit=digit,
        speaker_id=speaker_id,
        speaker_str=np.array(manifest["speaker"].tolist(), dtype=object),
        sex=sex,
        id_to_speaker=np.array(speakers_sorted, dtype=object),
        target_sr=target_sr,
        target_length=length,
        normalize=norm,
    )
    manifest.to_parquet(cp["manifest"])
    print(f"[preprocess] wrote {n} waveforms -> {cp['wav']} ({wav.nbytes/1e6:.0f} MB)")
    return cp


def load_cache(cfg: dict) -> dict:
    """Memory-map the cached waveforms and load aligned label/meta arrays."""
    if not cache_exists(cfg):
        raise FileNotFoundError(
            "Waveform cache missing. Run: python scripts/preprocess.py"
        )
    cp = cache_paths(cfg)
    wav = np.load(cp["wav"], mmap_mode="r")
    meta = np.load(cp["meta"], allow_pickle=True)
    manifest = pd.read_parquet(cp["manifest"])
    return {
        "wav": wav,
        "digit": meta["digit"],
        "speaker_id": meta["speaker_id"],
        "speaker_str": meta["speaker_str"],
        "sex": meta["sex"],
        "id_to_speaker": meta["id_to_speaker"],
        "manifest": manifest,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the AudioMNIST waveform cache.")
    ap.add_argument("--config", default=None, help="path to YAML config")
    ap.add_argument("--recompute", action="store_true", help="rebuild even if cache exists")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    build_cache(cfg, recompute=args.recompute)


if __name__ == "__main__":
    main()
