"""Mel-spectrogram cache, derived ONCE from the 8 kHz waveform cache.

The spectrogram 2D CNN is the only Week-2 model that changes the input
representation (a deliberate study axis). To keep that axis clean and cheap, we
compute log-mel spectrograms from the SAME cached 8 kHz waveforms (never
re-decoding audio, never touching raw files) and cache them row-for-row aligned
to the waveform cache / Day-1 manifest order — so manifest index i indexes mel
row i, exactly like the waveform cache.

Mel params live in cfg['spectrogram'] and are recorded in the cache meta so the
input representation is fully reproducible. Idempotent: skips if a cache built
with the SAME params already exists, unless ``recompute=True``.

Layout:
  <spec_cache>.npy        float32 [N, n_mels, T]   the log-mel spectrograms
  <spec_cache>_meta.npz   the mel params + shape (provenance / param check)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from ..utils.config import load_config, paths
from .preprocess import load_cache


def _spec_cfg(cfg: dict) -> dict:
    s = dict(cfg["spectrogram"])
    s.setdefault("fmin", 0)
    s.setdefault("fmax", cfg["preprocess"]["target_sr"] // 2)
    s.setdefault("power", 2.0)
    s.setdefault("log", True)
    return s


def spec_cache_paths(cfg: dict) -> dict[str, Path]:
    p = paths(cfg)
    base = p.processed_dir / cfg["spectrogram"]["cache_name"]
    return {"spec": base.with_suffix(".npy"), "meta": Path(str(base) + "_meta.npz")}


def _params_signature(cfg: dict) -> dict:
    s = _spec_cfg(cfg)
    return {
        "sr": int(cfg["preprocess"]["target_sr"]),
        "target_length": int(cfg["preprocess"]["target_length"]),
        "n_mels": int(s["n_mels"]),
        "n_fft": int(s["n_fft"]),
        "hop_length": int(s["hop_length"]),
        "fmin": float(s["fmin"]),
        "fmax": float(s["fmax"]),
        "power": float(s["power"]),
        "log": bool(s["log"]),
    }


def cache_exists(cfg: dict) -> bool:
    cp = spec_cache_paths(cfg)
    if not all(path.exists() for path in cp.values()):
        return False
    # ensure the existing cache was built with the SAME mel params
    meta = np.load(cp["meta"], allow_pickle=True)
    stored = {k: meta[k].item() for k in meta.files if k != "shape"}
    want = _params_signature(cfg)
    return all(stored.get(k) == v for k, v in want.items())


def _melspec(wav: np.ndarray, sig: dict) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sig["sr"],
        n_fft=sig["n_fft"],
        hop_length=sig["hop_length"],
        n_mels=sig["n_mels"],
        fmin=sig["fmin"],
        fmax=sig["fmax"],
        power=sig["power"],
    )
    if sig["log"]:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)


def build_spec_cache(cfg: dict, recompute: bool = False) -> dict[str, Path]:
    """Compute log-mel spectrograms for every cached waveform; write the memmap."""
    cp = spec_cache_paths(cfg)
    if cache_exists(cfg) and not recompute:
        print(f"[spectrogram] cache exists with matching params, skipping (use --recompute): {cp['spec']}")
        return cp

    sig = _params_signature(cfg)
    cache = load_cache(cfg)
    wav = cache["wav"]
    n = wav.shape[0]

    # frame count from a probe so the memmap is shaped correctly
    t = _melspec(np.asarray(wav[0], dtype=np.float32), sig).shape[1]

    cp["spec"].parent.mkdir(parents=True, exist_ok=True)
    spec = np.lib.format.open_memmap(
        cp["spec"], mode="w+", dtype=np.float32, shape=(n, sig["n_mels"], t)
    )
    for i in tqdm(range(n), desc="melspec"):
        spec[i] = _melspec(np.asarray(wav[i], dtype=np.float32), sig)
    spec.flush()

    np.savez(cp["meta"], shape=np.array([n, sig["n_mels"], t]), **{k: np.array(v) for k, v in sig.items()})
    print(f"[spectrogram] wrote {n} mel-spectrograms {spec.shape} -> {cp['spec']} ({spec.nbytes/1e6:.0f} MB)")
    return cp


def load_spec_cache(cfg: dict) -> dict:
    """Memory-map the cached mel-spectrograms aligned to the waveform cache order."""
    if not cache_exists(cfg):
        raise FileNotFoundError(
            "Mel-spectrogram cache missing or params changed. Run: "
            "python scripts/build_spectrograms.py [--recompute]"
        )
    cp = spec_cache_paths(cfg)
    spec = np.load(cp["spec"], mmap_mode="r")
    meta = np.load(cp["meta"], allow_pickle=True)
    return {"spec": spec, "n_mels": int(meta["n_mels"]), "T": int(spec.shape[2])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the mel-spectrogram cache from the waveform cache.")
    ap.add_argument("--config", default=None, help="path to YAML config (must have a spectrogram block)")
    ap.add_argument("--recompute", action="store_true", help="rebuild even if a matching cache exists")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    build_spec_cache(cfg, recompute=args.recompute)


if __name__ == "__main__":
    main()
