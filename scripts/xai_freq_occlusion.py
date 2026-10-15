"""Model-agnostic frequency-band OCCLUSION across the four core architectures.

Idea (necessity test): band-stop a frequency band in the INPUT WAVEFORM (once),
then measure each model's digit-accuracy drop. Because the perturbation is applied
to the shared 8 kHz waveform and the spectrogram model's mel input is RECOMPUTED
from that same band-stopped waveform (via the project's exact mel params), all four
architectures see the identical intervention -> the profiles are directly comparable.

Outputs (under --out-dir, default results/freq_occlusion/):
  * occlusion_results.csv  : rows (arch, band_lo, band_hi, group, baseline_acc,
                             occluded_acc, acc_drop, n)  for group in {all,<sex>...}
  * per-arch profile figure: acc-drop vs band, overall + per sex group.
  * run_meta.json          : config, bands, checkpoints, device, seed.

Reuses: load_cache, load_split_indices, si_checkpoints (seed-aware; defaults to the
seed-42 SI checkpoints), and the project's mel builder. Reads cached inputs only;
trains nothing. Needs a CUDA GPU for reasonable runtime (see README).

NOTE: si_checkpoints is not defined in this repository's src/probing/extract_week2.py
(see README, "Running the GPU-based analyses").

    python scripts/xai_freq_occlusion.py [--config CONFIG] [--regime speaker_independent]
        [--n-bands 16] [--device cuda|cpu] [--limit N] [--seed 42] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.preprocess import load_cache
from src.data.spectrogram import _melspec, _params_signature
from src.models.factory import build_model, input_kind
from src.probing.extract_week2 import si_checkpoints
from src.splits.make_splits import load_split_indices
from src.utils.config import load_config, resolve, resolve_arch_cfg

ARCH_ORDER = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]


def load_arch_model(cfg, arch, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config") or resolve_arch_cfg(cfg, arch=arch)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def predict_waveform(model, wav, device, batch=256):
    """wav: [M, L] float32 -> pred labels [M] for a waveform (Conv1d) model."""
    preds = []
    for i in range(0, len(wav), batch):
        x = torch.from_numpy(wav[i:i + batch]).unsqueeze(1).to(device)  # [b,1,L]
        preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def predict_spec(model, wav, sig, device, batch=256):
    """Recompute mel from (possibly band-stopped) waveforms, then predict."""
    preds = []
    for i in range(0, len(wav), batch):
        mel = np.stack([_melspec(w, sig) for w in wav[i:i + batch]]).astype(np.float32)
        x = torch.from_numpy(mel).unsqueeze(1).to(device)  # [b,1,n_mels,T]
        preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def bandstop(wav, sr, lo, hi):
    """Brick-wall band-stop [lo,hi) via rFFT zeroing. wav: [M, L] -> [M, L]."""
    L = wav.shape[-1]
    X = np.fft.rfft(wav, axis=-1)
    freqs = np.fft.rfftfreq(L, d=1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    X[:, mask] = 0.0
    return np.fft.irfft(X, n=L, axis=-1).astype(np.float32)


def accuracy(pred, true, grp_mask=None):
    m = np.ones(len(true), bool) if grp_mask is None else grp_mask
    return float((pred[m] == true[m]).mean()), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/week2_probing.yaml")
    ap.add_argument("--regime", default="speaker_independent")
    ap.add_argument("--n-bands", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="subsample test set (0=all; smoke only)")
    ap.add_argument("--seed", type=int, default=42, help="which SI seed's checkpoints")
    ap.add_argument("--out-dir", default="results/freq_occlusion",
                    help="output directory (relative paths resolve against the repo root)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    device = torch.device(args.device)
    sr = int(cfg["preprocess"]["target_sr"])
    sig = _params_signature(cfg)

    cache = load_cache(cfg)
    test_idx = load_split_indices(cfg, args.regime, cache)["test"]
    if args.limit:
        test_idx = test_idx[: args.limit]
    wav = np.asarray(cache["wav"])[test_idx].astype(np.float32)   # [M, L]
    true = np.asarray(cache["digit"])[test_idx].astype(np.int64)
    sex = np.asarray(cache["sex"])[test_idx]
    groups = ["all"] + sorted(set(sex.tolist()))

    ckpts = si_checkpoints(cfg)
    for a in ARCH_ORDER:
        assert a in ckpts and ckpts[a].exists(), f"missing checkpoint for {a}: {ckpts.get(a)}"

    # band edges over [0, Nyquist]
    edges = np.linspace(0, sr / 2, args.n_bands + 1)
    bands = list(zip(edges[:-1], edges[1:]))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for arch in ARCH_ORDER:
        acfg = resolve_arch_cfg(cfg, arch=arch)
        is_spec = input_kind(acfg) == "spectrogram"
        model = load_arch_model(cfg, arch, ckpts[arch], device)
        pred_fn = (lambda w: predict_spec(model, w, sig, device)) if is_spec \
            else (lambda w: predict_waveform(model, w, device))

        base_pred = pred_fn(wav)
        base_acc = {g: accuracy(base_pred, true, None if g == "all" else (sex == g))[0] for g in groups}
        print(f"[{arch}] baseline acc all={base_acc['all']:.4f}")

        for (lo, hi) in bands:
            wstop = bandstop(wav, sr, lo, hi)
            pred = pred_fn(wstop)
            for g in groups:
                gm = None if g == "all" else (sex == g)
                acc, n = accuracy(pred, true, gm)
                rows.append({
                    "arch": arch, "band_lo": round(float(lo), 1), "band_hi": round(float(hi), 1),
                    "group": g, "baseline_acc": round(base_acc[g], 5),
                    "occluded_acc": round(acc, 5), "acc_drop": round(base_acc[g] - acc, 5), "n": n,
                })
            print(f"  band [{lo:6.0f},{hi:6.0f}) Hz  all drop={base_acc['all']-accuracy(pred,true)[0]:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "occlusion_results.csv", index=False)

    # per-arch profile figure (overall + per group)
    for arch in ARCH_ORDER:
        sub = df[df["arch"] == arch]
        centers = (sub[sub["group"] == "all"]["band_lo"].to_numpy()
                   + sub[sub["group"] == "all"]["band_hi"].to_numpy()) / 2
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for g in groups:
            s = sub[sub["group"] == g].sort_values("band_lo")
            ax.plot((s["band_lo"] + s["band_hi"]) / 2, s["acc_drop"], marker="o", label=g)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("frequency band center (Hz)")
        ax.set_ylabel("digit-accuracy drop (baseline − occluded)")
        ax.set_title(f"Frequency-band occlusion reliance — {arch} (seed {args.seed} SI)")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out_dir / f"occlusion_profile_{arch}.png", dpi=150)
        plt.close(fig)

    with open(out_dir / "run_meta.json", "w") as fh:
        json.dump({"config": args.config, "regime": args.regime, "seed": args.seed,
                   "device": str(device), "n_bands": args.n_bands, "sr": sr,
                   "n_test": int(len(test_idx)), "groups": groups,
                   "checkpoints": {a: str(ckpts[a]) for a in ARCH_ORDER}}, fh, indent=2)
    print(f"\n[occlusion] wrote -> {out_dir}")


if __name__ == "__main__":
    main()
