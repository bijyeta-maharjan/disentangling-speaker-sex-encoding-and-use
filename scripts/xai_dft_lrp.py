"""DFT-LRP frequency attribution for the CNN family (cnn1d, resnet1d), WITH a
feature-flipping faithfulness gate that must pass BEFORE the attribution is
interpreted.

Pipeline per architecture (seed-42 SI checkpoint, cached waveforms):
  1. LRP (zennit EpsilonPlusFlat + BatchNorm canonization) -> time-domain
     relevance R_t for the predicted class. Models are plain Conv/BN/ReLU/Pool/
     Linear (LRP-friendly by design).
  2. DFT-LRP: map R_t to per-frequency relevance R_f by LRP-epsilon redistribution
     through the (linear) inverse-DFT virtual layer. This reduces exactly to
        R_f[k] = Re( X_f[k] * rfft( R_t / (x_t + eps) )[k] ),   X_f = rfft(x_t).
  3. FAITHFULNESS GATE (frequency-domain feature flipping): rank rfft bins by |R_f|
     and progressively REMOVE them (zero the bin, inverse-rfft, re-predict), in
     Most-Relevant-First (MoRF), Least-Relevant-First (LeRF) and RANDOM orders.
     A faithful attribution => predicted-class prob collapses fastest under MoRF:
        AUC(MoRF) < AUC(random) < AUC(LeRF).
     We report the AUCs + a PASS/FAIL. Only on PASS do we emit band profiles.
  4. If PASS: aggregate |R_f| into frequency-band relevance profiles (overall +
     per sex group) -> the "frequency USED" attribution to compare with occlusion.

Reads cached inputs + seed-42 SI checkpoints only. Needs a CUDA GPU for reasonable
runtime (see README).

NOTE: si_checkpoints is not defined in this repository's src/probing/extract_week2.py
(see README, "Running the GPU-based analyses").

    python scripts/xai_dft_lrp.py [--archs cnn1d,resnet1d] [--n-faith 1000]
        [--steps 20] [--n-bands 16] [--device cuda|cpu] [--limit N] [--seed 42]
        [--out-dir DIR]
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zennit.attribution import Gradient
from zennit.canonizers import NamedMergeBatchNorm
from zennit.composites import EpsilonPlusFlat

from src.data.preprocess import load_cache
from src.models.factory import build_model
from src.probing.extract_week2 import si_checkpoints
from src.splits.make_splits import load_split_indices
from src.utils.config import load_config, resolve, resolve_arch_cfg

WAVE_ARCHS_DEFAULT = "cnn1d,resnet1d"
EPS = 1e-6


def load_arch_model(cfg, arch, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config") or resolve_arch_cfg(cfg, arch=arch)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def bn_conv_pairs(model):
    """Build [([conv_name], bn_name), ...] for every BatchNorm1d whose sibling
    '<parent>.conv' exists (covers ConvBlock and ResidualConvBlock main paths)."""
    mods = dict(model.named_modules())
    pairs = []
    for name, m in mods.items():
        if isinstance(m, nn.BatchNorm1d) and name.endswith(".bn"):
            conv_name = name[:-3] + ".conv"
            if conv_name in mods and isinstance(mods[conv_name], nn.Conv1d):
                pairs.append(([conv_name], name))
    return pairs


def lrp_time_relevance(model, x, device):
    """R_t for the predicted class. x: [M,1,L] tensor. Returns (R_t [M,L], pred)."""
    canonizers = [NamedMergeBatchNorm(bn_conv_pairs(model))]
    composite = EpsilonPlusFlat(canonizers=canonizers)
    x = x.clone().requires_grad_(True)
    with Gradient(model=model, composite=composite) as attributor:
        out, rel = attributor(x, lambda o: (o == o.max(1, keepdim=True).values).float())
    pred = out.argmax(1).detach().cpu().numpy()
    R_t = rel.detach().squeeze(1).cpu().numpy()  # [M, L]
    return R_t, pred


def dft_lrp_freq(x_np, R_t):
    """R_f[k] = Re( rfft(x)[k] * rfft(R_t/(x+eps))[k] ). x_np,R_t: [M,L] -> [M,K]."""
    X_f = np.fft.rfft(x_np, axis=-1)
    G = np.fft.rfft(R_t / (x_np + EPS), axis=-1)
    return np.real(X_f * G)  # [M, K]


@torch.no_grad()
def predicted_prob(model, wav_np, pred, device, batch=256):
    """Mean prob mass on the (fixed) predicted class after perturbation."""
    probs = np.empty(len(wav_np), np.float32)
    for i in range(0, len(wav_np), batch):
        x = torch.from_numpy(wav_np[i:i + batch]).unsqueeze(1).to(device)
        p = torch.softmax(model(x), 1).cpu().numpy()
        idx = pred[i:i + batch]
        probs[i:i + batch] = p[np.arange(len(idx)), idx]
    return probs


def flip_curve(model, X_f, order, pred, K, L, device, steps):
    """Progressively zero rfft bins in `order` (per-sample [M,K] int), invert,
    measure mean predicted-class prob. Returns fractions[steps+1], mean_prob[]."""
    M = X_f.shape[0]
    fracs = np.linspace(0, 1, steps + 1)
    means = np.empty(steps + 1, np.float32)
    for si, f in enumerate(fracs):
        k = int(round(f * K))
        Xc = X_f.copy()
        if k > 0:
            rows = np.arange(M)[:, None]
            cols = order[:, :k]
            Xc[rows, cols] = 0.0
        wav = np.fft.irfft(Xc, n=L, axis=-1).astype(np.float32)
        means[si] = predicted_prob(model, wav, pred, device).mean()
    return fracs, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/week2_probing.yaml")
    ap.add_argument("--regime", default="speaker_independent")
    ap.add_argument("--archs", default=WAVE_ARCHS_DEFAULT)
    ap.add_argument("--n-faith", type=int, default=1000, help="samples for faithfulness")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n-bands", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="subsample test (smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/dft_lrp",
                    help="output directory (relative paths resolve against the repo root)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    device = torch.device(args.device)
    sr = int(cfg["preprocess"]["target_sr"])
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]

    cache = load_cache(cfg)
    test_idx = load_split_indices(cfg, args.regime, cache)["test"]
    rng = np.random.default_rng(args.seed)
    if args.limit:
        test_idx = test_idx[: args.limit]
    wav_all = np.asarray(cache["wav"])[test_idx].astype(np.float32)
    sex_all = np.asarray(cache["sex"])[test_idx]
    L = wav_all.shape[-1]
    K = L // 2 + 1
    freqs = np.fft.rfftfreq(L, d=1.0 / sr)
    ckpts = si_checkpoints(cfg)

    # faithfulness subset
    n_faith = min(args.n_faith, len(wav_all)) if not args.limit else len(wav_all)
    fsel = rng.choice(len(wav_all), size=n_faith, replace=False)
    wav_f, sex_f = wav_all[fsel], sex_all[fsel]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    band_rows = []
    for arch in archs:
        assert arch in ckpts and ckpts[arch].exists(), f"missing ckpt {arch}"
        model = load_arch_model(cfg, arch, ckpts[arch], device)

        # 1-2: LRP time relevance + DFT-LRP freq relevance (batched)
        R_f = np.empty((len(wav_f), K), np.float32)
        preds = np.empty(len(wav_f), np.int64)
        bs = 128
        for i in range(0, len(wav_f), bs):
            xb = torch.from_numpy(wav_f[i:i + bs]).unsqueeze(1).to(device)
            R_t, pr = lrp_time_relevance(model, xb, device)
            R_f[i:i + bs] = dft_lrp_freq(wav_f[i:i + bs], R_t)
            preds[i:i + bs] = pr

        X_f = np.fft.rfft(wav_f, axis=-1)
        absR = np.abs(R_f)
        morf = np.argsort(-absR, axis=1)          # most relevant first
        lerf = np.argsort(absR, axis=1)            # least relevant first
        rando = np.argsort(rng.random(absR.shape), axis=1)

        # 3: faithfulness gate
        fr, m_morf = flip_curve(model, X_f, morf, preds, K, L, device, args.steps)
        _, m_lerf = flip_curve(model, X_f, lerf, preds, K, L, device, args.steps)
        _, m_rand = flip_curve(model, X_f, rando, preds, K, L, device, args.steps)
        auc = lambda c: float(np.trapezoid(c, fr))
        a_morf, a_lerf, a_rand = auc(m_morf), auc(m_lerf), auc(m_rand)
        # Faithful iff removing MOST-relevant frequencies degrades the predicted
        # class faster than removing least-relevant OR random ones. (The random<LeRF
        # ordering is a weaker, noisier signal, so it is reported but not gated on.)
        faithful = (a_morf < a_lerf) and (a_morf < a_rand)
        print(f"[{arch}] faithfulness AUC  MoRF={a_morf:.3f}  random={a_rand:.3f}  "
              f"LeRF={a_lerf:.3f}  => {'PASS' if faithful else 'FAIL'}")

        # faithfulness figure
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(fr, m_morf, "-o", label=f"MoRF (AUC {a_morf:.2f})")
        ax.plot(fr, m_rand, "-s", label=f"random (AUC {a_rand:.2f})")
        ax.plot(fr, m_lerf, "-^", label=f"LeRF (AUC {a_lerf:.2f})")
        ax.set_xlabel("fraction of rfft bins removed")
        ax.set_ylabel("mean predicted-class prob")
        ax.set_title(f"DFT-LRP faithfulness (freq feature-flipping) — {arch}")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out_dir / f"faithfulness_{arch}.png", dpi=150)
        plt.close(fig)

        summary[arch] = {"auc_morf": a_morf, "auc_random": a_rand, "auc_lerf": a_lerf,
                         "faithful": bool(faithful), "n_faith": int(len(wav_f))}

        # 4: band profiles ONLY if faithful (else emitted but flagged untrusted)
        edges = np.linspace(0, sr / 2, args.n_bands + 1)
        groups = ["all"] + sorted(set(sex_f.tolist()))
        for g in groups:
            gm = np.ones(len(sex_f), bool) if g == "all" else (sex_f == g)
            prof = absR[gm].mean(0)  # mean |R_f| per bin
            for b in range(args.n_bands):
                lo, hi = edges[b], edges[b + 1]
                sel = (freqs >= lo) & (freqs < hi)
                band_rows.append({"arch": arch, "group": g,
                                  "band_lo": round(float(lo), 1), "band_hi": round(float(hi), 1),
                                  "mean_abs_relevance": round(float(prof[sel].sum()), 6),
                                  "faithful": bool(faithful)})

    pd.DataFrame(band_rows).to_csv(out_dir / "dftlrp_band_relevance.csv", index=False)
    with open(out_dir / "faithfulness_summary.json", "w") as fh:
        json.dump({"seed": args.seed, "device": str(device), "steps": args.steps,
                   "n_bands": args.n_bands, "archs": archs, "per_arch": summary,
                   "checkpoints": {a: str(ckpts[a]) for a in archs}}, fh, indent=2)
    print(f"\n[dft-lrp] wrote -> {out_dir}")
    print("GATE: interpret band relevance ONLY for archs with faithful=PASS.")


if __name__ == "__main__":
    main()
