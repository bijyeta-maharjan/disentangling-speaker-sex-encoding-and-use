"""Descriptive frequency shift under female fine-tuning, via the DFT-LRP
implementation in scripts/xai_dft_lrp.py. This script imports and reuses its
LRP/DFT-LRP/faithfulness functions verbatim (lrp_time_relevance, dft_lrp_freq,
flip_curve) rather than re-deriving the method, and drives them over the
fine-tuning protocol of scripts/female_finetune.py instead of the full SI test
split.

SCOPE -- stated on every artifact built from this script's output:
  - DESCRIPTIVE ONLY. Reports which frequency bands the model relies on
    differently after fine-tuning on held-out female speakers ("changed when
    adapting to women"). This is NOT a "female-specific" signature claim.
  - cnn1d ONLY. The DFT-LRP script's BatchNorm-canonization
    (bn_conv_pairs) matches nn.BatchNorm1d/nn.Conv1d pairs, and its DFT mapping
    operates on the raw waveform (rfft over the time axis). spec_cnn2d uses
    nn.BatchNorm2d/nn.Conv2d over a mel-spectrogram input -- structurally
    incompatible. DFT-LRP was never run on spec_cnn2d (the
    faithfulness_summary.json archs list is ["cnn1d", "resnet1d"]).
  - Uses the EXACT fine-tuning protocol: pinned male-only checkpoints
    (results/female_finetune/pinned_checkpoints.json), 4 female
    folds x seeds 42/43/44 (splits/female_finetune_localization_folds_v1.json),
    locked fine-tune schedule (2 epochs, lr=1e-4, batch=128, wd=0.0, Adam, CPU,
    seeded -- identical to scripts/female_finetune.py). Every
    (seed, fold) is its own before/after comparison; fold/seed spread is
    reported, NEVER pooled into one CI (locked decision, matches female_finetune_report.py).
  - Band relevance is interpreted ONLY where the faithfulness gate
    (MoRF < random AND MoRF < LeRF, by AUC) passes for BOTH the before model
    and the after model, on the same eval speech.
  - Male-control shift (--include-male-control) is CONTEXT ONLY, confounded
    (see female_finetune.py's male_control_asymmetry note) --
    never a "female-specific" claim, never subtracted from the female shift.

Usage:
  python scripts/female_finetune_band_shift.py [--device cuda] [--n-bands 16]
      [--steps 20] [--seeds 42,43,44] [--folds 0,1,2,3]
      [--include-male-control]
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

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[0]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import xai_dft_lrp as dftlrp  # noqa: E402  (reused, not re-derived)
from female_finetune import (  # noqa: E402
    FEMALE_FOLDS_PATH, MALE_FOLDS_PATH, FT_LR, FT_WEIGHT_DECAY, FT_BATCH_SIZE,
    load_pinned_model, load_cache_for, indices_for_speakers, finetune_,
    eval_on_speakers,
)
from src.models.factory import input_kind  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.utils.config import resolve
from src.utils.experiment import git_sha  # noqa: E402

ARCH = "cnn1d"  # spec_cnn2d structurally out of scope for this script -- see docstring
FT_EPOCHS = 2   # locked (pilot_epoch_gate.json)
OUT_DIR = resolve("results/female_finetune_band_shift")

PITCH_BAND_HZ = (0, 250)
FORMANT_BAND_HZ = (250, 750)


def band_region(lo: float, hi: float) -> str:
    if lo >= PITCH_BAND_HZ[0] and hi <= PITCH_BAND_HZ[1]:
        return "pitch"
    if lo >= FORMANT_BAND_HZ[0] and hi <= FORMANT_BAND_HZ[1]:
        return "formant"
    return "higher"


def band_profile(absR: np.ndarray, freqs: np.ndarray, n_bands: int, sr: int):
    edges = np.linspace(0, sr / 2, n_bands + 1)
    prof = absR.mean(0)
    out = np.empty(n_bands)
    for b in range(n_bands):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        out[b] = prof[sel].sum()
    return out, edges


def dftlrp_on_speech(model, wav: np.ndarray, device, steps: int) -> dict:
    """Adapts xai_dft_lrp.main()'s per-arch LRP -> DFT-LRP -> faithfulness body
    to a caller-supplied eval-speech batch, reusing its functions verbatim."""
    L = wav.shape[-1]
    K = L // 2 + 1
    R_f = np.empty((len(wav), K), np.float32)
    preds = np.empty(len(wav), np.int64)
    bs = 128
    for i in range(0, len(wav), bs):
        xb = torch.from_numpy(wav[i:i + bs]).unsqueeze(1).to(device)
        R_t, pr = dftlrp.lrp_time_relevance(model, xb, device)
        R_f[i:i + bs] = dftlrp.dft_lrp_freq(wav[i:i + bs], R_t)
        preds[i:i + bs] = pr

    X_f = np.fft.rfft(wav, axis=-1)
    absR = np.abs(R_f)
    morf = np.argsort(-absR, axis=1)
    lerf = np.argsort(absR, axis=1)
    rng = np.random.default_rng(0)
    rando = np.argsort(rng.random(absR.shape), axis=1)

    fr, m_morf = dftlrp.flip_curve(model, X_f, morf, preds, K, L, device, steps)
    _, m_lerf = dftlrp.flip_curve(model, X_f, lerf, preds, K, L, device, steps)
    _, m_rand = dftlrp.flip_curve(model, X_f, rando, preds, K, L, device, steps)
    auc = lambda c: float(np.trapezoid(c, fr))
    a_morf, a_lerf, a_rand = auc(m_morf), auc(m_lerf), auc(m_rand)
    faithful = (a_morf < a_lerf) and (a_morf < a_rand)
    return {
        "absR": absR, "faithful": bool(faithful),
        "auc_morf": a_morf, "auc_lerf": a_lerf, "auc_random": a_rand,
        "n": int(len(wav)),
    }


def run_condition(condition: str, folds_path: Path, seeds, n_bands, steps, device):
    with open(folds_path) as fh:
        spec = json.load(fh)

    rows = []
    consistency_checks = []
    for seed in seeds:
        cpu_model, cfg, entry = load_pinned_model(ARCH, seed, torch.device("cpu"))
        feature = input_kind(cfg)
        assert feature == "waveform", f"unexpected feature kind {feature} for {ARCH}"
        cache = load_cache_for(cfg)
        sr = int(cfg["preprocess"]["target_sr"])

        for fold in spec["folds"]:
            eval_spk, finetune_spk = fold["eval"], fold["finetune"]
            eval_idx = indices_for_speakers(cache, eval_spk)
            wav_eval = np.asarray(cache["wav"])[eval_idx].astype(np.float32)
            freqs = np.fft.rfftfreq(wav_eval.shape[-1], d=1.0 / sr)

            # ---- BEFORE: base pinned model, this fold's eval speech ----
            model_before = copy.deepcopy(cpu_model).to(device).eval()
            res_before = dftlrp_on_speech(model_before, wav_eval, device, steps)
            prof_before, edges = band_profile(res_before["absR"], freqs, n_bands, sr)
            del model_before

            # ---- fine-tune: exact female_finetune.py protocol (CPU, seeded, locked hparams) ----
            model_ft_cpu = copy.deepcopy(cpu_model)
            set_seed(seed)
            finetune_(model_ft_cpu, cache, finetune_spk, feature, epochs=FT_EPOCHS,
                      lr=FT_LR, batch_size=FT_BATCH_SIZE, weight_decay=FT_WEIGHT_DECAY,
                      seed=seed, device=torch.device("cpu"))

            after_acc = eval_on_speakers(model_ft_cpu, cache, eval_spk, feature,
                                          FT_BATCH_SIZE, torch.device("cpu"), seed=seed)
            consistency_checks.append({
                "condition": condition, "seed": seed, "fold": fold["fold"],
                "post_ft_eval_acc_point": after_acc["overall"]["point"],
            })

            # ---- AFTER: fine-tuned model, SAME eval speech ----
            model_after = model_ft_cpu.to(device).eval()
            res_after = dftlrp_on_speech(model_after, wav_eval, device, steps)
            prof_after, _ = band_profile(res_after["absR"], freqs, n_bands, sr)
            del model_after, model_ft_cpu

            both_faithful = res_before["faithful"] and res_after["faithful"]
            for b in range(n_bands):
                lo, hi = float(edges[b]), float(edges[b + 1])
                rows.append({
                    "condition": condition, "arch": ARCH, "seed": seed, "fold": fold["fold"],
                    "band_lo": round(lo, 1), "band_hi": round(hi, 1),
                    "band_region": band_region(lo, hi),
                    "relevance_before": round(float(prof_before[b]), 6),
                    "relevance_after": round(float(prof_after[b]), 6),
                    "delta": round(float(prof_after[b] - prof_before[b]), 6),
                    "faithful_before": res_before["faithful"], "faithful_after": res_after["faithful"],
                    "faithfulness_gate_pass": both_faithful,
                    "auc_morf_before": res_before["auc_morf"], "auc_lerf_before": res_before["auc_lerf"],
                    "auc_random_before": res_before["auc_random"],
                    "auc_morf_after": res_after["auc_morf"], "auc_lerf_after": res_after["auc_lerf"],
                    "auc_random_after": res_after["auc_random"],
                    "n_eval_samples": res_before["n"],
                })
            print(f"[band-shift] {condition} seed{seed} fold{fold['fold']}: "
                  f"faithful before={res_before['faithful']} after={res_after['faithful']}  "
                  f"post-FT eval acc={after_acc['overall']['point']:.4f}")

    return rows, consistency_checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-bands", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--include-male-control", action="store_true",
                     help="context only, confounded -- see docstring")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    fold_ids = {int(f) for f in args.folds.split(",")}

    def filter_folds(path):
        spec = json.load(open(path))
        spec["folds"] = [f for f in spec["folds"] if f["fold"] in fold_ids]
        tmp = OUT_DIR / f"_tmp_{path.stem}.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as fh:
            json.dump(spec, fh)
        return tmp

    female_path = filter_folds(FEMALE_FOLDS_PATH)
    all_rows, all_checks = run_condition("female", female_path, seeds, args.n_bands, args.steps, device)

    if args.include_male_control:
        male_path = filter_folds(MALE_FOLDS_PATH)
        rows_m, checks_m = run_condition("male_control", male_path, seeds, args.n_bands, args.steps, device)
        all_rows += rows_m
        all_checks += checks_m

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "band_relevance_shift.csv", index=False)
    with open(OUT_DIR / "consistency_check_vs_recovery.json", "w") as fh:
        json.dump(all_checks, fh, indent=2)
    with open(OUT_DIR / "run_meta.json", "w") as fh:
        json.dump({
            "git_sha": git_sha(),
            "arch": ARCH,
            "arch_scope_note": "cnn1d only -- spec_cnn2d structurally out of scope for the DFT-LRP script (BatchNorm1d/Conv1d canonization + raw-waveform rfft); DFT-LRP was never run on spec_cnn2d.",
            "scope_note": "DESCRIPTIVE ONLY: bands that changed when the model adapted to women. NOT a female-specific signature claim.",
            "ft_epochs": FT_EPOCHS, "ft_lr": FT_LR, "ft_batch_size": FT_BATCH_SIZE,
            "ft_weight_decay": FT_WEIGHT_DECAY,
            "seeds": seeds, "folds": sorted(fold_ids),
            "n_bands": args.n_bands, "steps": args.steps, "device": str(device),
            "male_control_included": args.include_male_control,
            "male_control_note": "context only, confounded (weak exposure + ceiling) -- see female_finetune.py male_control_asymmetry note. Never a female-specific claim.",
            "pitch_band_hz": list(PITCH_BAND_HZ), "formant_band_hz": list(FORMANT_BAND_HZ),
        }, fh, indent=2)
    print(f"\n[band-shift] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
