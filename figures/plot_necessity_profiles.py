#!/usr/bin/env python3
"""Overlaid, speech-band-annotated frequency-reliance (necessity) profiles.

Pure re-analysis of the seed-42 occlusion output (scripts/xai_freq_occlusion.py).
Reads results/freq_occlusion/occlusion_results.csv and produces:

  figures/output/necessity_all_arch.png    -- 4 archs' group=all necessity profiles,
                                              overlaid on shared axes, speech bands shaded
                                              (paper figure necessity_all_arch.png)
  figures/output/necessity_by_group.png    -- female/male profiles per arch (availability only)
  results/freq_occlusion/necessity_summary.csv
                                           -- per-arch: drop@pitch(0-250), drop@peak band,
                                              peak-band Hz, contrast Delta, Wilson 95% CIs

NO model is loaded, NO GPU is used, NO training/inference runs. Imports: numpy/pandas/matplotlib only.
Seed-42 SI only (that is all the occlusion data covers) -- confidence capped at Medium by design.

Hypotheses written down before running (this script tests them):
  H2-1  acc_drop(0-250 Hz pitch) < acc_drop(250-500 Hz formant) for every arch; single largest
        necessity band lies in 250-750 Hz (formant region), not 0-250 (pitch).
  H2-2  bands >=1750 Hz near-dispensable (acc_drop ~ 0) for all archs; spec_cnn2d partial exception.
This script only COMPUTES the numbers/figure; it does not edit the hypothesis or draw conclusions.
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- paths (anchored to project root = parent of figures/) ---
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT, "results", "freq_occlusion")
FIGS_DIR = os.path.join(PROJECT, "figures", "output")

ARCHS = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
Z = 1.959963984540054  # 95% normal quantile

# literature speech-band boundaries (Hz), pre-registered for annotation
SPEECH_BANDS = [
    ("F0 / pitch (speaker-diagnostic)", 85, 255, "0.55"),
    ("F1 (formant)", 300, 1000, "tab:blue"),
    ("F2 (formant)", 850, 2500, "tab:green"),
]
NYQUIST = 4000.0  # 8 kHz sample rate


def wilson_ci(p, n, z=Z):
    """Wilson score 95% interval for a binomial proportion (well-behaved near 0/1)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def find_occlusion_csv():
    path = os.path.join(RESULTS_DIR, "occlusion_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing -- run scripts/xai_freq_occlusion.py first")
    return path


def load_profiles(csv_path):
    df = pd.read_csv(csv_path)
    need = {"arch", "band_lo", "band_hi", "group", "baseline_acc", "occluded_acc", "acc_drop", "n"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"occlusion CSV missing columns: {missing}")
    df["center"] = (df["band_lo"] + df["band_hi"]) / 2.0
    # verify inputs resolve exactly as the pre-registration assumes
    for g in ["all", "female", "male"]:
        sub = df[df.group == g]
        got = sorted(sub.arch.unique())
        assert got == sorted(ARCHS), f"group={g}: expected archs {sorted(ARCHS)}, got {got}"
        for a in ARCHS:
            nb = len(sub[sub.arch == a])
            assert nb == 16, f"group={g} arch={a}: expected 16 bands, got {nb}"
    return df


def summarize(df):
    rows = []
    dall = df[df.group == "all"]
    for a in ARCHS:
        p = dall[dall.arch == a].sort_values("band_lo").reset_index(drop=True)
        base = float(p.baseline_acc.iloc[0])
        n = int(p.n.iloc[0])
        pitch = p[p.band_lo == 0.0].iloc[0]                       # 0-250 Hz
        peak = p.loc[p.acc_drop.idxmax()]                         # overall peak-necessity band
        occ_pitch, occ_peak = float(pitch.occluded_acc), float(peak.occluded_acc)
        wl_pitch, wh_pitch = wilson_ci(occ_pitch, n)
        wl_peak, wh_peak = wilson_ci(occ_peak, n)
        rows.append(dict(
            arch=a, baseline_acc=base, n=n,
            drop_pitch_0_250=float(pitch.acc_drop), occ_acc_pitch=occ_pitch,
            wilson_lo_pitch=wl_pitch, wilson_hi_pitch=wh_pitch,
            peak_band_lo=float(peak.band_lo), peak_band_hi=float(peak.band_hi),
            drop_peak=float(peak.acc_drop), occ_acc_peak=occ_peak,
            wilson_lo_peak=wl_peak, wilson_hi_peak=wh_peak,
            delta_peak_minus_pitch=float(peak.acc_drop) - float(pitch.acc_drop),
            peak_in_formant_250_750=bool(250.0 <= float(peak.band_lo) < 750.0),
        ))
    return pd.DataFrame(rows)


def plot_overlay(df):
    dall = df[df.group == "all"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for lbl, lo, hi, c in SPEECH_BANDS:
        ax.axvspan(lo, hi, color=c, alpha=0.12, lw=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], lbl, ha="center", va="bottom", fontsize=8, color="0.3")
    for a in ARCHS:
        p = dall[dall.arch == a].sort_values("center")
        ax.plot(p.center, p.acc_drop, marker="o", ms=4, lw=1.6, label=a)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("frequency band center (Hz)")
    ax.set_ylabel("digit-accuracy drop when band removed  (necessity)")
    ax.set_title("Frequency-reliance (necessity) profiles — seed 42 SI, n=6000, group=all\n"
                 "band-stop removes FREQUENCY (content+speaker jointly), not identity")
    ax.legend(title="architecture", loc="upper right")
    ax.text(0.99, -0.13, "8 kHz SR: Nyquist=4 kHz; fricative energy >4 kHz absent",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.4")
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "necessity_all_arch.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_by_group(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, a in zip(axes.ravel(), ARCHS):
        for g, style in [("female", dict(ls="--", marker="^")), ("male", dict(ls="-", marker="o"))]:
            p = df[(df.arch == a) & (df.group == g)].sort_values("center")
            ax.plot(p.center, p.acc_drop, ms=3, lw=1.3, label=g, **style)
        ax.axhline(0, color="0.8", lw=0.7)
        ax.set_title(a, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Per-group necessity (availability only)\n"
                 "female n small (3 test speakers): profiles noisier", fontsize=11)
    fig.supxlabel("frequency band center (Hz)")
    fig.supylabel("accuracy drop (necessity)")
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "necessity_by_group.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    csv_path = find_occlusion_csv()
    print(f"[input] {csv_path}")
    df = load_profiles(csv_path)
    summary = summarize(df)
    summary_path = os.path.join(RESULTS_DIR, "necessity_summary.csv")
    summary.to_csv(summary_path, index=False)

    # console readout for the record
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== per-arch necessity summary (group=all, seed 42 SI, n=6000) ===")
    show = summary[["arch", "drop_pitch_0_250", "peak_band_lo", "peak_band_hi", "drop_peak",
                    "delta_peak_minus_pitch", "peak_in_formant_250_750"]]
    print(show.to_string(index=False))
    print("\nWilson 95% CI on occluded accuracy (pitch band vs peak band):")
    for _, r in summary.iterrows():
        print(f"  {r.arch:11s} pitch occ={r.occ_acc_pitch:.4f} "
              f"[{r.wilson_lo_pitch:.4f},{r.wilson_hi_pitch:.4f}]  "
              f"peak occ={r.occ_acc_peak:.4f} [{r.wilson_lo_peak:.4f},{r.wilson_hi_peak:.4f}]")

    f1 = plot_overlay(df)
    f2 = plot_by_group(df)
    print(f"\n[out] {summary_path}\n[out] {f1}\n[out] {f2}")
    print("\nH2-1 check: delta>0 for all archs AND peak_in_formant_250_750 all True ->",
          bool((summary.delta_peak_minus_pitch > 0).all() and summary.peak_in_formant_250_750.all()))


if __name__ == "__main__":
    main()
