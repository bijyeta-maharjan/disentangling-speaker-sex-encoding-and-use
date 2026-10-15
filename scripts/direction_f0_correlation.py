"""Does the validated whitened sex direction track F0?

Goal: link the sex CONCEPT to a real acoustic property (gender ->
fundamental frequency), not just to the fact that it separates 3 women from 9
men. The re-split control (scripts/direction_resplit_control.py) proved the
whitened sex direction is separable from all 219 non-sex 3-vs-9 re-splits of
the same 12 SI test speakers in cnn1d and spec_cnn2d (resnet1d/crnn_gru sat
inside their nulls -- their direction is NOT validated and is OUT OF SCOPE
here). That control ruled out "generic 12-person contrast"; it did not rule
out "artifact of these particular 3 women" vs "genuine pitch axis" -- both
would beat a random re-split. This script disambiguates the two by asking
whether the projection onto the direction tracks F0, especially WITHIN MALES
ALONE, where the direction was never fit to do anything.

CLAIM BOUNDARY -- read before trusting any number this script prints:
  Female fold is n=3 speakers (1500 utterances). All cross-sex and
  within-female correlations are WITHIN-SAMPLE, not population estimates --
  the high-F0 end of the range is sampled by only 3 women. FLAG stamped on
  every artifact; see FLAG below.
  The WITHIN-MALE correlation (n=9 speakers, 4500 utterances, SI test fold)
  is the well-powered anchor and does NOT depend on the female count -- it is
  reported first and most prominently, and it is the number that decides this
  analysis.
  Per-utterance correlation is well-powered in general (thousands of
  utterances); this is why the analysis runs on the whitened-direction
  PROJECTION (a continuous score), not on individual unit activations (the
  unstable target the unit seed-stability null already ruled out).

NOTE on speaker counts: AudioMNIST is 48 male / 12 female speakers
DATASET-WIDE, but "Speaker-disjoint SI test fold for everything" is a CRITICAL
DISCIPLINE for this analysis (the whitened direction was fit ONLY on the 12 SI
test speakers, and generalizing the correlation beyond that fold would use
activations under a direction that was never validated against them). The SI
test fold has 9 male / 3 female speakers (4500 / 1500 utterances). This
script therefore reports n=9 male speakers, not 48, for the within-male
anchor -- flagged explicitly in the run report so the discrepancy with a
naive "48 speakers" read of the dataset is not silently swallowed.

Scope: cnn1d and spec_cnn2d ONLY (resnet1d, crnn_gru direction unvalidated --
excluded, not silently dropped: see EXCLUDED_ARCHS / run_meta).

Signal used for F0: the 8 kHz preprocessed waveform (data/processed/
waveforms_8k_8192.npy via src.data.preprocess.load_cache) -- the exact signal
the model consumed, NOT the original 48 kHz audio. F0 (85-255 Hz) is fully
preserved at 8 kHz (Nyquist 4 kHz).

Reuses (does not rebuild):
  - scripts/direction_resplit_control.py's fit_whitened_direction -- the
    EXACT whitened-direction estimator that passed the re-split control (fit
    directly on the 12 SI test speakers, real sex label). No persisted .npy
    of this vector existed on disk after that script ran (it only wrote
    cosines to CSV), so recomputing via this identical, deterministic
    (lbfgs, no randomness) function is a reuse, not a new fitting
    procedure. The resulting vector is cached here
    (whitened_direction_{arch}_seed{seed}.npy) so it need not be refit again.
  - scripts/unit_selectivity.py's ARCH_ORDER / penultimate_cache_path /
    si_test_sex_groups / OUT_DIR (same cached trained-penultimate features,
    NOT re-extracted).
  - src/probing/extract_week2.py's load_labels, src/probing/splits.py's
    load_si_spec, src/data/preprocess.py's load_cache.

Not done here: formants and cross-seed stability (direction_formant_correlation.py),
readout sensitivity, digit-conditioned splits, resnet1d/crnn_gru.

    python scripts/direction_f0_correlation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from scripts.direction_resplit_control import fit_whitened_direction
from scripts.unit_selectivity import OUT_DIR, penultimate_cache_path, si_test_sex_groups
from src.data.preprocess import load_cache
from src.probing.extract_week2 import load_labels as load_seed42_labels
from src.probing.splits import load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"

SCOPE_ARCHS = ["cnn1d", "spec_cnn2d"]
EXCLUDED_ARCHS = ["resnet1d", "crnn_gru"]
EXCLUDED_NOTE = (
    "resnet1d and crnn_gru EXCLUDED from this analysis: their whitened sex direction sat "
    "INSIDE the random-re-split null in the re-split control (not separable from "
    "an arbitrary 3-vs-9 grouping of the same 12 speakers), so there is no validated "
    "direction to project for them. Stated explicitly, not silently dropped."
)
PRIMARY_SEED = 42

FLAG = "n=3 female test speakers — within-sample, not a population claim"
SPEAKER_COUNT_NOTE = (
    "AudioMNIST is 48 male / 12 female speakers dataset-wide, but this analysis follows the "
    "'speaker-disjoint SI test fold for everything' discipline: the whitened direction was "
    "fit ONLY on the 12 SI test speakers, so correlations here use that same fold -- "
    "n=9 male / n=3 female speakers (4500 / 1500 utterances), not the dataset-wide 48/12."
)

F0_FMIN, F0_FMAX = 65.0, 300.0
F0_FRAME_LENGTH, F0_HOP_LENGTH = 1024, 256

ACOUSTICS_DIR = resolve("data/processed/acoustics_si")
F0_CACHE_PATH = ACOUSTICS_DIR / "f0_median_si_test_fold.parquet"
DIRECTION_CACHE_DIR = OUT_DIR  # results/sex_direction
OUT_CSV = OUT_DIR / "f0_correlation.csv"
FIGS_DIR = resolve("figures/output")
FIG_PATH = FIGS_DIR / "direction_vs_f0.png"
RUNS_DIR = resolve("results/sex_direction")


# --------------------------------------------------------------------------- #
# F0 extraction (idempotent cache), ordered/paired with the manifest
# --------------------------------------------------------------------------- #
def extract_f0_cache(cfg: dict, labels: pd.DataFrame, test_idx: np.ndarray) -> pd.DataFrame:
    if F0_CACHE_PATH.exists():
        df = pd.read_parquet(F0_CACHE_PATH)
        assert np.array_equal(df["row_idx"].to_numpy(), test_idx), (
            "cached F0 row_idx order does not match current SI test fold row order -- "
            "delete the cache and re-run if the manifest changed"
        )
        print(f"[f0] cache hit -> {F0_CACHE_PATH} ({len(df)} utterances)")
        return df

    print(f"[f0] cache miss, extracting median F0 for {len(test_idx)} SI test-fold utterances "
          f"from the 8kHz preprocessed waveform (pyin, fmin={F0_FMIN}, fmax={F0_FMAX}) ...")
    cache = load_cache(cfg)
    wav = cache["wav"]
    t0 = time.time()
    f0_med, voiced_frac = np.empty(len(test_idx), dtype=np.float64), np.empty(len(test_idx), dtype=np.float64)
    for i, row in enumerate(test_idx):
        x = np.asarray(wav[row], dtype=np.float64)
        f0, voiced_flag, _ = librosa.pyin(
            x, fmin=F0_FMIN, fmax=F0_FMAX, sr=cfg["preprocess"]["target_sr"],
            frame_length=F0_FRAME_LENGTH, hop_length=F0_HOP_LENGTH, fill_na=np.nan,
        )
        voiced = f0[voiced_flag]
        f0_med[i] = float(np.nanmedian(voiced)) if voiced.size > 0 else float("nan")
        voiced_frac[i] = float(np.mean(voiced_flag))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(test_idx)} utterances ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame({
        "row_idx": test_idx,
        "speaker": labels["speaker"].to_numpy()[test_idx],
        "sex": labels["sex"].to_numpy()[test_idx],
        "digit": labels["digit"].to_numpy()[test_idx],
        "f0_median_hz": f0_med,
        "voiced_frac": voiced_frac,
    })
    n_nan = int(df["f0_median_hz"].isna().sum())
    print(f"[f0] done in {time.time()-t0:.0f}s -- {n_nan}/{len(df)} utterances had no voiced F0 (dropped from correlation)")

    ACOUSTICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F0_CACHE_PATH)
    print(f"[f0] cache -> {F0_CACHE_PATH}")
    return df


# --------------------------------------------------------------------------- #
# whitened direction: reuse the exact re-split-control estimator, cache it
# --------------------------------------------------------------------------- #
def get_whitened_direction(arch: str, seed: int, X: np.ndarray, spk_str: np.ndarray,
                            female_spk: list[str], male_spk: list[str], cfg: dict) -> np.ndarray:
    path = DIRECTION_CACHE_DIR / f"whitened_direction_{arch}_seed{seed}.npy"
    if path.exists():
        print(f"[direction] {arch} seed={seed}: cache hit -> {path}")
        return np.load(path)
    w = fit_whitened_direction(X, spk_str, female_spk, male_spk, cfg)
    np.save(path, w)
    print(f"[direction] {arch} seed={seed}: fit via direction_resplit_control.fit_whitened_direction "
          f"(no prior persisted artifact found) -> cached to {path}")
    return w


# --------------------------------------------------------------------------- #
# correlation strata
# --------------------------------------------------------------------------- #
def stratum_stats(arch: str, stratum: str, f0: np.ndarray, score: np.ndarray,
                   speakers: np.ndarray, directional: bool) -> dict:
    rho, p_rho = spearmanr(f0, score)
    r, p_r = pearsonr(f0, score)
    return {
        "arch": arch, "stratum": stratum,
        "spearman_rho": round(float(rho), 4), "spearman_p": float(p_rho),
        "pearson_r": round(float(r), 4), "pearson_p": float(p_r),
        "n_utts": int(len(f0)), "n_speakers": int(pd.unique(speakers).shape[0]),
        "directional": directional, "flag": FLAG,
    }


def run_f0_correlation(cfg: dict) -> pd.DataFrame:
    si_spec = load_si_spec(cfg)
    female_spk, male_spk = si_test_sex_groups(si_spec)
    labels = load_seed42_labels(cfg)
    spk_str = labels["speaker"].astype(str).to_numpy()

    mask_f = np.isin(spk_str, female_spk)
    mask_m = np.isin(spk_str, male_spk)
    mask = mask_f | mask_m
    test_idx = np.where(mask)[0]
    assert len(female_spk) == 3 and len(male_spk) == 9, f"unexpected SI test sex split: {len(female_spk)}F/{len(male_spk)}M"
    assert mask.sum() == 6000, f"expected 6000 SI test-fold utterances, got {mask.sum()}"
    print(f"[f0] SI test fold: {len(female_spk)} female speakers ({int(mask_f.sum())} utts), "
          f"{len(male_spk)} male speakers ({int(mask_m.sum())} utts)")
    print(f"[f0] {SPEAKER_COUNT_NOTE}")
    print(f"[f0] {EXCLUDED_NOTE}")

    f0_df = extract_f0_cache(cfg, labels, test_idx)
    # ordered, paired alignment assertion: F0 row order == mask-derived test_idx order
    assert np.array_equal(f0_df["row_idx"].to_numpy(), test_idx), "F0 cache misaligned with SI test-fold row order"
    assert np.array_equal(f0_df["speaker"].to_numpy().astype(str), spk_str[test_idx]), "F0/speaker misalignment"
    assert np.array_equal(f0_df["sex"].to_numpy().astype(str), labels["sex"].to_numpy()[test_idx].astype(str)), "F0/sex misalignment"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, fig_data = [], {}

    for arch in SCOPE_ARCHS:
        X = np.load(penultimate_cache_path(PRIMARY_SEED, arch))
        assert X.shape[0] == len(labels), f"{arch}: penultimate row count {X.shape[0]} != labels {len(labels)}"

        w = get_whitened_direction(arch, PRIMARY_SEED, X, spk_str, female_spk, male_spk, cfg)
        score_all = X @ w
        score_test = score_all[f0_df["row_idx"].to_numpy()]  # SAME order as f0_df, by construction

        df = f0_df.copy()
        df["score"] = score_test
        valid = df["f0_median_hz"].notna()
        n_dropped = int((~valid).sum())
        df = df[valid].reset_index(drop=True)
        print(f"[f0] {arch}: {n_dropped} utterances dropped (no voiced F0), {len(df)} retained for correlation")

        male_df = df[df["sex"] == "male"]
        female_df = df[df["sex"] == "female"]

        r_male = stratum_stats(arch, "males_only", male_df["f0_median_hz"].to_numpy(),
                                male_df["score"].to_numpy(), male_df["speaker"].to_numpy(), directional=False)
        r_all = stratum_stats(arch, "all_cross_sex", df["f0_median_hz"].to_numpy(),
                               df["score"].to_numpy(), df["speaker"].to_numpy(), directional=True)
        r_female = stratum_stats(arch, "females_only", female_df["f0_median_hz"].to_numpy(),
                                  female_df["score"].to_numpy(), female_df["speaker"].to_numpy(), directional=True)
        rows.extend([r_male, r_all, r_female])

        print(f"[f0] {arch}: MALES-ONLY (anchor) spearman_rho={r_male['spearman_rho']:+.3f} "
              f"(n={r_male['n_utts']} utts, {r_male['n_speakers']} speakers)  |  "
              f"all rho={r_all['spearman_rho']:+.3f} (n=3F/9M, DIRECTIONAL)  |  "
              f"females-only rho={r_female['spearman_rho']:+.3f} (n=3 speakers, DIRECTIONAL)")

        fig_data[arch] = df

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[f0] table -> {OUT_CSV}")

    fig_path = headline_figure(fig_data)
    print(f"[f0] headline figure -> {fig_path}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "f0_correlation_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "git_sha": git_sha(), "seed": PRIMARY_SEED, "flag": FLAG,
            "speaker_count_note": SPEAKER_COUNT_NOTE, "excluded_note": EXCLUDED_NOTE,
            "scope_archs": SCOPE_ARCHS, "excluded_archs": EXCLUDED_ARCHS,
            "female_test_speakers": female_spk, "male_test_speakers": male_spk,
            "f0_extraction": {"tool": "librosa.pyin", "signal": "8kHz preprocessed waveform",
                               "fmin": F0_FMIN, "fmax": F0_FMAX,
                               "frame_length": F0_FRAME_LENGTH, "hop_length": F0_HOP_LENGTH},
            "rows": rows,
        }, fh, indent=2)

    return df_out


def headline_figure(fig_data: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, len(SCOPE_ARCHS), figsize=(7 * len(SCOPE_ARCHS), 6))
    if len(SCOPE_ARCHS) == 1:
        axes = [axes]
    for ax, arch in zip(axes, SCOPE_ARCHS):
        df = fig_data[arch]
        male_df = df[df["sex"] == "male"]
        female_df = df[df["sex"] == "female"]
        ax.scatter(male_df["f0_median_hz"], male_df["score"], s=10, alpha=0.35,
                   color="#2980b9", label=f"male (n={male_df['speaker'].nunique()} spk)")
        ax.scatter(female_df["f0_median_hz"], female_df["score"], s=10, alpha=0.55,
                   color="#c0392b", label=f"female (n={female_df['speaker'].nunique()} spk, n=3)")

        rho_m, _ = spearmanr(male_df["f0_median_hz"], male_df["score"])
        rho_all, _ = spearmanr(df["f0_median_hz"], df["score"])
        rho_f, _ = spearmanr(female_df["f0_median_hz"], female_df["score"])

        ax.set_title(arch, fontsize=12)
        ax.set_xlabel("median F0 (Hz), 8kHz signal")
        ax.set_ylabel("projection onto whitened sex direction (score = w·x)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        ax.text(0.02, 0.98, f"MALES-ONLY ρ = {rho_m:+.3f}  (anchor, n=9 speakers)",
                transform=ax.transAxes, va="top", ha="left", fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.text(0.02, 0.88,
                f"all (cross-sex) ρ = {rho_all:+.3f}  [n=3 female, directional]\n"
                f"females-only ρ = {rho_f:+.3f}  [n=3 female, directional]",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    fig.suptitle(
        "Does the validated whitened sex direction track F0 -- especially within males alone?\n"
        f"{FLAG}\n"
        "Scope: cnn1d + spec_cnn2d only (resnet1d/crnn_gru direction unvalidated, excluded).",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return FIG_PATH


def main():
    cfg = load_config(DEFAULT_CONFIG)
    df = run_f0_correlation(cfg)
    print("\n================ F0 CORRELATION ================")
    print(df.to_string(index=False))
    print(f"\n{FLAG}")
    print(SPEAKER_COUNT_NOTE)
    print(EXCLUDED_NOTE)
    print("\n--- HEADLINE: within-male F0 correlation (well-powered, n=9 speakers, does NOT depend on female count) ---")
    for arch in SCOPE_ARCHS:
        row = df[(df["arch"] == arch) & (df["stratum"] == "males_only")].iloc[0]
        verdict = "PITCH-LINKED" if abs(row["spearman_rho"]) >= 0.3 else "NOT CLEARLY PITCH-LINKED (weak/null within-male signal)"
        print(f"  {arch}: within-male spearman rho = {row['spearman_rho']:+.3f} (p={row['spearman_p']:.2e}, "
              f"n={row['n_utts']} utts / {row['n_speakers']} speakers) -> {verdict}")


if __name__ == "__main__":
    main()
