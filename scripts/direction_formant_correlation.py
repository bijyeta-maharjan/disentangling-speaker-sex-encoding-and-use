"""Which acoustic property does the validated sex direction load on (F0 vs
formants), and is the within-male F0 result stable across seeds?

Builds on scripts/direction_f0_correlation.py, which found the whitened
sex direction's within-male F0 correlation WEAK (rho ~0.13-0.19, both archs)
despite a strong cross-sex correlation driven mostly by a between-cluster
offset -- i.e. "sex-specific but not clearly pitch-linked", not a clean
pitch axis. This script asks two follow-on
questions: (a) does F0 still beat formants even though it's weak, i.e. is the
signal AT LEAST more pitch-like than formant-like; (b) is the weak within-male
F0 correlation at least stable across retraining seeds, or is even that a
one-run artifact.

AMENDMENT (valid-mask parity, REQUIRED before any F0-vs-formant comparison):
F0 (librosa pyin) and formants (parselmouth Burg) are independent extractors
with independent failure modes -- an utterance can have a defined F0 but an
undefined formant track, or vice versa. Comparing an F0 rho computed on "all
utterances where F0 was defined" against a formant rho computed on "all
utterances where that formant was defined" would confound acoustics with
sample composition: a difference in rho could be a difference in WHICH
utterances survived extraction, not a difference in what the direction
encodes. So every correlation in f0_vs_formant_correlation.csv --- F0, F1, F2,
AND F3 alike --- is computed on the SAME intersection mask (F0 valid AND F1
valid AND F2 valid AND F3 valid), computed once per architecture (the masks
are audio-derived, not model-derived, so numerically identical across cnn1d
and spec_cnn2d, but reported per architecture since they gate
each architecture's comparison independently). The intersection size, the
F0-only drop, the formant-only drop, and the final common-n are printed
before any comparison is trusted. If the intersection loses >25% of the SI
test-fold's 6000 utterances relative to either single-extractor mask, that is
flagged loudly rather than silently absorbed -- it would mean the two
extractors disagree enough that the common set may not represent the full
population the F0 analysis characterized.

CLAIM BOUNDARY: n=3 female test speakers -- within-sample, not a population
claim (FLAG, stamped on every artifact). The decisive number for the pitch-
vs-formant question is the WITHIN-MALE stratum (n=9 speakers, does not depend
on the female count), for the same reason it was decisive in the F0 analysis: cross-
sex and within-female numbers are directional only. Cross-seed stability is
evaluated on the within-male F0 stratum only (the number the F0 analysis
already flagged as decisive).

Signal: 8kHz preprocessed waveform, matching the F0 analysis (the exact
signal the model consumed). F3 sits near the 4kHz
Nyquist at this sample rate -- lower-confidence by construction; formants
extracted with maximum_formant=3800Hz (just under Nyquist) via
parselmouth.Sound.to_formant_burg, max_number_of_formants=5.

Reuses (does not rebuild): scripts/direction_f0_correlation.py's F0 cache,
whitened-direction cache (get_whitened_direction), stratum_stats, SCOPE_ARCHS
/ EXCLUDED_ARCHS / FLAG / SPEAKER_COUNT_NOTE; scripts/unit_selectivity.py's
penultimate_cache_path / si_test_sex_groups; the same trained-penultimate
caches for seeds 42/43/44 (NOT re-extracted).

Not done here: readout sensitivity, digit-conditioned splits,
resnet1d/crnn_gru (unvalidated direction).

    python scripts/direction_formant_correlation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import parselmouth
from scipy.stats import spearmanr

from scripts.unit_selectivity import penultimate_cache_path, si_test_sex_groups
from scripts.direction_f0_correlation import (
    ACOUSTICS_DIR,
    EXCLUDED_ARCHS,
    EXCLUDED_NOTE,
    F0_CACHE_PATH,
    FLAG,
    OUT_DIR,
    PRIMARY_SEED,
    SCOPE_ARCHS,
    SPEAKER_COUNT_NOTE,
    get_whitened_direction,
    stratum_stats,
)
from src.data.preprocess import load_cache
from src.probing.extract_week2 import load_labels as load_seed42_labels
from src.probing.splits import load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"

STABILITY_SEEDS = [43, 44]
ALL_STABILITY_SEEDS = [PRIMARY_SEED] + STABILITY_SEEDS

FORMANT_CACHE_PATH = ACOUSTICS_DIR / "formants_median_si_test_fold.parquet"
MAX_NUM_FORMANTS = 5
MAXIMUM_FORMANT_HZ = 3800.0  # just under the 4kHz Nyquist of the 8kHz signal

INTERSECTION_DROP_FLAG_THRESHOLD = 0.25

OUT_PITCH_VS_FORMANT_CSV = OUT_DIR / "f0_vs_formant_correlation.csv"
OUT_CROSSSEED_CSV = OUT_DIR / "f0_correlation_crossseed.csv"
FIGS_DIR = resolve("figures/output")
FIG_PATH = FIGS_DIR / "direction_vs_f0_and_formants.png"
RUNS_DIR = resolve("results/sex_direction")


# --------------------------------------------------------------------------- #
# formant extraction (idempotent cache), same manifest order as the F0 cache
# --------------------------------------------------------------------------- #
def extract_formant_cache(cfg: dict, labels: pd.DataFrame, test_idx: np.ndarray) -> pd.DataFrame:
    if FORMANT_CACHE_PATH.exists():
        df = pd.read_parquet(FORMANT_CACHE_PATH)
        assert np.array_equal(df["row_idx"].to_numpy(), test_idx), (
            "cached formant row_idx order does not match current SI test fold row order -- "
            "delete the cache and re-run if the manifest changed"
        )
        print(f"[formant] cache hit -> {FORMANT_CACHE_PATH} ({len(df)} utterances)")
        return df

    print(f"[formant] cache miss, extracting median F1-F3 for {len(test_idx)} SI test-fold utterances "
          f"from the 8kHz preprocessed waveform (parselmouth Burg, maximum_formant={MAXIMUM_FORMANT_HZ}Hz) ...")
    cache = load_cache(cfg)
    wav = cache["wav"]
    sr = cfg["preprocess"]["target_sr"]
    f1m = np.empty(len(test_idx), dtype=np.float64)
    f2m = np.empty(len(test_idx), dtype=np.float64)
    f3m = np.empty(len(test_idx), dtype=np.float64)
    t0 = time.time()
    for i, row in enumerate(test_idx):
        x = np.asarray(wav[row], dtype=np.float64)
        snd = parselmouth.Sound(x, sampling_frequency=sr)
        formant = snd.to_formant_burg(max_number_of_formants=MAX_NUM_FORMANTS, maximum_formant=MAXIMUM_FORMANT_HZ)
        n = formant.get_number_of_frames()
        f1s = np.empty(n); f2s = np.empty(n); f3s = np.empty(n)
        for k in range(1, n + 1):
            t = formant.get_time_from_frame_number(k)
            f1s[k - 1] = formant.get_value_at_time(1, t)
            f2s[k - 1] = formant.get_value_at_time(2, t)
            f3s[k - 1] = formant.get_value_at_time(3, t)
        f1m[i] = float(np.nanmedian(f1s)) if np.any(~np.isnan(f1s)) else float("nan")
        f2m[i] = float(np.nanmedian(f2s)) if np.any(~np.isnan(f2s)) else float("nan")
        f3m[i] = float(np.nanmedian(f3s)) if np.any(~np.isnan(f3s)) else float("nan")
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(test_idx)} utterances ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame({
        "row_idx": test_idx,
        "speaker": labels["speaker"].to_numpy()[test_idx],
        "sex": labels["sex"].to_numpy()[test_idx],
        "digit": labels["digit"].to_numpy()[test_idx],
        "f1_median_hz": f1m, "f2_median_hz": f2m, "f3_median_hz": f3m,
    })
    print(f"[formant] done in {time.time()-t0:.0f}s -- "
          f"{int(df['f1_median_hz'].isna().sum())}/{len(df)} F1 undefined, "
          f"{int(df['f2_median_hz'].isna().sum())}/{len(df)} F2 undefined, "
          f"{int(df['f3_median_hz'].isna().sum())}/{len(df)} F3 undefined")
    ACOUSTICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FORMANT_CACHE_PATH)
    print(f"[formant] cache -> {FORMANT_CACHE_PATH}")
    return df


# --------------------------------------------------------------------------- #
# valid-mask parity (AMENDMENT): independent F0 / formant masks, intersected
# --------------------------------------------------------------------------- #
def valid_mask_parity(f0_df: pd.DataFrame, formant_df: pd.DataFrame, arch: str) -> dict:
    assert np.array_equal(f0_df["row_idx"].to_numpy(), formant_df["row_idx"].to_numpy()), \
        "F0 cache and formant cache row order mismatch"
    n_total = len(f0_df)
    f0_valid = f0_df["f0_median_hz"].notna().to_numpy()
    formant_valid = (formant_df["f1_median_hz"].notna() & formant_df["f2_median_hz"].notna()
                      & formant_df["f3_median_hz"].notna()).to_numpy()
    both_valid = f0_valid & formant_valid
    f0_only_drop = int((~f0_valid & formant_valid).sum())
    formant_only_drop = int((f0_valid & ~formant_valid).sum())
    both_invalid_drop = int((~f0_valid & ~formant_valid).sum())
    n_intersection = int(both_valid.sum())
    pct_dropped = 1.0 - n_intersection / n_total
    flagged = pct_dropped > INTERSECTION_DROP_FLAG_THRESHOLD

    print(f"[valid-mask-parity] {arch}: n_total={n_total}  n_f0_valid={int(f0_valid.sum())}  "
          f"n_formant_valid={int(formant_valid.sum())}  n_intersection={n_intersection}  "
          f"f0_only_drop={f0_only_drop}  formant_only_drop={formant_only_drop}  "
          f"both_invalid_drop={both_invalid_drop}  pct_dropped={pct_dropped:.1%}")
    if flagged:
        print(f"[valid-mask-parity] *** FLAG: {arch} intersection drops {pct_dropped:.1%} of utterances "
              f"(> {INTERSECTION_DROP_FLAG_THRESHOLD:.0%} threshold) -- F0 and formant extractors disagree "
              f"enough that the common set may not represent the full F0-analysis population. Reported, not hidden. ***")
    else:
        print(f"[valid-mask-parity] {arch}: intersection drop {pct_dropped:.1%} is within the "
              f"{INTERSECTION_DROP_FLAG_THRESHOLD:.0%} tolerance.")

    return {
        "arch": arch, "n_total": n_total, "n_f0_valid": int(f0_valid.sum()),
        "n_formant_valid": int(formant_valid.sum()), "n_intersection": n_intersection,
        "f0_only_drop": f0_only_drop, "formant_only_drop": formant_only_drop,
        "both_invalid_drop": both_invalid_drop, "pct_dropped": round(pct_dropped, 4),
        "flagged_high_drop": bool(flagged), "mask": both_valid,
    }


# --------------------------------------------------------------------------- #
# (a) pitch vs formant, on the intersection mask, seed 42 (primary)
# --------------------------------------------------------------------------- #
def run_pitch_vs_formant(cfg: dict, labels: pd.DataFrame, spk_str: np.ndarray,
                          female_spk: list[str], male_spk: list[str],
                          f0_df: pd.DataFrame, formant_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    parity_rows = []
    for arch in SCOPE_ARCHS:
        parity = valid_mask_parity(f0_df, formant_df, arch)
        parity_rows.append({k: v for k, v in parity.items() if k != "mask"})
        mask = parity["mask"]

        X = np.load(penultimate_cache_path(PRIMARY_SEED, arch))
        w = get_whitened_direction(arch, PRIMARY_SEED, X, spk_str, female_spk, male_spk, cfg)
        score_all = X @ w
        row_idx = f0_df["row_idx"].to_numpy()[mask]
        score = score_all[row_idx]

        df = pd.DataFrame({
            "row_idx": row_idx, "speaker": f0_df["speaker"].to_numpy()[mask],
            "sex": f0_df["sex"].to_numpy()[mask], "score": score,
            "F0": f0_df["f0_median_hz"].to_numpy()[mask],
            "F1": formant_df["f1_median_hz"].to_numpy()[mask],
            "F2": formant_df["f2_median_hz"].to_numpy()[mask],
            "F3": formant_df["f3_median_hz"].to_numpy()[mask],
        })

        male_df = df[df["sex"] == "male"]
        female_df = df[df["sex"] == "female"]
        strata = {"males_only": (male_df, False), "all_cross_sex": (df, True), "females_only": (female_df, True)}

        best_formant_male_rho, best_formant_name = 0.0, None
        for feature in ["F0", "F1", "F2", "F3"]:
            for stratum, (sdf, directional) in strata.items():
                r = stratum_stats(arch, stratum, sdf[feature].to_numpy(), sdf["score"].to_numpy(),
                                   sdf["speaker"].to_numpy(), directional=directional)
                r["feature"] = feature
                r["n_intersection"] = parity["n_intersection"]
                rows.append(r)
                if feature != "F0" and stratum == "males_only" and abs(r["spearman_rho"]) > abs(best_formant_male_rho):
                    best_formant_male_rho, best_formant_name = r["spearman_rho"], feature

        f0_male_rho = next(r["spearman_rho"] for r in rows
                            if r["arch"] == arch and r["feature"] == "F0" and r["stratum"] == "males_only")
        verdict = ("F0 >> formant (present-not-needed convergence)" if abs(f0_male_rho) > abs(best_formant_male_rho)
                    else "FORMANT >= F0 -- STOP-AND-ESCALATE, digit-relevant band")
        print(f"[pitch-vs-formant] {arch} (n_intersection={parity['n_intersection']}): "
              f"males-only F0 rho={f0_male_rho:+.3f}  strongest formant males-only rho={best_formant_male_rho:+.3f} "
              f"({best_formant_name})  -> {verdict}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PITCH_VS_FORMANT_CSV, index=False)
    print(f"[pitch-vs-formant] table -> {OUT_PITCH_VS_FORMANT_CSV}")

    pd.DataFrame(parity_rows).to_csv(OUT_DIR / "f0_formant_valid_mask_parity.csv", index=False)
    print(f"[pitch-vs-formant] valid-mask parity diagnostics -> {OUT_DIR / 'f0_formant_valid_mask_parity.csv'}")
    return df_out


# --------------------------------------------------------------------------- #
# (b) cross-seed stability of the within-male F0 correlation
# --------------------------------------------------------------------------- #
def run_crossseed_f0(cfg: dict, spk_str: np.ndarray, female_spk: list[str], male_spk: list[str],
                      f0_df: pd.DataFrame) -> pd.DataFrame:
    f0_valid = f0_df["f0_median_hz"].notna().to_numpy()
    row_idx = f0_df["row_idx"].to_numpy()[f0_valid]
    sex = f0_df["sex"].to_numpy()[f0_valid]
    speaker = f0_df["speaker"].to_numpy()[f0_valid]
    f0 = f0_df["f0_median_hz"].to_numpy()[f0_valid]
    male_sel = sex == "male"

    rows = []
    for arch in SCOPE_ARCHS:
        rhos = []
        for seed in ALL_STABILITY_SEEDS:
            X = np.load(penultimate_cache_path(seed, arch))
            w = get_whitened_direction(arch, seed, X, spk_str, female_spk, male_spk, cfg)
            score = (X @ w)[row_idx]
            r = stratum_stats(arch, "males_only", f0[male_sel], score[male_sel], speaker[male_sel], directional=False)
            r["seed"] = seed
            rows.append(r)
            rhos.append(r["spearman_rho"])
        rho_range = max(rhos) - min(rhos)
        print(f"[crossseed] {arch}: within-male F0 rho across seeds {ALL_STABILITY_SEEDS} = "
              f"{[f'{x:+.3f}' for x in rhos]}  (range={rho_range:.3f})")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CROSSSEED_CSV, index=False)
    print(f"[crossseed] table -> {OUT_CROSSSEED_CSV}")
    return df_out


def headline_figure(pf_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(SCOPE_ARCHS), figsize=(7 * len(SCOPE_ARCHS), 5.5))
    if len(SCOPE_ARCHS) == 1:
        axes = [axes]
    for ax, arch in zip(axes, SCOPE_ARCHS):
        sub = pf_df[(pf_df["arch"] == arch) & (pf_df["stratum"] == "males_only")]
        feats = ["F0", "F1", "F2", "F3"]
        vals = [float(sub[sub["feature"] == f]["spearman_rho"].iloc[0]) for f in feats]
        colors = ["#c0392b" if f == "F0" else "#7f8c8d" for f in feats]
        ax.bar(feats, vals, color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{arch} -- males-only |ρ| by acoustic feature", fontsize=11)
        ax.set_ylabel("Spearman ρ (score vs feature, males-only)")
        n_int = int(sub["n_intersection"].iloc[0])
        ax.text(0.02, 0.02, f"n_intersection={n_int} (valid-mask-parity)", transform=ax.transAxes,
                fontsize=8, va="bottom")
    fig.suptitle(
        "F0 vs formants: which band does the validated sex direction load on? (males-only, decisive stratum)\n"
        f"{FLAG}  |  Same intersection utterance set used for every bar (valid-mask parity amendment).",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return FIG_PATH


def main():
    cfg = load_config(DEFAULT_CONFIG)
    si_spec = load_si_spec(cfg)
    female_spk, male_spk = si_test_sex_groups(si_spec)
    labels = load_seed42_labels(cfg)
    spk_str = labels["speaker"].astype(str).to_numpy()

    assert F0_CACHE_PATH.exists(), f"F0 cache missing ({F0_CACHE_PATH}) -- run scripts/direction_f0_correlation.py first"
    f0_df = pd.read_parquet(F0_CACHE_PATH)
    test_idx = f0_df["row_idx"].to_numpy()
    formant_df = extract_formant_cache(cfg, labels, test_idx)

    print(f"[formant] {SPEAKER_COUNT_NOTE}")
    print(f"[formant] {EXCLUDED_NOTE}")

    print("\n================ (a) PITCH vs FORMANT (valid-mask parity) ================")
    pf_df = run_pitch_vs_formant(cfg, labels, spk_str, female_spk, male_spk, f0_df, formant_df)

    print("\n================ (b) CROSS-SEED WITHIN-MALE F0 STABILITY ================")
    cs_df = run_crossseed_f0(cfg, spk_str, female_spk, male_spk, f0_df)

    fig_path = headline_figure(pf_df)
    print(f"\n[formant] headline figure -> {fig_path}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "formant_correlation_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "git_sha": git_sha(), "primary_seed": PRIMARY_SEED, "stability_seeds": STABILITY_SEEDS,
            "flag": FLAG, "speaker_count_note": SPEAKER_COUNT_NOTE, "excluded_note": EXCLUDED_NOTE,
            "scope_archs": SCOPE_ARCHS, "excluded_archs": EXCLUDED_ARCHS,
            "intersection_drop_flag_threshold": INTERSECTION_DROP_FLAG_THRESHOLD,
            "formant_extraction": {"tool": "parselmouth.to_formant_burg", "signal": "8kHz preprocessed waveform",
                                    "max_number_of_formants": MAX_NUM_FORMANTS, "maximum_formant_hz": MAXIMUM_FORMANT_HZ,
                                    "note": "F3 lower-confidence: maximum_formant set just under the 4kHz Nyquist"},
        }, fh, indent=2)

    print("\n================ SUMMARY ================")
    print(pf_df[pf_df["stratum"] == "males_only"][
        ["arch", "feature", "spearman_rho", "n_utts", "n_intersection", "flag"]
    ].to_string(index=False))
    print()
    print(cs_df[["arch", "seed", "spearman_rho", "n_utts", "flag"]].to_string(index=False))
    print(f"\n{FLAG}")


if __name__ == "__main__":
    main()
