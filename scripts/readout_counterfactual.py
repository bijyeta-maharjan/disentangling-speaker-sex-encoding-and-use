"""Confirm the analytic "not used at the readout" result behaviourally
(counterfactual perturbation) and check it survives retraining (cross-seed),
plus make the mean-Jacobian averaging auditable via the per-point magnitude
distribution.

scripts/readout_sensitivity.py found every digit's null-
normalized sensitivity s_k = |J_mean_k . u_hat| within (mostly well below)
the 1000-random-direction null, seed 42, both architectures -- consistent
with "encoded but not used at the readout." Three follow-on checks close the
gap between that analytic result and a confirmatory claim:

  (a) DISTRIBUTION AUDIT. s_k averages the Jacobian
      over utterances BEFORE dotting with u_hat, which can hide cancellation:
      if a digit's local sensitivity direction flips sign across utterances,
      s_k collapses toward zero even if each utterance individually has
      non-trivial |J_k(x).u_hat|. readout_sensitivity.py's sign_agreement_frac column already
      hinted at this (several digits near 0.5 = coin-flip agreement). Here we
      report the per-point |J_k(x).u_hat| distribution directly (mean, median,
      std, p10/p90) AND a matched null-normalized abs-magnitude statistic
      (mean_x|J_k(x).v| against 1000 random v, same recipe as the signed
      null in readout_sensitivity.py) so "low because genuinely small" and "low because of cancellation
      that a bigger single nudge would still not reveal" are told apart.
  (b) COUNTERFACTUAL. The behavioral shadow of the analytic result: take
      correctly-classified SI test utterances, add delta.u_hat to the
      (raw-space) penultimate representation for delta swept as a multiple of
      the empirical male<->female projection gap (delta=1 = "move by the
      average sex gap"), record mean Delta(correct-digit logit) and resulting
      accuracy. A direction the readout ignores should show a flat logit
      curve and accuracy that does not collapse even at delta=+-1 (or beyond).
  (c) CROSS-SEED. Repeat the analytic sensitivity for seeds 43/44;
      report the per-digit range across seeds 42/43/44. Stable-near-null
      across seeds is the confirmatory pattern; a seed-unstable near-null
      result would undercut the claim even if seed 42 alone looked
      clean.

If magnitudes are uniformly small (not just the signed mean, the raw
per-point distribution too), the counterfactual sweep is flat, and the
cross-seed range stays inside the null -- the analytic read holds, not a
seed-42 accident.

CLAIM BOUNDARY (identical to readout_sensitivity.py, restated -- every artifact must carry
it): n=3 female test speakers -- within-sample, not a population claim. This
test barely depends on female n; its confidence comes from null-
normalization, cross-seed stability, and cross-architecture agreement.
Bounds ONLY the linear readout's sensitivity to the penultimate sex
direction -- does NOT establish upstream routing. Scope: cnn1d + spec_cnn2d
only (resnet1d/crnn_gru direction unvalidated, excluded).

Reuses (does not rebuild): scripts/readout_sensitivity.py's
load_whitened_direction, load_trained_head (asserts W pulled from the
trained head), head_jacobian (the same Jacobian-as-W operationalization); persisted
whitened directions and cached penultimate features; scripts/
unit_selectivity.py's si_test_sex_groups.

Not done here: input-relevance attribution, acoustic work, by-group
generalization, resnet1d/crnn_gru.

    python scripts/readout_counterfactual.py
    python scripts/readout_counterfactual.py --redraw   # counterfactual figure only, from the saved table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.unit_selectivity import OUT_DIR, penultimate_cache_path, si_test_sex_groups
from scripts.readout_sensitivity import (
    BOUND_NOTE,
    EXCLUDED_ARCHS,
    EXCLUDED_NOTE,
    FLAG,
    HEAD_METHOD_NOTE,
    N_DIGITS,
    N_NULL,
    NULL_RNG_SEED,
    PRIMARY_SEED,
    SCOPE_ARCHS,
    SPEAKER_COUNT_NOTE,
    head_jacobian,
    load_trained_head,
    load_whitened_direction,
)
from src.probing.extract_week2 import load_labels as load_seed42_labels
from src.probing.splits import load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"

STABILITY_SEEDS = [43, 44]
ALL_SEEDS = [PRIMARY_SEED] + STABILITY_SEEDS
DELTA_MULTIPLES = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]

ARCH_COLORS = {"cnn1d": "#2a78d6", "spec_cnn2d": "#008300"}  # fixed hues, matches geom_* convention

OUT_DISTRIBUTION_CSV = OUT_DIR / "readout_distribution_seed42.csv"
OUT_COUNTERFACTUAL_CSV = OUT_DIR / "readout_counterfactual.csv"
OUT_CROSSSEED_CSV = OUT_DIR / "readout_crossseed.csv"
FIGS_DIR = resolve("figures/output")
FIG_DIST_PATH = FIGS_DIR / "readout_distribution.png"
FIG_CF_PATH = FIGS_DIR / "counterfactual_shift.png"
RUNS_DIR = resolve("results/sex_direction")


# --------------------------------------------------------------------------- #
# shared setup
# --------------------------------------------------------------------------- #
def si_test_fold(cfg):
    si_spec = load_si_spec(cfg)
    female_spk, male_spk = si_test_sex_groups(si_spec)
    labels = load_seed42_labels(cfg)
    spk_str = labels["speaker"].astype(str).to_numpy()
    mask = np.isin(spk_str, female_spk) | np.isin(spk_str, male_spk)
    test_idx = np.where(mask)[0]
    assert mask.sum() == 6000 and len(female_spk) == 3 and len(male_spk) == 9
    sex = labels["sex"].to_numpy()[test_idx]
    digit = labels["digit"].to_numpy()[test_idx]
    return labels, test_idx, sex, digit, female_spk, male_spk


# --------------------------------------------------------------------------- #
# (a) per-point magnitude distribution, null-normalized abs-magnitude stat
# --------------------------------------------------------------------------- #
def distribution_audit(cfg, device, test_idx, seed=PRIMARY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(NULL_RNG_SEED)
    rows = []
    fig_data = {}
    for arch in SCOPE_ARCHS:
        u = load_whitened_direction(arch, seed)
        d = u.shape[0]
        X = np.load(penultimate_cache_path(seed, arch))[test_idx]
        model = load_trained_head(cfg, arch, seed, device)
        J = head_jacobian(model, X, device)  # (10, N, d)
        J_mean = J.mean(axis=1)

        V = rng.normal(size=(N_NULL, d)).astype(np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        per_digit_abs = {}
        for k in range(N_DIGITS):
            dots = J[k] @ u  # (N,)
            abs_dots = np.abs(dots)
            per_digit_abs[k] = abs_dots

            s_signed = float(np.abs(J_mean[k] @ u))
            s_abs_mean = float(abs_dots.mean())

            null_dots = J[k] @ V.T  # (N, N_NULL)
            null_abs_mean = np.abs(null_dots).mean(axis=0)  # (N_NULL,)
            pct_abs = float((null_abs_mean <= s_abs_mean).mean() * 100)

            null_signed = np.abs(J_mean[k] @ V.T)  # (N_NULL,)
            pct_signed = float((null_signed <= s_signed).mean() * 100)

            rows.append({
                "arch": arch, "digit": k, "seed": seed,
                "s_k_signed": round(s_signed, 5), "s_k_signed_percentile": round(pct_signed, 2),
                "abs_dot_mean": round(s_abs_mean, 5), "abs_dot_percentile": round(pct_abs, 2),
                "abs_dot_median": round(float(np.median(abs_dots)), 5),
                "abs_dot_std": round(float(abs_dots.std()), 5),
                "abs_dot_p10": round(float(np.percentile(abs_dots, 10)), 5),
                "abs_dot_p90": round(float(np.percentile(abs_dots, 90)), 5),
                "sign_agreement_frac": round(float((np.sign(dots) == np.sign(J_mean[k] @ u)).mean()), 3),
                "n_utts": int(len(dots)), "flag": FLAG,
            })
            print(f"[distribution] {arch} digit={k}: signed s_k={s_signed:.4f} (pct={pct_signed:.1f})  "
                  f"|  per-point |dot| mean={s_abs_mean:.4f} median={np.median(abs_dots):.4f} "
                  f"(pct={pct_abs:.1f}, matched null)")
        fig_data[arch] = {"per_digit_abs": per_digit_abs, "s_signed": {k: rows[-N_DIGITS + k]["s_k_signed"] for k in range(N_DIGITS)}}

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DISTRIBUTION_CSV, index=False)
    print(f"[distribution] table -> {OUT_DISTRIBUTION_CSV}")
    distribution_figure(fig_data)
    return df


def distribution_figure(fig_data: dict) -> Path:
    fig, axes = plt.subplots(1, len(SCOPE_ARCHS), figsize=(12, 4.5), sharey=False)
    if len(SCOPE_ARCHS) == 1:
        axes = [axes]
    for ax, arch in zip(axes, SCOPE_ARCHS):
        d = fig_data[arch]
        data = [d["per_digit_abs"][k] for k in range(N_DIGITS)]
        bp = ax.boxplot(data, positions=range(N_DIGITS), widths=0.6, showfliers=False,
                         patch_artist=True)
        for box in bp["boxes"]:
            box.set(facecolor=ARCH_COLORS[arch], alpha=0.25, edgecolor=ARCH_COLORS[arch])
        for med in bp["medians"]:
            med.set(color=ARCH_COLORS[arch], linewidth=1.5)
        signed = [d["s_signed"][k] for k in range(N_DIGITS)]
        ax.scatter(range(N_DIGITS), signed, marker="D", s=45, color="#c0392b", zorder=5,
                   label="signed |J_mean_k . u_hat| (s_k)")
        ax.set_title(arch, fontsize=11)
        ax.set_xlabel("digit")
        ax.set_ylabel("|J_k(x) . u_hat|  (per utterance)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Per-point sensitivity distribution vs. the signed mean s_k (seed 42) -- is the low s_k\n"
        "genuinely small, or small because of cancellation across utterances?\n"
        f"{FLAG}",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIST_PATH, dpi=150)
    plt.close(fig)
    print(f"[distribution] figure -> {FIG_DIST_PATH}")
    return FIG_DIST_PATH


# --------------------------------------------------------------------------- #
# (b) counterfactual perturbation, seed 42 (primary)
# --------------------------------------------------------------------------- #
def counterfactual(cfg, device, test_idx, sex, digit, seed=PRIMARY_SEED) -> pd.DataFrame:
    rows = []
    fig_data = {}
    for arch in SCOPE_ARCHS:
        u = load_whitened_direction(arch, seed)
        X = np.load(penultimate_cache_path(seed, arch))[test_idx]
        model = load_trained_head(cfg, arch, seed, device)

        score = X @ u
        gap = float(score[sex == "female"].mean() - score[sex == "male"].mean())
        print(f"[counterfactual] {arch}: empirical male<->female projection gap = {gap:+.4f} "
              f"(female mean={score[sex=='female'].mean():+.4f}, male mean={score[sex=='male'].mean():+.4f})")

        with torch.no_grad():
            x0 = torch.tensor(X, dtype=torch.float32, device=device)
            logits0 = model.classifier(x0).cpu().numpy()
        pred0 = logits0.argmax(axis=1)
        correct0 = pred0 == digit
        n_correct = int(correct0.sum())
        acc0 = float(correct0.mean())
        print(f"[counterfactual] {arch}: baseline accuracy={acc0:.4f}, "
              f"{n_correct}/{len(digit)} correctly-classified utterances used for the sweep")

        Xc = X[correct0]
        yc = digit[correct0]
        correct_logit0 = logits0[correct0, yc]

        deltas, mean_dlogit, std_dlogit, acc_at_delta = [], [], [], []
        for m in DELTA_MULTIPLES:
            delta_raw = m * gap
            Xp = Xc + delta_raw * u[None, :]
            with torch.no_grad():
                xp = torch.tensor(Xp, dtype=torch.float32, device=device)
                logits_p = model.classifier(xp).cpu().numpy()
            correct_logit_p = logits_p[np.arange(len(yc)), yc]
            d_logit = correct_logit_p - correct_logit0
            pred_p = logits_p.argmax(axis=1)
            acc_p = float((pred_p == yc).mean())

            deltas.append(m)
            mean_dlogit.append(float(d_logit.mean()))
            std_dlogit.append(float(d_logit.std()))
            acc_at_delta.append(acc_p)

            rows.append({
                "arch": arch, "delta_over_gap": m, "delta_raw": round(delta_raw, 5),
                "mean_delta_correct_logit": round(float(d_logit.mean()), 5),
                "std_delta_correct_logit": round(float(d_logit.std()), 5),
                "accuracy_at_delta": round(acc_p, 5),
                "delta_accuracy": round(acc_p - 1.0, 5),  # baseline acc on this subset is 1.0 by construction
                "n_utts_correct_baseline": n_correct, "empirical_projection_gap": round(gap, 5),
                "flag": FLAG,
            })
        print(f"[counterfactual] {arch}: at delta=+-1 (full sex-gap swap), "
              f"mean_dlogit={mean_dlogit[deltas.index(1)]:+.4f}/{mean_dlogit[deltas.index(-1)]:+.4f}  "
              f"accuracy={acc_at_delta[deltas.index(1)]:.4f}/{acc_at_delta[deltas.index(-1)]:.4f}")

        fig_data[arch] = {"delta": deltas, "mean_dlogit": mean_dlogit, "std_dlogit": std_dlogit, "acc": acc_at_delta}

    df = pd.DataFrame(rows)
    df.to_csv(OUT_COUNTERFACTUAL_CSV, index=False)
    print(f"[counterfactual] table -> {OUT_COUNTERFACTUAL_CSV}")
    counterfactual_figure(fig_data)
    return df


def counterfactual_figure(fig_data: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for arch in SCOPE_ARCHS:
        d = fig_data[arch]
        delta = np.array(d["delta"])
        mdl = np.array(d["mean_dlogit"])
        sdl = np.array(d["std_dlogit"])
        color = ARCH_COLORS[arch]
        ax1.plot(delta, mdl, marker="o", color=color, label=arch)
        ax1.fill_between(delta, mdl - sdl, mdl + sdl, color=color, alpha=0.12)
        ax2.plot(delta, d["acc"], marker="o", color=color, label=arch)

    for ax in (ax1, ax2):
        ax.axvline(1, color="#888888", ls="--", lw=1)
        ax.axvline(-1, color="#888888", ls="--", lw=1)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("delta (multiples of the empirical male<->female projection gap)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    ax1.set_ylabel("mean Delta(correct-digit logit)")
    ax1.set_title("Logit response to moving along u_hat")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy under the same perturbation")
    ax2.set_ylim(-0.02, 1.02)

    fig.suptitle(
        "Counterfactual: does perturbing along the validated sex direction move the digit readout?\n"
        f"{FLAG}  |  dashed lines mark delta=+-1 (\"swap sex position\")",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(FIG_CF_PATH, dpi=150)
    plt.close(fig)
    print(f"[counterfactual] figure -> {FIG_CF_PATH}")
    return FIG_CF_PATH


# --------------------------------------------------------------------------- #
# (c) cross-seed analytic sensitivity (same formula, seeds 42/43/44)
# --------------------------------------------------------------------------- #
def crossseed(cfg, device, test_idx) -> pd.DataFrame:
    rng = np.random.default_rng(NULL_RNG_SEED)
    rows = []
    for arch in SCOPE_ARCHS:
        per_seed = {}
        for seed in ALL_SEEDS:
            u = load_whitened_direction(arch, seed)
            d = u.shape[0]
            X = np.load(penultimate_cache_path(seed, arch))[test_idx]
            model = load_trained_head(cfg, arch, seed, device)
            J = head_jacobian(model, X, device)
            J_mean = J.mean(axis=1)

            V = rng.normal(size=(N_NULL, d)).astype(np.float32)
            V /= np.linalg.norm(V, axis=1, keepdims=True)

            s = np.abs(J_mean @ u)
            null_s = np.abs(J_mean @ V.T)
            for k in range(N_DIGITS):
                pct = float((null_s[k] <= s[k]).mean() * 100)
                rows.append({"arch": arch, "digit": k, "seed": seed, "s_k": round(float(s[k]), 5),
                             "percentile": round(pct, 2), "exceeds_95": bool(pct > 95), "flag": FLAG})
                per_seed.setdefault(k, {})[seed] = pct

        for k in range(N_DIGITS):
            pcts = [per_seed[k][sd] for sd in ALL_SEEDS]
            rng_range = max(pcts) - min(pcts)
            print(f"[crossseed] {arch} digit={k}: percentile across seeds {ALL_SEEDS} = "
                  f"{[f'{p:.1f}' for p in pcts]}  (range={rng_range:.1f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CROSSSEED_CSV, index=False)
    print(f"[crossseed] table -> {OUT_CROSSSEED_CSV}")
    return df


def redraw_counterfactual_from_saved() -> Path:
    """Redraw the counterfactual figure from results/sex_direction/readout_counterfactual.csv."""
    df = pd.read_csv(OUT_COUNTERFACTUAL_CSV)
    fig_data = {}
    for arch in SCOPE_ARCHS:
        sub = df[df["arch"] == arch].sort_values("delta_over_gap")
        fig_data[arch] = {"delta": sub["delta_over_gap"].tolist(),
                          "mean_dlogit": sub["mean_delta_correct_logit"].tolist(),
                          "std_dlogit": sub["std_delta_correct_logit"].tolist(),
                          "acc": sub["accuracy_at_delta"].tolist()}
    return counterfactual_figure(fig_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redraw", action="store_true", help="only redraw the counterfactual figure from the saved table")
    args = ap.parse_args()
    if args.redraw:
        redraw_counterfactual_from_saved()
        return
    cfg = load_config(DEFAULT_CONFIG)
    device = torch.device("cpu")
    labels, test_idx, sex, digit, female_spk, male_spk = si_test_fold(cfg)

    print(f"[counterfactual] {SPEAKER_COUNT_NOTE}")
    print(f"[counterfactual] {EXCLUDED_NOTE}")
    print(f"[counterfactual] {HEAD_METHOD_NOTE}")

    print("\n================ (a) DISTRIBUTION AUDIT (seed 42) ================")
    dist_df = distribution_audit(cfg, device, test_idx)

    print("\n================ (b) COUNTERFACTUAL SWEEP (seed 42) ================")
    cf_df = counterfactual(cfg, device, test_idx, sex, digit)

    print("\n================ (c) CROSS-SEED STABILITY (42/43/44) ================")
    cs_df = crossseed(cfg, device, test_idx)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "readout_counterfactual_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "git_sha": git_sha(), "primary_seed": PRIMARY_SEED, "stability_seeds": STABILITY_SEEDS,
            "flag": FLAG, "speaker_count_note": SPEAKER_COUNT_NOTE, "excluded_note": EXCLUDED_NOTE,
            "bound_note": BOUND_NOTE, "head_method_note": HEAD_METHOD_NOTE,
            "scope_archs": SCOPE_ARCHS, "excluded_archs": EXCLUDED_ARCHS,
            "delta_multiples": DELTA_MULTIPLES, "n_null": N_NULL, "null_rng_seed": NULL_RNG_SEED,
        }, fh, indent=2)

    print("\n================ SUMMARY ================")
    print("-- distribution audit (seed 42): signed vs per-point |dot| percentile --")
    print(dist_df[["arch", "digit", "s_k_signed_percentile", "abs_dot_percentile", "sign_agreement_frac"]]
          .to_string(index=False))

    print("\n-- counterfactual at delta=+-1 (full sex-gap swap) --")
    for arch in SCOPE_ARCHS:
        sub = cf_df[cf_df["arch"] == arch]
        for m in (1, -1):
            r = sub[sub["delta_over_gap"] == m].iloc[0]
            print(f"  {arch} delta={m:+d}: mean_delta_correct_logit={r['mean_delta_correct_logit']:+.4f}  "
                  f"accuracy={r['accuracy_at_delta']:.4f} (delta_accuracy={r['delta_accuracy']:+.4f})")

    print("\n-- cross-seed: any digit exceeding 95th percentile in ANY seed? --")
    exceptions = cs_df[cs_df["exceeds_95"]]
    if exceptions.empty:
        print("  none -- all digits, all seeds, both architectures stayed within the null.")
    else:
        print(exceptions[["arch", "digit", "seed", "percentile"]].to_string(index=False))

    print(f"\n{FLAG}")
    print(SPEAKER_COUNT_NOTE)
    print(EXCLUDED_NOTE)
    print(BOUND_NOTE)

    print("\n--- VERDICT INPUTS ---")
    for arch in SCOPE_ARCHS:
        max_abs_pct = float(dist_df[dist_df["arch"] == arch]["abs_dot_percentile"].max())
        cf_sub = cf_df[cf_df["arch"] == arch]
        max_abs_dlogit = float(cf_sub["mean_delta_correct_logit"].abs().max())
        min_acc = float(cf_sub["accuracy_at_delta"].min())
        cs_sub = cs_df[cs_df["arch"] == arch]
        any_exceed = bool(cs_sub["exceeds_95"].any())
        print(f"  {arch}: max per-point-magnitude null percentile (any digit) = {max_abs_pct:.1f}  |  "
              f"max |mean Delta(correct logit)| over sweep = {max_abs_dlogit:.4f}  |  "
              f"min accuracy over sweep = {min_acc:.4f}  |  "
              f"any seed/digit exceeds 95th pct = {any_exceed}")


if __name__ == "__main__":
    main()
