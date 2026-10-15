"""Direction-stability fork: is the sex CONCEPT a stable direction in a rotating
basis, even though the unit analysis (unit_selectivity.py) showed the UNITS expressing it are not stable to
retraining (top-20 Jaccard at/below chance, self-rho~=0, seeds 42/43/44)?

Expectation (written down before looking): cnn1d/resnet1d/spec_cnn2d
(real global sex signal) predicted to show HIGH aligned direction-cosine across
seeds (>> permutation floor) despite unstable units. crnn_gru (near-chance
global sex signal) predicted to show an UNSTABLE direction too -- an internal
negative control: you cannot have a stable direction for a signal that is
barely present.

CLAIM BOUNDARY:
  - n=3 female test speakers, IDENTICAL across seeds 42/43/44 (verified in
    unit_selectivity.py). FLAG stamped on every artifact.
  - A stable aligned cosine proves "stable to RETRAINING" only. It is the same
    3 women every seed, so it does NOT prove "stable across female samples."
    It rules out training-run noise as the explanation; it does not rule out
    three-women idiosyncrasy. That confound is explicitly carried to the F0 analysis.
  - Unit indices and raw (unaligned) direction vectors are NOT comparable
    across architectures (different learned bases). All stability comparisons
    here are WITHIN-architecture, across-SEED only.

METHOD NOTES (state these on the output, not just in code comments):
  - Two direction estimators, penultimate, unit-normalized:
      (1) mean-difference / CAV (primary): mean(female X) - mean(male X) on
          the SI test fold (n=3 female / 9 male speakers) -- literally "which
          way is female," the vector the downstream acoustic work will use.
      (2) probe-weight (secondary, corroboration only): the existing
          speaker-disjoint sex-probe's logistic-regression weight vector
          (src/probing/probe.py, fit on SI train+val, verbatim), un-scaled
          back into raw penultimate coordinates (coef / StandardScaler.scale_)
          so it lives in the SAME coordinate system as the CAV. Sign flipped
          so it also points toward "more female," matching the CAV convention
          (encode_labels sorts alphabetically -> female=0/male=1, so raw
          sklearn coef_ points toward MALE; negated here).
  - Cross-seed alignment (orthogonal Procrustes) is fit on the FULL 30k-row,
    mean-centered penultimate matrix (all speakers, not just the SI test
    fold): the best-conditioned estimate of the basis change between two
    training runs of the SAME architecture, and it applies uniformly to a
    test-fold-derived direction (the CAV) and a train-fold-derived one (the
    probe weight) alike. Rotating one seed's direction through this R into
    the other seed's space is what makes the subsequent cosine meaningful --
    an unaligned cosine between two different learned bases is not.
  - Permutation floor (mean-difference estimator only, for compute cost):
    redraw which 3 of the 12 SI test speakers are "pseudo-female" (same
    group sizes as real), recompute the pseudo-CAV in both seeds' RAW spaces,
    rotate through the REAL (label-independent) Procrustes R, cosine. 200
    reps x 3 seed-pairs, averaged the same way the real cosine is averaged
    over pairs -- an apples-to-apples floor for the specific estimator +
    alignment procedure, not just a dimensionality-based guess.
  - Alignment quality travels with every cosine: normalized Procrustes
    residual + linear CKA, per pair, averaged per architecture. A cosine
    computed through a bad alignment is not trustworthy on its own.

Reuses (does not rebuild):
  - src/probing/probe.py (build_probe), src/probing/splits.py (load_si_spec,
    generalizable_split), src/probing/extract_week2.py (load_labels).
  - scripts/unit_selectivity.py's per-seed cache-path convention and SI
    test-fold sex grouping (si_test_sex_groups) -- same cached trained
    penultimate features, NOT re-extracted.
  - scripts/week2_cka_funnel.py's linear_cka math (reimplemented locally, matching
    the existing repo convention of a small local copy per script).

Out of scope: F0/formant work, segmentation, CRP/DFT, readout-sensitivity,
digit-conditioned splits, by-group generalization claims. This script only
asks whether the sex direction is stable across retraining, within each
architecture.

    python scripts/direction_stability.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

from scripts.unit_selectivity import (
    ARCH_ORDER,
    FLAG,
    OUT_DIR,
    SEEDS,
    penultimate_cache_path,
    si_test_sex_groups,
)
from src.probing import probe as P
from src.probing.extract_week2 import load_labels as load_seed42_labels
from src.probing.splits import generalizable_split, load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"
N_PERM = 200
RUNS_DIR = resolve("results/sex_direction")
FIGS_DIR = resolve("figures/output")
UNIT_STABILITY_CSV = OUT_DIR / "unit_seed_stability.csv"

GRU_NOTE = ("crnn_gru is the PRE-REGISTERED negative control: near-chance global sex-decodability "
            "means it should show an UNSTABLE direction too, not just unstable units.")


# --------------------------------------------------------------------------- #
# loading + explicit cross-seed row-alignment check (no re-extraction)
# --------------------------------------------------------------------------- #
def load_penultimate(seed: int, arch: str) -> np.ndarray:
    return np.load(penultimate_cache_path(seed, arch))


def assert_cross_seed_alignment(labels: pd.DataFrame, arch: str) -> int:
    """Every cached seed array for this arch must have exactly N rows, in the
    SAME manifest order. The row-for-row digit/speaker equality against this
    same label table was already asserted at cache-WRITE time (shuffle=False,
    ensure_trained_penultimate in scripts/unit_selectivity.py); this is the
    cheap invariant re-check required before any cross-seed operation here."""
    n = len(labels)
    for seed in SEEDS:
        X = load_penultimate(seed, arch)
        assert X.shape[0] == n, (
            f"cross-seed misalignment: seed={seed} arch={arch} has {X.shape[0]} rows, "
            f"expected {n} (labels table) -- do not compare directions until this is fixed"
        )
    return n


# --------------------------------------------------------------------------- #
# direction estimators (raw penultimate coordinates, unit-normalized)
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def mean_diff_direction(X: np.ndarray, spk_str: np.ndarray, female_spk: list[str],
                         male_spk: list[str]) -> np.ndarray:
    f_mask = np.isin(spk_str, female_spk)
    m_mask = np.isin(spk_str, male_spk)
    d = X[f_mask].mean(axis=0) - X[m_mask].mean(axis=0)
    return _unit(d)


def probe_weight_direction(X: np.ndarray, labels: pd.DataFrame, si_spec: dict, cfg: dict) -> np.ndarray:
    """Existing speaker-disjoint sex-probe (train on SI train+val), weight
    vector un-scaled back to raw penultimate coordinates, sign-flipped to
    point toward FEMALE (encode_labels: female=0/male=1 alphabetically, so
    sklearn's raw coef_ points toward male)."""
    train_pos, _test_pos = generalizable_split(labels, "sex", si_spec)
    y, classes = P.encode_labels(labels["sex"].tolist())
    assert classes == ["female", "male"], f"unexpected class order {classes}"
    pipe = P.build_probe("linear", cfg, standardize=True)
    pipe.fit(X[train_pos], y[train_pos])
    clf = pipe.named_steps["logisticregression"]
    scaler = pipe.named_steps["standardscaler"]
    w_raw = clf.coef_.ravel() / scaler.scale_
    return _unit(-w_raw)  # flip: point toward female, matching the CAV convention


# --------------------------------------------------------------------------- #
# cross-seed orthogonal Procrustes alignment (fit on the FULL dataset)
# --------------------------------------------------------------------------- #
def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    Xc, Yc = _center(X), _center(Y)
    hsic_xy = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, "fro")
    hsic_yy = np.linalg.norm(Yc.T @ Yc, "fro")
    return float(hsic_xy / (hsic_xx * hsic_yy))


def procrustes_fit(Xa_full: np.ndarray, Xb_full: np.ndarray):
    """R (orthogonal) minimizing ||Xa_c @ R - Xb_c||_F on the full, mean-
    centered dataset. Returns R plus two alignment-quality diagnostics that
    must travel with any cosine computed through R."""
    Xa_c, Xb_c = _center(Xa_full), _center(Xb_full)
    R, _ = orthogonal_procrustes(Xa_c, Xb_c)
    resid = float(np.linalg.norm(Xa_c @ R - Xb_c, "fro") / np.linalg.norm(Xb_c, "fro"))
    cka = linear_cka(Xa_c, Xb_c)
    return R, resid, cka


def aligned_cosine(dir_a: np.ndarray, R: np.ndarray, dir_b: np.ndarray) -> float:
    rotated = _unit(dir_a @ R)
    return float(np.dot(rotated, dir_b))


# --------------------------------------------------------------------------- #
# permutation floor (mean-difference estimator; speaker-level, group-size
# preserving draws, mirrors the unit analysis's chance-floor logic)
# --------------------------------------------------------------------------- #
def permutation_floor_pair(Xa: np.ndarray, Xb: np.ndarray, spk_str: np.ndarray,
                            test_spk: list[str], n_female: int, R: np.ndarray,
                            n_perm: int, rng: np.random.Generator) -> np.ndarray:
    cosines = np.empty(n_perm, dtype=float)
    test_spk_arr = np.array(test_spk)
    for i in range(n_perm):
        perm = rng.permutation(test_spk_arr)
        pseudo_f, pseudo_m = list(perm[:n_female]), list(perm[n_female:])
        da = mean_diff_direction(Xa, spk_str, pseudo_f, pseudo_m)
        db = mean_diff_direction(Xb, spk_str, pseudo_f, pseudo_m)
        cosines[i] = aligned_cosine(da, R, db)
    return cosines


# --------------------------------------------------------------------------- #
# WHY the permutation floor is high, not just THAT it is: a rotation-sanity
# diagnostic. Pure random unit vectors through the SAME R should show ~0
# aligned cosine if R is a genuine, non-degenerate alignment; if group-
# mean-difference vectors (real OR pseudo speaker splits) instead show high
# aligned cosine while random vectors don't, that localizes the effect to the
# ESTIMATOR (small-sample group contrasts on an extremely low-rank
# representation), not to a broken or degenerate rotation.
# --------------------------------------------------------------------------- #
def rotation_sanity_check(R: np.ndarray, n_draws: int, rng: np.random.Generator) -> dict:
    d = R.shape[0]
    cos = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        va, vb = _unit(rng.normal(size=d)), _unit(rng.normal(size=d))
        cos[i] = aligned_cosine(va, R, vb)
    return {"mean": float(cos.mean()), "std": float(cos.std())}


def top_pc_variance_fraction(X: np.ndarray, k: int = 10) -> float:
    Xc = _center(X)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    var = s ** 2
    return float(var[:k].sum() / var.sum())


# --------------------------------------------------------------------------- #
# direction stability across seeds
# --------------------------------------------------------------------------- #
def run_direction_stability(cfg: dict) -> dict:
    si_spec = load_si_spec(cfg)
    female_spk, male_spk = si_test_sex_groups(si_spec)
    test_spk = si_spec["roles"]["test"]
    labels = load_seed42_labels(cfg)
    spk_str = labels["speaker"].astype(str).to_numpy()
    pairs = [(42, 43), (42, 44), (43, 44)]
    rng = np.random.default_rng(cfg["seed"])

    unit_stab = pd.read_csv(UNIT_STABILITY_CSV) if UNIT_STABILITY_CSV.exists() else None
    if unit_stab is None:
        print(f"[direction] WARNING: {UNIT_STABILITY_CSV} not found -- unit_jaccard/unit_chance "
              "columns will be NaN in the output table.")

    rows, fig_data = [], {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for arch in ARCH_ORDER:
        n = assert_cross_seed_alignment(labels, arch)
        X = {seed: load_penultimate(seed, arch) for seed in SEEDS}
        print(f"[direction] {arch}: cross-seed row alignment OK (n={n}, seeds={SEEDS}, "
              "digit/speaker equality vs the shared label table was asserted at cache-write time)")

        mean_dir = {seed: mean_diff_direction(X[seed], spk_str, female_spk, male_spk) for seed in SEEDS}
        probe_dir = {seed: probe_weight_direction(X[seed], labels, si_spec, cfg) for seed in SEEDS}
        top10_pc_var = top_pc_variance_fraction(X[42], k=10)

        pair_cos_mean, pair_cos_probe, pair_resid, pair_cka = [], [], [], []
        pair_floor_mean, pair_floor_p95, pair_floor_std = [], [], []
        pair_rand_sanity = []
        for a, b in pairs:
            R, resid, cka = procrustes_fit(X[a], X[b])
            pair_resid.append(resid)
            pair_cka.append(cka)
            pair_cos_mean.append(aligned_cosine(mean_dir[a], R, mean_dir[b]))
            pair_cos_probe.append(aligned_cosine(probe_dir[a], R, probe_dir[b]))
            floor_cosines = permutation_floor_pair(X[a], X[b], spk_str, test_spk, len(female_spk),
                                                    R, N_PERM, rng)
            pair_floor_mean.append(float(floor_cosines.mean()))
            pair_rand_sanity.append(rotation_sanity_check(R, 30, rng))
            pair_floor_p95.append(float(np.percentile(floor_cosines, 95)))
            pair_floor_std.append(float(floor_cosines.std()))
            print(f"  {arch} seeds {a}v{b}: aligned_cosine(mean-diff)={pair_cos_mean[-1]:+.3f}  "
                  f"aligned_cosine(probe-weight)={pair_cos_probe[-1]:+.3f}  "
                  f"perm_null(mean-diff, n={N_PERM}): mean={pair_floor_mean[-1]:+.3f} "
                  f"std={pair_floor_std[-1]:.3f} p95={pair_floor_p95[-1]:+.3f}  "
                  f"resid={resid:.3f}  CKA={cka:.3f}")

        mean_aligned_cosine = float(np.mean(pair_cos_mean))
        mean_aligned_cosine_probe = float(np.mean(pair_cos_probe))
        perm_floor_mean = float(np.mean(pair_floor_mean))
        perm_floor_p95 = float(np.mean(pair_floor_p95))  # the floor a real cosine must CLEAR
        alignment_quality_resid = float(np.mean(pair_resid))
        alignment_quality_cka = float(np.mean(pair_cka))
        rand_sanity_mean = float(np.mean([r["mean"] for r in pair_rand_sanity]))
        rand_sanity_std = float(np.mean([r["std"] for r in pair_rand_sanity]))

        if unit_stab is not None and arch in unit_stab["arch"].values:
            urow = unit_stab.set_index("arch").loc[arch]
            unit_jaccard, unit_chance = float(urow["mean_jaccard_stable_to_retraining"]), float(urow["chance_jaccard_null"])
        else:
            unit_jaccard, unit_chance = float("nan"), float("nan")

        # Verdict uses the 95th-pct permutation null (a proper one-sided test), NOT an
        # arbitrary margin over the mean floor. BUT: the rotation-sanity check shows the
        # rotation itself is clean (pure-random-vector aligned cosine ~= 0), while the
        # permutation null is high (~0.7-0.9) for ANY group-mean-difference vector among
        # these same 12 fixed speakers, real split or arbitrary re-split. With ~94% of
        # variance in the top 10 PCs (extreme low rank), a small-sample (n=12) group
        # contrast is dominated by that shared, cross-seed-reproducible subspace almost
        # regardless of WHICH grouping is chosen. So "does not clear the permutation floor"
        # here does not cleanly mean "no real sex direction" -- it means the raw-CAV cosine
        # test cannot distinguish a real sex axis from a generic reproducible contrast of
        # THESE 12 individuals. Report as a metric-level confound, not a clean stop signal.
        direction_clears_floor = mean_aligned_cosine > perm_floor_p95
        units_stable = (not np.isnan(unit_jaccard)) and unit_jaccard > unit_chance * 1.5
        metric_caveat = ("permutation null itself is high (0.7-0.9) for ANY 3-vs-9 re-split of these "
                          "12 speakers -- extreme representation low-rank (see top10_pc_variance_fraction) "
                          "means the raw mean-diff CAV cosine cannot cleanly separate a sex-specific "
                          "direction from a generic reproducible individual-grouping contrast at n=12; "
                          "rotation itself is clean (random-vector sanity check ~0, see rotation_sanity_mean)")
        if direction_clears_floor and not units_stable:
            pattern = "STABLE DIRECTION, UNSTABLE UNITS (predicted pattern for real-signal archs)"
        elif not direction_clears_floor:
            pattern = "DOES NOT CLEAR PERMUTATION FLOOR (primary estimator inconclusive -- see metric_caveat)"
        else:
            pattern = "stable direction AND stable units (units result would need re-examination)"

        rows.append({
            "arch": arch,
            "mean_aligned_cosine_meandiff": round(mean_aligned_cosine, 4),
            "mean_aligned_cosine_probeweight": round(mean_aligned_cosine_probe, 4),
            "perm_floor_mean": round(perm_floor_mean, 4),
            "perm_floor_p95": round(perm_floor_p95, 4),
            "alignment_quality_resid": round(alignment_quality_resid, 4),
            "alignment_quality_cka": round(alignment_quality_cka, 4),
            "rotation_sanity_mean": round(rand_sanity_mean, 4), "rotation_sanity_std": round(rand_sanity_std, 4),
            "top10_pc_variance_fraction": round(top10_pc_var, 4),
            "metric_caveat": metric_caveat,
            "unit_jaccard_top20": unit_jaccard, "unit_chance_jaccard": unit_chance,
            "n_female_speakers": len(female_spk), "n_male_speakers": len(male_spk),
            "n_perm": N_PERM,
            "pattern": pattern,
            "gru_note": GRU_NOTE if arch == "crnn_gru" else "",
            "flag": FLAG,
        })
        fig_data[arch] = {
            "unit_jaccard": unit_jaccard, "unit_chance": unit_chance,
            "dir_cosine": mean_aligned_cosine, "dir_floor": perm_floor_p95,
            "dir_cosine_probe": mean_aligned_cosine_probe,
        }
        print(f"[direction] {arch}: mean_aligned_cosine(CAV)={mean_aligned_cosine:+.3f} "
              f"vs perm_null mean={perm_floor_mean:+.3f}/p95={perm_floor_p95:+.3f}  |  "
              f"probe-weight cosine={mean_aligned_cosine_probe:+.3f} "
              f"(no floor computed, compute-cost tradeoff)  |  alignment resid={alignment_quality_resid:.3f} "
              f"CKA={alignment_quality_cka:.3f}  |  units: jaccard={unit_jaccard} vs chance={unit_chance}")
        print(f"  rotation sanity (pure-random-vector aligned cosine, should be ~0): "
              f"mean={rand_sanity_mean:+.3f} std={rand_sanity_std:.3f}  |  "
              f"top-10-PC variance fraction (seed 42)={top10_pc_var:.3f}")
        print(f"  -> {pattern}")
        if arch == "crnn_gru":
            print(f"  NOTE: {GRU_NOTE}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "direction_stability.csv"
    df.to_csv(out_csv, index=False)

    fig_path = headline_figure(fig_data, FIGS_DIR / "direction_vs_unit_stability.png")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "direction_stability_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "git_sha": git_sha(), "seeds": SEEDS, "flag": FLAG, "n_perm": N_PERM,
            "female_test_speakers": female_spk, "male_test_speakers": male_spk,
            "procrustes_fit_on": "full 30k-row mean-centered penultimate matrix (all speakers)",
            "framing_note": "aligned cosine proves stable-to-RETRAINING only; same 3 female "
                             "speakers every seed, so this does NOT prove stable-across-female-samples "
                             "(that confound is unresolved and carries to the F0 analysis)",
            "basis_caveat": "unit indices and raw (unaligned) direction vectors are not comparable "
                             "across architectures; all comparisons here are within-arch, across-seed",
            "gru_note": GRU_NOTE,
            "diagnostic_finding": "pure-random-vector aligned cosine ~=0 (rotation is not degenerate), "
                                   "but group-mean-difference vectors from ANY 3-vs-9 re-split of the 12 "
                                   "fixed SI test speakers -- real sex split or arbitrary -- show high "
                                   "(0.7-0.96) cross-seed aligned cosine. Localizes the high permutation "
                                   "floor to the ESTIMATOR (small-sample group contrast on an extremely "
                                   "low-rank representation, ~94% variance in top 10 PCs), not to a broken "
                                   "alignment. See per-arch rotation_sanity_mean/top10_pc_variance_fraction "
                                   "and metric_caveat below.",
            "rows": rows,
        }, fh, indent=2)
    print(f"\n[direction] table -> {out_csv}")
    print(f"[direction] headline figure -> {fig_path}")
    return {"table": df, "fig": str(fig_path)}


def headline_figure(fig_data: dict, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, arch in zip(axes.flat, ARCH_ORDER):
        d = fig_data[arch]
        labels_x = ["units\n(top-20 Jaccard)", "direction\n(aligned cosine, CAV)"]
        observed = [d["unit_jaccard"], d["dir_cosine"]]
        floor = [d["unit_chance"], d["dir_floor"]]
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, observed, width=w, color="#2980b9", label="observed")
        ax.bar(x + w / 2, floor, width=w, color="#bdc3c7", label="chance floor / perm. null p95")
        ax.axhline(0, color="black", lw=0.8)
        ax.scatter([1], [d["dir_cosine_probe"]], color="#e74c3c", zorder=5, marker="D", s=40,
                   label="probe-weight cosine (secondary)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=9)
        title = arch + ("  [negative control]" if arch == "crnn_gru" else "")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.3, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Units unstable to retraining vs. is the sex DIRECTION stable? — seeds 42/43/44 SI, trained penultimate\n"
        f"{FLAG}\n"
        "Aligned cosine = stable-to-RETRAINING only (same 3 female speakers every seed) — NOT stable-across-female-samples.\n"
        "Permutation floor is itself high for ANY 3-vs-9 re-split of these 12 speakers (extreme representation "
        "low-rank) — see metric_caveat.",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg = load_config(args.config)
    h = run_direction_stability(cfg)
    print("\n================ DIRECTION STABILITY ================")
    print(h["table"][["arch", "mean_aligned_cosine_meandiff", "mean_aligned_cosine_probeweight",
                       "perm_floor_mean", "perm_floor_p95", "rotation_sanity_mean",
                       "top10_pc_variance_fraction", "unit_jaccard_top20", "unit_chance_jaccard",
                       "pattern"]].to_string(index=False))
    print(f"\n{FLAG}")
    print("\nSee metric_caveat column / diagnostic_finding in direction_stability_run_meta.json for why "
          "the permutation floor is high (estimator confound, not a broken alignment).")


if __name__ == "__main__":
    main()
