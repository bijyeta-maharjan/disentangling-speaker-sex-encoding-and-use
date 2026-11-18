"""Exhaustive re-split control for the sex direction: does the whitened
sex-probe direction reproduce across seeds in a way an arbitrary 3-vs-9
re-split of the SAME 12 SI test speakers does not?

Motivation (see direction_stability.py): the raw mean-difference CAV
turned out confounded -- penultimate space is near-rank-10 (94-98% variance in
the top 10 PCs), so ANY 3-vs-9 speaker grouping's mean-difference vector
reproduces across seeds via the same Procrustes alignment (0.6-0.96 aligned
cosine), indistinguishable from the real sex split. The whitened (probe-weight)
estimator partially escaped this in direction_stability.py (per-unit
standardisation with StandardScaler before fitting). This script tests whether that escape is
real: locate the real sex direction's cross-seed cosine within the FULL
distribution of what every OTHER possible 3-vs-9 split of these same 12
speakers produces.

Design choice (stated up front, not hidden): there are only C(12,3) = 220
distinct ways to choose which 3 of the 12 SI test speakers form the minority
group. A ~500-draw Monte Carlo of a 220-point discrete space would just
resample most partitions twice for no added information. We therefore
ENUMERATE ALL 220 partitions and drop the one that equals the true sex split,
leaving n=219 -- the exact null distribution, strictly better than any
subsample of it.

CLAIM BOUNDARY -- TWO CAVEATS THAT TRAVEL TOGETHER, ALWAYS:
  (1) n=3 female test speakers -- selectivity is within-sample, not a
      population claim. FLAG stamped on every artifact.
  (2) This control removes the LOW-RANK / GENERIC-12-PERSON-CONTRAST confound
      ONLY. A positive result (sex cosine >> random re-split distribution)
      means "the whitened sex direction is reproducible in a way an arbitrary
      split of these people is not" -- it does NOT mean the direction
      generalizes beyond these three women. That confound is a sample-size
      problem no amount of within-sample control can fix.
  Both caveats are printed and written into every artifact; neither one
  substitutes for the other.

Reuses (does not rebuild): scripts/direction_stability.py's Procrustes
alignment (procrustes_fit, aligned_cosine, _unit), per-seed cache loading and
cross-seed alignment assertion, and scripts/unit_selectivity.py's SI
test-fold sex grouping. src/probing/probe.py's build_probe (the existing
harness) for the whitened logistic-regression direction.

Whitened direction here is fit DIRECTLY on the 12 test speakers' 6000
utterances (label = which side of the current 3-vs-9 split), NOT via the
train+val generalization split used by direction_stability.py's probe_weight_direction --
that is what makes the real-split and random-resplit numbers apples-to-apples
(same population, same procedure, only the group LABEL changes). This is a
within-sample direction estimate; no generalization claim is made or implied.

Out of scope: F0/formant work, segmentation, CRP/DFT, readout-sensitivity,
digit-conditioned splits, by-group generalization claims, and the raw
mean-difference CAV as a working direction (it appears ONLY as a labelled
reference line in the headline figure, carried over from direction_stability.py).

    python scripts/direction_resplit_control.py
    python scripts/direction_resplit_control.py --redraw   # figure only, from the saved per-point file
"""
from __future__ import annotations

import argparse
import itertools
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

from scripts.direction_stability import _unit, aligned_cosine, procrustes_fit
from scripts.unit_selectivity import ARCH_ORDER, FLAG, OUT_DIR, SEEDS, penultimate_cache_path, si_test_sex_groups
from src.probing import probe as P
from src.probing.extract_week2 import load_labels as load_seed42_labels
from src.probing.splits import load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"
PAIRS = [(42, 43), (42, 44), (43, 44)]
RUNS_DIR = resolve("results/sex_direction")
FIGS_DIR = resolve("figures/output")
DIRECTION_STABILITY_CSV = OUT_DIR / "direction_stability.csv"

THREE_WOMEN_CAVEAT = (
    "This control removes the low-rank / generic-12-person-contrast confound ONLY. "
    "It does NOT remove the three-women confound: a positive result means the whitened "
    "direction is reproducible in a way an arbitrary split of THESE 12 people is not -- "
    "it does not mean the direction generalizes beyond these three women."
)


# --------------------------------------------------------------------------- #
# loading + alignment (verbatim convention from direction_stability.py)
# --------------------------------------------------------------------------- #
def load_penultimate(seed: int, arch: str) -> np.ndarray:
    return np.load(penultimate_cache_path(seed, arch))


def assert_cross_seed_alignment(labels: pd.DataFrame, arch: str) -> int:
    n = len(labels)
    for seed in SEEDS:
        X = load_penultimate(seed, arch)
        assert X.shape[0] == n, (
            f"cross-seed misalignment: seed={seed} arch={arch} has {X.shape[0]} rows, expected {n}"
        )
    return n


# --------------------------------------------------------------------------- #
# whitened direction fit DIRECTLY on the 12 test speakers (group_a vs group_b)
# --------------------------------------------------------------------------- #
def fit_whitened_direction(X: np.ndarray, spk_str: np.ndarray, group_a: list[str],
                            group_b: list[str], cfg: dict) -> np.ndarray:
    mask_a = np.isin(spk_str, group_a)
    mask_b = np.isin(spk_str, group_b)
    mask = mask_a | mask_b
    y = mask_a[mask].astype(int)  # 1 = group_a ("positive"/female-analogue), 0 = group_b
    pipe = P.build_probe("linear", cfg, standardize=True)
    pipe.fit(X[mask], y)
    clf = pipe.named_steps["logisticregression"]
    scaler = pipe.named_steps["standardscaler"]
    w_raw = clf.coef_.ravel() / scaler.scale_
    return _unit(w_raw)


def enumerate_resplits(test_spk: list[str], female_spk: list[str]) -> list[tuple[list[str], list[str]]]:
    """All C(12,3)=220 ways to pick the 3-member group, minus the true sex split."""
    female_set = frozenset(female_spk)
    test_set = frozenset(test_spk)
    splits = []
    for combo in itertools.combinations(sorted(test_spk), 3):
        group_a = frozenset(combo)
        if group_a == female_set:
            continue
        group_b = test_set - group_a
        splits.append((sorted(group_a), sorted(group_b)))
    return splits


# --------------------------------------------------------------------------- #
# re-split control
# --------------------------------------------------------------------------- #
def run_resplit_control(cfg: dict) -> dict:
    si_spec = load_si_spec(cfg)
    female_spk, male_spk = si_test_sex_groups(si_spec)
    test_spk = si_spec["roles"]["test"]
    labels = load_seed42_labels(cfg)
    spk_str = labels["speaker"].astype(str).to_numpy()

    resplits = enumerate_resplits(test_spk, female_spk)
    n_resplits = len(resplits)
    assert n_resplits == 219, f"expected 219 non-sex 3-vs-9 partitions, got {n_resplits}"
    print(f"[resplit] enumerated {n_resplits} distinct 3-vs-9 speaker partitions "
          f"(C(12,3)-1=219, exact null -- see module docstring for why this replaces a Monte Carlo)")

    raw_cav_table = pd.read_csv(DIRECTION_STABILITY_CSV) if DIRECTION_STABILITY_CSV.exists() else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, fig_data, point_rows = [], {}, []
    t_start = time.time()

    for arch in ARCH_ORDER:
        n = assert_cross_seed_alignment(labels, arch)
        X = {seed: load_penultimate(seed, arch) for seed in SEEDS}
        print(f"[resplit] {arch}: cross-seed row alignment OK (n={n}, seeds={SEEDS})")

        # Procrustes R per pair, fit ONCE (label-independent), reused for every split.
        R, resid, cka = {}, {}, {}
        for a, b in PAIRS:
            R[(a, b)], resid[(a, b)], cka[(a, b)] = procrustes_fit(X[a], X[b])
        alignment_quality_resid = float(np.mean(list(resid.values())))
        alignment_quality_cka = float(np.mean(list(cka.values())))

        # real sex direction, per seed, then aligned cosine averaged over pairs
        sex_dir = {seed: fit_whitened_direction(X[seed], spk_str, female_spk, male_spk, cfg) for seed in SEEDS}
        sex_pair_cos = [aligned_cosine(sex_dir[a], R[(a, b)], sex_dir[b]) for a, b in PAIRS]
        sex_whitened_cosine = float(np.mean(sex_pair_cos))

        # null: every other 3-vs-9 partition of the same 12 speakers
        null_cosines = np.empty(n_resplits, dtype=float)
        t0 = time.time()
        for i, (ga, gb) in enumerate(resplits):
            dirs = {seed: fit_whitened_direction(X[seed], spk_str, ga, gb, cfg) for seed in SEEDS}
            pair_cos = [aligned_cosine(dirs[a], R[(a, b)], dirs[b]) for a, b in PAIRS]
            null_cosines[i] = np.mean(pair_cos)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {arch}: {i+1}/{n_resplits} resplits done ({elapsed:.0f}s elapsed)")

        random_p50 = float(np.percentile(null_cosines, 50))
        random_p95 = float(np.percentile(null_cosines, 95))
        sex_percentile = float((null_cosines <= sex_whitened_cosine).mean() * 100)
        verdict = "SEPARABLE (sex-specific)" if sex_whitened_cosine > random_p95 else "NOT SEPARABLE (inside generic-contrast null)"

        raw_cav_cosine = float("nan")
        if raw_cav_table is not None and arch in raw_cav_table["arch"].values:
            raw_cav_cosine = float(raw_cav_table.set_index("arch").loc[arch, "mean_aligned_cosine_meandiff"])

        rows.append({
            "arch": arch,
            "sex_whitened_cosine": round(sex_whitened_cosine, 4),
            "random_p50": round(random_p50, 4), "random_p95": round(random_p95, 4),
            "sex_percentile": round(sex_percentile, 2),
            "n_resplits": n_resplits,
            "alignment_quality_resid": round(alignment_quality_resid, 4),
            "alignment_quality_cka": round(alignment_quality_cka, 4),
            "raw_cav_cosine_reference": round(raw_cav_cosine, 4) if not np.isnan(raw_cav_cosine) else None,
            "verdict": verdict,
            "n_female_speakers": len(female_spk), "n_male_speakers": len(male_spk),
            "flag": FLAG, "three_women_caveat": THREE_WOMEN_CAVEAT,
        })
        fig_data[arch] = {"null": null_cosines, "sex_cosine": sex_whitened_cosine, "raw_cav": raw_cav_cosine}
        point_rows.append({"arch": arch, "kind": "sex", "resplit_index": -1,
                           "group_a": ";".join(sorted(female_spk)), "cosine": sex_whitened_cosine})
        point_rows += [{"arch": arch, "kind": "null", "resplit_index": i, "group_a": ";".join(ga),
                        "cosine": float(c)} for i, ((ga, _gb), c) in enumerate(zip(resplits, null_cosines))]
        print(f"[resplit] {arch}: sex_whitened_cosine={sex_whitened_cosine:+.3f}  "
              f"random p50={random_p50:+.3f} p95={random_p95:+.3f}  "
              f"sex_percentile={sex_percentile:.1f}  alignment resid={alignment_quality_resid:.3f} "
              f"CKA={alignment_quality_cka:.3f}  -> {verdict}")

    print(f"\n[resplit] total elapsed: {time.time() - t_start:.0f}s")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "resplit_control.csv"
    df.to_csv(out_csv, index=False)

    points = pd.DataFrame(point_rows)
    points.to_csv(NULL_POINTS_CSV, index=False)
    null_figure(points, df)
    fig_path = NULL_POINTS_CSV

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "resplit_control_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "git_sha": git_sha(), "seeds": SEEDS, "flag": FLAG, "three_women_caveat": THREE_WOMEN_CAVEAT,
            "n_resplits": n_resplits,
            "design_note": "enumerated all C(12,3)-1=219 non-sex 3-vs-9 partitions (exact null) "
                            "rather than a 500-draw Monte Carlo of the same 220-point discrete space",
            "female_test_speakers": female_spk, "male_test_speakers": male_spk,
            "rows": rows,
        }, fh, indent=2)
    print(f"[resplit] table -> {out_csv}")
    print(f"[resplit] per-point null cosines -> {fig_path}")
    return {"table": df, "fig": str(fig_path)}


NULL_POINTS_CSV = OUT_DIR / "resplit_null_cosines.csv"


def null_figure(points: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Paper Fig. A1c: per model, the 219 re-split null cosines, the observed sex-direction
    cosine and the confounded raw mean-difference reference (IEEE style)."""
    from figures import ieee_style as S

    fig, axes = S.figure(3.0, 2.2, 2, 2, sharex=True, sharey=True)
    bins = np.linspace(min(-0.2, points["cosine"].min()), 1.0, 31)
    summ = summary.set_index("arch")
    for ax, arch in zip(axes.flat, S.MODEL_ORDER):
        null = points[(points["arch"] == arch) & (points["kind"] == "null")]["cosine"]
        sex = float(points[(points["arch"] == arch) & (points["kind"] == "sex")]["cosine"].iloc[0])
        ax.hist(null, bins=bins, color=S.NULL_GREY, lw=0, label="219 re-splits")
        ax.axvline(sex, color=S.OBSERVED, lw=1.0, label="Sex direction")
        raw = summ.loc[arch, "raw_cav_cosine_reference"]
        if pd.notna(raw):
            ax.axvline(float(raw), color=S.REFERENCE_GREY, lw=0.8, ls="--", label="Raw mean difference")
        ax.set_title(S.MODEL_NAME[arch])
        S.light_grid(ax)
    for ax in axes[1, :]:
        ax.set_xlabel("Cross-seed cosine")
    for ax in axes[:, 0]:
        ax.set_ylabel("Count")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, handlelength=1.4)
    S.save(fig, "whitened_direction_null")


def redraw_from_saved() -> None:
    null_figure(pd.read_csv(NULL_POINTS_CSV, keep_default_na=False), pd.read_csv(OUT_DIR / "resplit_control.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--redraw", action="store_true",
                    help="only redraw the figure from results/sex_direction/resplit_null_cosines.csv")
    args = ap.parse_args()
    if args.redraw:
        redraw_from_saved()
        return
    cfg = load_config(args.config)
    h = run_resplit_control(cfg)
    print("\n================ RESPLIT CONTROL ================")
    print(h["table"][["arch", "sex_whitened_cosine", "random_p50", "random_p95",
                       "sex_percentile", "n_resplits", "verdict"]].to_string(index=False))
    print(f"\n{FLAG}")
    print(THREE_WOMEN_CAVEAT)


if __name__ == "__main__":
    main()
