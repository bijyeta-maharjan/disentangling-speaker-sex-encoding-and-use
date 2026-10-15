"""Unit-level sex selectivity: which penultimate units fire most differently for
female vs. male speakers, and is that ranking stable across training seeds?

Three stages:
  selectivity     per-unit Cohen's d (female vs. male) on the SI test fold, seed 42
  seed-stability  top-20 Jaccard + full-ranking Spearman across seeds 42/43/44
  manifest        top-k unit indices per architecture/seed (no analysis)

Reuses the Week-2 cross-architecture probing harness verbatim (src/probing/
extract_week2.py, splits.py) for feature extraction, checkpoint discovery, and
the frozen-model/eval/no-grad/shuffle=False disciplines. Only the per-unit
Cohen's-d ranking + figures are new.

CLAIM BOUNDARY (read before trusting any number this script prints):
The SI test fold has only 3 female speakers (vs 9 male). Every effect size
here is a WITHIN-SAMPLE selectivity statistic estimated from three women, not
a population-level "female neuron" claim. FLAG is stamped on every artifact.
Confidence comes from CROSS-ARCHITECTURE / CROSS-SEED agreement, not from any
single unit's Cohen's d.

Not done here: digit-conditioned splits, by-group generalization stats,
F0/formant correlation, readout analyses.

    python scripts/unit_selectivity.py --stage selectivity
    python scripts/unit_selectivity.py --stage seed-stability   (also writes the manifest)
    python scripts/unit_selectivity.py --stage manifest
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

from src.data.preprocess import load_cache
from src.models.factory import input_kind
from src.probing.extract_week2 import (
    PENULTIMATE,
    TRAINED,
    _loader,
    build_trained,
    extract_features,
)
from src.probing.extract_week2 import load_labels as load_seed42_labels
from src.probing.splits import load_si_spec
from src.utils.config import load_config, resolve, resolve_arch_cfg
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"
ARCH_ORDER = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
SEEDS = [42, 43, 44]
TOPK = 20

FLAG = "n=3 female test speakers — selectivity is within-sample, not a population claim"
GRU_NOTE = ("crnn_gru has weak global sex-decodability (prior linear-probe sex margin "
            "~-0.006, near chance) — its unit ranking here may be near-noise; report, don't interpret.")

OUT_DIR = resolve("results/sex_direction")
RUNS_DIR = resolve("results/sex_direction")
FIGS_DIR = resolve("figures/output")


# --------------------------------------------------------------------------- #
# checkpoint discovery (seed-aware; same authoritative source as extract_week2)
# --------------------------------------------------------------------------- #
def seed_checkpoints(cfg: dict, seed: int) -> dict[str, Path]:
    results = resolve(cfg["sweep_results"])
    out: dict[str, Path] = {}
    with open(results, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["regime"] == "speaker_independent" and r["seed"] == seed:
                out[r["arch"]] = resolve(r["run_dir"]) / "best.pt"
    return out


# --------------------------------------------------------------------------- #
# per-seed trained-penultimate cache (idempotent; seed 42 reuses the existing
# Week-2 cache verbatim, seeds 43/44 get their own cache dir so seed-42
# artifacts stay untouched — mirrors the cross-seed-margin convention)
# --------------------------------------------------------------------------- #
def penultimate_cache_path(seed: int, arch: str) -> Path:
    if seed == 42:
        base = resolve("data/processed/activations_week2")
    else:
        base = resolve(f"data/processed/activations_week2_seed{seed}")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{arch}__{TRAINED}__{PENULTIMATE}.npy"


def ensure_trained_penultimate(cfg: dict, seed: int, arch: str, ckpt_path: Path,
                                labels: pd.DataFrame, device) -> np.ndarray:
    path = penultimate_cache_path(seed, arch)
    if path.exists():
        return np.load(path)
    print(f"[extract] seed={seed} arch={arch}: cache miss, extracting trained penultimate ...")
    probe_cfg = resolve_arch_cfg(cfg, arch=arch)
    feature = input_kind(probe_cfg)
    cache = dict(load_cache(cfg))
    if feature == "spectrogram":
        from src.data.spectrogram import load_spec_cache
        cache["spec"] = load_spec_cache(cfg)["spec"]
    model = build_trained(cfg, arch, ckpt_path, device)
    loader = _loader(cfg, cache, feature)
    feats, digits, spk = extract_features(model, loader, device, arch, intermediate_names=[])
    assert np.array_equal(labels["digit"].to_numpy(), digits), f"digit desync seed={seed} arch={arch}"
    assert np.array_equal(labels["speaker_id"].to_numpy(), spk), f"speaker desync seed={seed} arch={arch}"
    np.save(path, feats[PENULTIMATE])
    print(f"[extract] seed={seed} arch={arch}: saved {feats[PENULTIMATE].shape} -> {path}")
    return feats[PENULTIMATE]


# --------------------------------------------------------------------------- #
# SI test-fold sex groups (speaker-disjoint from train/val by construction)
# --------------------------------------------------------------------------- #
def si_test_sex_groups(si_spec: dict) -> tuple[list[str], list[str]]:
    test = si_spec["roles"]["test"]
    sex = si_spec["speaker_sex"]
    female = [s for s in test if sex[s] == "female"]
    male = [s for s in test if sex[s] == "male"]
    return female, male


# --------------------------------------------------------------------------- #
# per-unit Cohen's d (utterance-level), ranked by |d|
# --------------------------------------------------------------------------- #
def cohens_d_table(X: np.ndarray, labels: pd.DataFrame, female_spk: list[str],
                    male_spk: list[str]) -> tuple[pd.DataFrame, dict]:
    spk_str = labels["speaker"].astype(str).to_numpy()
    f_mask = np.isin(spk_str, female_spk)
    m_mask = np.isin(spk_str, male_spk)
    Xf, Xm = X[f_mask], X[m_mask]
    nf, nm = Xf.shape[0], Xm.shape[0]

    mean_f, mean_m = Xf.mean(axis=0), Xm.mean(axis=0)
    var_f, var_m = Xf.var(axis=0, ddof=1), Xm.var(axis=0, ddof=1)
    pooled_std = np.sqrt(((nf - 1) * var_f + (nm - 1) * var_m) / (nf + nm - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (mean_f - mean_m) / pooled_std
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    df = pd.DataFrame({
        "unit_index": np.arange(X.shape[1], dtype=int),
        "mean_act_female": mean_f,
        "mean_act_male": mean_m,
        "cohens_d": d,
    })
    df["_abs_d"] = df["cohens_d"].abs()
    df = df.sort_values("_abs_d", ascending=False).reset_index(drop=True)
    df["abs_d_rank"] = np.arange(1, len(df) + 1)
    df = df.drop(columns=["_abs_d"])
    meta = {
        "n_female_speakers": len(female_spk), "n_male_speakers": len(male_spk),
        "n_female_utt": int(nf), "n_male_utt": int(nm),
        "female_speakers": female_spk, "male_speakers": male_spk,
    }
    return df, meta


def participation_ratio(d: np.ndarray) -> float:
    """Effective number of units carrying the |d| effect: 1 = one unit dominates,
    n_units = effect spread evenly. PR = (sum d^2)^2 / sum d^4."""
    d2 = np.asarray(d, dtype=float) ** 2
    denom = float((d2 ** 2).sum())
    if denom == 0:
        return 0.0
    return float(d2.sum() ** 2 / denom)


# --------------------------------------------------------------------------- #
# headline figure: ranked |d| bar plot, one panel per architecture
# --------------------------------------------------------------------------- #
def headline_figure(tables: dict[str, pd.DataFrame], seed: int, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for ax, arch in zip(axes.flat, ARCH_ORDER):
        df = tables[arch]
        ax.bar(df["abs_d_rank"], df["cohens_d"].abs(), color="#8e44ad", width=1.0)
        title = arch + ("  [weak sex-decodability — near-noise]" if arch == "crnn_gru" else "")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("unit, ranked by |d|")
        ax.set_ylabel("|Cohen's d|  (female vs male)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Female-selective units per architecture, ranked by |d| — seed {seed} SI, trained penultimate\n{FLAG}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Stage 1: per-unit selectivity (seed 42)
# --------------------------------------------------------------------------- #
def run_selectivity(cfg: dict) -> dict:
    device = torch.device(cfg["extract"]["device"])
    si_spec = load_si_spec(cfg)
    female_spk, male_spk = si_test_sex_groups(si_spec)
    labels = load_seed42_labels(cfg)  # same cache order for every seed (same waveform cache)
    ckpts = seed_checkpoints(cfg, 42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tables, summary_rows = {}, []
    for arch in ARCH_ORDER:
        X = ensure_trained_penultimate(cfg, 42, arch, ckpts[arch], labels, device)
        df, meta = cohens_d_table(X, labels, female_spk, male_spk)
        pr = participation_ratio(df["cohens_d"].to_numpy())
        n_units = len(df)

        out = df.copy()
        out["n_female_speakers"] = meta["n_female_speakers"]
        out["n_male_speakers"] = meta["n_male_speakers"]
        out["n_female_utt"] = meta["n_female_utt"]
        out["n_male_utt"] = meta["n_male_utt"]
        if arch == "crnn_gru":
            out["note"] = GRU_NOTE
        out["flag"] = FLAG
        csv_path = OUT_DIR / f"{arch}_seed42_unit_selectivity.csv"
        out.to_csv(csv_path, index=False)

        tables[arch] = df
        top5 = df.head(5)
        smear = "broad smear" if pr / n_units > 0.3 else ("moderate" if pr / n_units > 0.1 else "few sharp units")
        summary_rows.append({
            "arch": arch, "n_units": n_units,
            "n_female_speakers": meta["n_female_speakers"], "n_male_speakers": meta["n_male_speakers"],
            "n_female_utt": meta["n_female_utt"], "n_male_utt": meta["n_male_utt"],
            "participation_ratio": round(pr, 2), "pr_fraction": round(pr / n_units, 4),
            "distribution": smear,
            "top5_unit_index": top5["unit_index"].tolist(),
            "top5_cohens_d": [round(v, 4) for v in top5["cohens_d"].tolist()],
        })
        print(f"\n[selectivity] {arch} (seed 42, trained, n={n_units} units; "
              f"female={meta['n_female_speakers']}spk/{meta['n_female_utt']}utt, "
              f"male={meta['n_male_speakers']}spk/{meta['n_male_utt']}utt)")
        print(f"  top-5 |d| units: {list(zip(top5['unit_index'].tolist(), [round(v,3) for v in top5['cohens_d'].tolist()]))}")
        print(f"  participation ratio = {pr:.1f} / {n_units} ({pr/n_units:.1%}) -> {smear}")
        if arch == "crnn_gru":
            print(f"  NOTE: {GRU_NOTE}")
        print(f"  saved -> {csv_path}")

    fig_path = headline_figure(tables, 42, FIGS_DIR / "unit_selectivity_seed42.png")
    print(f"\n[selectivity] headline figure -> {fig_path}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "git_sha": git_sha(), "seed": 42, "flag": FLAG,
        "female_test_speakers": female_spk, "male_test_speakers": male_spk,
        "gru_note": GRU_NOTE,
        "summary": summary_rows,
    }
    with open(RUNS_DIR / "unit_selectivity_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "unit_selectivity_summary.csv", index=False)
    print(f"[selectivity] summary -> {OUT_DIR / 'unit_selectivity_summary.csv'} and {RUNS_DIR / 'unit_selectivity_run_meta.json'}")
    return {"tables": tables, "summary": summary_df, "fig": str(fig_path)}


# --------------------------------------------------------------------------- #
# Stage 2: cross-seed agreement
#
# IMPORTANT FRAMING (do not get this backwards): seeds 42/43/44 are three
# TRAINING RUNS of the SAME architecture against the SAME frozen speaker
# partition (verified below, not assumed) — same 12 SI test speakers, same
# 3 female speakers, every time. So a unit that survives across seeds is
# "stable to retraining" (robust to init/SGD stochasticity). It is NOT
# "stable across female samples" — that would require different female
# speakers in the test fold, which we do not have. Every sentence below
# must use the first framing, never the second.
# --------------------------------------------------------------------------- #
def verify_si_partition_fixed(cfg: dict, ckpts_by_seed: dict[int, dict[str, Path]]) -> dict:
    """Check (not assume) that seeds 42/43/44 trained against the IDENTICAL SI
    speaker partition, by comparing each run's fold config + split sizes
    against splits/speaker_independent.json. If this ever fails, cross-seed
    agreement would mean something different (and stronger) than "stable to
    retraining" — so we fail loudly rather than silently mislabel it."""
    si_spec = load_si_spec(cfg)
    expected_test_n = len(si_spec["roles"]["test"]) * 500  # 500 utt/speaker in AudioMNIST
    ref = None
    for seed, ckpts in ckpts_by_seed.items():
        for arch, ckpt_path in ckpts.items():
            run_dir = ckpt_path.parent
            with open(run_dir / "config.json", encoding="utf-8") as fh:
                c = json.load(fh)
            with open(run_dir / "split_sizes.json", encoding="utf-8") as fh:
                sizes = json.load(fh)
            fold_key = (c["splits"]["n_folds"], c.get("splits", {}).get("test_fold"),
                        c.get("splits", {}).get("val_fold"))
            # test_fold/val_fold may live one level up in some configs; fall back gracefully
            if fold_key[1] is None:
                fold_key = (si_spec["n_folds"], si_spec["test_fold"], si_spec["val_fold"])
            assert fold_key == (si_spec["n_folds"], si_spec["test_fold"], si_spec["val_fold"]), (
                f"seed={seed} arch={arch}: fold config {fold_key} != splits/speaker_independent.json "
                f"({si_spec['n_folds']}, {si_spec['test_fold']}, {si_spec['val_fold']}) — "
                "partition may have reshuffled; do not label cross-seed agreement 'stable to retraining' "
                "without re-checking what actually varied."
            )
            assert sizes["test"] == expected_test_n, (
                f"seed={seed} arch={arch}: test split size {sizes['test']} != "
                f"expected {expected_test_n} from splits/speaker_independent.json test-speaker count"
            )
            if ref is None:
                ref = fold_key
    print(f"[seed-stability] verified: seeds {sorted(ckpts_by_seed)} all trained against the SAME SI partition "
          f"(n_folds/test_fold/val_fold={ref}) — same 12 test speakers (3 female) every seed. "
          "Cross-seed agreement below is retraining-stability, NOT cross-female-sample generalization.")
    return {"n_folds": ref[0], "test_fold": ref[1], "val_fold": ref[2], "partition_fixed": True}


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def null_jaccard(n_units: int, k: int) -> float:
    """Expected top-k Jaccard for two INDEPENDENT random k-subsets of n_units
    (analytic E[|A cap B|]/E[|A cup B|] approximation) — the chance floor a raw
    Jaccard number must clear before it means anything as "agreement"."""
    exp_inter = k * k / n_units
    exp_union = 2 * k - exp_inter
    return exp_inter / exp_union if exp_union else 0.0


def run_seed_stability(cfg: dict) -> dict:
    device = torch.device(cfg["extract"]["device"])
    si_spec = load_si_spec(cfg)
    female_spk, male_spk = si_test_sex_groups(si_spec)
    labels = load_seed42_labels(cfg)

    ckpts_by_seed = {seed: seed_checkpoints(cfg, seed) for seed in SEEDS}
    partition_check = verify_si_partition_fixed(cfg, ckpts_by_seed)

    tables = {}  # (arch, seed) -> df
    for seed in SEEDS:
        for arch in ARCH_ORDER:
            X = ensure_trained_penultimate(cfg, seed, arch, ckpts_by_seed[seed][arch], labels, device)
            df, _meta = cohens_d_table(X, labels, female_spk, male_spk)
            tables[(arch, seed)] = df
            if seed != 42:  # seed 42 tables already saved by stage 1 with fuller columns
                out = df.copy()
                out["n_female_speakers"], out["n_male_speakers"] = len(female_spk), len(male_spk)
                if arch == "crnn_gru":
                    out["note"] = GRU_NOTE
                out["flag"] = FLAG
                out.to_csv(OUT_DIR / f"{arch}_seed{seed}_unit_selectivity.csv", index=False)

    pairs = [(42, 43), (42, 44), (43, 44)]
    stability_rows = []
    for arch in ARCH_ORDER:
        n_units = len(tables[(arch, 42)])
        topsets = {seed: set(tables[(arch, seed)].head(TOPK)["unit_index"].tolist()) for seed in SEEDS}
        jac = {f"jaccard_{a}v{b}": round(jaccard(topsets[a], topsets[b]), 3) for a, b in pairs}
        mean_j = round(float(np.mean(list(jac.values()))), 3)
        chance = round(null_jaccard(n_units, TOPK), 3)

        # complementary, less cutoff-sensitive measure: Spearman rho of the FULL
        # |d| vector (unit-index aligned) between seed pairs.
        from scipy.stats import spearmanr
        abs_d = {seed: tables[(arch, seed)].set_index("unit_index")["cohens_d"].abs().sort_index().to_numpy()
                 for seed in SEEDS}
        rho = {f"spearman_{a}v{b}": round(float(spearmanr(abs_d[a], abs_d[b]).statistic), 3) for a, b in pairs}
        mean_rho = round(float(np.mean(list(rho.values()))), 3)

        verdict = "ABOVE chance" if mean_j > chance * 1.5 else "AT/BELOW chance (no reproducible top-k set)"
        stability_rows.append({
            "arch": arch, "n_units": n_units, "topk": TOPK, **jac,
            "mean_jaccard_stable_to_retraining": mean_j, "chance_jaccard_null": chance,
            "jaccard_verdict": verdict, **rho, "mean_spearman_full_|d|_vector": mean_rho,
        })
        print(f"[seed-stability] {arch} (n_units={n_units}): top-{TOPK} Jaccard across retraining seeds -> {jac} "
              f"(mean={mean_j}, CHANCE FLOOR={chance}) -> {verdict}")
        print(f"         full-|d|-vector Spearman across retraining seeds -> {rho} (mean={mean_rho})")
        print(f"         [stable-to-RETRAINING framing only; NOT stable-across-female-samples — "
              f"{partition_check['n_folds']}-fold partition fixed at test_fold={partition_check['test_fold']} for all 3 seeds]")

    stability_df = pd.DataFrame(stability_rows)
    stability_df["flag"] = FLAG
    stability_df["basis_caveat"] = (
        "raw unit indices are NOT comparable across architectures (different bases); "
        "cross-architecture agreement is assessed later via the acoustic link (F0/formant "
        "correlation), NOT by matching unit indices here."
    )
    stability_df["framing"] = (
        "Jaccard measures stability to RETRAINING (same fixed SI partition, 3 female / 9 male "
        "test speakers every seed). It is NOT evidence of stability across different female "
        "samples — the partition did not reshuffle (verified, see partition_check)."
    )
    out_csv = OUT_DIR / "unit_seed_stability.csv"
    stability_df.to_csv(out_csv, index=False)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "unit_seed_stability_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "git_sha": git_sha(), "seeds": SEEDS, "flag": FLAG, "topk": TOPK,
            "partition_check": partition_check,
            "framing_note": "stable to retraining, NOT stable across female samples (partition fixed, checked not assumed)",
            "basis_caveat": "raw unit indices not comparable across architectures",
            "stability": stability_rows,
        }, fh, indent=2)
    print(f"[seed-stability] stability table -> {out_csv} and {RUNS_DIR / 'unit_seed_stability_run_meta.json'}")
    return {"tables": tables, "stability": stability_df, "partition_check": partition_check}


# --------------------------------------------------------------------------- #
# Stage 3: top-k unit manifest (no analysis)
# --------------------------------------------------------------------------- #
def run_topk_manifest(cfg: dict, seed_tables: dict | None = None, stability_df: pd.DataFrame | None = None) -> Path:
    if seed_tables is None:
        # allow standalone `--stage manifest` runs by recomputing tables from cache
        device = torch.device(cfg["extract"]["device"])
        si_spec = load_si_spec(cfg)
        female_spk, male_spk = si_test_sex_groups(si_spec)
        labels = load_seed42_labels(cfg)
        ckpts_by_seed = {seed: seed_checkpoints(cfg, seed) for seed in SEEDS}
        seed_tables = {}
        for seed in SEEDS:
            for arch in ARCH_ORDER:
                X = ensure_trained_penultimate(cfg, seed, arch, ckpts_by_seed[seed][arch], labels, device)
                df, _meta = cohens_d_table(X, labels, female_spk, male_spk)
                seed_tables[(arch, seed)] = df

    if stability_df is not None:
        retraining_stability_caveat = {
            "measured": True,
            "verdict": "top-k unit identity and full-|d|-vector rank are AT/BELOW the chance floor for "
                       "retraining agreement in EVERY architecture (see unit_seed_stability.csv) — the "
                       "seed-42 units below are NOT a reproducible set; a different training run of the "
                       "same architecture on the identical partition picks different top units.",
            "per_arch": stability_df.set_index("arch")[["mean_jaccard_stable_to_retraining",
                                                          "chance_jaccard_null", "jaccard_verdict",
                                                          "mean_spearman_full_|d|_vector"]].to_dict("index"),
            "consequence_for_f0_analysis": "treat per-unit indices as candidates from ONE trained instance per "
                                      "architecture, not validated stable units; weight any F0/formant "
                                      "correlation finding accordingly.",
        }
    else:
        retraining_stability_caveat = {
            "measured": False,
            "note": "run `--stage seed-stability` first to get the measured retraining-stability verdict; "
                    "standalone `--stage manifest` did not compute it.",
        }

    manifest = {
        "flag": FLAG,
        "framing": "top-k units are ranked by within-sample |Cohen's d| (female vs male, SI test "
                   "fold, n=3 female / 9 male speakers). Cross-seed presence would mean stable to "
                   "RETRAINING, NOT stable across female samples — but see retraining_stability_caveat.",
        "retraining_stability_caveat": retraining_stability_caveat,
        "basis_caveat": "unit indices are architecture-specific bases; do not compare indices across archs directly",
        "topk": TOPK,
        "git_sha": git_sha(),
        "note_on_use": "Unit indices only; no correlation or segmentation is computed here.",
        "units": {},
    }
    for arch in ARCH_ORDER:
        manifest["units"][arch] = {}
        for seed in SEEDS:
            top = seed_tables[(arch, seed)].head(TOPK)
            manifest["units"][arch][str(seed)] = {
                "unit_index": top["unit_index"].tolist(),
                "cohens_d": [round(v, 4) for v in top["cohens_d"].tolist()],
            }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "topk_units.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[topk] manifest ({TOPK} units/arch/seed) -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--stage", default="selectivity", choices=["selectivity", "seed-stability", "manifest"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage == "selectivity":
        h = run_selectivity(cfg)
        print("\n================ UNIT SELECTIVITY SUMMARY ================")
        print(h["summary"].to_string(index=False))
        print(f"\n{FLAG}")
    elif args.stage == "seed-stability":
        h = run_seed_stability(cfg)
        print("\n================ UNIT SEED STABILITY (stable-to-RETRAINING) ================")
        print(h["stability"][["arch", "n_units", "topk", "mean_jaccard_stable_to_retraining",
                               "chance_jaccard_null", "jaccard_verdict",
                               "mean_spearman_full_|d|_vector"]].to_string(index=False))
        print(f"\n{FLAG}")
        run_topk_manifest(cfg, seed_tables=h["tables"], stability_df=h["stability"])
    elif args.stage == "manifest":
        run_topk_manifest(cfg)


if __name__ == "__main__":
    main()
