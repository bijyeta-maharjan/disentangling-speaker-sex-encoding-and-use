"""Within-digit conditional geometry of speaker information.

Reuses the cross-architecture probing harness verbatim (src/probing/probe.py, splits.py,
extract_week2.py) for measure (a). No model training, no GPU: everything reads
already-cached penultimate features (data/processed/activations_week2[_seed{43,44}]).

Three measures, per architecture (hypotheses written down before running):

  (a) CONDITIONAL DECODABILITY — per digit, linear probe for sex (speaker-
      disjoint SI test fold) and speaker-identity (60-way identity split),
      filtered to that digit. A GENERALIZATION claim -> stays on the held-out
      test fold. Seed-42 primary; pooled over 42/43/44 where cached.

  (b) SEX-DIRECTION CONSISTENCY ACROSS DIGITS — pairwise cosine similarity of
      the per-digit sex-probe weight vectors. A REPRESENTATION-STRUCTURE claim
      (not generalization) -> fit on the SI train+val speakers (48 spk, ~12
      female), NOT the 3-female test fold. Seed-42 only.

  (c) ORTHOGONALITY TO THE DIGIT SUBSPACE — principal angle between the shared
      sex direction and the span of per-digit class-mean vectors. Same
      train-fold data source as (b). Seed-42 only.

DISCIPLINE (load-bearing): (b)/(c) require directions from
different digits to live in the SAME coordinate system to be comparable, so a
single StandardScaler is fit ONCE on the whole train-fold (all digits) and
applied uniformly -- NOT `build_probe`'s per-call, per-digit-local scaler, which
would silently put every digit's direction in its own private scaled basis.

GATE (mandatory): (b)/(c) are computed for an architecture only if that
architecture's GLOBAL (not per-digit) sex probe significantly clears the 0.75
majority-chance baseline (bootstrap CI lower bound > chance), reusing the
existing seed-42 penultimate probe_results.csv rather than recomputing. crnn_gru is
pre-registered as the likely not-estimable case (global trained sex acc 0.737
vs 0.75 chance).

Outputs are written to results/within_digit_geometry/.

Seeds 43/44 (measure (a) only) read configs/week2_probing_seed43.yaml and
_seed44.yaml, which are not included in this repository (see README, Known
limitations); run with --seeds 42 without them.

Usage:
    python scripts/geom_within_digit.py [--config CONFIG] [--seeds 42,43,44]
        [--archs cnn1d,resnet1d,crnn_gru,spec_cnn2d] [--n-boot-direction 200]
        [--smoke-test]

    --smoke-test restricts to 1 architecture x 1 digit x 20 direction-bootstrap
    reps, for verifying the pipeline + timing a single fit before committing to
    the full run. Does not touch the frozen hypothesis or write to the real
    run folder (writes to results/within_digit_geometry/_smoke/ instead).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.probing import probe as P
from src.probing.extract_week2 import load_features, load_labels
from src.probing.splits import generalizable_split, identity_split, load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

ARCH_ORDER = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
N_DIGITS = 10
CHANCE_SEX = 0.75  # majority baseline on the SI test fold (9 male / 3 female speakers)

SEED_CONFIG = {
    42: "configs/week2_probing.yaml",
    43: "configs/week2_probing_seed43.yaml",
    44: "configs/week2_probing_seed44.yaml",
}


# --------------------------------------------------------------------------- #
# gate: reuse the existing seed-42 penultimate probing result, do not recompute
# --------------------------------------------------------------------------- #
def load_global_sex_gate(cfg42: dict) -> pd.DataFrame:
    """Global (not per-digit) trained sex-probe accuracy + CI per arch, from the
    existing penultimate probing run (experiments/week2_probing/latest.txt). Falls
    back to a fresh fit_eval only if that artifact is unexpectedly missing."""
    runs_dir = resolve(cfg42["paths"]["runs_dir"])
    latest_path = runs_dir / "latest.txt"
    if latest_path.exists():
        run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
        csv_path = run_dir / "probe_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            sub = df[(df["attribute"] == "sex") & (df["model"] == "trained")].copy()
            sub["gate_pass"] = sub["ci_lo"] > sub["chance"]
            return sub[["arch", "acc", "ci_lo", "ci_hi", "chance", "gate_pass"]]
    # fallback: recompute (should not trigger given the cached artifact exists)
    print("[geom_within_digit] WARNING: latest.txt/probe_results.csv not found, recomputing gate")
    labels = load_labels(cfg42)
    si_spec = load_si_spec(cfg42)
    y, _ = P.encode_labels(labels["sex"].tolist())
    spk = labels["speaker_id"].to_numpy()
    train_pos, test_pos = generalizable_split(labels, "sex", si_spec)
    rows = []
    for arch in ARCH_ORDER:
        X = load_features(cfg42, arch, "trained")
        r = P.fit_eval(X, y, spk, train_pos, test_pos, "linear", cfg42, standardize=True)
        rows.append({"arch": arch, "acc": r["acc"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                     "chance": r["chance"], "gate_pass": r["ci_lo"] > r["chance"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# measure (a): per-digit conditional decodability (generalization claim ->
# stays on the existing speaker-disjoint / identity TEST folds)
# --------------------------------------------------------------------------- #
def conditional_decodability(cfg: dict, seed: int, arch: str, attribute: str) -> list[dict]:
    labels = load_labels(cfg)
    digit = labels["digit"].to_numpy()
    X = load_features(cfg, arch, "trained")
    spk = labels["speaker_id"].to_numpy()

    if attribute == "sex":
        si_spec = load_si_spec(cfg)
        y, _ = P.encode_labels(labels["sex"].tolist())
        train_pos, test_pos = generalizable_split(labels, "sex", si_spec)
        kind = "generalizable"
    else:  # speaker_id
        y, _ = P.encode_labels(labels["speaker_id"].tolist())
        train_pos, test_pos = identity_split(labels, cfg["identity_split"]["test_size"], cfg["seed"])
        kind = "identity"

    y_rand = P.make_random_labels(y, spk, np.concatenate([train_pos, test_pos]), kind, seed=cfg["seed"] + 100)

    out = []
    for d in range(N_DIGITS):
        mask_d = digit == d
        tr_d = np.intersect1d(train_pos, np.flatnonzero(mask_d))
        te_d = np.intersect1d(test_pos, np.flatnonzero(mask_d))
        n_female_test = int((labels.loc[te_d, "sex"] == "female").sum()) if attribute == "sex" else None
        n_female_test_speakers = (
            int(labels.loc[te_d, "speaker"].loc[labels.loc[te_d, "sex"] == "female"].nunique())
            if attribute == "sex" else None
        )
        real = P.fit_eval(X, y, spk, tr_d, te_d, "linear", cfg, standardize=True)
        rand = P.fit_eval(X, y_rand, spk, tr_d, te_d, "linear", cfg, standardize=True)
        out.append({
            "seed": seed, "arch": arch, "attribute": attribute, "digit": d,
            "acc": real["acc"], "acc_speaker_mean": real["acc_speaker_mean"],
            "ci_lo": real["ci_lo"], "ci_hi": real["ci_hi"], "chance": real["chance"],
            "selectivity": real["acc"] - rand["acc"],
            "n_train": real["n_train"], "n_test": real["n_test"],
            "n_test_speakers": real["n_test_speakers"],
            "n_female_test_rows": n_female_test, "n_female_test_speakers": n_female_test_speakers,
        })
    return out


# --------------------------------------------------------------------------- #
# measures (b)/(c): representation-structure claims -> TRAIN-FOLD only, ONE
# shared scaler across all 10 digits (load-bearing: see module docstring)
# --------------------------------------------------------------------------- #
def _train_fold_mask(labels: pd.DataFrame, si_spec: dict) -> np.ndarray:
    train_spk = set(si_spec["roles"]["train"]) | set(si_spec["roles"]["val"])
    return labels["speaker"].astype(str).isin(train_spk).to_numpy()


def _fit_digit_direction(Xs: np.ndarray, y_sex: np.ndarray, pos: np.ndarray, cfg: dict) -> np.ndarray:
    """Fit LogisticRegression directly on ALREADY-scaled features (no internal
    scaler -- that is what keeps every digit's direction in the same basis)."""
    p = cfg["probe"]["linear"]
    clf = LogisticRegression(C=p["C"], max_iter=p["max_iter"], n_jobs=-1)
    clf.fit(Xs[pos], y_sex[pos])
    return clf.coef_.ravel()


def direction_geometry(cfg: dict, arch: str, n_boot: int, digits: list[int] | None = None,
                        seed: int = 0) -> dict:
    labels = load_labels(cfg)
    si_spec = load_si_spec(cfg)
    fold_mask = _train_fold_mask(labels, si_spec)
    digit = labels["digit"].to_numpy()
    y_sex, _ = P.encode_labels(labels["sex"].tolist())
    spk_str = labels["speaker"].astype(str).to_numpy()

    X = load_features(cfg, arch, "trained")
    scaler = StandardScaler().fit(X[fold_mask])
    Xs = scaler.transform(X)

    digits = digits if digits is not None else list(range(N_DIGITS))
    pos_by_digit = {d: np.flatnonzero(fold_mask & (digit == d)) for d in digits}
    directions = {d: _fit_digit_direction(Xs, y_sex, pos_by_digit[d], cfg) for d in digits}

    # (b) pairwise cosine similarity
    pairs = list(itertools.combinations(digits, 2))
    cos_point = {}
    for i, j in pairs:
        vi, vj = directions[i], directions[j]
        cos_point[(i, j)] = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12))
    mean_cos_point = float(np.mean(list(cos_point.values())))

    # (c) digit-discriminative subspace (class-mean span) + principal angle to
    # the shared direction (mean of the per-digit unit directions)
    class_means = np.stack([Xs[pos_by_digit[d]].mean(axis=0) for d in digits])
    M = class_means - class_means.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    rank = min(len(digits) - 1, Vt.shape[0])
    basis = Vt[:rank]  # orthonormal basis of the digit subspace

    unit_dirs = np.stack([directions[d] / (np.linalg.norm(directions[d]) + 1e-12) for d in digits])
    shared_dir = unit_dirs.mean(axis=0)
    shared_dir /= (np.linalg.norm(shared_dir) + 1e-12)
    # angle between shared_dir and its projection onto the digit subspace:
    # 0 deg = shared_dir lies IN the subspace (entangled), 90 deg = orthogonal
    # (the clean-disentanglement case H1c predicts).
    proj_norm = float(np.clip(np.linalg.norm(basis @ shared_dir), 0.0, 1.0))
    angle_deg_point = float(np.degrees(np.arccos(proj_norm)))

    # the digit-subspace direction that REALIZES the principal angle above (the orthogonal
    # projection of shared_dir onto the subspace, renormalized) -- persisted as an artifact so a
    # downstream consumer can plot/verify without refitting.
    proj_vec = basis.T @ (basis @ shared_dir)
    principal_direction = proj_vec / (np.linalg.norm(proj_vec) + 1e-12)
    _check_angle = float(np.degrees(np.arccos(np.clip(np.dot(shared_dir, principal_direction), -1.0, 1.0))))
    assert abs(_check_angle - angle_deg_point) < 1e-6, (
        f"principal_direction does not realize the computed angle for {arch}: "
        f"{_check_angle:.6f} vs {angle_deg_point:.6f} -- geometry bug")

    # bootstrap CI: resample TRAIN-FOLD speakers with replacement, refit
    # per-digit directions on the resampled row-set (scaler held fixed at the
    # observed fold -- a stated simplification), recompute.
    rng = np.random.default_rng(seed)
    train_speakers = np.array(sorted(set(spk_str[fold_mask])))
    cos_boot, angle_boot = [], []
    for b in range(n_boot):
        drawn = rng.choice(train_speakers, size=len(train_speakers), replace=True)
        # duplicate rows for speakers drawn >1x so they carry proportional weight
        boot_pos_by_digit = {}
        for d in digits:
            rows = []
            for s in drawn:
                rows.append(np.flatnonzero((spk_str == s) & (digit == d)))
            rows = np.concatenate(rows) if rows else np.array([], dtype=int)
            boot_pos_by_digit[d] = rows
        if any(len(boot_pos_by_digit[d]) == 0 or len(np.unique(y_sex[boot_pos_by_digit[d]])) < 2 for d in digits):
            continue  # degenerate resample (a digit lost one sex entirely) -> skip, do not fabricate
        boot_dirs = {d: _fit_digit_direction(Xs, y_sex, boot_pos_by_digit[d], cfg) for d in digits}
        boot_pairs_cos = [
            float(np.dot(boot_dirs[i], boot_dirs[j]) /
                  (np.linalg.norm(boot_dirs[i]) * np.linalg.norm(boot_dirs[j]) + 1e-12))
            for i, j in pairs
        ]
        cos_boot.append(np.mean(boot_pairs_cos))
        boot_unit = np.stack([boot_dirs[d] / (np.linalg.norm(boot_dirs[d]) + 1e-12) for d in digits])
        boot_shared = boot_unit.mean(axis=0)
        boot_shared /= (np.linalg.norm(boot_shared) + 1e-12)
        boot_proj = min(np.linalg.norm(basis @ boot_shared), 1.0)
        angle_boot.append(float(np.degrees(np.arccos(np.clip(boot_proj, 0.0, 1.0)))))

    def _ci(vals, point):
        if len(vals) < 10:
            return {"lo": float("nan"), "hi": float("nan"), "n_boot_used": len(vals)}
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"lo": float(lo), "hi": float(hi), "n_boot_used": len(vals)}

    return {
        "arch": arch,
        "mean_cosine_point": mean_cos_point,
        "mean_cosine_ci": _ci(cos_boot, mean_cos_point),
        "orthogonality_angle_deg_point": angle_deg_point,
        "orthogonality_angle_deg_ci": _ci(angle_boot, angle_deg_point),
        "n_train_fold_speakers": int(len(train_speakers)),
        "digits": digits,
        "pairwise_cosine": {f"{i}-{j}": v for (i, j), v in cos_point.items()},
        # stripped out and persisted to .npz by main() before this dict is JSON-dumped -- the
        # objects the conclusions above were computed from, not just the conclusions themselves.
        "_artifacts": {
            "digit_directions": np.stack([directions[d] for d in digits]),
            "digit_index": np.array(digits),
            "shared_dir": shared_dir,
            "digit_subspace_basis": basis,
            "principal_direction": principal_direction,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/week2_probing.yaml", help="seed-42 (primary) config")
    ap.add_argument("--seeds", default="42,43,44", help="comma list; used for measure (a) pooling only")
    ap.add_argument("--archs", default=",".join(ARCH_ORDER))
    ap.add_argument("--n-boot-direction", type=int, default=200)
    ap.add_argument("--smoke-test", action="store_true",
                     help="1 arch x 2 digits x 20 boot reps, writes to results/within_digit_geometry/_smoke/, "
                          "for verifying the pipeline + timing before the real run")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    archs = [a for a in args.archs.split(",") if a]
    cfg42 = load_config(args.config)

    if args.smoke_test:
        archs = archs[:1]
        seeds = [42]
        n_boot = 20
        smoke_digits = [0, 1]
    else:
        n_boot = args.n_boot_direction
        smoke_digits = None

    t0 = time.time()
    gate_df = load_global_sex_gate(cfg42)
    print("[geom_within_digit] probe-quality gate (global trained sex probe vs 0.75 chance):")
    print(gate_df.to_string(index=False))

    # measure (a): pooled across requested seeds
    decode_rows = []
    for seed in seeds:
        cfg = load_config(SEED_CONFIG[seed])
        for arch in archs:
            decode_rows += conditional_decodability(cfg, seed, arch, "sex")
            decode_rows += conditional_decodability(cfg, seed, arch, "speaker_id")
    decode_df = pd.DataFrame(decode_rows)

    # measures (b)/(c): seed-42 only, gated
    geom_rows = []
    for arch in archs:
        gate_pass = bool(gate_df.loc[gate_df["arch"] == arch, "gate_pass"].iloc[0])
        if not gate_pass:
            geom_rows.append({"arch": arch, "estimable": False,
                               "reason": "global sex probe does not clear chance (probe-quality gate)"})
            continue
        g = direction_geometry(cfg42, arch, n_boot=n_boot, digits=smoke_digits, seed=42)
        g["estimable"] = True
        geom_rows.append(g)

    elapsed = time.time() - t0
    print(f"\n[geom_within_digit] done in {elapsed:.1f}s "
          f"({'SMOKE TEST' if args.smoke_test else 'FULL RUN'})")

    # persist
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sha = git_sha()
    base = resolve("results/within_digit_geometry")
    run_dir = (base / "_smoke") if args.smoke_test else base
    run_dir.mkdir(parents=True, exist_ok=True)

    # persist the direction-vector artifacts (per estimable arch) BEFORE stripping them for the
    # JSON dump -- the objects direction_geometry()'s numbers were computed from, not just the
    # numbers.
    artifacts_dir = run_dir / "direction_vectors"
    for g in geom_rows:
        if not g.get("estimable"):
            continue
        artifacts = g.pop("_artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        npz_path = artifacts_dir / f"{g['arch']}_directions.npz"
        np.savez(npz_path, **artifacts)
        g["directions_npz"] = str(npz_path.relative_to(run_dir))

    gate_df.to_csv(run_dir / "sex_probe_gate.csv", index=False)
    decode_df.to_csv(run_dir / "conditional_decodability.csv", index=False)
    with open(run_dir / "direction_geometry.json", "w", encoding="utf-8") as fh:
        json.dump(geom_rows, fh, indent=2)
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"git_sha": sha, "created": stamp, "seeds": seeds, "archs": archs,
                   "n_boot_direction": n_boot, "smoke_test": args.smoke_test,
                   "elapsed_sec": round(elapsed, 1)}, fh, indent=2)
    print(f"[geom_within_digit] wrote outputs -> {run_dir}")


if __name__ == "__main__":
    main()
