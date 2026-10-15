"""Layer-wise digit-gradient vs. sex-subspace angle (whole-pipeline geometric
generalization of the readout capstone).

FROZEN MODELS ONLY. Nothing is trained or modified. Reuses the pre-existing
speaker_independent (SI) checkpoints for cnn1d and spec_cnn2d, seeds 42/43/44
(the two architectures with a validated sex direction).

Question. The readout capstone (results/sex_direction/) showed the
PENULTIMATE sex direction sits ~83 degrees off the digit-discriminative
subspace and the readout is insensitive to it -- a claim about one layer. This
generalizes it across EVERY layer: does the digit decision's gradient
direction stay near-orthogonal to the sex subspace throughout the network, or
does it align with sex at some earlier layer? Geometric, not causal (no
perturbation here).

Method (per architecture x seed x layer, layer = every conv block output +
penultimate):
  1. SEX SUBSPACE: fit a whitened (StandardScaler + LogisticRegression) sex
     direction on the GAP-pooled activation at that layer, ON THE SI TEST FOLD
     (12 speakers, 9 male / 3 female) -- per the pre-registered claim boundary,
     NOT the train+val fold. This differs from geom_within_digit.py's (b)/(c)
     measures (which deliberately use train+val for statistical power); here
     the test fold is used because the digit-gradient (item 2 below) is also
     computed on the test fold and both objects must live in the SAME
     evaluation population for the angle between them to mean anything.
     k=1 (rank-1): a binary (female-vs-male) linear contrast has exactly one
     degree of freedom between the two group means, so there is no
     well-defined SECOND sex-discriminative direction independent of the
     first with this estimator -- k=1 for every layer, reported explicitly
     (sex_subspace_dim column), not silently assumed.
  2. DIGIT-GRADIENT DIRECTION: one forward+backward pass over the SAME SI test
     utterances (frozen model, eval()); d(correct-digit logit)/d(activation)
     at that layer, per utterance (computed via a single torch.autograd.grad
     call across all layers per batch -- correct because summing independent
     per-example logits before backward yields exactly the per-example
     Jacobian row for each, with no cross-example terms). Gradients are taken
     w.r.t. the RAW (pre-GAP) block output for intermediate layers -- that is
     the actual graph node on the path to the logits, not a redundant pooled
     copy -- and then mean-pooled over the spatial/time dimension exactly as
     Grad-CAM (Selvaraju et al., 2017) pools a channel's gradient map into a
     per-channel importance weight (alpha_c = mean_t d(logit)/d(A_{t,c})); for
     the penultimate layer, which IS already the pooled (B,C) tensor actually
     used to compute the logits, the gradient is taken directly, no separate
     pooling step. Different utterances have different correct digits, so
     their gradient vectors point in genuinely different directions; a plain
     (signed) mean over utterances would partially cancel. Instead: normalize
     each utterance's gradient to a unit vector, average the OUTER PRODUCTS
     (mean_i u_i u_i^T, sign-invariant since u u^T == (-u)(-u)^T), and take
     the top eigenvector of that matrix as THE digit-gradient direction --
     this is "per-point [direction] then averaged as magnitude" per the
     pre-registration, avoiding signed cancellation.
  3. ANGLE: reuses the exact vector-to-subspace formula already used in
     this repo (scripts/geom_within_digit.py:
     angle = degrees(arccos(clip(norm(basis @ v), 0, 1))) for an orthonormal
     `basis` of the reference subspace) -- not rebuilt, just applied with the
     sex direction as a rank-1 `basis` and the digit-gradient direction as `v`.
  4. NULL: for that layer's dimension C, draw 300 random unit vectors and
     compute the SAME angle formula between the (fixed) digit-gradient
     direction and each random direction; report the 5th/95th percentile of
     that distribution as the calibration band. "Near-orthogonal" is judged
     against this null, not eyeballed against 90 degrees.

CLAIM BOUNDARY (read before citing):
  - The sex subspace at each layer is estimated using the SI TEST fold, whose
    female side is n=3. Every artifact carries the standing flag "n=3 female
    test speakers -- within-sample, not a population claim."
  - Confidence in this analysis comes from CROSS-LAYER, CROSS-ARCHITECTURE,
    CROSS-SEED CONSISTENCY, not from the female sample size. A single
    layer/seed/arch angle should not be over-read; the pattern across all of
    them is the evidence.
  - This is a GEOMETRIC claim: near-orthogonality means the digit-decision
    gradient direction does not ALIGN with the sex subspace at that layer. It
    bounds alignment/pathway, not strict necessity -- establishing necessity
    would require an intervention (perturbation/ablation), which is out of
    scope here (frozen-model analysis only).

Pre-registered expectation (CONFIRMATORY, written before running): the
digit-gradient stays >=~80 degrees from the sex subspace at every layer in
both architectures, consistent with the penultimate capstone result -- i.e. no
layer where the decision aligns with sex. Any layer that drops well below
that (well below the null band) is the interesting exception to flag.

Usage: python scripts/layer_sex_gradient_angle.py [--archs cnn1d spec_cnn2d]
    [--seeds 42 43 44] [--n-null 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import AudioMNISTCacheDataset
from src.models.factory import build_model, input_kind
from src.splits.make_splits import load_split_indices
from src.utils.config import load_config, resolve, resolve_arch_cfg
from src.utils.experiment import git_sha
from scripts.cross_sex_generalization import load_cache_for, SI_BASELINE_CHECKPOINTS

ARCHS = ["cnn1d", "spec_cnn2d"]
SEEDS = [42, 43, 44]
N_NULL_DEFAULT = 300
N_FLAG = "n=3 female test speakers -- within-sample, not a population claim"
OUT_DIR = resolve("results/layer_angle")


def gap(feat: torch.Tensor) -> torch.Tensor:
    """Global-average-pool a raw (B,C,T) or (B,C,H,W) block output to (B,C).
    Already-pooled (B,C) tensors (penultimate) pass through unchanged."""
    if feat.dim() == 2:
        return feat
    return feat.mean(dim=tuple(range(2, feat.dim())))


def angle_vector_to_subspace(basis: np.ndarray, v: np.ndarray) -> float:
    """Same formula as scripts/geom_within_digit.py:
    angle = degrees(arccos(clip(||basis @ v||, 0, 1))) for orthonormal `basis`
    (rows are basis vectors) and unit vector v."""
    proj_norm = float(np.clip(np.linalg.norm(basis @ v), 0.0, 1.0))
    return float(np.degrees(np.arccos(proj_norm)))


def random_unit_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


def layer_names_for(model) -> list[str]:
    return list(model.intermediate_names) + ["penultimate"]


def process_arch_seed(arch: str, seed: int, repo_root: Path, n_null: int) -> list[dict]:
    print(f"\n--- {arch} seed {seed} ---")
    cfg = resolve_arch_cfg(load_config("configs/week2_architectures.yaml"), arch=arch, seed=seed)
    feature = input_kind(cfg)
    device = torch.device("cpu")

    cache = load_cache_for(cfg)
    idx = load_split_indices(cfg, "speaker_independent", cache)
    test_idx = idx["test"]
    sex = np.asarray(cache["sex"])[test_idx]
    spk_str = np.asarray(cache["speaker_str"])[test_idx]
    n_female_utts = int((sex == "female").sum())
    n_male_utts = int((sex == "male").sum())
    n_female_spk = int(len(set(spk_str[sex == "female"])))
    n_male_spk = int(len(set(spk_str[sex == "male"])))
    print(f"  SI test fold: n={len(test_idx)} utterances "
          f"(female utts={n_female_utts}/spk={n_female_spk}, "
          f"male utts={n_male_utts}/spk={n_male_spk})")

    ckpt_dir = resolve(SI_BASELINE_CHECKPOINTS[(arch, seed)])
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    layers = layer_names_for(model)
    ds = AudioMNISTCacheDataset(cache, test_idx, feature=feature)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    pooled_feats = {l: [] for l in layers}
    grad_units = {l: [] for l in layers}
    sex_labels = []

    for x, y, _spk in loader:
        x = x.to(device)
        y = y.to(device)
        logits, feats = model(x, return_features=True)
        # feats[l] is the ACTUAL graph node on the path to logits (raw, pre-GAP
        # for intermediate blocks; already-pooled for penultimate) -- gradients
        # must be taken w.r.t. THIS tensor, not a redundant pooled copy (a
        # separately-computed pooled tensor would be a dead-end branch that
        # never influences logits, and autograd.grad would error on it).
        correct_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        grads = torch.autograd.grad(
            outputs=correct_logit.sum(),
            inputs=[feats[l] for l in layers],
            retain_graph=False,
        )
        for l, g in zip(layers, grads):
            # Grad-CAM-style channel weight: mean-pool the GRADIENT over the
            # spatial/time dims for raw blocks; already (B,C) for penultimate.
            g_pooled = gap(g).detach().numpy()
            norms = np.linalg.norm(g_pooled, axis=1, keepdims=True)
            ok = norms.flatten() > 1e-10
            u = np.zeros_like(g_pooled)
            u[ok] = g_pooled[ok] / norms[ok]
            grad_units[l].append(u[ok])
            pooled_feats[l].append(gap(feats[l]).detach().numpy())

    # sex labels, in the SAME row order as the loader (shuffle=False over test_idx)
    sex_labels = (np.asarray(cache["sex"])[test_idx] == "female").astype(int)

    rows = []
    rng = np.random.default_rng(seed)
    for depth_i, l in enumerate(layers):
        X = np.concatenate(pooled_feats[l], axis=0)
        U = np.concatenate(grad_units[l], axis=0)
        assert X.shape[0] == len(sex_labels), f"{arch}/{l}: feature rows != label rows"

        # --- (1) sex subspace, fit on the SI TEST fold (per pre-registration) ---
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1)
        clf.fit(Xs, sex_labels)
        sex_dir = clf.coef_.ravel()
        sex_dir_unit = sex_dir / (np.linalg.norm(sex_dir) + 1e-12)
        fit_acc = float(clf.score(Xs, sex_labels))

        # --- (2) digit-gradient direction: top eigenvector of mean outer product ---
        M = (U.T @ U) / U.shape[0]
        eigvals, eigvecs = np.linalg.eigh(M)
        top = eigvecs[:, -1]
        top_frac = float(eigvals[-1] / (eigvals.sum() + 1e-12))
        digit_grad_dir = top / (np.linalg.norm(top) + 1e-12)

        # --- (3) angle (reused formula) ---
        basis = sex_dir_unit.reshape(1, -1)
        angle = angle_vector_to_subspace(basis, digit_grad_dir)

        # --- (4) null: fixed digit-grad direction vs random directions ---
        dim = X.shape[1]
        rand_vecs = random_unit_vectors(n_null, dim, rng)
        null_angles = np.array([angle_vector_to_subspace(digit_grad_dir.reshape(1, -1), v)
                                 for v in rand_vecs])
        null_p5, null_p95 = np.percentile(null_angles, [5, 95])

        rows.append({
            "arch": arch, "layer": l, "depth_norm": depth_i / (len(layers) - 1),
            "seed": seed, "principal_angle_deg": angle,
            "null_p5": float(null_p5), "null_p95": float(null_p95),
            "null_mean": float(null_angles.mean()),
            "sex_subspace_dim": 1,
            "digit_grad_top_eigval_frac": top_frac,
            "sex_direction_fit_acc_on_test_fold": fit_acc,
            "n_test_utterances": int(X.shape[0]),
            "n_test_utts_female": n_female_utts, "n_test_utts_male": n_male_utts,
            "n_test_speakers_female": n_female_spk, "n_test_speakers_male": n_male_spk,
            "layer_dim": dim,
            "below_null_p5": bool(angle < null_p5),
        })
        print(f"  {l:12s} dim={dim:4d}  angle={angle:6.2f} deg  "
              f"null=[{null_p5:.1f},{null_p95:.1f}]  sex_fit_acc={fit_acc:.3f}  "
              f"{'<<< BELOW NULL' if angle < null_p5 else ''}")

    return rows


def make_headline_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"cnn1d": "tab:blue", "spec_cnn2d": "tab:orange"}

    for arch in df["arch"].unique():
        sub = df[df["arch"] == arch]
        layers_ordered = sub.drop_duplicates("layer").sort_values("depth_norm")["layer"].tolist()
        depths = sub.drop_duplicates("layer").sort_values("depth_norm")["depth_norm"].tolist()

        mean_angle = sub.groupby("layer")["principal_angle_deg"].mean().reindex(layers_ordered)
        min_angle = sub.groupby("layer")["principal_angle_deg"].min().reindex(layers_ordered)
        max_angle = sub.groupby("layer")["principal_angle_deg"].max().reindex(layers_ordered)
        null_lo = sub.groupby("layer")["null_p5"].min().reindex(layers_ordered)
        null_hi = sub.groupby("layer")["null_p95"].max().reindex(layers_ordered)

        c = colors.get(arch, "gray")
        ax.plot(depths, mean_angle.values, "-o", color=c, label=f"{arch} (mean of {sub['seed'].nunique()} seeds)")
        ax.fill_between(depths, min_angle.values, max_angle.values, color=c, alpha=0.2,
                         label=f"{arch} seed range")
        ax.fill_between(depths, null_lo.values, null_hi.values, color=c, alpha=0.08, hatch="//",
                         label=f"{arch} random-subspace null (5-95pct)")

    ax.axhline(90, color="black", linestyle=":", linewidth=1, label="90 deg (exact orthogonality)")
    ax.set_xlabel("Normalized layer depth (0=first conv block, 1=penultimate)")
    ax.set_ylabel("Principal angle: digit-gradient vs. sex subspace (deg)")
    ax.set_ylim(0, 95)
    ax.set_title("Layer-wise digit-gradient vs. sex-subspace angle\n"
                "GEOMETRIC claim (alignment, not necessity) -- " + N_FLAG, fontsize=9)
    ax.legend(fontsize=7, loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=ARCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-null", type=int, default=N_NULL_DEFAULT)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for arch in args.archs:
        for seed in args.seeds:
            all_rows += process_arch_seed(arch, seed, repo_root, args.n_null)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "layer_angle_table.csv", index=False)
    make_headline_figure(df, OUT_DIR / "headline_layer_angle.png")

    # decision rule
    per_layer = df.groupby(["arch", "layer"]).agg(
        depth_norm=("depth_norm", "first"),
        mean_angle=("principal_angle_deg", "mean"),
        min_angle=("principal_angle_deg", "min"),
        max_angle=("principal_angle_deg", "max"),
        min_null_p5=("null_p5", "min"),
        n_seeds_below_null=("below_null_p5", "sum"),
        n_seeds=("seed", "count"),
    ).reset_index()

    print("\n=== per-layer summary ===")
    print(per_layer.to_string(index=False))

    exceptions = per_layer[per_layer["n_seeds_below_null"] == per_layer["n_seeds"]]
    inconclusive = per_layer[(per_layer["n_seeds_below_null"] > 0) &
                              (per_layer["n_seeds_below_null"] < per_layer["n_seeds"])]

    if len(exceptions) == 0 and len(inconclusive) == 0:
        overall_verdict = "ALL_LAYERS_NEAR_ORTHOGONAL"
        overall_text = (
            "Every layer, every seed, both architectures: the digit-gradient "
            "direction stays at/above the random-subspace null relative to the "
            "sex subspace (never seed-stably below it). The digit decision's "
            "gradient does not align with sex anywhere in the pipeline -- "
            "whole-pipeline geometric support for 'not used', broadening the "
            "penultimate-only capstone. This is a geometric/alignment claim, "
            "not a necessity claim (no perturbation was run here)."
        )
    elif len(exceptions) > 0:
        overall_verdict = "EXCEPTION_LAYER_FOUND"
        overall_text = (
            f"Seed-stable exception at: {exceptions[['arch','layer']].to_dict('records')} "
            "-- the digit-gradient direction aligns with the sex subspace beyond "
            "chance at this layer, in every seed. This does NOT establish that "
            "the decision NEEDS sex here (alignment != necessity; that requires "
            "perturbation, out of scope) but it is the interesting exception to "
            "flag for follow-up, not smoothed over."
        )
    else:
        overall_verdict = "SOME_LAYERS_INCONCLUSIVE"
        overall_text = (
            f"Seed-unstable (some seeds below null, some not) at: "
            f"{inconclusive[['arch','layer']].to_dict('records')} -- reported as "
            "inconclusive at those layers, not forced to a verdict."
        )

    print(f"\n[layer_sex_gradient_angle] VERDICT: {overall_verdict}\n{overall_text}")

    run_meta = {
        "created": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "git_sha": git_sha(),
        "archs": args.archs,
        "seeds": args.seeds,
        "n_null": args.n_null,
        "n_flag": N_FLAG,
        "sex_subspace_dim_k": 1,
        "sex_subspace_dim_rationale": (
            "Binary female-vs-male contrast has exactly one degree of freedom "
            "between the two group means -- the whitened logistic-regression "
            "direction is rank-1 by construction, consistent with every other "
            "sex-direction estimator in this repo. k=1 reported explicitly per "
            "row (sex_subspace_dim column), not silently assumed."
        ),
        "sex_subspace_fit_population": "SI test fold (12 speakers, 9M/3F) -- per pre-registration, "
                                        "NOT the train+val fold used by geom_within_digit.py's (b)/(c) "
                                        "measures (that choice trades power for evaluation-population "
                                        "consistency with the digit-gradient, computed on the same fold).",
        "digit_gradient_population": "SI test fold, same 12 speakers / 6000 utterances as the sex-subspace fit.",
        "digit_gradient_aggregation": "unit-normalize each utterance's gradient, top eigenvector of the "
                                       "mean outer-product matrix (sign-invariant, avoids signed cancellation "
                                       "across utterances with different correct digits).",
        "angle_formula_reused_from": "scripts/geom_within_digit.py "
                                      "(angle = degrees(arccos(clip(||basis @ v||,0,1))))",
        "pre_registered_expectation": (
            "CONFIRMATORY: the digit-gradient stays >=~80 degrees from the sex "
            "subspace at every layer in both architectures, consistent with the "
            "penultimate result -- no layer where the decision aligns with sex. "
            "Written before running; watching for any layer that drops well "
            "below that as the interesting exception."
        ),
        "claim_boundary": (
            "GEOMETRIC claim only: near-orthogonality means the decision axis "
            "does not align with sex at that layer. Bounds alignment/pathway, "
            "not strict necessity (would need perturbation, out of scope here). "
            "Confidence comes from cross-layer/cross-architecture/cross-seed "
            "consistency, not from the female sample size (n=3)."
        ),
        "decision_rule": (
            "All layers near-orthogonal (angle >= null, both archs, seed-stable) "
            "-> whole-pipeline geometric support for 'not used'. A seed-stable "
            "layer below the null -> interesting exception, flagged, not "
            "overclaimed as necessity. Seed-unstable -> inconclusive at that layer."
        ),
        "overall_verdict": overall_verdict,
        "overall_verdict_text": overall_text,
    }
    with open(OUT_DIR / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, default=str)
    per_layer.to_csv(OUT_DIR / "per_layer_summary.csv", index=False)

    print(f"\n[layer_sex_gradient_angle] wrote outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
