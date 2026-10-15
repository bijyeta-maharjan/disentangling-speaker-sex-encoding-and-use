"""Does the digit READOUT use the validated whitened sex direction? The
"encoded -> used" question at the penultimate layer.

Established so far (see results/sex_direction/ and
scripts/direction_resplit_control.py, scripts/direction_*_correlation.py): sex
is encoded at penultimate (cross-architecture probing), distributed rather than
a single unit (unit seed-stability null), and a WHITENED direction there is validated as
sex-specific -- separable from all 219 non-sex re-splits of the same 12 SI
test speakers -- in cnn1d and spec_cnn2d only. Prior geometry (scripts/
geom_within_digit.py) put that direction ~83 degrees from a digit-discriminative
SUBSPACE, but that subspace was a class-mean/LDA-style construct, not the
model's own classifier weights -- it doesn't bind what the trained readout
actually does with the representation. This script asks the direct question:
moving a penultimate representation along the validated sex direction, does
the ACTUAL trained classifier head's output move more than it would along a
random direction?

METHODOLOGY NOTE -- read before trusting any number here (flagged up front,
not discovered by the reader): both cnn1d and spec_cnn2d's classifier head is
a 3-layer MLP (Linear(d->128) -> ReLU -> Dropout -> Linear(128->64) -> ReLU ->
Dropout -> Linear(64->10)), NOT a single linear map from the 256/128-dim
penultimate (GAP) representation to the 10 logits. So "the actual final-layer
classifier weights W (shape 10xd at the penultimate->logit map)" does not
exist as one stored parameter tensor -- the map is piecewise-linear (ReLU-
gated), not globally linear. Design decision:
W is operationalized as the JACOBIAN of the trained head's logits w.r.t. the
penultimate representation, d(logits)/d(penultimate), computed via autograd
through the REAL loaded classifier submodule (model.classifier) -- frozen
weights, eval(), no optimizer step; only the input requires grad. This is the
literal answer to "how much does logit_k move per unit step along u-hat" for
the ACTUAL trained readout, generalized correctly to a nonlinear head, and it
keeps u-hat in the persisted 256/128-dim GAP space (no refit needed -- the
alternative, redefining u-hat in the 64-dim last-hidden-layer space, would
have broken "reuse u-hat" and abandoned the space every prior artifact was
computed in). J_mean = mean of the per-utterance Jacobian over the SI
test fold (12 speakers, 6000 utterances -- the readout's typical operating
regime) is the single 10xd matrix used in s_k = |J_mean_k . u_hat|, the direct
analog of the linear-readout formula |W_k . u_hat|. A per-utterance sign-agreement diagnostic
is also reported: averaging Jacobians before dotting can mask cancellation if
the local sensitivity direction flips sign across utterances.

CLAIM BOUNDARY:
  n=3 female test speakers -- within-sample, not a population claim (FLAG,
  every artifact). But this test barely depends on female n: it uses the
  classifier weights (via J_mean) and the validated u-hat, not a female-vs-
  male sample statistic. Confidence here comes from null-normalization
  (1000 random directions), cross-seed stability (readout_counterfactual.py), and cross-
  architecture agreement -- NOT from sample size. Report those as the
  confidence source.
  This test bounds ONLY the linear (well: locally-linearized) readout's
  sensitivity to the penultimate sex direction. It does NOT bound whether
  EARLIER layers route sex-correlated features upstream of digit features --
  that is a distinct, unaddressed question. Stated on every output.
  Scope: cnn1d + spec_cnn2d only (resnet1d/crnn_gru direction unvalidated,
  excluded -- stated explicitly, not silently dropped).

Reuses (does not rebuild): persisted whitened directions
(whitened_direction_{arch}_seed42.npy, from scripts/direction_f0_correlation.py);
cached trained-penultimate features (data/processed/activations_week2_seed42/);
scripts/unit_selectivity.py's seed_checkpoints / si_test_sex_groups;
src/probing/extract_week2.py's build_trained (loads the REAL trained head).

Not done here: counterfactual perturbation + cross-seed stability
(readout_counterfactual.py), input-relevance attribution, acoustic work,
by-group generalization, resnet1d/crnn_gru.

    python scripts/readout_sensitivity.py
    python scripts/readout_sensitivity.py --redraw   # figure only, from the saved table
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

from scripts.unit_selectivity import OUT_DIR, penultimate_cache_path, seed_checkpoints, si_test_sex_groups
from src.probing.extract_week2 import build_trained
from src.probing.extract_week2 import load_labels as load_seed42_labels
from src.probing.splits import load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"

SCOPE_ARCHS = ["cnn1d", "spec_cnn2d"]
EXCLUDED_ARCHS = ["resnet1d", "crnn_gru"]
EXCLUDED_NOTE = (
    "resnet1d and crnn_gru EXCLUDED from this analysis: their whitened sex direction sat "
    "INSIDE the random-re-split null in the re-split control, so there is no "
    "validated direction whose readout sensitivity would be meaningful. Stated explicitly."
)
PRIMARY_SEED = 42
N_DIGITS = 10
N_NULL = 1000
NULL_RNG_SEED = 0  # fixed for reproducibility across reruns

FLAG = "n=3 female test speakers — within-sample, not a population claim"
SPEAKER_COUNT_NOTE = (
    "AudioMNIST is 48 male / 12 female speakers dataset-wide; this analysis, like the rest of "
    "the sex-direction analyses, follows the speaker-disjoint SI test fold (n=9 male / n=3 female, 4500/1500 "
    "utterances). Note this test barely depends on female n -- see module docstring."
)
BOUND_NOTE = (
    "This test bounds ONLY the linear readout's sensitivity to the penultimate sex direction. "
    "It does NOT establish whether earlier layers route sex-correlated features upstream of "
    "digit features -- that is a distinct, unaddressed question."
)
HEAD_METHOD_NOTE = (
    "Both classifier heads are 3-layer MLPs (Linear-ReLU-Dropout x2 then Linear), not a single "
    "linear map from penultimate to logits, so 'W' is operationalized as the Jacobian "
    "d(logits)/d(penultimate) of the REAL trained head, averaged over the SI test fold "
    "(J_mean, 10xd) -- s_k = |J_mean_k . u_hat|, the direct analog of the linear-readout formula."
)

DIRECTION_DIR = OUT_DIR  # results/sex_direction
OUT_CSV = OUT_DIR / "readout_sensitivity_seed42.csv"
FIGS_DIR = resolve("figures/output")
FIG_PATH = FIGS_DIR / "readout_jacobian_heatmap.png"
RUNS_DIR = resolve("results/sex_direction")


def load_whitened_direction(arch: str, seed: int) -> np.ndarray:
    path = DIRECTION_DIR / f"whitened_direction_{arch}_seed{seed}.npy"
    assert path.exists(), (
        f"persisted whitened direction missing: {path} -- this analysis REUSES it "
        "(run scripts/direction_f0_correlation.py first; do not refit here)"
    )
    u = np.load(path)
    assert abs(np.linalg.norm(u) - 1.0) < 1e-6, f"{path} is not unit-normalized"
    return u


def load_trained_head(cfg: dict, arch: str, seed: int, device: torch.device):
    ckpts = seed_checkpoints(cfg, seed)
    assert arch in ckpts, f"no SI seed-{seed} checkpoint for {arch}"
    ckpt_raw = torch.load(ckpts[arch], map_location=device, weights_only=False)
    model = build_trained(cfg, arch, ckpts[arch], device)
    model.eval()
    # DISCIPLINE: assert W was actually pulled from the trained head, not re-initialized.
    for name, param in model.classifier.state_dict().items():
        ckpt_param = ckpt_raw["model_state"][f"classifier.{name}"]
        assert torch.equal(param.cpu(), ckpt_param.cpu()), (
            f"{arch} seed={seed}: classifier.{name} does not match checkpoint -- head not "
            "actually loaded from the trained model"
        )
    print(f"[readout] {arch} seed={seed}: classifier head verified loaded from {ckpts[arch]} "
          f"({sum(p.numel() for p in model.classifier.parameters())} params)")
    return model


# --------------------------------------------------------------------------- #
# head-only Jacobian d(logits)/d(penultimate), batched via the "sum trick"
# (valid because Linear/ReLU/Dropout(eval=identity) act per-sample independently
# -- no BatchNorm in these classifier heads, so no cross-sample coupling)
# --------------------------------------------------------------------------- #
def head_jacobian(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    logits = model.classifier(x)  # (N, 10)
    assert logits.shape[1] == N_DIGITS
    J = np.empty((N_DIGITS, X.shape[0], X.shape[1]), dtype=np.float32)
    for k in range(N_DIGITS):
        grad = torch.autograd.grad(logits[:, k].sum(), x, retain_graph=(k < N_DIGITS - 1))[0]
        J[k] = grad.detach().cpu().numpy()
    return J  # (10, N, d)


def run_readout_sensitivity(cfg: dict) -> pd.DataFrame:
    device = torch.device("cpu")
    si_spec = load_si_spec(cfg)
    female_spk, male_spk = si_test_sex_groups(si_spec)
    labels = load_seed42_labels(cfg)
    spk_str = labels["speaker"].astype(str).to_numpy()
    mask = np.isin(spk_str, female_spk) | np.isin(spk_str, male_spk)
    test_idx = np.where(mask)[0]
    assert mask.sum() == 6000 and len(female_spk) == 3 and len(male_spk) == 9
    print(f"[readout] SI test fold: {len(female_spk)} female speakers ({int(np.isin(spk_str, female_spk).sum())} utts), "
          f"{len(male_spk)} male speakers ({int(np.isin(spk_str, male_spk).sum())} utts)")
    print(f"[readout] {SPEAKER_COUNT_NOTE}")
    print(f"[readout] {EXCLUDED_NOTE}")
    print(f"[readout] {HEAD_METHOD_NOTE}")

    rng = np.random.default_rng(NULL_RNG_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    fig_data = {}

    for arch in SCOPE_ARCHS:
        u = load_whitened_direction(arch, PRIMARY_SEED)
        d = u.shape[0]

        X_full = np.load(penultimate_cache_path(PRIMARY_SEED, arch))
        X_test = X_full[test_idx]
        assert X_test.shape[1] == d, f"{arch}: penultimate dim {X_test.shape[1]} != u_hat dim {d}"

        model = load_trained_head(cfg, arch, PRIMARY_SEED, device)
        J = head_jacobian(model, X_test, device)  # (10, N, d)
        J_mean = J.mean(axis=1)  # (10, d) -- the "W" analog

        # null: 1000 random unit directions, SAME J_mean reused (direction-independent Jacobian)
        V = rng.normal(size=(N_NULL, d)).astype(np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        s = np.abs(J_mean @ u)                      # (10,)
        null_s = np.abs(J_mean @ V.T)                # (10, N_NULL)

        whole_s = float(np.linalg.norm(J_mean @ u))
        whole_null = np.linalg.norm(J_mean @ V.T, axis=0)  # (N_NULL,)
        whole_pct = float((whole_null <= whole_s).mean() * 100)

        print(f"[readout] {arch}: whole-readout sensitivity ||J_mean . u_hat||={whole_s:.4f}  "
              f"null p50={np.percentile(whole_null,50):.4f} p95={np.percentile(whole_null,95):.4f}  "
              f"percentile={whole_pct:.1f}")

        for k in range(N_DIGITS):
            null_p50 = float(np.percentile(null_s[k], 50))
            null_p95 = float(np.percentile(null_s[k], 95))
            percentile = float((null_s[k] <= s[k]).mean() * 100)
            exceeds_95 = bool(s[k] > null_p95)

            # diagnostic: does the per-utterance local sensitivity sign agree with J_mean's?
            per_pt_dot = J[k] @ u  # (N,)
            mean_sign = np.sign(J_mean[k] @ u)
            sign_agree_frac = float((np.sign(per_pt_dot) == mean_sign).mean()) if mean_sign != 0 else float("nan")
            mean_abs_dot = float(np.mean(np.abs(per_pt_dot)))  # secondary/robustness magnitude

            rows.append({
                "arch": arch, "digit": k, "s_k": round(float(s[k]), 5),
                "null_p50": round(null_p50, 5), "null_p95": round(null_p95, 5),
                "percentile": round(percentile, 2), "exceeds_95": exceeds_95,
                "sign_agreement_frac": round(sign_agree_frac, 3),
                "mean_abs_dot_diagnostic": round(mean_abs_dot, 5),
                "whole_readout_s": round(whole_s, 5), "whole_readout_percentile": round(whole_pct, 2),
                "n_male_speakers": 9, "n_female_speakers": 3,
                "n_utts": int(mask.sum()), "flag": FLAG,
            })

        fig_data[arch] = {"percentile": np.array([r["percentile"] for r in rows if r["arch"] == arch]),
                           "exceeds_95": np.array([r["exceeds_95"] for r in rows if r["arch"] == arch])}

        exceptions = [r["digit"] for r in rows if r["arch"] == arch and r["exceeds_95"]]
        verdict = "WITHIN NULL for all digits" if not exceptions else f"EXCEEDS 95th pct on digit(s) {exceptions}"
        print(f"[readout] {arch}: {verdict}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[readout] table -> {OUT_CSV}")

    fig_path = headline_figure(fig_data)
    print(f"[readout] headline figure -> {fig_path}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "readout_sensitivity_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "git_sha": git_sha(), "seed": PRIMARY_SEED, "flag": FLAG,
            "speaker_count_note": SPEAKER_COUNT_NOTE, "excluded_note": EXCLUDED_NOTE,
            "bound_note": BOUND_NOTE, "head_method_note": HEAD_METHOD_NOTE,
            "scope_archs": SCOPE_ARCHS, "excluded_archs": EXCLUDED_ARCHS,
            "n_null": N_NULL, "null_rng_seed": NULL_RNG_SEED,
            "female_test_speakers": female_spk, "male_test_speakers": male_spk,
            "rows": rows,
        }, fh, indent=2)

    return df


def headline_figure(fig_data: dict) -> Path:
    """Sequential single-hue heatmap (percentile is a magnitude, not a polarity):
    one hue light->dark, never a rainbow."""
    archs = SCOPE_ARCHS
    mat = np.stack([fig_data[a]["percentile"] for a in archs])  # (n_arch, 10)
    exceeds = np.stack([fig_data[a]["exceeds_95"] for a in archs])

    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(N_DIGITS))
    ax.set_xticklabels(range(N_DIGITS))
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs)
    ax.set_xlabel("digit")

    for i in range(len(archs)):
        for j in range(N_DIGITS):
            val = mat[i, j]
            txt_color = "white" if val > 55 else "#1a1a1a"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9, color=txt_color)
            if exceeds[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor="#c0392b", lw=2.5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("null percentile (vs. 1000 random directions)")

    fig.suptitle(
        "Readout sensitivity to the validated sex direction, per digit (seed 42)\n"
        f"{FLAG}\n"
        "Bounds ONLY the linear readout's sensitivity to the penultimate sex direction --\n"
        "does NOT establish whether earlier layers route sex-correlated features upstream of digit features.\n"
        "Red outline = exceeds the 95th percentile of the random-direction null. Scope: cnn1d + spec_cnn2d only.",
        fontsize=8.8, x=0.5, ha="center",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.74])
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return FIG_PATH


def redraw_from_saved() -> Path:
    """Redraw the heatmap from results/sex_direction/readout_sensitivity_seed42.csv."""
    df = pd.read_csv(OUT_CSV)
    fig_data = {arch: {"percentile": df[df["arch"] == arch].sort_values("digit")["percentile"].to_numpy(),
                       "exceeds_95": df[df["arch"] == arch].sort_values("digit")["exceeds_95"].to_numpy()}
                for arch in SCOPE_ARCHS}
    return headline_figure(fig_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redraw", action="store_true", help="only redraw the figure from the saved table")
    args = ap.parse_args()
    if args.redraw:
        print(f"[readout] figure -> {redraw_from_saved()}")
        return
    cfg = load_config(DEFAULT_CONFIG)
    df = run_readout_sensitivity(cfg)
    print("\n================ READOUT SENSITIVITY ================")
    print(df[["arch", "digit", "s_k", "null_p50", "null_p95", "percentile", "exceeds_95"]].to_string(index=False))
    print(f"\n{FLAG}")
    print(SPEAKER_COUNT_NOTE)
    print(EXCLUDED_NOTE)
    print(BOUND_NOTE)
    print("\n--- HEADLINE ---")
    for arch in SCOPE_ARCHS:
        sub = df[df["arch"] == arch]
        exceptions = sub[sub["exceeds_95"]]["digit"].tolist()
        whole_pct = float(sub["whole_readout_percentile"].iloc[0])
        if not exceptions:
            print(f"  {arch}: ALL digits within the random-direction null (whole-readout percentile={whole_pct:.1f}) "
                  "-> consistent with 'not used at the readout'")
        else:
            print(f"  {arch}: digit(s) {exceptions} EXCEED the 95th-percentile null "
                  f"(whole-readout percentile={whole_pct:.1f}) -> H1 candidate, localized readout sensitivity")


if __name__ == "__main__":
    main()
