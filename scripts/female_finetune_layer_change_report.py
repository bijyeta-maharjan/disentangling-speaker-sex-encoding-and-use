"""Build the layer-change report (layer-wise localization of what fine-tuning changed)
from results/female_finetune/layer_change_raw_results.json.

Reporting frame is locked in scripts/female_finetune_layer_change.py's
docstring and repeated here in run-meta: the female profile is a DESCRIPTIVE
localization ("what changed to read women"), not a female-specific claim.
The male-control profile is a confounded secondary reference (see
scripts/female_finetune.py for the two structural confounds:
novel-vs-familiar exposure, and ceiling/no-headroom -- also documented in
recovery_run_meta.json's male_control_asymmetry). The female-minus-male
double-difference is reported but is NOT interpretable as specificity in
either direction on this dataset (AudioMNIST has no clean alternative
control -- only 12 female speakers total, all already used).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import resolve
from src.utils.experiment import git_sha

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = resolve("results/female_finetune")
FIGS_DIR = REPO_ROOT / "figures" / "output"

DESCRIPTIVE_ONLY_CAVEAT = (
    "The female layer-change profile is reported as DESCRIPTIVE localization -- "
    "what the model changed, at each depth, to read women it was not trained on "
    "-- explicitly NOT as a 'female-specific' mechanism claim. The male-control "
    "profile is a confounded secondary reference (weak novel-vs-familiar exposure "
    "asymmetry + ceiling/no-headroom, see recovery_run_meta.json male_control_asymmetry): "
    "the 10 male-control speakers had zero gradient updates but weak checkpoint-"
    "selection exposure, and the male-only model is already near-ceiling on them "
    "pre-fine-tune, leaving far less room to move. Because of this, the female-minus-"
    "male double-difference by layer is NOT interpretable as specificity in EITHER "
    "direction on this dataset: a difference could be the confounds, not sex; "
    "agreement could be masked by the male pool's ceiling, not genuine convergence. "
    "AudioMNIST has no clean alternative control (only 12 female speakers total, all "
    "already used) -- female-vs-generic specificity is structurally NOT RESOLVABLE "
    "on this data, stated explicitly rather than smoothed over."
)

PRE_REGISTERED_EXPECTATION = (
    "Pre-registered (per the recovery gate, both archs verdict LARGER_THAN_MALE_CONTROL_"
    "CONFOUNDED): the layer-change analysis asks whether fine-tuning retunes LOW-level layers (early "
    "conv/spectrogram blocks -- acoustic/frequency-level retuning) or HIGH-level "
    "layers (late blocks / penultimate -- decision-level retuning). This question is "
    "answerable from the female profile ALONE (before-vs-after the female fine-tune, "
    "same eval speech) and does not require the male control to be valid. If relative "
    "change is flat or higher early -> low-level acoustic retuning; if it rises "
    "monotonically with depth -> high-level decision-boundary retuning. The female-"
    "minus-male comparison is computed for completeness but is pre-registered as "
    "NOT informative about female-specificity (see DESCRIPTIVE_ONLY_CAVEAT)."
)


def main() -> None:
    with open(OUT_DIR / "layer_change_raw_results.json") as fh:
        raw = json.load(fh)

    rows = []
    for r in raw:
        layer_names = list(r["relative_change_by_layer"].keys())
        n_layers = len(layer_names)
        for i, ln in enumerate(layer_names):
            rows.append({
                "arch": r["arch"], "seed": r["seed"], "fold": r["fold"],
                "condition": r["condition"], "layer": ln,
                "layer_index": i, "n_layers": n_layers,
                "normalized_depth": i / (n_layers - 1),
                "n_finetune": r["n_finetune"], "n_eval": r["n_eval"],
                "relative_change": r["relative_change_by_layer"][ln],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "layer_change_per_run_layer.csv", index=False)

    profile = df.groupby(["arch", "condition", "layer", "layer_index", "normalized_depth"]).agg(
        n_runs=("relative_change", "size"),
        mean=("relative_change", "mean"),
        min=("relative_change", "min"),
        max=("relative_change", "max"),
    ).reset_index().sort_values(["arch", "condition", "layer_index"])
    profile.to_csv(OUT_DIR / "layer_change_profile.csv", index=False)

    # Female-minus-male double difference, by matched layer_index (reported, flagged uninterpretable for specificity)
    dd_rows = []
    for arch in sorted(df["arch"].unique()):
        fem = profile[(profile["arch"] == arch) & (profile["condition"] == "female")].set_index("layer_index")
        male = profile[(profile["arch"] == arch) & (profile["condition"] == "male_control")].set_index("layer_index")
        for li in sorted(fem.index):
            dd_rows.append({
                "arch": arch, "layer_index": li, "layer": fem.loc[li, "layer"],
                "normalized_depth": fem.loc[li, "normalized_depth"],
                "female_mean": fem.loc[li, "mean"], "male_control_mean": male.loc[li, "mean"],
                "double_diff_female_minus_male": fem.loc[li, "mean"] - male.loc[li, "mean"],
            })
    double_diff = pd.DataFrame(dd_rows)
    double_diff.to_csv(OUT_DIR / "layer_change_double_diff_by_layer.csv", index=False)

    # Per-arch verdict on WHERE change concentrates (female profile only, per pre-registered question)
    verdicts = {}
    for arch in sorted(df["arch"].unique()):
        fem = profile[(profile["arch"] == arch) & (profile["condition"] == "female")].sort_values("layer_index")
        depths = fem["normalized_depth"].to_numpy()
        means = fem["mean"].to_numpy()
        peak_idx = int(np.argmax(means))
        peak_depth = float(depths[peak_idx])
        early_mean = float(means[: len(means) // 2].mean())
        late_mean = float(means[len(means) // 2:].mean())
        if peak_depth <= 1e-9:
            shape = "FLAT_OR_EARLY_PEAK"
        elif peak_idx == len(means) - 1 or late_mean > early_mean * 1.15:
            shape = "MONOTONIC_INCREASE_WITH_DEPTH"
        else:
            shape = "MID_PEAK_THEN_DECLINE"
        verdicts[arch] = {
            "shape": shape,
            "peak_layer": str(fem.iloc[peak_idx]["layer"]),
            "peak_normalized_depth": peak_depth,
            "peak_mean_relative_change": float(means[peak_idx]),
            "early_half_mean": early_mean,
            "late_half_mean": late_mean,
            "interpretation": (
                "Relative activation change from fine-tuning is largest at or increasing toward the "
                "PENULTIMATE / late layers -- consistent with high-level, decision-level retuning rather "
                "than low-level acoustic re-encoding."
                if shape in ("MONOTONIC_INCREASE_WITH_DEPTH",)
                else "Relative activation change from fine-tuning peaks or stays elevated at MID-depth then "
                     "declines toward the penultimate -- consistent with retuning concentrated in intermediate "
                     "feature-combination layers rather than either raw acoustic front-end or final decision layer."
                if shape == "MID_PEAK_THEN_DECLINE"
                else "Relative activation change is flat or largest at the earliest layer -- consistent with "
                     "low-level acoustic/frequency-level retuning."
            ),
        }

    # Figure: relative change vs normalized depth, female vs male-control, per arch (mean +/- min/max across seeds*folds)
    archs = sorted(df["arch"].unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 5), sharey=False)
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        for cond, color, marker in [("female", "tab:red", "o"), ("male_control", "tab:blue", "s")]:
            sub = profile[(profile["arch"] == arch) & (profile["condition"] == cond)].sort_values("layer_index")
            ax.errorbar(sub["normalized_depth"], sub["mean"],
                        yerr=[sub["mean"] - sub["min"], sub["max"] - sub["mean"]],
                        fmt=marker, color=color, capsize=4, markersize=8, label=cond,
                        linestyle="-", alpha=0.85)
        ax.set_title(arch)
        ax.set_xlabel("normalized depth (0=first layer, 1=penultimate)")
        ax.set_xlim(-0.05, 1.05)
    axes[0].set_ylabel("relative L2 activation change (fine-tuned vs. pinned, eval set)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Layer-wise activation change from fine-tuning, female vs. male-control\n"
                 "(error bars = min/max across seeds 42/43/44 x folds -- NOT a pooled CI; "
                 "female-vs-male comparison is confounded, see run-meta)", fontsize=9)
    fig.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_DIR / "layer_change_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    run_meta = {
        "git_sha": git_sha(),
        "analysis": "layerwise_localization",
        "depends_on": "results/female_finetune/recovery_run_meta.json (both archs: "
                       "LARGER_THAN_MALE_CONTROL_CONFOUNDED -> proceed to localization)",
        "epochs_locked": 2,
        "lr": 1e-4, "batch_size": 128, "weight_decay": 0.0, "optimizer": "Adam",
        "archs": archs,
        "seeds": sorted(df["seed"].unique().tolist()),
        "n_folds_female": 4, "n_folds_male_control": 4,
        "n_runs_expected": len(archs) * 3 * 4 * 2,
        "n_runs_found": len(raw),
        "metric": "relative_change = sqrt(sum ||act_after - act_before||^2) / sqrt(sum ||act_before||^2), "
                  "streamed per batch over the fold's eval speakers, per intermediate layer + penultimate, "
                  "in depth order, normalized to [0, 1].",
        "pinned_checkpoints_manifest": "results/female_finetune/pinned_checkpoints.json",
        "pre_registered_expectation": PRE_REGISTERED_EXPECTATION,
        "descriptive_only_caveat": DESCRIPTIVE_ONLY_CAVEAT,
        "male_control_is_confounded": True,
        "female_vs_generic_specificity_resolvable_on_this_data": False,
        "specificity_not_resolvable_reason": (
            "Structural: no novel (never-before-seen) male speakers are available to fine-tune on "
            "(all males are already in the male-only training pool), and the male-control pool's "
            "pre-fine-tune accuracy is already near ceiling, leaving no headroom -- there is no clean "
            "alternative control on AudioMNIST (only 12 female speakers total, all used)."
        ),
        "verdicts": verdicts,
    }
    with open(OUT_DIR / "layer_change_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, default=str)

    print(json.dumps(verdicts, indent=2))
    print(f"\nwrote {OUT_DIR}/layer_change_run_meta.json, layer_change_profile.csv, layer_change_per_run_layer.csv, "
          f"layer_change_double_diff_by_layer.csv, layer_change_profile.png")


if __name__ == "__main__":
    main()
