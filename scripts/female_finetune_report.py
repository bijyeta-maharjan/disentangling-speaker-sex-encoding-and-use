"""Build the fine-tuning recovery report (did fine-tuning help, and is it female-specific?)
from results/female_finetune/recovery_raw_results.json.

Per-fold spread is reported, never pooled into one CI across folds (locked
out-of-scope decision). See scripts/female_finetune.py for the
fine-tuning protocol and the male-control asymmetry this report must stamp
on every artifact.
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

BOUNDED_CLAIM = (
    "Fine-tuning shows what the model CAN RECRUIT to help with a new group, "
    "not what the original model was USING. This is true for both conditions."
)
MALE_CONTROL_ASYMMETRY = (
    "The male-control pool is NOT a clean analogue of the female pool, in TWO "
    "confounded ways: (1) exposure -- the 10 val-only males had zero gradient "
    "updates but DID influence checkpoint selection (weak exposure, not true "
    "zero exposure like the 12 women); (2) ceiling -- the male-only model's "
    "pre-fine-tune accuracy on these males is already very high (0.92-0.99, "
    "vs 0.79-0.93 for the women), leaving far less headroom for fine-tuning "
    "to show ANY improvement regardless of mechanism. Both confounds bias the "
    "male-control condition toward showing a SMALLER delta than the female "
    "condition even under a purely generic-adaptation story. Therefore: if "
    "female and male-control signatures MATCH (layer-change / band-shift analyses), that is STRONG "
    "evidence for generic adaptation (agreement despite two biases pushing "
    "toward disagreement). If they DIFFER, that CANNOT be reported as a clean "
    "female-specific finding -- it is confounded with novel-vs-familiar "
    "exposure AND with ceiling/headroom, not sex alone. Report differences as "
    "'suggestive, control-confounded', never as a clean female-specific claim."
)


def main() -> None:
    with open(OUT_DIR / "recovery_raw_results.json") as fh:
        raw = json.load(fh)
    df = pd.DataFrame([{
        "arch": r["arch"], "seed": r["seed"], "fold": r["fold"], "condition": r["condition"],
        "n_finetune": len(r["finetune_speakers"]), "n_eval": len(r["eval_speakers"]),
        "before_point": r["before"]["point"], "before_lo": r["before"]["lo"], "before_hi": r["before"]["hi"],
        "after_point": r["after"]["point"], "after_lo": r["after"]["lo"], "after_hi": r["after"]["hi"],
        "delta": r["delta_point"],
    } for r in raw])
    df.to_csv(OUT_DIR / "recovery_per_run.csv", index=False)

    fold_avg = df.groupby(["arch", "condition", "fold"]).agg(
        n_finetune=("n_finetune", "first"), n_eval=("n_eval", "first"),
        before_mean=("before_point", "mean"), after_mean=("after_point", "mean"),
        delta_mean=("delta", "mean"), delta_min=("delta", "min"), delta_max=("delta", "max"),
    ).reset_index()
    fold_avg.to_csv(OUT_DIR / "recovery_fold_seed_avg.csv", index=False)

    summary_rows = []
    for arch in sorted(df["arch"].unique()):
        for cond in ["female", "male_control"]:
            sub = df[(df["arch"] == arch) & (df["condition"] == cond)]
            fa = fold_avg[(fold_avg["arch"] == arch) & (fold_avg["condition"] == cond)]
            summary_rows.append({
                "arch": arch, "condition": cond,
                "n_folds": fa.shape[0], "n_runs": sub.shape[0],
                "before_mean": sub["before_point"].mean(),
                "after_mean": sub["after_point"].mean(),
                "delta_mean_all_runs": sub["delta"].mean(),
                "delta_fold_mean_min": fa["delta_mean"].min(),
                "delta_fold_mean_max": fa["delta_mean"].max(),
                "frac_runs_positive": (sub["delta"] > 0).mean(),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "recovery_summary_table.csv", index=False)

    # Gate verdicts per architecture
    verdicts = {}
    for arch in sorted(df["arch"].unique()):
        fem = summary[(summary["arch"] == arch) & (summary["condition"] == "female")].iloc[0]
        male = summary[(summary["arch"] == arch) & (summary["condition"] == "male_control")].iloc[0]
        female_reliable = fem["delta_fold_mean_min"] > 0 and fem["frac_runs_positive"] >= 0.9
        if not female_reliable:
            verdict = "NULL_NOTHING_TO_LOCALIZE"
            text = "Female fine-tuning does not reliably improve female eval accuracy across folds -- nothing to localize."
        else:
            comparable = abs(fem["delta_mean_all_runs"] - male["delta_mean_all_runs"]) < 0.02
            if comparable:
                verdict = "COMPARABLE_TO_MALE_CONTROL"
                text = ("Female fine-tuning helps, but the male-control condition shows a similarly-sized "
                        "improvement -- consistent with generic adaptation, not a distinct female mechanism, "
                        "at the BEHAVIORAL level. Proceed to the layer-change / band-shift analyses to check whether the underlying "
                        "frequency/layer shift also converges.")
            else:
                verdict = "LARGER_THAN_MALE_CONTROL_CONFOUNDED"
                text = ("Female fine-tuning helps substantially more than the male-control condition at the "
                        "behavioral level, BUT this asymmetry is expected even under generic adaptation because "
                        "of the exposure and ceiling confounds (see male_control_asymmetry note) -- this alone "
                        "does NOT establish a female-specific mechanism. Proceed to the layer-change / band-shift analyses (frequency/layer "
                        "double-difference) to determine whether the mechanism is genuinely different, reporting "
                        "any such finding as suggestive and control-confounded, never clean.")
        verdicts[arch] = {
            "verdict": verdict, "text": text,
            "female_delta_mean": float(fem["delta_mean_all_runs"]),
            "female_delta_fold_range": [float(fem["delta_fold_mean_min"]), float(fem["delta_fold_mean_max"])],
            "female_frac_positive": float(fem["frac_runs_positive"]),
            "male_control_delta_mean": float(male["delta_mean_all_runs"]),
            "male_control_delta_fold_range": [float(male["delta_fold_mean_min"]), float(male["delta_fold_mean_max"])],
            "male_control_frac_positive": float(male["frac_runs_positive"]),
        }

    # Figure: per-fold delta means, female vs male-control, per arch
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, arch in zip(axes, sorted(df["arch"].unique())):
        for cond, color, marker in [("female", "tab:red", "o"), ("male_control", "tab:blue", "s")]:
            fa = fold_avg[(fold_avg["arch"] == arch) & (fold_avg["condition"] == cond)].sort_values("fold")
            ax.errorbar(fa["fold"], fa["delta_mean"],
                        yerr=[fa["delta_mean"] - fa["delta_min"], fa["delta_max"] - fa["delta_mean"]],
                        fmt=marker, color=color, capsize=4, markersize=8, label=cond,
                        linestyle="-", alpha=0.85)
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_title(arch)
        ax.set_xlabel("fold")
        ax.set_xticks(sorted(df["fold"].unique()))
    axes[0].set_ylabel("eval accuracy delta (after - before fine-tune)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Fine-tune eval-accuracy improvement, female vs. male-control, per fold\n"
                 "(error bars = min/max across seeds 42/43/44 within that fold -- NOT a pooled CI)", fontsize=10)
    fig.text(0.5, -0.02, BOUNDED_CLAIM, ha="center", fontsize=7.5, style="italic", wrap=True)
    fig.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_DIR / "finetune_recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    run_meta = {
        "git_sha": git_sha(),
        "analysis": "finetune_recovery",
        "epochs_locked": 2,
        "epoch_gate_pilot": "results/female_finetune/pilot_epoch_gate.json (fold0 cnn1d seed42: "
                             "epochs=2 delta=+0.149 [CI non-overlapping], epochs=3 delta=+0.148 -- no added benefit, "
                             "2 epochs locked)",
        "lr": 1e-4, "batch_size": 128, "weight_decay": 0.0, "optimizer": "Adam",
        "archs": sorted(df["arch"].unique().tolist()),
        "seeds": sorted(df["seed"].unique().tolist()),
        "n_folds_female": 4, "n_folds_male_control": 4,
        "pinned_checkpoints_manifest": "results/female_finetune/pinned_checkpoints.json",
        "bounded_claim": BOUNDED_CLAIM,
        "male_control_asymmetry": MALE_CONTROL_ASYMMETRY,
        "n_flag_female": "fine-tune female n<=9, eval female n=3 per fold -- within-sample, not a population claim",
        "n_flag_male_control": "male-control fine-tune n<=8, eval n<=3 per fold, drawn from val-only (weak-exposure) "
                                "males -- within-sample, control-confounded, not a clean zero-exposure analogue",
        "pooling_note": "Fold spread is reported (min/max of the seed-averaged per-fold delta); folds are NEVER "
                         "pooled into one tighter CI across folds (locked out-of-scope decision).",
        "pre_registered_gate": "If female fine-tuning doesn't reliably improve female accuracy, the 'what changed' "
                                "analysis has nothing to localize -- flag as null. If it does improve, compare "
                                "magnitude to male-control: comparable -> generic adaptation at the behavioral "
                                "level; larger -> proceed to the layer-change / band-shift analyses but do NOT claim "
                                "female-specificity from accuracy alone, since the male-control asymmetry above predicts a smaller male-"
                                "control delta even under pure generic adaptation.",
        "verdicts": verdicts,
    }
    with open(OUT_DIR / "recovery_run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, default=str)

    print(json.dumps(verdicts, indent=2))
    print(f"\nwrote {OUT_DIR}/recovery_run_meta.json, recovery_summary_table.csv, recovery_fold_seed_avg.csv, "
          f"recovery_per_run.csv, finetune_recovery.png")


if __name__ == "__main__":
    main()
