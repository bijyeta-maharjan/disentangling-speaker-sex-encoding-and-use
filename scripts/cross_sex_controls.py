"""Two controls on the cross-sex behavioural gap.

Reuses scripts/cross_sex_generalization.py's harness (load_cache_for,
SI_BASELINE_CHECKPOINTS, infer/bootstrap helpers) and
scripts/make_male_subset_control_split.py. Trains NEW models only for the
same-size control; the leave-one-out check is reconstructed entirely from
results/cross_sex_generalization's already-saved per-digit CSV -- no
retraining, no new inference (leave-one-out WITHIN the existing 3
genuinely-held-out female speakers, not a full 5-fold rotation).

Same-size control (male->male). Question: is the cross-sex gap about SEX, or
just about training on fewer/different speakers? Expectation (written before
looking): if the cross-sex gap is sex-specific, the male->male generalization
drop will be substantially smaller than the ~5-point cross-sex gap. If
male->male drops comparably, the cross-sex gap is mostly a
sample-size/distribution-shift artifact, not sex. This control CAN falsify the
behavioural sex claim; that is the point of running it.

Leave-one-out (within the fixed 3). Question: is the cross-sex gap a one-off
driven by one of the 3 held-out women, or robust to which of them you look
at? Expectation: if the effect is real, the gap stays negative across (most)
rotations; if it flips sign / swings wildly, it is speaker-driven and
fragile. NOT better-powered than the original n=3 (still those same 3 women,
sub-sampled) -- rotations are reported individually, never pooled into a
single tighter CI.

Claim boundaries (both checks):
  - The same-size control is well-powered (n=9 held-out males) -- its
    MAGNITUDE can be trusted.
  - The leave-one-out check is NOT better-powered (<=3 women); it tests
    DIRECTION-robustness, not magnitude. Every leave-one-out artifact carries
    "n=3 female test speakers -- within-sample, not a population claim."
  - All accuracies carry speaker-bootstrap 95% CIs.

Not done here: no train-female->test-male, no balanced dataset, no
perturbation/CRP, no new architectures, no hyperparameter tuning, no pooling of
leave-one-out rotations into one CI.

Usage: python scripts/cross_sex_controls.py [--archs cnn1d spec_cnn2d]
       python scripts/cross_sex_controls.py --redraw   # male_only_gap figure from the saved table
"""
from __future__ import annotations

import argparse
import itertools
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fairness.bootstrap import bootstrap_speaker_ci
from src.models.factory import build_model, input_kind
from src.splits.make_splits import load_split_indices
from src.train.trainer import train as train_model
from src.utils.config import load_config, resolve, resolve_arch_cfg
from src.utils.experiment import git_sha
from scripts.cross_sex_generalization import (
    infer, load_cache_for, SI_BASELINE_CHECKPOINTS, per_speaker_overall, bootstrap_diff_ci,
)

ARCHS = ["cnn1d", "spec_cnn2d"]
SEEDS = [42, 43, 44]
MALE_SUBSET_REGIME = "male_subset_control_male_train_male_test"
CROSS_SEX_RESULTS_DIR = resolve("results/cross_sex_generalization")
OUT_DIR = resolve("results/cross_sex_controls")
FIGS_DIR = resolve("figures/output")
FEMALE_HELD_OUT = ["36", "43", "47"]
N_FLAG_FEMALE = "n=3 female test speakers -- within-sample, not a population claim"


# --------------------------------------------------------------------------- #
# Same-size control: male -> male
# --------------------------------------------------------------------------- #
def run_same_size_control(archs: list[str], repo_root: Path, seeds: list[int] | None = None,
              reuse_run_dirs: dict | None = None) -> dict:
    """reuse_run_dirs: optional {(arch, seed): run_dir_str} to skip retraining
    a combo that already completed (has train_summary.json) in a prior,
    interrupted invocation -- resumability after a killed background job."""
    seeds = seeds if seeds is not None else SEEDS
    reuse_run_dirs = reuse_run_dirs or {}
    print(f"\n{'=' * 70}\nSame-size control: male-subset models (seeds={seeds})\n{'=' * 70}")
    results = {"by_arch": {}}
    for arch in archs:
        results["by_arch"][arch] = {"seeds": {}}
        for seed in seeds:
            print(f"\n--- same-size control {arch} seed {seed} ---")
            cfg = resolve_arch_cfg(load_config("configs/week2_architectures.yaml"), arch=arch, seed=seed)
            feature = input_kind(cfg)
            device = torch.device("cpu")

            reuse_dir = reuse_run_dirs.get((arch, seed))
            if reuse_dir and (Path(reuse_dir) / "train_summary.json").exists():
                print(f"  reusing already-completed run: {reuse_dir}")
                run_dir = Path(reuse_dir)
            else:
                summary = train_model(cfg, MALE_SUBSET_REGIME)
                run_dir = Path(summary["run_dir"])

            cache = load_cache_for(cfg)
            idx = load_split_indices(cfg, MALE_SUBSET_REGIME, cache)
            test_idx = idx["test"]  # 9 SI-held-out males

            ckpt_a = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            model_a = build_model(cfg).to(device)
            model_a.load_state_dict(ckpt_a["model_state"])

            si_dir = resolve(SI_BASELINE_CHECKPOINTS[(arch, seed)])
            ckpt_c = torch.load(si_dir / "best.pt", map_location="cpu", weights_only=False)
            model_c = build_model(cfg).to(device)
            model_c.load_state_dict(ckpt_c["model_state"])

            yt_a, yp_a, spk_a = infer(model_a, cache, test_idx, device, feature, cfg["train"]["batch_size"])
            yt_c, yp_c, spk_c = infer(model_c, cache, test_idx, device, feature, cfg["train"]["batch_size"])
            assert np.array_equal(spk_a, spk_c)

            per_spk_a = per_speaker_overall(yt_a, yp_a, spk_a)
            per_spk_c = per_speaker_overall(yt_c, yp_c, spk_c)

            results["by_arch"][arch]["seeds"][seed] = {
                "train_run_dir": str(run_dir),
                "si_baseline_run_dir": str(si_dir),
                "held_out_male_speakers": sorted(np.unique(spk_a).tolist()),
                "male_subset_on_held_out_males": {
                    "overall": bootstrap_speaker_ci(per_spk_a["accuracy"].to_numpy(), seed=seed),
                    "per_speaker": per_spk_a,
                },
                "si_baseline_on_held_out_males": {
                    "overall": bootstrap_speaker_ci(per_spk_c["accuracy"].to_numpy(), seed=seed),
                    "per_speaker": per_spk_c,
                },
            }
            a_pt = results["by_arch"][arch]["seeds"][seed]["male_subset_on_held_out_males"]["overall"]["point"]
            c_pt = results["by_arch"][arch]["seeds"][seed]["si_baseline_on_held_out_males"]["overall"]["point"]
            print(f"  male-subset|male-subset-trained={a_pt:.4f}  male|SI-baseline={c_pt:.4f}  gap={a_pt - c_pt:+.4f}")

        # seed-pool
        seeds = SEEDS
        a_frames = [results["by_arch"][arch]["seeds"][s]["male_subset_on_held_out_males"]["per_speaker"].set_index("speaker")
                    for s in seeds]
        c_frames = [results["by_arch"][arch]["seeds"][s]["si_baseline_on_held_out_males"]["per_speaker"].set_index("speaker")
                    for s in seeds]
        a_pooled = pd.concat(a_frames, axis=1).mean(axis=1).sort_index()
        c_pooled = pd.concat(c_frames, axis=1).mean(axis=1).sort_index()
        diff_ci = bootstrap_diff_ci(a_pooled.to_numpy(), c_pooled.to_numpy(), seed=42)
        pooled_a_ci = bootstrap_speaker_ci(a_pooled.to_numpy(), seed=42)
        pooled_c_ci = bootstrap_speaker_ci(c_pooled.to_numpy(), seed=42)
        per_seed_gap = {
            s: (results["by_arch"][arch]["seeds"][s]["male_subset_on_held_out_males"]["overall"]["point"]
                - results["by_arch"][arch]["seeds"][s]["si_baseline_on_held_out_males"]["overall"]["point"])
            for s in seeds
        }
        results["by_arch"][arch]["pooled"] = {
            "male_subset_acc": pooled_a_ci, "si_baseline_acc": pooled_c_ci,
            "gap_ci": diff_ci, "per_seed_gap": per_seed_gap,
        }
        print(f"  same-size control {arch} seed-pooled gap: {diff_ci}")
    return results


# --------------------------------------------------------------------------- #
# Leave-one-out rotation within the fixed 3 (no retraining)
# --------------------------------------------------------------------------- #
def per_speaker_overall_from_digit_csv(digit_df: pd.DataFrame, arch: str, seed, condition: str) -> pd.Series:
    sub = digit_df[(digit_df["arch"] == arch) & (digit_df["seed"] == seed) & (digit_df["condition"] == condition)]
    return sub.groupby("speaker")["accuracy"].mean()


def run_leave_one_out(archs: list[str]) -> dict:
    print(f"\n{'=' * 70}\nLeave-one-out rotation (within the fixed 3, no retraining)\n{'=' * 70}")
    digit_csv = CROSS_SEX_RESULTS_DIR / "results_per_digit.csv"
    if not digit_csv.exists():
        raise FileNotFoundError(f"{digit_csv} missing -- run scripts/cross_sex_generalization.py first")
    digit_df = pd.read_csv(digit_csv, dtype={"speaker": str})

    rotations = [
        {"rotation_id": "full_3", "held_out_speakers": FEMALE_HELD_OUT},
        {"rotation_id": "drop_36", "held_out_speakers": [s for s in FEMALE_HELD_OUT if s != "36"]},
        {"rotation_id": "drop_43", "held_out_speakers": [s for s in FEMALE_HELD_OUT if s != "43"]},
        {"rotation_id": "drop_47", "held_out_speakers": [s for s in FEMALE_HELD_OUT if s != "47"]},
    ]

    results = {"by_arch": {}}
    for arch in archs:
        results["by_arch"][arch] = {"rotations": {}}
        # seed-pool per speaker (across 42/43/44) for both conditions, ONCE
        a_by_seed = {s: per_speaker_overall_from_digit_csv(digit_df, arch, s, "male_only_on_female") for s in SEEDS}
        c_by_seed = {s: per_speaker_overall_from_digit_csv(digit_df, arch, s, "si_baseline_on_female") for s in SEEDS}
        a_pooled_all = pd.concat(a_by_seed.values(), axis=1).mean(axis=1)
        c_pooled_all = pd.concat(c_by_seed.values(), axis=1).mean(axis=1)

        for rot in rotations:
            speakers = sorted(rot["held_out_speakers"])
            a_sub = a_pooled_all.loc[speakers].to_numpy()
            c_sub = c_pooled_all.loc[speakers].to_numpy()
            gap_ci = bootstrap_diff_ci(a_sub, c_sub, seed=42)
            a_ci = bootstrap_speaker_ci(a_sub, seed=42)
            c_ci = bootstrap_speaker_ci(c_sub, seed=42)
            results["by_arch"][arch]["rotations"][rot["rotation_id"]] = {
                "held_out_speakers": speakers, "n_test_speakers": len(speakers),
                "male_only_on_female": a_ci, "si_baseline_on_female": c_ci,
                "gap_ci": gap_ci,
            }
            print(f"  leave-one-out {arch} rotation={rot['rotation_id']} speakers={speakers} "
                  f"gap={gap_ci['point']:+.4f} [{gap_ci['lo']:+.4f},{gap_ci['hi']:+.4f}]")

        gaps = [results["by_arch"][arch]["rotations"][r["rotation_id"]]["gap_ci"]["point"] for r in rotations]
        all_negative = all(g < 0 for g in gaps)
        all_positive = all(g > 0 for g in gaps)
        if all_negative:
            verdict = "DIRECTION_ROBUST_NEGATIVE"
            text = "Gap stays negative across all rotations (full-3 + all 3 leave-one-out) -- direction-robust, not a single-speaker artifact, though still n<=3 and NOT better-powered than the original comparison."
        elif all_positive:
            verdict = "DIRECTION_ROBUST_POSITIVE"
            text = "Gap stays positive across all rotations -- inconsistent with the original negative finding; direction-robust but in the opposite direction, worth a closer look."
        else:
            verdict = "SPEAKER_DRIVEN_FRAGILE"
            text = "Gap sign FLIPS across rotations -- speaker-driven and fragile, not a robust cross-sex effect. Reported as such, not smoothed into a single verdict."
        results["by_arch"][arch]["verdict"] = verdict
        results["by_arch"][arch]["verdict_text"] = text
        print(f"  leave-one-out {arch} VERDICT: {verdict} -- {text}")
    return results


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_same_size_figure(same_size: dict, cross_sex_female_gap: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    archs = list(same_size["by_arch"].keys())
    x = np.arange(len(archs))
    width = 0.35

    male_gaps = [same_size["by_arch"][a]["pooled"]["gap_ci"] for a in archs]
    female_gaps = [cross_sex_female_gap[a] for a in archs]

    ax.bar(x - width / 2, [g["point"] for g in male_gaps], width, label="male->male gap (n=9, well-powered)",
           yerr=[[g["point"] - g["lo"] for g in male_gaps], [g["hi"] - g["point"] for g in male_gaps]],
           capsize=5, color="tab:green")
    ax.bar(x + width / 2, [g["point"] for g in female_gaps], width, label="cross-sex female gap (n=3, held-out-only)",
           yerr=[[g["point"] - g["lo"] for g in female_gaps], [g["hi"] - g["point"] for g in female_gaps]],
           capsize=5, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel("Accuracy gap (subset-trained minus SI-baseline, same held-out speakers)")
    ax.set_title("Same-size male control vs. male-only model on held-out women\n"
                "male gap well-powered (n=9); female gap n=3, directional")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_leave_one_out_figure(leave_one_out: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(leave_one_out["by_arch"]), figsize=(6 * len(leave_one_out["by_arch"]), 5), squeeze=False)
    for i, (arch, res) in enumerate(leave_one_out["by_arch"].items()):
        ax = axes[0][i]
        rot_ids = list(res["rotations"].keys())
        pts = [res["rotations"][r]["gap_ci"]["point"] for r in rot_ids]
        los = [res["rotations"][r]["gap_ci"]["lo"] for r in rot_ids]
        his = [res["rotations"][r]["gap_ci"]["hi"] for r in rot_ids]
        ns = [res["rotations"][r]["n_test_speakers"] for r in rot_ids]
        x = np.arange(len(rot_ids))
        ax.errorbar(x, pts, yerr=[[p - lo for p, lo in zip(pts, los)], [hi - p for p, hi in zip(pts, his)]],
                     fmt="o", capsize=5, color="tab:red", markersize=8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n(n={n})" for r, n in zip(rot_ids, ns)], fontsize=8)
        ax.set_ylabel("Gap (male-only minus SI-baseline)")
        ax.set_title(f"{arch}: {res['verdict']}", fontsize=10)
    fig.suptitle("Leave-one-out rotation within the fixed 3 held-out women -- "
                f"{N_FLAG_FEMALE}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def redraw_from_saved(archs: list[str]) -> None:
    """Redraw the same-size control figure from results/cross_sex_controls/same_size_control.csv."""
    df = pd.read_csv(OUT_DIR / "same_size_control.csv", dtype={"seed": str})
    same_size, female_gap = {"by_arch": {}}, {}
    for arch in archs:
        m = df[(df["arch"] == arch) & (df["test_type"] == "male") & (df["seed"] == "pooled")].iloc[0]
        f = df[(df["arch"] == arch) & (df["test_type"] == "female")].iloc[0]
        same_size["by_arch"][arch] = {"pooled": {"gap_ci": {"point": m["gap"], "lo": m["ci_low"], "hi": m["ci_high"]}}}
        female_gap[arch] = {"point": f["gap"], "lo": f["ci_low"], "hi": f["ci_high"]}
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    make_same_size_figure(same_size, female_gap, FIGS_DIR / "male_only_gap.png")
    print(f"[cross_sex_controls] figure -> {FIGS_DIR / 'male_only_gap.png'}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=ARCHS)
    ap.add_argument("--redraw", action="store_true", help="only redraw the male_only_gap figure from the saved table")
    ap.add_argument("--reuse-run-dirs-json", default=None,
                     help="optional JSON file {'arch_seed': run_dir} of already-completed "
                          "same-size control runs to skip retraining (resumability after an interrupted run)")
    args = ap.parse_args()
    if args.redraw:
        redraw_from_saved(args.archs)
        return

    repo_root = Path(__file__).resolve().parents[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reuse_run_dirs = {}
    if args.reuse_run_dirs_json:
        with open(args.reuse_run_dirs_json) as fh:
            raw = json.load(fh)
        for k, v in raw.items():
            a, s = k.rsplit("_", 1)
            reuse_run_dirs[(a, int(s))] = v

    # cross-sex female gap (held-out-only, n=3) for the same-size comparison
    cs_meta = json.load(open(CROSS_SEX_RESULTS_DIR / "run_meta.json"))
    cs_overall = pd.read_csv(CROSS_SEX_RESULTS_DIR / "results_overall.csv")
    cross_sex_female_gap = {}
    for arch in args.archs:
        row = cs_overall[(cs_overall["arch"] == arch) & (cs_overall["seed"] == "pooled[si_genuinely_held_out_n3]") &
                          (cs_overall["condition"] == "paired_diff_A_minus_C")]
        if len(row) == 0:
            raise RuntimeError(f"could not find held-out-only pooled diff for {arch} in {CROSS_SEX_RESULTS_DIR}")
        r = row.iloc[0]
        cross_sex_female_gap[arch] = {"point": r["accuracy"], "lo": r["ci_lo"], "hi": r["ci_hi"]}

    same_size = run_same_size_control(args.archs, repo_root, reuse_run_dirs=reuse_run_dirs)
    leave_one_out = run_leave_one_out(args.archs)

    make_same_size_figure(same_size, cross_sex_female_gap, FIGS_DIR / "male_only_gap.png")
    make_leave_one_out_figure(leave_one_out, FIGS_DIR / "leave_one_out_gaps.png")

    # tables
    same_size_rows = []
    for arch, res in same_size["by_arch"].items():
        for seed, sres in res["seeds"].items():
            same_size_rows.append({
                "test_type": "male", "arch": arch, "seed": seed,
                "gap": sres["male_subset_on_held_out_males"]["overall"]["point"] -
                       sres["si_baseline_on_held_out_males"]["overall"]["point"],
                "ci_low": np.nan, "ci_high": np.nan,  # per-seed CI not the target estimand; pooled row below carries it
                "n_test_speakers": 9,
            })
        p = res["pooled"]["gap_ci"]
        same_size_rows.append({"test_type": "male", "arch": arch, "seed": "pooled",
                            "gap": p["point"], "ci_low": p["lo"], "ci_high": p["hi"], "n_test_speakers": 9})
        fg = cross_sex_female_gap[arch]
        same_size_rows.append({"test_type": "female", "arch": arch, "seed": "pooled(cross_sex, held_out_only)",
                            "gap": fg["point"], "ci_low": fg["lo"], "ci_high": fg["hi"], "n_test_speakers": 3})
    pd.DataFrame(same_size_rows).to_csv(OUT_DIR / "same_size_control.csv", index=False)

    leave_one_out_rows = []
    for arch, res in leave_one_out["by_arch"].items():
        for rot_id, r in res["rotations"].items():
            g = r["gap_ci"]
            leave_one_out_rows.append({
                "rotation_id": rot_id, "held_out_speakers": ";".join(r["held_out_speakers"]),
                "arch": arch, "seed": "pooled(42,43,44)", "gap": g["point"],
                "ci_low": g["lo"], "ci_high": g["hi"], "n_test_speakers": r["n_test_speakers"],
            })
    pd.DataFrame(leave_one_out_rows).to_csv(OUT_DIR / "leave_one_out.csv", index=False)

    # same-size control verdict
    same_size_verdicts = {}
    for arch in args.archs:
        male_gap = same_size["by_arch"][arch]["pooled"]["gap_ci"]["point"]
        female_gap = cross_sex_female_gap[arch]["point"]
        ratio_note = f"male gap={male_gap:+.4f} vs female gap={female_gap:+.4f}"
        if abs(male_gap) < abs(female_gap) * 0.5:
            v = "SEX_SPECIFIC_SURVIVES"
            t = f"Male->male gap is much smaller than the cross-sex female gap ({ratio_note}) -- the cross-sex gap is sex-specific; the behavioral sex claim SURVIVES its strongest control."
        elif abs(male_gap) >= abs(female_gap) * 0.8:
            v = "OVERTURNED_SAMPLE_SIZE_ARTIFACT"
            t = f"Male->male gap is comparable to (or larger than) the cross-sex female gap ({ratio_note}) -- the cross-sex gap looks like a sample-size/distribution-shift artifact, NOT a sex effect. This OVERTURNS the behavioral sex interpretation; reported loudly, not softened."
        else:
            v = "AMBIGUOUS"
            t = f"Male->male gap is intermediate between 'much smaller' and 'comparable' ({ratio_note}) -- does not cleanly support or overturn the sex-specific interpretation."
        same_size_verdicts[arch] = {"verdict": v, "text": t}
        print(f"\n[cross_sex_controls] same-size control {arch} VERDICT: {v}\n  {t}")

    run_meta = {
        "created": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "git_sha": git_sha(),
        "archs": args.archs,
        "seeds": SEEDS,
        "same_size_control": {
            "pre_registered_expectation": (
                "If the cross-sex gap is sex-specific, male->male generalization "
                "drop will be substantially smaller than the ~5-point cross-sex "
                "gap. If male->male drops comparably, the cross-sex gap is "
                "mostly sample-size/distribution-shift, not sex -- in which "
                "case this control OVERTURNS the sex interpretation and the "
                "representational 'not used' story stands cleaner. This "
                "control CAN falsify the behavioral sex claim; written before "
                "looking."
            ),
            "decision_rule": (
                "male gap << cross-sex gap -> sex-specific, behavioral sex "
                "claim survives its strongest control. male gap ~= cross-sex "
                "gap -> gap is sample-size/shift, not sex; behavioral sex "
                "claim overturned, reported loudly. Operationalized here as: "
                "|male_gap| < 0.5*|female_gap| -> survives; "
                "|male_gap| >= 0.8*|female_gap| -> overturned; else ambiguous."
            ),
            "split_file": "splits/male_subset_control_male_train_male_test.json",
            "held_out_male_speakers": json.load(open(repo_root / "splits" / "male_subset_control_male_train_male_test.json"))["roles"]["test"],
            "confound_note": "None needed: the 9 held-out males are the SI regime's OWN test fold, "
                              "genuinely held out from BOTH the male-subset model and the SI baseline by construction.",
            "verdicts": same_size_verdicts,
        },
        "leave_one_out": {
            "pre_registered_expectation": (
                "If the effect is real, the gap stays negative across (most) "
                "rotations; if it flips sign / swings wildly, it is "
                "speaker-driven and fragile. Written before looking."
            ),
            "decision_rule": (
                "gaps negative across all rotations -> direction-robust, not a "
                "3-speaker artifact. gaps flip sign / swing -> speaker-driven, "
                "fragile, reported as such. NOT pooled into one CI -- reported "
                "per rotation, read qualitatively."
            ),
            "scope_decision": (
                "Leave-one-out WITHIN the existing 3 genuinely-held-out female "
                "speakers (36,43,47) -- NOT a full 5-fold rotation requiring "
                "retrained baselines (that would need 24 new training runs; "
                "declined in favour of this cheaper, no-retrain "
                "design that reuses every checkpoint already on disk). This "
                "means the leave-one-out check tests robustness to WHICH of the fixed 3 women "
                "drives the signal, not generalization to a genuinely "
                "different held-out female group."
            ),
            "n_flag": N_FLAG_FEMALE,
            "verdicts": {arch: {"verdict": leave_one_out["by_arch"][arch]["verdict"],
                                 "text": leave_one_out["by_arch"][arch]["verdict_text"]}
                         for arch in args.archs},
        },
        "out_of_scope_confirmed_not_run": [
            "train-female -> test-male", "balanced 50/50 dataset",
            "perturbation/CRP", "new architectures beyond cnn1d/spec_cnn2d",
            "hyperparameter tuning", "pooling leave-one-out rotations into one CI",
        ],
    }
    with open(OUT_DIR / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, default=str)

    print(f"\n[cross_sex_controls] wrote outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
