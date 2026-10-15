"""Behavioural cross-check on "encoded but not used": train on men only, test on women.

Scientific question. If the digit classifier is trained on MALE speakers only
and tested on FEMALE speakers, does digit accuracy hold? A behavioral angle on
"sex isn't load-bearing," complementing the representational "not used" result
in results/sex_direction/.

PRE-REGISTERED EXPECTATION (written before any run in this script looked at a
result): female digit accuracy under male-only training will stay near the
female accuracy of the normally (sex-mixed) trained model on the SAME female
speakers -- i.e. no significant, seed-stable drop. This would be behavioral
support for "sex is not load-bearing for the digit decision," corroborating
the representational readout-sensitivity capstone
(results/sex_direction/readout_sensitivity_run_meta.json). It is NOT expected to establish population generality (female
n is tiny) and does NOT supersede the representational capstone.

Locked scope:
  - train=male -> test=female ONLY. NOT female->male (<=12 female training
    speakers makes any accuracy drop a sample-size confound, not a sex effect).
  - NOT a balanced 50/50 dataset (12 female speakers cap it).
  - Headline architectures: cnn1d, spec_cnn2d (the two with a validated
    downstream "used" finding). resnet1d/crnn_gru only
    if --archs names them explicitly.
  - Reuses configs/week2_architectures.yaml verbatim: same preprocessing
    (48k->8k, pad/trim 8192, per-utterance peak norm), same train protocol
    (Adam lr=1e-3, batch 128, 8 epochs max, early-stop patience 3), same
    device (cpu -- matches every existing checkpoint in this repo; NOT
    switched to the available MPS backend, to keep this run directly
    comparable to the pre-existing SI baseline checkpoints). No tuning.
  - Seeds 42/43/44, reusing this repo's own convention: the SPLIT
    itself (which speakers are train/val/test) is fixed once
    (splits/cross_sex_male_train_female_test.json, built by
    scripts/make_cross_sex_split.py); only the training seed (weight init +
    data order) varies across 42/43/44.

Method. For each architecture in --archs and each seed in 42/43/44:
  (A) Train a male-only model on cross_sex_male_train_female_test (via
      src.train.trainer.train -- the exact function scripts/train.py calls;
      not reimplemented).
  (B) Evaluate that checkpoint on its own FEMALE test set (12 speakers) and
      its own held-out MALE val set (10 speakers, in-distribution reference).
  (C) Evaluate the pre-existing, matching-seed SPEAKER-INDEPENDENT (normal,
      sex-mixed-trained) checkpoint for the same architecture on the SAME 12
      female speakers -- so the comparison is female-vs-female, isolating the
      train-composition change (see run_meta for the located checkpoint
      paths). NOTE: unlike condition (A)'s female test set, condition (C)'s
      baseline model was itself partly TRAINED on some of these 12 female
      speakers (the SI split only holds out 3 of them) -- this is reported,
      not hidden; it means the (C) baseline may carry a small memorization
      advantage the (A) model cannot have, which if anything works AGAINST
      finding "male-only training generalizes as well as normal training".

Every accuracy is a speaker-bootstrap 95% CI (this repo's existing
bootstrap_speaker_ci, n_boot=1000, matching cfg['eval']) over the relevant
speaker group -- never a single point.

Decision rule (per architecture, seed-pooled by averaging each of the 12
female speakers' per-digit accuracy across the 3 training seeds, then
bootstrapping the PAIRED per-speaker difference acc_A - acc_C over speakers):
  - paired-diff 95% CI includes 0, AND all 3 individual seeds agree in sign
    (or are also non-significant) -> CORROBORATED: female accuracy under
    male-only training is statistically indistinguishable from the same
    female speakers under normal training.
  - paired-diff 95% CI excludes 0 on the negative side (male-only is worse)
    AND all 3 individual seeds show the same direction -> TENSION_FLAG:
    genuine tension with the representational capstone; do not smooth over.
  - anything else (CI excludes 0 but seeds disagree in sign; or the CI is
    wide enough that both "no effect" and "large effect" remain plausible)
    -> UNDERPOWERED_DIRECTIONAL_ONLY: report as an honest non-result.

Output: results/cross_sex_generalization/ (run_meta.json with this
pre-registration + the decision rule + the n flag; per-arch/seed/digit CSVs)
and one PNG per architecture in figures/output/.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_cache
from src.fairness.bootstrap import bootstrap_speaker_ci
from src.models.factory import build_model, input_kind
from src.splits.make_splits import load_split_indices
from src.train.trainer import train as train_model
from src.utils.config import load_config, paths, resolve, resolve_arch_cfg
from src.utils.experiment import git_sha

SI_SPLIT_PATH = Path(__file__).resolve().parents[1] / "splits" / "speaker_independent.json"


def female_speaker_si_roles(female_speakers: list[str]) -> dict[str, str]:
    """For each female test speaker, which SI role (train/val/test) it held in
    the pre-existing normal (sex-mixed) checkpoints. IMPORTANT CONFOUND: the
    SI-baseline model (condition C) was itself TRAINED on any speaker whose
    role is 'train', and used for early-stop selection if 'val' -- only
    role=='test' speakers are genuinely held out from that model. A "female
    accuracy" computed over all 12 speakers therefore mixes true generalization
    with plain memorization for condition C (never for condition A, which
    trained on zero female speakers). This is disentangled below by also
    reporting the decision rule restricted to the role=='test' subset."""
    with open(SI_SPLIT_PATH) as fh:
        si_spec = json.load(fh)
    role_of = {}
    for role, spks in si_spec["roles"].items():
        for s in spks:
            role_of[s] = role
    return {s: role_of.get(s, "?") for s in female_speakers}

REGIME = "cross_sex_male_train_female_test"
HEADLINE_ARCHS = ["cnn1d", "spec_cnn2d"]
SEEDS = [42, 43, 44]
OUT_DIR = resolve("results/cross_sex_generalization")
FIGS_DIR = resolve("figures/output")

N_FLAG = "female test speakers n=12 (of 60 total; <=12 dataset-wide) -- directional, not a population claim"

# Pre-existing normal (speaker_independent, sex-mixed-trained) checkpoints,
# located by inspecting experiments/*/config.json for arch + regime + seed.
# Recorded explicitly here (not re-discovered at run time) so the run-meta is
# reproducible and the provenance is auditable. These are the run directories
# of the original training sweep (see results/training/results.jsonl); after
# retraining, point them at your own experiments/<run_id> directories.
SI_BASELINE_CHECKPOINTS = {
    ("cnn1d", 42): "experiments/speaker_independent_seed42_f2f2b59_20260623-193637",
    ("cnn1d", 43): "experiments/speaker_independent_seed43_f2f2b59_20260623-212141",
    ("cnn1d", 44): "experiments/speaker_independent_seed44_f2f2b59_20260624-004537",
    ("resnet1d", 42): "experiments/speaker_independent_seed42_f2f2b59_20260623-195922",
    ("resnet1d", 43): "experiments/speaker_independent_seed43_f2f2b59_20260623-225321",
    ("resnet1d", 44): "experiments/speaker_independent_seed44_f2f2b59_20260624-011355",
    ("crnn_gru", 42): "experiments/speaker_independent_seed42_f2f2b59_20260623-204342",
    ("crnn_gru", 43): "experiments/speaker_independent_seed43_f2f2b59_20260623-235311",
    ("crnn_gru", 44): "experiments/speaker_independent_seed44_f2f2b59_20260624-020410",
    ("spec_cnn2d", 42): "experiments/speaker_independent_seed42_f2f2b59_20260623-211909",
    ("spec_cnn2d", 43): "experiments/speaker_independent_seed43_f2f2b59_20260624-003955",
    ("spec_cnn2d", 44): "experiments/speaker_independent_seed44_f2f2b59_20260624-023926",
}


def load_cache_for(cfg: dict) -> dict:
    cache = load_cache(cfg)
    if input_kind(cfg) == "spectrogram":
        from src.data.spectrogram import load_spec_cache

        cache["spec"] = load_spec_cache(cfg)["spec"]
    return cache


@torch.no_grad()
def infer(model, cache, idx, device, feature, batch_size=128):
    """Run inference; return y_true, y_pred, speaker_str (parallel arrays)."""
    from src.data.dataset import AudioMNISTCacheDataset

    ds = AudioMNISTCacheDataset(cache, idx, feature=feature)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, ys = [], []
    for x, y, _spk in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    speaker_str = np.asarray(cache["speaker_str"])[np.asarray(idx)]
    return y_true, y_pred, speaker_str


def per_speaker_overall(y_true, y_pred, speaker_str) -> pd.DataFrame:
    df = pd.DataFrame({"speaker": speaker_str, "correct": (y_true == y_pred).astype(float)})
    return df.groupby("speaker")["correct"].mean().reset_index(name="accuracy")


def per_speaker_by_digit(y_true, y_pred, speaker_str) -> pd.DataFrame:
    df = pd.DataFrame({"speaker": speaker_str, "digit": y_true, "correct": (y_true == y_pred).astype(float)})
    return df.groupby(["speaker", "digit"])["correct"].mean().reset_index(name="accuracy")


def load_checkpoint_model(cfg: dict, run_dir: Path, device):
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def bootstrap_diff_ci(a_vals: np.ndarray, b_vals: np.ndarray, n_boot=1000, ci=0.95, seed=0):
    """95% CI on the PAIRED difference a-b, resampling speakers (a_vals/b_vals
    must be aligned to the same speaker order)."""
    diff = a_vals - b_vals
    return bootstrap_speaker_ci(diff, n_boot=n_boot, ci=ci, seed=seed)


def run_arch(arch: str, root_cfg_path: str, repo_root: Path) -> dict:
    print(f"\n{'=' * 70}\n[cross_sex] architecture: {arch}\n{'=' * 70}")
    device = torch.device("cpu")
    results = {"arch": arch, "seeds": {}}

    for seed in SEEDS:
        print(f"\n--- {arch} seed {seed} ---")
        cfg = resolve_arch_cfg(load_config(root_cfg_path), arch=arch, seed=seed)
        feature = input_kind(cfg)

        # (A) train male-only model
        summary = train_model(cfg, REGIME)
        run_dir = Path(summary["run_dir"])

        # shared cache + split indices for this arch's feature representation
        cache = load_cache_for(cfg)
        idx = load_split_indices(cfg, REGIME, cache)

        model_a, ckpt_a = load_checkpoint_model(cfg, run_dir, device)

        # (B) male-only model on its own female test + male val (reference)
        yt_f, yp_f, spk_f = infer(model_a, cache, idx["test"], device, feature, cfg["train"]["batch_size"])
        yt_m, yp_m, spk_m = infer(model_a, cache, idx["val"], device, feature, cfg["train"]["batch_size"])

        # (C) matching-seed SI baseline model on the SAME female test speakers
        si_dir = resolve(SI_BASELINE_CHECKPOINTS[(arch, seed)])
        model_c, ckpt_c = load_checkpoint_model(cfg, si_dir, device)
        yt_fc, yp_fc, spk_fc = infer(model_c, cache, idx["test"], device, feature, cfg["train"]["batch_size"])
        assert np.array_equal(spk_f, spk_fc), "female test speaker order mismatch between A and C"

        per_spk_a = per_speaker_overall(yt_f, yp_f, spk_f)
        per_spk_b = per_speaker_overall(yt_m, yp_m, spk_m)
        per_spk_c = per_speaker_overall(yt_fc, yp_fc, spk_fc)

        digit_a = per_speaker_by_digit(yt_f, yp_f, spk_f)
        digit_b = per_speaker_by_digit(yt_m, yp_m, spk_m)
        digit_c = per_speaker_by_digit(yt_fc, yp_fc, spk_fc)

        results["seeds"][seed] = {
            "train_run_dir": str(run_dir),
            "train_summary": summary,
            "si_baseline_run_dir": str(si_dir),
            "si_baseline_best_epoch": ckpt_c.get("epoch"),
            "female_test_speakers": sorted(np.unique(spk_f).tolist()),
            "male_val_speakers": sorted(np.unique(spk_m).tolist()),
            "condA_male_only_on_female": {
                "overall": bootstrap_speaker_ci(per_spk_a["accuracy"].to_numpy(), seed=seed),
                "per_speaker": per_spk_a,
                "per_digit": digit_a,
            },
            "condB_male_only_on_male_val": {
                "overall": bootstrap_speaker_ci(per_spk_b["accuracy"].to_numpy(), seed=seed),
                "per_speaker": per_spk_b,
                "per_digit": digit_b,
            },
            "condC_si_baseline_on_female": {
                "overall": bootstrap_speaker_ci(per_spk_c["accuracy"].to_numpy(), seed=seed),
                "per_speaker": per_spk_c,
                "per_digit": digit_c,
            },
        }
        a_pt = results["seeds"][seed]["condA_male_only_on_female"]["overall"]["point"]
        c_pt = results["seeds"][seed]["condC_si_baseline_on_female"]["overall"]["point"]
        b_pt = results["seeds"][seed]["condB_male_only_on_male_val"]["overall"]["point"]
        print(f"[cross_sex] {arch} seed{seed}: female|male-only={a_pt:.4f}  "
              f"female|SI-baseline={c_pt:.4f}  male-val|male-only(ref)={b_pt:.4f}")

    return results


def seed_pool_and_decide(arch_results: dict, speaker_subset: list[str] | None = None,
                          label: str = "all_12") -> dict:
    """Average each female speaker's per-speaker accuracy across the 3 seeds
    for conditions A and C, then bootstrap the paired difference over speakers
    (optionally restricted to `speaker_subset`). Also checks per-seed sign
    consistency within that subset."""
    seeds = sorted(arch_results["seeds"].keys())
    a_frames = [arch_results["seeds"][s]["condA_male_only_on_female"]["per_speaker"].set_index("speaker") for s in seeds]
    c_frames = [arch_results["seeds"][s]["condC_si_baseline_on_female"]["per_speaker"].set_index("speaker") for s in seeds]

    a_pooled = pd.concat(a_frames, axis=1).mean(axis=1).sort_index()
    c_pooled = pd.concat(c_frames, axis=1).mean(axis=1).sort_index()
    assert list(a_pooled.index) == list(c_pooled.index)

    if speaker_subset is not None:
        a_pooled = a_pooled.loc[sorted(speaker_subset)]
        c_pooled = c_pooled.loc[sorted(speaker_subset)]

    pooled_a_ci = bootstrap_speaker_ci(a_pooled.to_numpy(), seed=42)
    pooled_c_ci = bootstrap_speaker_ci(c_pooled.to_numpy(), seed=42)
    diff_ci = bootstrap_diff_ci(a_pooled.to_numpy(), c_pooled.to_numpy(), seed=42)

    per_seed_signs = []
    for s in seeds:
        a_frame_s = arch_results["seeds"][s]["condA_male_only_on_female"]["per_speaker"].set_index("speaker")
        c_frame_s = arch_results["seeds"][s]["condC_si_baseline_on_female"]["per_speaker"].set_index("speaker")
        if speaker_subset is not None:
            a_frame_s = a_frame_s.loc[sorted(speaker_subset)]
            c_frame_s = c_frame_s.loc[sorted(speaker_subset)]
        per_seed_signs.append(float(a_frame_s["accuracy"].mean() - c_frame_s["accuracy"].mean()))

    ci_excludes_zero = diff_ci["lo"] > 0 or diff_ci["hi"] < 0
    seeds_agree_negative = all(d < 0 for d in per_seed_signs)
    seeds_agree_nonneg = all(d >= -1e-9 for d in per_seed_signs)

    if ci_excludes_zero and diff_ci["hi"] < 0 and seeds_agree_negative:
        verdict = "TENSION_FLAG"
        verdict_text = (
            "Female accuracy drops under male-only training, and the drop is "
            "seed-stable (all 3 seeds negative). This is a genuine tension "
            "with the representational 'not used' capstone -- flagged, not "
            "smoothed over."
        )
    elif not ci_excludes_zero and (seeds_agree_nonneg or all(d <= 1e-9 for d in per_seed_signs)):
        verdict = "CORROBORATED"
        verdict_text = (
            "Female accuracy under male-only training is statistically "
            "indistinguishable from the same female speakers under normal "
            "(sex-mixed) training -- behavioral corroboration of 'sex not "
            "load-bearing for the digit decision', consistent with the "
            "representational readout-sensitivity capstone."
        )
    else:
        verdict = "UNDERPOWERED_DIRECTIONAL_ONLY"
        verdict_text = (
            "The paired-difference CI and/or per-seed signs are inconsistent "
            "or inconclusive at this sample size. Reported as an honest "
            "non-result, not a forced verdict: underpowered at n=12 female "
            "speakers, directional only."
        )

    return {
        "label": label,
        "speakers_used": sorted(speaker_subset) if speaker_subset is not None else sorted(a_pooled.index.tolist()),
        "pooled_female_acc_male_only_trained": pooled_a_ci,
        "pooled_female_acc_si_baseline": pooled_c_ci,
        "paired_diff_ci_A_minus_C": diff_ci,
        "per_seed_diff_A_minus_C": dict(zip(seeds, per_seed_signs)),
        "verdict": verdict,
        "verdict_text": verdict_text,
    }


def make_headline_figure(arch: str, arch_results: dict, pooled: dict, out_path: Path) -> None:
    seeds = sorted(arch_results["seeds"].keys())
    fig, ax = plt.subplots(figsize=(7, 5))

    conditions = [
        ("Female | male-only trained", "condA_male_only_on_female", "tab:blue"),
        ("Female | normal (SI) trained", "condC_si_baseline_on_female", "tab:orange"),
    ]
    x_positions = [0, 1]
    for (label, key, color), x in zip(conditions, x_positions):
        pts = [arch_results["seeds"][s][key]["overall"]["point"] for s in seeds]
        los = [arch_results["seeds"][s][key]["overall"]["lo"] for s in seeds]
        his = [arch_results["seeds"][s][key]["overall"]["hi"] for s in seeds]
        for i, s in enumerate(seeds):
            ax.errorbar(x + (i - 1) * 0.06, pts[i], yerr=[[pts[i] - los[i]], [his[i] - pts[i]]],
                        fmt="o", color=color, alpha=0.5, capsize=3, markersize=5)
        pooled_key = "pooled_female_acc_male_only_trained" if key == "condA_male_only_on_female" else "pooled_female_acc_si_baseline"
        p = pooled[pooled_key]
        ax.errorbar(x, p["point"], yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]],
                    fmt="D", color=color, markersize=10, capsize=6,
                    label=f"{label} (seed-pooled)")

    # male-val reference line (mean across seeds, condition B)
    male_ref_pts = [arch_results["seeds"][s]["condB_male_only_on_male_val"]["overall"]["point"] for s in seeds]
    ax.axhline(np.mean(male_ref_pts), color="gray", linestyle="--", linewidth=1.5,
              label=f"Male-only model, own held-out MALE val (ref, mean of {len(seeds)} seeds)")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([c[0] for c in conditions])
    ax.set_ylabel("Digit accuracy")
    ax.set_title(f"Cross-sex generalization -- {arch}\n"
                f"verdict: {pooled['verdict']}", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.02)
    fig.text(0.5, -0.02, N_FLAG, ha="center", fontsize=8, style="italic", wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dataframe_dump(all_results: dict, out_dir: Path, si_roles: dict) -> None:
    overall_rows, digit_rows = [], []
    for arch, res in all_results["by_arch"].items():
        for seed, sres in res["seeds"].items():
            for cond_key, cond_label in [
                ("condA_male_only_on_female", "male_only_on_female"),
                ("condB_male_only_on_male_val", "male_only_on_male_val_ref"),
                ("condC_si_baseline_on_female", "si_baseline_on_female"),
            ]:
                c = sres[cond_key]
                o = c["overall"]
                overall_rows.append({
                    "arch": arch, "seed": seed, "condition": cond_label,
                    "accuracy": o["point"], "ci_lo": o["lo"], "ci_hi": o["hi"],
                    "n_speakers": o["n_speakers"], "n_flag": N_FLAG if "female" in cond_label else "",
                })
                for _, r in c["per_digit"].iterrows():
                    digit_rows.append({
                        "arch": arch, "seed": seed, "condition": cond_label,
                        "digit": int(r["digit"]), "speaker": r["speaker"], "accuracy": r["accuracy"],
                        "speaker_si_role": si_roles.get(r["speaker"], "") if "female" in cond_label else "",
                    })
        for pooled_key, pooled in [("pooled", res["pooled"]), ("pooled_held_out_only", res["pooled_held_out_only"])]:
            n_lbl = pooled["label"]
            overall_rows.append({
                "arch": arch, "seed": f"pooled[{n_lbl}]", "condition": "male_only_on_female",
                "accuracy": pooled["pooled_female_acc_male_only_trained"]["point"],
                "ci_lo": pooled["pooled_female_acc_male_only_trained"]["lo"],
                "ci_hi": pooled["pooled_female_acc_male_only_trained"]["hi"],
                "n_speakers": pooled["pooled_female_acc_male_only_trained"]["n_speakers"],
                "n_flag": N_FLAG,
            })
            overall_rows.append({
                "arch": arch, "seed": f"pooled[{n_lbl}]", "condition": "si_baseline_on_female",
                "accuracy": pooled["pooled_female_acc_si_baseline"]["point"],
                "ci_lo": pooled["pooled_female_acc_si_baseline"]["lo"],
                "ci_hi": pooled["pooled_female_acc_si_baseline"]["hi"],
                "n_speakers": pooled["pooled_female_acc_si_baseline"]["n_speakers"],
                "n_flag": N_FLAG,
            })
            overall_rows.append({
                "arch": arch, "seed": f"pooled[{n_lbl}]", "condition": "paired_diff_A_minus_C",
                "accuracy": pooled["paired_diff_ci_A_minus_C"]["point"],
                "ci_lo": pooled["paired_diff_ci_A_minus_C"]["lo"],
                "ci_hi": pooled["paired_diff_ci_A_minus_C"]["hi"],
                "n_speakers": pooled["paired_diff_ci_A_minus_C"]["n_speakers"],
                "n_flag": N_FLAG,
                "verdict": pooled["verdict"],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall_rows).to_csv(out_dir / "results_overall.csv", index=False)
    pd.DataFrame(digit_rows).to_csv(out_dir / "results_per_digit.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=HEADLINE_ARCHS,
                    help="architectures to run (default: headline cnn1d spec_cnn2d)")
    ap.add_argument("--config", default="configs/week2_architectures.yaml")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    split_path = repo_root / "splits" / f"{REGIME}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} missing -- run scripts/make_cross_sex_split.py first")
    with open(split_path) as fh:
        split_spec = json.load(fh)

    female_speakers = split_spec["roles"]["test"]
    si_roles = female_speaker_si_roles(female_speakers)
    genuinely_held_out = sorted([s for s, r in si_roles.items() if r == "test"])
    si_seen = sorted([s for s, r in si_roles.items() if r in ("train", "val")])
    print(f"\n[cross_sex] CONFOUND CHECK: of the {len(female_speakers)} female test speakers, "
          f"{len(si_seen)} were seen by the SI-baseline model during its own training/val "
          f"({si_roles}), and only {len(genuinely_held_out)} ({genuinely_held_out}) are "
          f"genuinely held out from it. The full-12 comparison mixes true generalization "
          f"with memorization for condition C; a second, held-out-only (n={len(genuinely_held_out)}) "
          f"comparison is computed below to disentangle this.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {"by_arch": {}}

    for arch in args.archs:
        res = run_arch(arch, args.config, repo_root)
        pooled_all12 = seed_pool_and_decide(res, speaker_subset=None, label="all_12_female")
        pooled_heldout = seed_pool_and_decide(res, speaker_subset=genuinely_held_out, label="si_genuinely_held_out_n3")
        res["pooled"] = pooled_all12
        res["pooled_held_out_only"] = pooled_heldout
        all_results["by_arch"][arch] = res
        make_headline_figure(arch, res, pooled_all12, FIGS_DIR / f"male_only_{arch}.png")
        print(f"\n[cross_sex] {arch} VERDICT (all 12 female, mixes memorization for C): {pooled_all12['verdict']}\n  {pooled_all12['verdict_text']}")
        print(f"  female|male-only  (seed-pooled, n=12): {pooled_all12['pooled_female_acc_male_only_trained']}")
        print(f"  female|SI-baseline(seed-pooled, n=12): {pooled_all12['pooled_female_acc_si_baseline']}")
        print(f"  paired diff A-C   (seed-pooled, n=12): {pooled_all12['paired_diff_ci_A_minus_C']}")
        print(f"\n[cross_sex] {arch} VERDICT (SI-genuinely-held-out only, n={len(genuinely_held_out)}, no memorization confound): {pooled_heldout['verdict']}\n  {pooled_heldout['verdict_text']}")
        print(f"  female|male-only  (seed-pooled, n={len(genuinely_held_out)}): {pooled_heldout['pooled_female_acc_male_only_trained']}")
        print(f"  female|SI-baseline(seed-pooled, n={len(genuinely_held_out)}): {pooled_heldout['pooled_female_acc_si_baseline']}")
        print(f"  paired diff A-C   (seed-pooled, n={len(genuinely_held_out)}): {pooled_heldout['paired_diff_ci_A_minus_C']}")

    dataframe_dump(all_results, OUT_DIR, si_roles)

    run_meta = {
        "created": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "git_sha": git_sha(),
        "regime": REGIME,
        "split_file": str(split_path.relative_to(repo_root)),
        "split_spec_roles_n": {k: len(v) for k, v in split_spec["roles"].items()},
        "female_test_speakers": split_spec["roles"]["test"],
        "male_train_speakers": split_spec["roles"]["train"],
        "male_val_speakers": split_spec["roles"]["val"],
        "archs_run": args.archs,
        "seeds": SEEDS,
        "device": "cpu",
        "device_note": "Locked to cpu, matching every existing checkpoint in this repo "
                        "(MPS is available on this machine but was not used, to keep this "
                        "run comparable to the pre-existing SI baseline checkpoints and to "
                        "avoid an unstated protocol change).",
        "pre_registered_expectation": (
            "Female digit accuracy under male-only training will stay near the "
            "female accuracy of the normally (sex-mixed) trained model on the "
            "SAME female speakers -- no significant, seed-stable drop. Behavioral "
            "corroboration of 'sex not load-bearing', consistent with the "
            "representational readout-sensitivity capstone "
            "(results/sex_direction/readout_sensitivity_run_meta.json). Written before any run in this script looked at a result."
        ),
        "decision_rule": (
            "Per architecture: average each of the 12 female speakers' accuracy "
            "across seeds 42/43/44 for both male-only-trained and SI-baseline "
            "models; bootstrap the PAIRED per-speaker difference over speakers. "
            "CI includes 0 (& seeds agree/non-significant) -> CORROBORATED. "
            "CI excludes 0 on the negative side & all 3 seeds negative -> "
            "TENSION_FLAG (flagged loudly, not smoothed over). Anything else "
            "(CI excludes 0 but seeds disagree in sign, or genuinely wide) -> "
            "UNDERPOWERED_DIRECTIONAL_ONLY (honest non-result)."
        ),
        "excluded_by_design": split_spec["excluded_from_experiment"],
        "n_flag": N_FLAG,
        "si_baseline_checkpoints_used": {f"{a}_{s}": SI_BASELINE_CHECKPOINTS[(a, s)]
                                          for a in args.archs for s in SEEDS},
        "confound_note": (
            "The SI-baseline model (condition C) was itself TRAINED on 7 of "
            "the 12 female test speakers and used 2 more for early-stop "
            "selection (val) -- only 3 (the SI regime's own test fold) are "
            "genuinely held out from it. See female_speaker_si_roles below. "
            "A full-12 comparison therefore mixes true cross-sex "
            "generalization with plain speaker memorization for condition C "
            "only (condition A trained on zero female speakers, so it has no "
            "such advantage) -- this can by itself produce an apparent "
            "female-accuracy 'drop' under male-only training that has "
            "nothing to do with sex. Two verdicts are reported per "
            "architecture: 'verdict_all_12' (confounded, n=12) and "
            "'verdict_held_out_only' (clean but underpowered, n=3, the SI "
            "regime's own 3-speaker test fold) -- the held-out-only verdict "
            "is the one that actually speaks to cross-sex generalization; "
            "the all-12 verdict is reported for transparency, not as the "
            "headline."
        ),
        "female_speaker_si_roles": si_roles,
        "female_speakers_si_genuinely_held_out": genuinely_held_out,
        "female_speakers_si_seen_train_or_val": si_seen,
        "verdicts_all_12_confounded": {arch: all_results["by_arch"][arch]["pooled"]["verdict"] for arch in args.archs},
        "verdicts_held_out_only_n3": {arch: all_results["by_arch"][arch]["pooled_held_out_only"]["verdict"] for arch in args.archs},
    }
    with open(OUT_DIR / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, default=str)

    print(f"\n[cross_sex] wrote outputs to {OUT_DIR}")
    print("verdicts_all_12_confounded:", json.dumps(run_meta["verdicts_all_12_confounded"], indent=2))
    print("verdicts_held_out_only_n3:", json.dumps(run_meta["verdicts_held_out_only_n3"], indent=2))


if __name__ == "__main__":
    main()
