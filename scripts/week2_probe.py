"""Cross-architecture probing at the penultimate layer: trained vs. untrained margins.

Reuses the Day-3 probe harness (src/probing/probe.py, splits.py, labels.py)
verbatim; only the per-architecture feature extraction is new (extract_week2).

Computed (sex first, then identity):
  * linear probe SEX (speaker-disjoint) + IDENTITY (utterance) per architecture,
    for the trained seed-42 SI model AND its untrained control;
  * TRAINED-MINUS-UNTRAINED margin per architecture (does the digit task ADD
    demographic decodability beyond random init?);
  * margin table + headline figures (sex margin bar, identity margin bar).

    python scripts/week2_probe.py [--config CONFIG]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.probing import probe as P
from src.probing.extract_week2 import (
    PENULTIMATE,
    TRAINED,
    UNTRAINED,
    load_features,
    load_labels,
    run_extraction,
)
from src.probing.labels import ATTR_COLUMN
from src.probing.splits import generalizable_split, identity_split, load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"
ARCH_ORDER = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
# leakage gaps (SD-SI, seed 42) from results/training/arch_table.md — for cross-reference
LEAKAGE_GAP = {"cnn1d": 0.0327, "resnet1d": 0.0543, "crnn_gru": 0.0137, "spec_cnn2d": 0.0287}


def _attr_setup(labels, attr, cfg, si_spec):
    """Return (train_pos, test_pos, y_full, y_rand, classes) for an attribute."""
    name, kind = attr["name"], attr["kind"]
    col = ATTR_COLUMN[name]
    if kind == "identity":
        train_pos, test_pos = identity_split(labels, cfg["identity_split"]["test_size"], cfg["seed"])
        valid = labels.index.to_numpy()
    else:
        train_pos, test_pos = generalizable_split(labels, col, si_spec)
        valid = labels.index[labels[col].notna()].to_numpy()
    y_full = np.full(len(labels), -1, dtype=np.int64)
    y_valid, classes = P.encode_labels(labels.loc[valid, col].tolist())
    y_full[valid] = y_valid
    spk_full = labels["speaker_id"].to_numpy()
    y_rand = P.make_random_labels(y_full, spk_full, np.concatenate([train_pos, test_pos]),
                                  kind, seed=cfg["seed"] + 100)
    return train_pos, test_pos, y_full, y_rand, classes


def _probe_one(cfg, arch, model_kind, attr, setup, standardize, probe_kind="linear"):
    train_pos, test_pos, y_full, y_rand, classes = setup
    spk_full = load_labels(cfg)["speaker_id"].to_numpy()
    X = load_features(cfg, arch, model_kind, PENULTIMATE)
    real = P.fit_eval(X, y_full, spk_full, train_pos, test_pos, probe_kind, cfg, standardize)
    rand = P.fit_eval(X, y_rand, spk_full, train_pos, test_pos, probe_kind, cfg, standardize)
    return {
        "arch": arch, "attribute": attr["name"], "model": model_kind, "probe": probe_kind,
        "acc": round(real["acc"], 5),
        "acc_speaker_mean": round(real["acc_speaker_mean"], 5),
        "ci_lo": round(real["ci_lo"], 5), "ci_hi": round(real["ci_hi"], 5),
        "chance": round(real["chance"], 5),
        "random_acc": round(rand["acc"], 5),
        "selectivity": round(real["acc"] - rand["acc"], 5),
        "n_classes": len(classes), "n_train": real["n_train"], "n_test": real["n_test"],
        "n_test_speakers": real["n_test_speakers"],
    }


def _margin_rows(df_attr) -> pd.DataFrame:
    """Collapse trained/untrained rows -> one margin row per (arch, attribute, probe)."""
    out = []
    for (arch, attribute, probe), g in df_attr.groupby(["arch", "attribute", "probe"]):
        tr = g[g["model"] == TRAINED].iloc[0]
        un = g[g["model"] == UNTRAINED].iloc[0]
        out.append({
            "arch": arch, "attribute": attribute, "probe": probe,
            "chance": tr["chance"],
            "trained_acc": tr["acc"], "untrained_acc": un["acc"],
            "margin": round(tr["acc"] - un["acc"], 5),
            "trained_selectivity": tr["selectivity"],
            "untrained_selectivity": un["selectivity"],
            "trained_random_acc": tr["random_acc"],
            "leakage_gap_sd_si": LEAKAGE_GAP.get(arch, float("nan")),
        })
    df = pd.DataFrame(out)
    df["arch"] = pd.Categorical(df["arch"], [a for a in ARCH_ORDER if a in df["arch"].values])
    return df.sort_values(["attribute", "arch"]).reset_index(drop=True)


def _margin_figure(margin_df, attribute, out_path, title):
    sub = margin_df[(margin_df["attribute"] == attribute) & (margin_df["probe"] == "linear")]
    sub = sub.set_index("arch").reindex([a for a in ARCH_ORDER if a in sub["arch"].astype(str).values])
    archs = list(sub.index.astype(str))
    margins = sub["margin"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = ["#c0392b" if m > 0.02 else ("#e67e22" if m > 0.005 else "#7f8c8d") for m in margins]
    bars = ax.bar(archs, margins, color=colors, width=0.62)
    ax.axhline(0, color="black", lw=1.1)
    for b, m in zip(bars, margins):
        ax.text(b.get_x() + b.get_width() / 2, m + (0.004 if m >= 0 else -0.012),
                f"{m:+.3f}", ha="center", va="bottom" if m >= 0 else "top", fontsize=10)
    ax.set_ylabel("trained − untrained linear-probe accuracy\n(penultimate; >0 ⇒ task INDUCES encoding)")
    ax.set_title(title)
    lo, hi = min(0, margins.min()), max(0.02, margins.max())
    pad = 0.02 + 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_margins(cfg) -> dict:
    print("[week2-probe] ensuring per-architecture features are extracted ...")
    run_extraction(cfg)  # idempotent

    labels = load_labels(cfg)
    si_spec = load_si_spec(cfg)
    standardize = cfg["probe"]["standardize"]
    archs = [a for a in ARCH_ORDER if a in cfg["architectures"]]
    figs = resolve(cfg["paths"]["figs_dir"])
    attr_by_name = {a["name"]: a for a in cfg["attributes"]}

    all_rows = []
    headline = {}
    # SEX FIRST (the headline), then IDENTITY.
    for attr_name in ("sex", "speaker_id"):
        attr = attr_by_name[attr_name]
        setup = _attr_setup(labels, attr, cfg, si_spec)
        print(f"\n=== {attr_name} ({attr['kind']}) | classes={setup[4]} | "
              f"train={len(setup[0])} test={len(setup[1])} ===")
        rows = []
        for arch in archs:
            for model_kind in (TRAINED, UNTRAINED):
                r = _probe_one(cfg, arch, model_kind, attr, setup, standardize, "linear")
                rows.append(r)
                print(f"  {arch:11s} {model_kind:9s} linear | acc={r['acc']:.3f} "
                      f"chance={r['chance']:.3f} sel={r['selectivity']:+.3f}")
        all_rows.extend(rows)
        margin_df = _margin_rows(pd.DataFrame(rows))

        fig_name = "week2_induced_margin_sex.png" if attr_name == "sex" else "week2_induced_margin_identity.png"
        title = ("Is SEX encoding INDUCED by the digit task? (trained − untrained)"
                 if attr_name == "sex"
                 else "Is SPEAKER-IDENTITY encoding INDUCED? (trained − untrained)")
        path = _margin_figure(margin_df, attr_name, figs / fig_name, title)
        print(f"[week2-probe] saved figure -> {path}")
        if attr_name == "sex":
            headline["sex_margin_df"] = margin_df
            headline["sex_fig"] = str(path)
            _write_partial_table(cfg, pd.DataFrame(all_rows))  # save sex margins immediately

    # full margin table (sex + identity)
    full = pd.DataFrame(all_rows)
    margin_full = _margin_rows(full)
    run_dir = _persist(cfg, full, margin_full)
    headline["margin_full"] = margin_full
    headline["run_dir"] = str(run_dir)
    return headline


def _table_md(margin_full) -> str:
    lines = ["# Week-2 trained−untrained decodability margins (penultimate, linear, SEED 42 SI)", "",
             "Margin = trained − untrained linear-probe accuracy at the shared `penultimate` "
             "embedding. **Margin > 0 ⇒ the digit task INDUCES the encoding; margin ≈ 0 ⇒ "
             "INHERITED** (present at random init). Sex = speaker-disjoint; identity = "
             "per-speaker-utterance. Leakage gap (SD−SI) from results/training/arch_table.md.", "",
             "| arch | attribute | chance | trained | untrained | margin | trained selectivity | leakage gap (SD−SI) |",
             "|------|-----------|-------:|--------:|----------:|-------:|--------------------:|--------------------:|"]
    for r in margin_full.itertuples():
        lines.append(f"| {r.arch} | {r.attribute} | {r.chance:.3f} | {r.trained_acc:.3f} | "
                     f"{r.untrained_acc:.3f} | {r.margin:+.3f} | {r.trained_selectivity:+.3f} | "
                     f"{r.leakage_gap_sd_si:+.3f} |")
    lines.append("")
    return "\n".join(lines)


def _write_partial_table(cfg, partial_rows):
    out = resolve(f"results/probing_penultimate/margin_table_seed{cfg['seed']}.md")
    _margin = _margin_rows(partial_rows)
    out.write_text(_table_md(_margin), encoding="utf-8")


def _persist(cfg, full_df, margin_full) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sha = git_sha()
    run_dir = resolve(cfg["paths"]["runs_dir"]) / f"week2probe_{sha}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(run_dir / "probe_results.csv", index=False)
    margin_full.to_csv(run_dir / "margins.csv", index=False)
    table_stem = f"results/probing_penultimate/margin_table_seed{cfg['seed']}"
    resolve(f"{table_stem}.md").write_text(_table_md(margin_full), encoding="utf-8")
    margin_full.to_csv(resolve(f"{table_stem}.csv"), index=False)
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"git_sha": sha, "created": stamp, "seed": cfg["seed"],
                   "attributes": ["sex", "speaker_id"]}, fh, indent=2)
    with open(resolve(cfg["paths"]["runs_dir"]) / "latest.txt", "w", encoding="utf-8") as fh:
        fh.write(str(run_dir))
    print(f"[week2-probe] wrote tables -> {run_dir} and {table_stem}.md")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg = load_config(args.config)
    h = run_margins(cfg)
    print("\n================ MARGIN TABLE ================")
    print(h["margin_full"].to_string(index=False))


if __name__ == "__main__":
    main()
