"""Cross-architecture probing, secondary analyses: CKA and the layer-wise funnel.

(6) CKA matrix across the 4 architectures' TRAINED penultimate representations
    (linear CKA, subsample 4000 utterances) -> do architectures converge or
    diverge internally? Save figures/output/week2_cka_matrix.png and
    results/probing_penultimate/cka_matrix.csv.
(7) Layer-wise FUNNEL for the conv-family archs (cnn1d, resnet1d, spec_cnn2d)
    whose intermediates are cached: linear-probe decodability of sex + identity
    vs normalized depth, trained vs untrained -> where does demographic info
    peak/decline? Save figures/output/week2_funnel_<arch>.png + a tidy CSV.

Reuses the Day-3 probe harness + the Week-2 feature cache verbatim.

    python scripts/week2_cka_funnel.py [--config CONFIG] [--skip-cka] [--skip-funnel]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.factory import build_model
from src.probing import probe as P
from src.probing.extract_week2 import PENULTIMATE, TRAINED, UNTRAINED, load_features, load_labels
from src.probing.labels import ATTR_COLUMN
from src.probing.splits import generalizable_split, identity_split, load_si_spec
from src.utils.config import load_config, resolve, resolve_arch_cfg

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "week2_probing.yaml"
ARCH_ORDER = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
CONV_FAMILY = ["cnn1d", "resnet1d", "spec_cnn2d"]


# --------------------------------------------------------------------------- #
# (6) linear CKA
# --------------------------------------------------------------------------- #
def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    """Kornblith et al. (2019) linear CKA: invariant to orthogonal transf. + isotropic scale."""
    Xc, Yc = _center(X), _center(Y)
    hsic_xy = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, "fro")
    hsic_yy = np.linalg.norm(Yc.T @ Yc, "fro")
    return float(hsic_xy / (hsic_xx * hsic_yy))


def run_cka(cfg, archs, sub_idx):
    feats = {a: load_features(cfg, a, TRAINED, PENULTIMATE)[sub_idx].astype(np.float64) for a in archs}
    n = len(archs)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = linear_cka(feats[archs[i]], feats[archs[j]])
            M[i, j] = M[j, i] = c
    # figure
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(archs, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(archs)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=11)
    ax.set_title(f"Linear CKA — trained penultimate\n(n={len(sub_idx)} utterances, seed 42 SI)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CKA (1 = identical geometry)")
    fig.tight_layout()
    out = resolve(cfg["paths"]["figs_dir"]) / "week2_cka_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    cka_df = pd.DataFrame(M, index=archs, columns=archs)
    out_csv = resolve("results/probing_penultimate/cka_matrix.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cka_df.to_csv(out_csv)
    print(f"[cka] CKA matrix:\n{cka_df.round(3)}")
    print(f"[cka] saved -> {out} and {out_csv}")
    return M


# --------------------------------------------------------------------------- #
# (7) layer-wise funnel
# --------------------------------------------------------------------------- #
def _attr_setup(labels, name, kind, cfg, si_spec):
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
    return train_pos, test_pos, y_full, classes


def arch_block_names(cfg, arch):
    m = build_model(resolve_arch_cfg(cfg, arch=arch))
    return list(getattr(m, "intermediate_names", []))


def run_funnel(cfg, labels, si_spec):
    spk_full = labels["speaker_id"].to_numpy()
    standardize = cfg["probe"]["standardize"]
    attrs = [("sex", "generalizable"), ("speaker_id", "identity")]
    setups = {name: _attr_setup(labels, name, kind, cfg, si_spec) for name, kind in attrs}

    rows = []
    for arch in CONV_FAMILY:
        blocks = arch_block_names(cfg, arch)
        keys = blocks + [PENULTIMATE]  # ordered shallow -> deep, penultimate last
        depths = [(i + 1) / len(keys) for i in range(len(keys))]  # normalized depth in (0,1]
        for model_kind in (TRAINED, UNTRAINED):
            for key, depth in zip(keys, depths):
                X = load_features(cfg, arch, model_kind, key)
                for name, _kind in attrs:
                    train_pos, test_pos, y_full, _classes = setups[name]
                    r = P.fit_eval(X, y_full, spk_full, train_pos, test_pos, "linear", cfg, standardize)
                    rows.append({"arch": arch, "model": model_kind, "attribute": name,
                                 "block": key, "depth": round(depth, 4), "acc": round(r["acc"], 5),
                                 "chance": round(r["chance"], 5)})
                    print(f"  {arch:11s} {model_kind:9s} {name:10s} {key:12s} d={depth:.2f} acc={r['acc']:.3f}")
        _funnel_figure(cfg, arch, pd.DataFrame([x for x in rows if x["arch"] == arch]))
    df = pd.DataFrame(rows)
    out_csv = resolve("results/probing_penultimate/funnel.csv")
    df.to_csv(out_csv, index=False)
    print(f"[funnel] funnel table -> {out_csv}")
    return df


def _funnel_figure(cfg, arch, df):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    styles = {("sex", TRAINED): ("#c0392b", "-", "o"), ("sex", UNTRAINED): ("#c0392b", "--", "o"),
              ("speaker_id", TRAINED): ("#2c3e50", "-", "s"), ("speaker_id", UNTRAINED): ("#2c3e50", "--", "s")}
    for (name, model), (color, ls, mk) in styles.items():
        sub = df[(df.attribute == name) & (df.model == model)].sort_values("depth")
        if sub.empty:
            continue
        ax.plot(sub["depth"], sub["acc"], ls, color=color, marker=mk, ms=5,
                label=f"{name} ({model})")
    # chance lines
    ax.axhline(0.75, color="#c0392b", lw=0.8, alpha=0.4)
    ax.axhline(0.017, color="#2c3e50", lw=0.8, alpha=0.4)
    ax.set_xlabel("normalized depth (shallow → penultimate=1.0)")
    ax.set_ylabel("linear-probe accuracy")
    ax.set_title(f"Layer-wise decodability funnel — {arch} (seed 42 SI)\n"
                 "red=sex (chance .75), navy=identity (chance .017); solid=trained, dashed=untrained")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    out = resolve(cfg["paths"]["figs_dir"]) / f"week2_funnel_{arch}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[funnel] saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--skip-cka", action="store_true")
    ap.add_argument("--skip-funnel", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    labels = load_labels(cfg)
    si_spec = load_si_spec(cfg)
    archs = [a for a in ARCH_ORDER if a in cfg["architectures"]]

    n = len(labels)
    rng = np.random.default_rng(cfg["seed"])
    sub_idx = np.sort(rng.choice(n, size=min(cfg["cka"]["subsample"], n), replace=False))

    if not args.skip_cka:
        print("[cka] === CKA (trained penultimate) ===")
        run_cka(cfg, archs, sub_idx)
    if not args.skip_funnel:
        print("[funnel] === layer-wise funnel (conv-family) ===")
        run_funnel(cfg, labels, si_spec)


if __name__ == "__main__":
    main()
