"""Embedding visualizations — SUPPORTING / illustrative only.

PCA (with variance explained), t-SNE, and UMAP (if installed) of a block's GAP
features, colored by digit / sex / speaker-id. These ILLUSTRATE what the probes
prove; t-SNE/UMAP geometry is non-quantitative and tunable, so every such panel
carries a caveat in its caption.

Optional quantitative companion: KMeans on the embedding + mutual information
between clusters and {digit, sex, speaker} — a number to anchor the picture.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="white", context="talk")

CAVEAT = (
    "Caveat: t-SNE/UMAP geometry is non-quantitative and tunable (perplexity, "
    "seed) — it illustrates the probe results, it is not evidence on its own."
)


def _subsample(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def compute_embeddings(X: np.ndarray, cfg: dict, seed: int):
    """Standardize -> PCA(report var) -> {pca2, tsne2, (umap2)}. Returns dict."""
    v = cfg["viz"]
    sel = _subsample(len(X), v["tsne"]["subsample"], seed)
    Xs = StandardScaler().fit_transform(X[sel])

    n_comp = min(v["pca_components"], Xs.shape[1], Xs.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed)
    Xp = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_

    out = {
        "sel": sel,
        "pca2": Xp[:, :2],
        "pca_evr": evr,
        "pca_evr_2d": float(evr[:2].sum()),
        "pca_evr_cum": float(evr.sum()),
    }
    out["tsne2"] = TSNE(
        n_components=2, perplexity=v["tsne"]["perplexity"],
        max_iter=v["tsne"]["max_iter"], init="pca", random_state=seed,
    ).fit_transform(Xp)

    try:  # UMAP is optional; use it only if installed
        import umap

        out["umap2"] = umap.UMAP(random_state=seed).fit_transform(Xp)
    except Exception:
        out["umap2"] = None
    return out


def _scatter(ax, xy, color_vals, title, categorical=True):
    if categorical:
        cats = pd.Categorical(color_vals)
        cmap = sns.color_palette("tab10", len(cats.categories)) if len(cats.categories) <= 10 \
            else sns.color_palette("husl", len(cats.categories))
        for i, c in enumerate(cats.categories):
            m = cats.codes == i
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=cmap[i], label=str(c), alpha=0.6)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=6, c=color_vals, cmap="viridis", alpha=0.6)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])


def plot_embedding_grid(emb: dict, labels_sub: pd.DataFrame, block: str,
                        color_by, save_path: Path | None = None) -> plt.Figure:
    """Rows = embedding method (PCA/t-SNE[/UMAP]); cols = coloring (digit/sex/speaker)."""
    methods = [("PCA", emb["pca2"]), ("t-SNE", emb["tsne2"])]
    if emb.get("umap2") is not None:
        methods.append(("UMAP", emb["umap2"]))

    color_specs = {
        "digit": (labels_sub["digit"].to_numpy(), True),
        "sex": (labels_sub["sex"].to_numpy(), True),
        "speaker_id": (labels_sub["speaker_id"].to_numpy(), True),
    }
    cols = [c for c in color_by if c in color_specs]
    nrow, ncol = len(methods), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.0 * nrow), squeeze=False)

    for r, (mname, xy) in enumerate(methods):
        for c, cname in enumerate(cols):
            vals, cat = color_specs[cname]
            title = f"{mname} — colored by {cname}"
            if mname == "PCA" and r == 0:
                title += f"  (2D var={emb['pca_evr_2d']:.1%})"
            # legend only for sex (small) and digit; speaker has 60 -> skip legend
            _scatter(axes[r][c], xy, vals, title, categorical=cat)
            if cname in ("sex", "digit"):
                axes[r][c].legend(markerscale=2, fontsize=8, loc="best", framealpha=0.8)

    fig.suptitle(
        f"Embeddings of trained SI {block} GAP features (n={len(labels_sub)} subsampled)\n{CAVEAT}",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def cluster_mi(X: np.ndarray, labels_sub: pd.DataFrame, sel: np.ndarray,
               n_clusters: int, seed: int) -> dict:
    """KMeans on (subsampled) features; MI between clusters and each attribute."""
    Xs = StandardScaler().fit_transform(X[sel])
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(Xs)
    cl = km.labels_
    out = {}
    for attr in ("digit", "sex", "speaker_id"):
        truth = labels_sub[attr].to_numpy()
        out[attr] = {
            "nmi": round(float(normalized_mutual_info_score(truth, cl)), 4),
            "ami": round(float(adjusted_mutual_info_score(truth, cl)), 4),
        }
    out["n_clusters"] = n_clusters
    return out
