"""Figure helpers for evaluation. Each returns the saved path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SEX_COLORS = {"male": "#4C72B0", "female": "#C44E52"}


def plot_confusion(cm: np.ndarray, out: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_accuracy_by_sex(by_sex: dict, out: Path, title: str) -> Path:
    """Bar of per-sex accuracy (mean of per-speaker means) with bootstrap 95% CI."""
    sexes = sorted(by_sex.keys())
    points = [by_sex[s]["point"] for s in sexes]
    lo = [by_sex[s]["point"] - by_sex[s]["lo"] for s in sexes]
    hi = [by_sex[s]["hi"] - by_sex[s]["point"] for s in sexes]
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.arange(len(sexes))
    ax.bar(x, points, color=[_SEX_COLORS.get(s, "#888") for s in sexes],
           yerr=[lo, hi], capsize=8, width=0.6)
    for xi, s in zip(x, sexes):
        b = by_sex[s]
        # label inside the bar, clear of the error cap and the title
        ax.text(xi, 0.50, f"{b['point']:.3f}\n[{b['lo']:.3f}, {b['hi']:.3f}]\nn={b['n_speakers']}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sexes)
    ax.set_ylabel("Accuracy (per-speaker mean)")
    ax.set_ylim(0, 1.15)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_reliability(bins_by_group: dict, ece_by_group: dict, out: Path, title: str) -> Path:
    """Reliability diagram; one line per group (plus 'overall')."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for group, bins in bins_by_group.items():
        xs = [b["conf"] for b in bins if b["count"] > 0]
        ys = [b["acc"] for b in bins if b["count"] > 0]
        color = _SEX_COLORS.get(group, None)
        ax.plot(xs, ys, marker="o", ms=4, label=f"{group} (ECE={ece_by_group[group]:.3f})",
                color=color)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_confidence_dist(conf_by_group: dict, out: Path, title: str) -> Path:
    """Confidence (max-prob) distribution per group."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for group, conf in conf_by_group.items():
        ax.hist(conf, bins=30, range=(0, 1), histtype="step", density=True,
                lw=2, label=f"{group} (med={np.median(conf):.3f})",
                color=_SEX_COLORS.get(group, None))
    ax.set_xlabel("Max softmax probability (confidence)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_leakage_gap(acc_si: float, acc_sd: float, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    labels = ["speaker_independent\n(no leakage)", "speaker_dependent\n(leaky diagnostic)"]
    vals = [acc_si, acc_sd]
    bars = ax.bar(labels, vals, color=["#4C72B0", "#DD8452"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Leakage gap = {acc_sd - acc_si:+.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
