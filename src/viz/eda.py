"""Reusable EDA plotting functions.

Every function takes prepared DataFrames and returns the matplotlib ``Figure``.
``figures/plot_eda.py`` calls them, so the figures in ``figures/output/`` are
reproducible from a script.

Critical discipline: speaker-level demographic plots aggregate over SPEAKERS
(n=60), not utterances. Function signatures take a speaker-level df where
needed and an utterance-level df where needed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "output"


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------- Utterance-level ----------

def plot_digit_balance(manifest: pd.DataFrame, save: bool = True) -> plt.Figure:
    counts = manifest["digit"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color="steelblue")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Utterance count")
    ax.set_title(f"Utterances per digit (n={len(manifest):,} total)")
    expected = len(manifest) / 10
    ax.axhline(expected, ls="--", color="gray", label=f"expected ({expected:.0f})")
    ax.legend()
    if save:
        _save(fig, "eda_digit_balance.png")
    return fig


# ---------- Speaker-level demographic plots ----------

def plot_sex_distribution_speakers(speakers: pd.DataFrame, save: bool = True) -> plt.Figure:
    counts = speakers["gender"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(int(v)), ha="center")
    ax.set_ylabel("# speakers")
    ax.set_title(f"Sex distribution over SPEAKERS (n={len(speakers)})")
    if save:
        _save(fig, "eda_sex_distribution_speakers.png")
    return fig


def plot_accent_distribution_speakers(speakers: pd.DataFrame, save: bool = True) -> plt.Figure:
    counts = speakers["accent"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if c >= 3 else "#C44E52" for c in counts.values]
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("# speakers")
    ax.set_title(
        f"Accent distribution over SPEAKERS (n={len(speakers)})\n"
        "Red bars = <3 speakers (exploratory only)"
    )
    if save:
        _save(fig, "eda_accent_distribution_speakers.png")
    return fig


def plot_age_distribution_speakers(
    speakers: pd.DataFrame, age_col: str = "age", save: bool = True
) -> plt.Figure:
    # Exclude the speaker-45 anomaly (age=1234 in source metadata).
    valid = speakers[speakers[age_col] < 120]
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = [20, 25, 30, 35, 40, 45, 50, 55, 65]
    ax.hist(valid[age_col], bins=bins, color="seagreen", edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("# speakers")
    n_excl = len(speakers) - len(valid)
    title = f"Age distribution over SPEAKERS (n={len(valid)}"
    if n_excl:
        title += f"; excluded {n_excl} anomalous)"
    else:
        title += ")"
    ax.set_title(title)
    if save:
        _save(fig, "eda_age_distribution_speakers.png")
    return fig


def plot_sex_accent_heatmap_speakers(speakers: pd.DataFrame, save: bool = True) -> plt.Figure:
    ct = pd.crosstab(speakers["accent"], speakers["gender"])
    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar_kws={"label": "# speakers"})
    ax.set_title(f"Sex × Accent over SPEAKERS (n={len(speakers)})")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Accent")
    if save:
        _save(fig, "eda_sex_accent_heatmap_speakers.png")
    return fig


def plot_accent_origin_heatmap_speakers(speakers: pd.DataFrame, save: bool = True) -> plt.Figure:
    df = speakers.copy()
    df["origin_short"] = df["origin"].apply(lambda s: s.split(",")[0].strip() if s else "")
    ct = pd.crosstab(df["accent"], df["origin_short"])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrBr", ax=ax, cbar_kws={"label": "# speakers"})
    ax.set_title(f"Accent × Origin-region over SPEAKERS (n={len(speakers)})")
    ax.set_xlabel("Origin region")
    ax.set_ylabel("Accent")
    if save:
        _save(fig, "eda_accent_origin_heatmap_speakers.png")
    return fig


def plot_sex_agebin_heatmap_speakers(speakers: pd.DataFrame, save: bool = True) -> plt.Figure:
    valid = speakers[speakers["age"] < 120].copy()
    bins = [20, 25, 30, 35, 40, 65]
    labels = ["20-24", "25-29", "30-34", "35-39", "40+"]
    valid["age_bin"] = pd.cut(valid["age"], bins=bins, labels=labels, right=False)
    ct = pd.crosstab(valid["gender"], valid["age_bin"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="PuBu", ax=ax, cbar_kws={"label": "# speakers"})
    ax.set_title(f"Sex × Age-bin over SPEAKERS (n={len(valid)})")
    ax.set_xlabel("Age bin")
    ax.set_ylabel("Sex")
    if save:
        _save(fig, "eda_sex_agebin_heatmap_speakers.png")
    return fig


# ---------- Acoustic distribution shift (utterance-level features, but
# labeled with the speaker-level demographic of their speaker) ----------

def plot_f0_by_sex(feats: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        data=feats.dropna(subset=["f0_median_hz"]),
        x="f0_median_hz",
        hue="gender",
        fill=True,
        common_norm=False,
        ax=ax,
    )
    ax.set_xlabel("Median F0 (Hz) per utterance")
    ax.set_title("F0 distribution by sex (utterances; sex inherited from speaker)")
    if save:
        _save(fig, "eda_f0_by_sex.png")
    return fig


def plot_centroid_by_sex(feats: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=feats, x="gender", y="spectral_centroid_mean", ax=ax)
    ax.set_ylabel("Mean spectral centroid (Hz)")
    ax.set_title("Spectral centroid by sex")
    if save:
        _save(fig, "eda_centroid_by_sex.png")
    return fig


def plot_duration_by_sex(feats: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=feats, x="gender", y="duration_s", ax=ax)
    ax.set_ylabel("Duration (s)")
    ax.set_title("Utterance duration by sex")
    if save:
        _save(fig, "eda_duration_by_sex.png")
    return fig


def plot_centroid_by_accent(feats: pd.DataFrame, save: bool = True) -> plt.Figure:
    # Only accents with >= 3 speakers, to avoid n=1 sample 'distributions'.
    accent_speaker_counts = feats.groupby("accent")["speaker"].nunique()
    keep = accent_speaker_counts[accent_speaker_counts >= 3].index.tolist()
    sub = feats[feats["accent"].isin(keep)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=sub, x="accent", y="spectral_centroid_mean", ax=ax)
    ax.set_ylabel("Mean spectral centroid (Hz)")
    ax.set_title(
        f"Spectral centroid by accent (only accents with ≥3 speakers; "
        f"kept {len(keep)}/{feats['accent'].nunique()})"
    )
    if save:
        _save(fig, "eda_centroid_by_accent.png")
    return fig


# ---------- Recording-session confound check ----------

def plot_recordingdate_by_sex(speakers: pd.DataFrame, save: bool | None = True):
    """Parse recordingdate (YY-MM-DD-hh-mm-ss) into a date and show by sex.

    Returns the Figure, or None if no parseable dates exist.
    """
    df = speakers.copy()
    df["recdate"] = pd.to_datetime(
        df["recordingdate"], format="%y-%m-%d-%H-%M-%S", errors="coerce"
    )
    valid = df.dropna(subset=["recdate"])
    if valid.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    for sex, color in [("male", "#4C72B0"), ("female", "#DD8452")]:
        sub = valid[valid["gender"] == sex]
        ax.scatter(
            sub["recdate"], np.full(len(sub), sex), label=f"{sex} (n={len(sub)})",
            color=color, alpha=0.7, s=80,
        )
    ax.set_yticks(["male", "female"])
    ax.set_xlabel("Recording date")
    ax.set_title("Recording date by sex (each dot = one speaker session)")
    ax.legend()
    fig.autofmt_xdate()
    if save:
        _save(fig, "eda_recordingdate_by_sex.png")
    return fig
