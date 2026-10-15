"""Evaluation metrics for the digit classifier.

All group-level metrics are aggregated from PER-SPEAKER means and given CIs via
the speaker-level bootstrap (see bootstrap.py). Utterances are never pooled as if
independent for a group metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from .bootstrap import bootstrap_speaker_ci


# --------------------------------------------------------------------------- #
# basic, utterance-level (overall reporting only)
# --------------------------------------------------------------------------- #
def overall_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_prf(y_true, y_pred, n_classes: int = 10) -> pd.DataFrame:
    labels = list(range(n_classes))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"digit": labels, "precision": p, "recall": r, "f1": f1, "support": support}
    )


def confusion(y_true, y_pred, n_classes: int = 10) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


# --------------------------------------------------------------------------- #
# per-speaker + group (speaker is the unit of analysis)
# --------------------------------------------------------------------------- #
def per_speaker_accuracy(y_true, y_pred, speaker_id, sex=None) -> pd.DataFrame:
    """One row per speaker: speaker_id, accuracy, n, (sex)."""
    df = pd.DataFrame(
        {"speaker_id": np.asarray(speaker_id),
         "correct": (np.asarray(y_true) == np.asarray(y_pred)).astype(float)}
    )
    if sex is not None:
        df["sex"] = np.asarray(sex)
        grp = df.groupby("speaker_id").agg(accuracy=("correct", "mean"),
                                           n=("correct", "size"),
                                           sex=("sex", "first"))
    else:
        grp = df.groupby("speaker_id").agg(accuracy=("correct", "mean"),
                                           n=("correct", "size"))
    return grp.reset_index()


def accuracy_by_sex(per_speaker: pd.DataFrame, n_boot=1000, ci=0.95, seed=0) -> dict:
    """Per-sex accuracy = mean of per-speaker accuracies, with speaker-bootstrap CI."""
    out = {}
    for sex, sub in per_speaker.groupby("sex"):
        out[str(sex)] = bootstrap_speaker_ci(
            sub["accuracy"].to_numpy(), n_boot=n_boot, ci=ci, seed=seed
        )
    return out


def worst_speaker(per_speaker: pd.DataFrame) -> dict:
    row = per_speaker.loc[per_speaker["accuracy"].idxmin()]
    return {
        "speaker_id": int(row["speaker_id"]),
        "accuracy": float(row["accuracy"]),
        "n": int(row["n"]),
        "sex": str(row["sex"]) if "sex" in per_speaker.columns else None,
    }


def worst_group(by_sex: dict) -> dict:
    sex = min(by_sex, key=lambda s: by_sex[s]["point"])
    return {"sex": sex, **by_sex[sex]}


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(y_true, y_prob, n_bins: int = 15):
    """ECE plus per-bin (lo, hi, acc, conf, count) for a reliability diagram."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)

    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt > 0:
            acc = float(correct[m].mean())
            c = float(conf[m].mean())
            ece += (cnt / n) * abs(acc - c)
            bins.append({"lo": lo, "hi": hi, "acc": acc, "conf": c, "count": cnt})
        else:
            bins.append({"lo": lo, "hi": hi, "acc": float("nan"),
                         "conf": float("nan"), "count": 0})
    return float(ece), bins


def confidence_values(y_prob) -> np.ndarray:
    return np.asarray(y_prob).max(axis=1)
