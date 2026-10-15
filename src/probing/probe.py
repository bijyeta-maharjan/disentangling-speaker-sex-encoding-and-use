"""Probe training/evaluation with the rigor controls of discipline E.

Per (attribute x block x model x probe-type) we report:
  - test accuracy (utterance-level headline);
  - the correct CHANCE baseline (majority-class frequency in the test set; for
    speaker-id this equals ~1/n_speakers because the classes are balanced);
  - SELECTIVITY = real-label acc - random-label acc, using a control task whose
    labels are randomized at the SPEAKER level for generalizable attributes
    (per-utterance for speaker-id). A probe that scores high on random labels is
    too powerful to trust; selectivity strips that out.
  - speaker-bootstrap 95% CI on the speaker-mean accuracy (group summary).

Primary probe = LINEAR (logistic regression). A small MLP is run as a clearly
labeled nonlinear comparison.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..fairness.bootstrap import bootstrap_speaker_ci


# --------------------------------------------------------------------------- #
# label helpers
# --------------------------------------------------------------------------- #
def encode_labels(values) -> tuple[np.ndarray, list]:
    """Map arbitrary label values -> 0..K-1 ints. Returns (y, classes)."""
    classes = sorted({v for v in values})
    lut = {c: i for i, c in enumerate(classes)}
    y = np.array([lut[v] for v in values], dtype=np.int64)
    return y, classes


def majority_chance(y_test: np.ndarray) -> float:
    """Accuracy of always guessing the most frequent class in the test set."""
    if len(y_test) == 0:
        return float("nan")
    _, counts = np.unique(y_test, return_counts=True)
    return float(counts.max() / counts.sum())


def make_random_labels(y_full, spk_full, positions, kind: str, seed: int) -> np.ndarray:
    """Control-task labels aligned to `y_full` rows (only `positions` are used).

    generalizable: permute the speaker->label map (each speaker keeps ONE random
        label across its utterances; class balance preserved). A probe can only
        fit this by memorizing speakers, which cannot generalize to held-out
        speakers -> random-label test acc collapses to ~chance.
    identity: permute per-utterance labels.
    """
    rng = np.random.default_rng(seed)
    y_rand = y_full.copy()
    if kind == "identity":
        sub = y_full[positions].copy()
        rng.shuffle(sub)
        y_rand[positions] = sub
        return y_rand
    # generalizable: consistent random label per speaker
    speakers = np.unique(spk_full[positions])
    # draw each speaker's true label, then permute labels across speakers
    true_per_spk = np.array([y_full[positions][spk_full[positions] == s][0] for s in speakers])
    permuted = true_per_spk.copy()
    rng.shuffle(permuted)
    spk_to_rand = dict(zip(speakers.tolist(), permuted.tolist()))
    for s, lab in spk_to_rand.items():
        y_rand[(spk_full == s)] = lab
    return y_rand


# --------------------------------------------------------------------------- #
# probe construction
# --------------------------------------------------------------------------- #
def build_probe(kind: str, cfg: dict, standardize: bool):
    p = cfg["probe"]
    if kind == "linear":
        clf = LogisticRegression(
            C=p["linear"]["C"], max_iter=p["linear"]["max_iter"], n_jobs=-1
        )
    elif kind == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(p["mlp"]["hidden_layer_sizes"]),
            max_iter=p["mlp"]["max_iter"],
            alpha=p["mlp"]["alpha"],
            random_state=cfg["seed"],
        )
    else:
        raise ValueError(f"unknown probe kind: {kind}")
    steps = [StandardScaler(), clf] if standardize else [clf]
    return make_pipeline(*steps)


# --------------------------------------------------------------------------- #
# per-speaker accuracy + bootstrap
# --------------------------------------------------------------------------- #
def per_speaker_accuracy(y_true, y_pred, spk) -> np.ndarray:
    """One accuracy value per speaker present in the test set."""
    out = []
    for s in np.unique(spk):
        m = spk == s
        out.append(float((y_true[m] == y_pred[m]).mean()))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# one probe fit+eval
# --------------------------------------------------------------------------- #
def fit_eval(X, y, spk, train_pos, test_pos, kind_probe, cfg, standardize) -> dict:
    Xtr, Xte = X[train_pos], X[test_pos]
    ytr, yte = y[train_pos], y[test_pos]
    spk_te = spk[test_pos]

    probe = build_probe(kind_probe, cfg, standardize)
    probe.fit(Xtr, ytr)
    pred = probe.predict(Xte)

    acc = float((pred == yte).mean())
    per_spk = per_speaker_accuracy(yte, pred, spk_te)
    ci = bootstrap_speaker_ci(
        per_spk,
        n_boot=cfg["probe"]["bootstrap"]["n_boot"],
        ci=cfg["probe"]["bootstrap"]["ci"],
        seed=cfg["seed"],
    )
    return {
        "acc": acc,
        "acc_speaker_mean": ci["point"],
        "ci_lo": ci["lo"],
        "ci_hi": ci["hi"],
        "chance": majority_chance(yte),
        "n_train": int(len(train_pos)),
        "n_test": int(len(test_pos)),
        "n_classes_test": int(len(np.unique(yte))),
        "n_test_speakers": int(len(np.unique(spk_te))),
    }
