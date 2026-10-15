"""Evaluate a trained run on its test set and write metrics + figures.

Produces, per run:
  - overall accuracy, per-class P/R/F1, confusion matrix figure
  - per-speaker accuracy (CSV) and accuracy-by-sex from per-speaker means with
    speaker-level bootstrap 95% CIs; worst-speaker and worst-group
  - calibration: ECE + reliability diagram and confidence distribution, overall
    and per sex group
Everything is aggregated at the SPEAKER level for group metrics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.dataset import AudioMNISTCacheDataset
from ..data.preprocess import load_cache
from ..models.factory import build_model, input_kind
from ..splits.make_splits import load_split_indices
from ..utils.config import paths
from . import metrics as M
from . import plots as P


def latest_run_dir(experiments_dir: Path, regime: str) -> Path:
    prefix = f"{regime}_seed"
    cands = sorted(p for p in Path(experiments_dir).iterdir()
                   if p.is_dir() and p.name.startswith(prefix))
    if not cands:
        raise FileNotFoundError(f"no run dir for regime {regime!r} under {experiments_dir}")
    return cands[-1]


@torch.no_grad()
def collect_predictions(model, cache, test_idx, device, batch_size=128, feature="waveform"):
    """Run inference on the test set; return aligned arrays."""
    ds = AudioMNISTCacheDataset(cache, test_idx, feature=feature)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    probs, preds, ys, spk = [], [], [], []
    for x, y, s in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(1).cpu().numpy())
        ys.append(y.numpy())
        spk.append(s.numpy())
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    speaker_id = np.concatenate(spk)
    sex = np.asarray(cache["sex"])[np.asarray(test_idx)]
    return y_true, y_pred, y_prob, speaker_id, sex


def evaluate_run(cfg: dict, regime: str, run_dir: Path | None = None) -> dict:
    p = paths(cfg)
    run_dir = Path(run_dir) if run_dir else latest_run_dir(p.experiments_dir, regime)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device(cfg["train"]["device"])

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    cache = load_cache(cfg)
    feature = input_kind(cfg)
    if feature == "spectrogram":
        from ..data.spectrogram import load_spec_cache

        cache["spec"] = load_spec_cache(cfg)["spec"]
    test_idx = load_split_indices(cfg, regime, cache)["test"]
    y_true, y_pred, y_prob, speaker_id, sex = collect_predictions(
        model, cache, test_idx, device, cfg["train"]["batch_size"], feature
    )

    ev = cfg["eval"]
    n_classes = cfg["model"]["n_classes"]
    figs = p.figs_dir
    figs.mkdir(parents=True, exist_ok=True)

    # ---- overall + per class ----
    overall = M.overall_accuracy(y_true, y_pred)
    prf = M.per_class_prf(y_true, y_pred, n_classes)
    cm = M.confusion(y_true, y_pred, n_classes)
    P.plot_confusion(cm, figs / f"confusion_{regime}.png", f"Confusion — {regime}")

    # ---- per-speaker + by sex (speaker is the unit) ----
    per_spk = M.per_speaker_accuracy(y_true, y_pred, speaker_id, sex)
    by_sex = M.accuracy_by_sex(per_spk, n_boot=ev["n_bootstrap"], ci=ev["ci"], seed=cfg["seed"])
    worst_spk = M.worst_speaker(per_spk)
    worst_grp = M.worst_group(by_sex)
    P.plot_accuracy_by_sex(by_sex, figs / f"acc_by_sex_{regime}.png",
                           f"Accuracy by sex (speaker-level) — {regime}")

    # ---- calibration overall + per group ----
    ece_overall, bins_overall = M.expected_calibration_error(y_true, y_prob, ev["ece_bins"])
    bins_by_group = {"overall": bins_overall}
    ece_by_group = {"overall": ece_overall}
    conf_by_group = {}
    for sx in sorted(set(sex)):
        m = sex == sx
        e, b = M.expected_calibration_error(y_true[m], y_prob[m], ev["ece_bins"])
        ece_by_group[sx] = e
        bins_by_group[sx] = b
        conf_by_group[sx] = M.confidence_values(y_prob[m])
    P.plot_reliability(bins_by_group, ece_by_group, figs / f"reliability_{regime}.png",
                       f"Reliability — {regime}")
    P.plot_confidence_dist(conf_by_group, figs / f"confidence_dist_{regime}.png",
                           f"Confidence distribution by sex — {regime}")

    # ---- persist ----
    per_spk.to_csv(run_dir / "per_speaker_accuracy.csv", index=False)
    prf.to_csv(run_dir / "per_class_prf.csv", index=False)
    np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    summary = {
        "regime": regime,
        "run_dir": str(run_dir),
        "git_sha": ckpt.get("git_sha"),
        "seed": cfg["seed"],
        "best_epoch": ckpt.get("epoch"),
        "n_test_utterances": int(len(y_true)),
        "n_test_speakers": int(per_spk["speaker_id"].nunique()),
        "overall_accuracy": overall,
        "macro_f1": float(prf["f1"].mean()),
        "accuracy_by_sex": by_sex,
        "worst_speaker": worst_spk,
        "worst_group": worst_grp,
        "ece_overall": ece_overall,
        "ece_by_group": {k: float(v) for k, v in ece_by_group.items()},
        "per_class_f1": {int(r.digit): float(r.f1) for r in prf.itertuples()},
    }
    import json

    with open(run_dir / "eval_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary
