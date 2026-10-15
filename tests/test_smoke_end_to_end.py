"""CPU smoke test: one tiny experiment end to end, no dataset or GPU needed.

Builds a small synthetic AudioMNIST-shaped waveform cache (12 speakers, 3 female /
9 male, 10 digits x 4 repetitions, 8 kHz, 8,192 samples) plus a speaker-independent
split, then runs the real code path of the paper's core experiment:

  1. train cnn1d for four short epochs (src.train.trainer.train, same protocol code),
  2. evaluate per-speaker accuracy and per-sex accuracy with speaker-bootstrap CIs,
  3. probe sex from the penultimate layer of the trained and an untrained copy
     (speaker-disjoint, standardised logistic regression) and report the margin.

Runs in well under a minute on a laptop CPU:

    pytest tests/test_smoke_end_to_end.py -q -s
    python tests/test_smoke_end_to_end.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import AudioMNISTCacheDataset  # noqa: E402
from src.data.preprocess import load_cache  # noqa: E402
from src.fairness import metrics as M  # noqa: E402
from src.fairness.evaluate import collect_predictions  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.probing.probe import encode_labels, fit_eval  # noqa: E402
from src.splits.make_splits import load_split_indices  # noqa: E402
from src.train.trainer import train  # noqa: E402
from src.utils.config import load_config, resolve_arch_cfg  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

SR, LENGTH = 8000, 8192
SPEAKERS = [f"{i:02d}" for i in range(1, 13)]
FEMALE = {"02", "07", "11"}
ROLES = {"train": ["01", "02", "03", "04", "05", "06", "08"], "val": ["07", "09"], "test": ["10", "11", "12"]}
N_REPS = 4


def synth_clip(digit: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    """Digit-specific 'formant' tones + speaker pitch harmonics under a speech-like envelope."""
    t = np.arange(LENGTH) / SR
    env = np.exp(-((t - 0.5) ** 2) / (2 * 0.15 ** 2))
    formants = sum(np.sin(2 * np.pi * (300 + 70 * digit + 400 * k) * t) / (k + 1) for k in range(3))
    pitch = sum(0.4 * np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 4))
    x = env * (formants + pitch) + 0.05 * rng.standard_normal(LENGTH)
    return (x / np.abs(x).max()).astype(np.float32)


def write_synthetic_cache(processed: Path, cache_name: str) -> None:
    rng = np.random.default_rng(0)
    rows, wav, sex = [], [], []
    for spk in SPEAKERS:
        f0 = rng.uniform(190, 240) if spk in FEMALE else rng.uniform(100, 140)
        for digit in range(10):
            for rep in range(N_REPS):
                wav.append(synth_clip(digit, f0, rng))
                rows.append({"filepath": f"synthetic/{digit}_{spk}_{rep}.wav", "speaker": spk,
                             "digit": digit, "repetition": rep})
                sex.append("female" if spk in FEMALE else "male")
    manifest = pd.DataFrame(rows)
    processed.mkdir(parents=True, exist_ok=True)
    base = processed / cache_name
    np.save(base.with_suffix(".npy"), np.stack(wav))
    spk_to_id = {s: i for i, s in enumerate(SPEAKERS)}
    np.savez(
        str(base) + "_meta.npz",
        digit=manifest["digit"].to_numpy(np.int64),
        speaker_id=manifest["speaker"].map(spk_to_id).to_numpy(np.int64),
        speaker_str=np.array(manifest["speaker"].tolist(), dtype=object),
        sex=np.array(sex, dtype=object),
        id_to_speaker=np.array(SPEAKERS, dtype=object),
        target_sr=SR, target_length=LENGTH, normalize="peak",
    )
    manifest.to_parquet(str(base) + "_manifest.parquet")


def write_split(splits_dir: Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    spec = {"regime": "speaker_independent", "unit": "speaker", "roles": ROLES}
    (splits_dir / "speaker_independent.json").write_text(json.dumps(spec), encoding="utf-8")


@torch.no_grad()
def penultimate(model, cache, idx) -> np.ndarray:
    loader = torch.utils.data.DataLoader(AudioMNISTCacheDataset(cache, idx), batch_size=64, shuffle=False)
    model.eval()
    return np.concatenate([model(x, return_features=True)[1]["penultimate"].numpy() for x, _, _ in loader])


def run_smoke(tmp: Path) -> dict:
    cfg = resolve_arch_cfg(load_config(REPO / "configs" / "week2_architectures.yaml"), arch="cnn1d", seed=42)
    cfg["paths"] = {k: str(tmp / k) for k in ("raw_root", "processed_dir", "splits_dir", "experiments_dir", "figs_dir")}
    cfg["train"].update({"epochs": 4, "batch_size": 16, "early_stopping_patience": 4, "device": "cpu",
                         "num_workers": 0})
    cfg["eval"]["n_bootstrap"] = 200
    write_synthetic_cache(tmp / "processed_dir", cfg["preprocess"]["cache_name"])
    write_split(tmp / "splits_dir")

    # 1. train
    summary = train(cfg, "speaker_independent")
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "best.pt").exists()

    # 2. evaluate on held-out speakers, per sex with speaker-bootstrap CIs
    cache = load_cache(cfg)
    idx = load_split_indices(cfg, "speaker_independent", cache)
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model_state"])
    y_true, y_pred, _, speaker_id, sex = collect_predictions(model, cache, idx["test"], torch.device("cpu"))
    per_spk = M.per_speaker_accuracy(y_true, y_pred, speaker_id, sex)
    by_sex = M.accuracy_by_sex(per_spk, n_boot=cfg["eval"]["n_bootstrap"], ci=cfg["eval"]["ci"], seed=42)

    # 3. sex probe on the penultimate layer, trained vs. untrained (speaker-disjoint)
    probe_cfg = load_config(REPO / "configs" / "week2_probing.yaml")
    probe_cfg["probe"]["bootstrap"]["n_boot"] = 200
    probe_idx = np.concatenate([idx["train"], idx["val"], idx["test"]])
    n_fit = len(idx["train"]) + len(idx["val"])
    train_pos, test_pos = np.arange(n_fit), np.arange(n_fit, len(probe_idx))
    y, _ = encode_labels(np.asarray(cache["sex"])[probe_idx])
    spk = np.asarray(cache["speaker_str"])[probe_idx]
    set_seed(42)
    untrained = build_model(cfg)
    probes = {name: fit_eval(penultimate(m, cache, probe_idx), y, spk, train_pos, test_pos, "linear",
                             probe_cfg, standardize=True)
              for name, m in (("trained", model), ("untrained", untrained))}
    return {"summary": summary, "by_sex": by_sex, "probes": probes}


def test_smoke_end_to_end(tmp_path):
    out = run_smoke(tmp_path)
    assert 0.0 <= out["summary"]["best_val_acc"] <= 1.0
    assert set(out["by_sex"]) == {"female", "male"}
    assert out["by_sex"]["female"]["n_speakers"] == 1 and out["by_sex"]["male"]["n_speakers"] == 2
    for name, r in out["probes"].items():
        assert 0.0 <= r["acc"] <= 1.0, name
        assert r["n_test_speakers"] == 3
    margin = out["probes"]["trained"]["acc"] - out["probes"]["untrained"]["acc"]
    print(f"\n[smoke] val acc {out['summary']['best_val_acc']:.3f} | "
          + " | ".join(f"{s} {b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]" for s, b in out["by_sex"].items())
          + f" | sex probe trained {out['probes']['trained']['acc']:.3f}"
          f" untrained {out['probes']['untrained']['acc']:.3f} margin {margin:+.3f}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_smoke_end_to_end(Path(d))
    print("smoke test passed")
