"""Training loop: CrossEntropy + Adam, early stopping on VAL accuracy.

CPU-only. Reads waveforms from the memmap cache (never re-decodes audio). Each
run is logged to its own experiments/ folder, tagged with split regime, git SHA
and seed, with per-epoch CSV + a best checkpoint.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import AudioMNISTCacheDataset
from ..data.preprocess import load_cache
from ..models.factory import build_model, input_kind
from ..splits.make_splits import load_split_indices
from ..utils.config import paths
from ..utils.experiment import RunLogger
from ..utils.seeding import seed_worker, set_seed


def _loader(cache, indices, batch_size, shuffle, num_workers, seed, feature="waveform"):
    ds = AudioMNISTCacheDataset(cache, indices, feature=feature)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None):
    """Return (accuracy, mean_loss) over a loader."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y, _spk in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    acc = correct / total
    return acc, (loss_sum / total if criterion is not None else float("nan"))


def train(cfg: dict, regime: str) -> dict:
    seed = cfg["seed"]
    set_seed(seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device(cfg["train"]["device"])

    cache = load_cache(cfg)
    # Spectrogram models consume a row-for-row-aligned mel cache instead of the
    # waveform; waveform models all share the identical waveform cache.
    feature = input_kind(cfg)
    if feature == "spectrogram":
        from ..data.spectrogram import load_spec_cache

        cache["spec"] = load_spec_cache(cfg)["spec"]
    idx = load_split_indices(cfg, regime, cache)
    print(
        f"[train] arch={cfg['model'].get('arch', 'cnn1d')} input={feature} regime={regime} | "
        f"train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])} utterances"
    )

    t = cfg["train"]
    train_loader = _loader(cache, idx["train"], t["batch_size"], True, t["num_workers"], seed, feature)
    val_loader = _loader(cache, idx["val"], t["batch_size"], False, t["num_workers"], seed, feature)

    model = build_model(cfg).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    logger = RunLogger(paths(cfg).experiments_dir, regime, seed, cfg)
    logger.write_json("split_sizes.json", {k: int(len(v)) for k, v in idx.items()})
    print(f"[train] run dir: {logger.dir}")
    print(f"[train] parameters: {n_params:,}")

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    last_epoch = 0

    for epoch in range(1, t["epochs"] + 1):
        last_epoch = epoch
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for x, y, _spk in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        train_acc, train_loss = correct / total, loss_sum / total
        val_acc, val_loss = evaluate_loader(model, val_loader, device, criterion)

        improved = val_acc > best_val_acc
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "best": improved,
        }
        logger.log_epoch(row)
        print(
            f"[train] epoch {epoch:2d}/{t['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            + ("  <- best" if improved else "")
        )

        if improved:
            best_val_acc, best_epoch, epochs_no_improve = val_acc, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "regime": regime,
                    "seed": seed,
                    "git_sha": logger.sha,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                logger.checkpoint_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= t["early_stopping_patience"]:
                print(f"[train] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    summary = {
        "arch": cfg["model"].get("arch", "cnn1d"),
        "input": feature,
        "regime": regime,
        "seed": seed,
        "git_sha": logger.sha,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "epochs_to_stop": last_epoch,  # epochs actually run before early-stop / max (a Week-2 covariate)
        "max_epochs": t["epochs"],
        "run_dir": str(logger.dir),
        "checkpoint": str(logger.checkpoint_path),
    }
    logger.write_json("train_summary.json", summary)
    print(
        f"[train] done. best val_acc={best_val_acc:.4f} @ epoch {best_epoch} "
        f"(ran {last_epoch} epochs, {n_params:,} params)"
    )
    return summary
