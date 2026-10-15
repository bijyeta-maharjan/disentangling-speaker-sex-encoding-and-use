"""Lightweight, offline experiment tracking.

A run is a folder under experiments/ tagged with the split regime, git SHA and
seed. We log the config snapshot, environment, per-epoch metrics (CSV) and final
metrics (JSON), and save the best checkpoint there. No W&B / network calls.
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

from .config import REPO_ROOT


def git_sha(short: bool = True) -> str:
    """Current commit SHA, or 'nogit' if unavailable."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "nogit"


class RunLogger:
    """Creates experiments/<regime>_<seed>_<sha>_<timestamp>/ and writes to it."""

    def __init__(self, experiments_dir: Path, regime: str, seed: int, config: dict):
        self.regime = regime
        self.seed = seed
        self.sha = git_sha()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{regime}_seed{seed}_{self.sha}_{stamp}"
        self.dir = Path(experiments_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.epochs_csv = self.dir / "epochs.csv"
        self._epoch_fields: list[str] | None = None

        self.write_json("config.json", config)
        self.write_json(
            "run_meta.json",
            {
                "run_id": self.run_id,
                "regime": regime,
                "seed": seed,
                "git_sha": self.sha,
                "git_sha_full": git_sha(short=False),
                "created": stamp,
            },
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.dir / "best.pt"

    def write_json(self, name: str, obj) -> Path:
        path = self.dir / name
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
        return path

    def log_epoch(self, row: dict) -> None:
        """Append one epoch row to epochs.csv (header written on first call)."""
        if self._epoch_fields is None:
            self._epoch_fields = list(row.keys())
            with open(self.epochs_csv, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self._epoch_fields).writeheader()
        with open(self.epochs_csv, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self._epoch_fields).writerow(row)
