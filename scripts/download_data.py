"""Download AudioMNIST into data/raw/.

Idempotent: re-running is a no-op when the dataset already looks complete.
Uses `git clone --depth 1` of soerenab/AudioMNIST. The repo bundles audio under
its own `data/` subdir (one folder per speaker 01..60) plus
`audioMNIST_meta.txt`. We materialize that as
`data/raw/AudioMNIST/{01..60,audioMNIST_meta.txt}`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/soerenab/AudioMNIST.git"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import resolve  # noqa: E402

RAW_DIR = resolve("data/raw")
TARGET_DIR = RAW_DIR / "AudioMNIST"
EXPECTED_SPEAKER_DIRS = 60
EXPECTED_METADATA = "audioMNIST_meta.txt"


def looks_complete(target: Path) -> bool:
    if not target.is_dir():
        return False
    if not (target / EXPECTED_METADATA).is_file():
        return False
    speaker_dirs = [d for d in target.iterdir() if d.is_dir() and d.name.isdigit()]
    return len(speaker_dirs) >= EXPECTED_SPEAKER_DIRS


def clone(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / "_AudioMNIST_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"Cloning {REPO_URL} -> {tmp} (this may take a few minutes)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(tmp)],
        check=True,
    )
    inner_data = tmp / "data"
    if not inner_data.is_dir():
        raise RuntimeError(f"Expected {inner_data} after clone; layout changed?")
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(inner_data), str(target))
    shutil.rmtree(tmp)


def verify(target: Path) -> None:
    speaker_dirs = sorted(d for d in target.iterdir() if d.is_dir() and d.name.isdigit())
    n_speakers = len(speaker_dirs)
    n_wav = sum(1 for _ in target.rglob("*.wav"))
    print(f"Speakers found: {n_speakers}")
    print(f"WAV files found: {n_wav}")
    if not (target / EXPECTED_METADATA).is_file():
        raise RuntimeError(f"Missing metadata file at {target / EXPECTED_METADATA}")
    if n_speakers != EXPECTED_SPEAKER_DIRS:
        raise RuntimeError(f"Expected {EXPECTED_SPEAKER_DIRS} speaker dirs, got {n_speakers}")


def main() -> int:
    if looks_complete(TARGET_DIR):
        print(f"AudioMNIST already present at {TARGET_DIR} — skipping download.")
        verify(TARGET_DIR)
        return 0
    clone(TARGET_DIR)
    verify(TARGET_DIR)
    print(f"Done. Data at: {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
