"""Skip tests that need AudioMNIST (or its waveform cache) when it is not present.

The dataset is not redistributed. After `python scripts/download_data.py` and
`python scripts/preprocess.py` every test runs; on a fresh clone only the
dataset-free tests (architectures, smoke test) run and the rest are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import resolve  # noqa: E402

RAW_META = resolve("data/raw/AudioMNIST/audioMNIST_meta.txt")
WAVEFORM_CACHE = resolve("data/processed/waveforms_8k_8192.npy")

NEEDS_RAW = {"test_index.py", "test_metadata.py", "test_splits.py"}
NEEDS_CACHE = {"test_si_split_reuse_matches_week1", "test_si_partition_identical_across_architectures"}


def pytest_collection_modifyitems(config, items):
    skip_raw = pytest.mark.skip(reason=f"AudioMNIST not found at {RAW_META.parent} (run scripts/download_data.py)")
    skip_cache = pytest.mark.skip(reason=f"waveform cache not found at {WAVEFORM_CACHE} (run scripts/preprocess.py)")
    for item in items:
        if Path(str(item.fspath)).name in NEEDS_RAW and not RAW_META.exists():
            item.add_marker(skip_raw)
        if item.name in NEEDS_CACHE and not WAVEFORM_CACHE.exists():
            item.add_marker(skip_cache)
