"""Sanity tests for the audio manifest."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.index import build_manifest


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return build_manifest()


def test_total_count(df: pd.DataFrame) -> None:
    assert len(df) == 30_000, f"Expected 30000 WAVs, got {len(df)}"


def test_500_per_speaker(df: pd.DataFrame) -> None:
    counts = df.groupby("speaker").size()
    assert (counts == 500).all(), f"Per-speaker counts not uniform: {counts[counts != 500].to_dict()}"


def test_ten_digits(df: pd.DataFrame) -> None:
    assert set(df["digit"].unique()) == set(range(10))


def test_60_speakers(df: pd.DataFrame) -> None:
    assert df["speaker"].nunique() == 60
