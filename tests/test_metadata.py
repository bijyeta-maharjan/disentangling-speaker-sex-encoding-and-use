"""Sanity tests for metadata parsing."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.metadata import load_metadata


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_metadata()


def test_exactly_60_speakers(df: pd.DataFrame) -> None:
    assert len(df) == 60, f"Expected 60 speakers, got {len(df)}"


def test_speaker_ids_unique_and_padded(df: pd.DataFrame) -> None:
    assert df["speaker"].is_unique
    assert all(isinstance(s, str) and len(s) == 2 for s in df["speaker"])


def test_gender_populated(df: pd.DataFrame) -> None:
    assert df["gender"].notna().all()
    assert (df["gender"].str.len() > 0).all()
    assert set(df["gender"].unique()).issubset({"male", "female"})


def test_accent_populated(df: pd.DataFrame) -> None:
    assert df["accent"].notna().all()
    assert (df["accent"].str.len() > 0).all()


def test_age_populated(df: pd.DataFrame) -> None:
    assert df["age"].notna().all()
    assert (df["age"] > 0).all()


def test_known_age_anomaly_speaker_45(df: pd.DataFrame) -> None:
    # Source metadata records speaker 45's age as 1234 — almost certainly a typo
    # in the original dataset. We surface it via a test rather than silently
    # repair it so it can't be forgotten when computing age stats.
    bad = df[df["age"] > 120]["speaker"].tolist()
    assert bad == ["45"], f"Unexpected out-of-range ages: {bad}"
