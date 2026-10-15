"""Index every AudioMNIST WAV into a tidy manifest DataFrame.

Filename convention: ``{digit}_{speaker}_{repetition}.wav``
Columns: filepath, speaker, digit, repetition.

Prints any missing speakers or unparseable filenames it sees.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..utils.config import resolve

DEFAULT_RAW_ROOT = resolve("data/raw/AudioMNIST")
FILENAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.wav$")
EXPECTED_DIGITS = set(range(10))
EXPECTED_SPEAKERS = {f"{i:02d}" for i in range(1, 61)}
EXPECTED_TOTAL = 30_000
EXPECTED_PER_SPEAKER = 500


def build_manifest(root: Path | str = DEFAULT_RAW_ROOT) -> pd.DataFrame:
    root = Path(root)
    rows = []
    bad_names: list[str] = []
    for wav in root.rglob("*.wav"):
        m = FILENAME_RE.match(wav.name)
        if not m:
            bad_names.append(str(wav))
            continue
        digit, speaker, repetition = m.groups()
        rows.append(
            {
                "filepath": str(wav),
                "speaker": speaker,
                "digit": int(digit),
                "repetition": int(repetition),
            }
        )
    df = pd.DataFrame(rows).sort_values(["speaker", "digit", "repetition"]).reset_index(drop=True)

    if bad_names:
        print(f"[index] {len(bad_names)} files had unexpected names; first 5: {bad_names[:5]}")

    missing_speakers = EXPECTED_SPEAKERS - set(df["speaker"].unique())
    if missing_speakers:
        print(f"[index] Missing speaker dirs: {sorted(missing_speakers)}")

    per_speaker = df.groupby("speaker").size()
    short = per_speaker[per_speaker != EXPECTED_PER_SPEAKER]
    if not short.empty:
        print(f"[index] Speakers with non-500 utterance count:\n{short}")

    digit_counts = df["digit"].value_counts().sort_index()
    if set(digit_counts.index) != EXPECTED_DIGITS:
        print(f"[index] Unexpected digit set: {sorted(digit_counts.index)}")

    return df


def summary(df: pd.DataFrame) -> dict:
    return {
        "n_files": len(df),
        "n_speakers": df["speaker"].nunique(),
        "n_digits": df["digit"].nunique(),
        "per_speaker_min": int(df.groupby("speaker").size().min()),
        "per_speaker_max": int(df.groupby("speaker").size().max()),
    }


if __name__ == "__main__":
    df = build_manifest()
    print(summary(df))
