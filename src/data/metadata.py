"""Parse AudioMNIST speaker metadata into a tidy DataFrame.

One row per speaker (60 rows). Columns are normalized:
- speaker: zero-padded string "01".."60"
- gender: lowercase
- accent: title-case, whitespace-trimmed
- age: int (NaN if missing)
- native_speaker: "yes"/"no" (lowercase)
- origin, recordingdate, recordingroom: as given
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from ..utils.config import resolve

DEFAULT_META_PATH = resolve("data/raw/AudioMNIST/audioMNIST_meta.txt")


def _coerce_age(value) -> float:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return float("nan")


def _normalize_record(speaker: str, raw: dict) -> dict:
    accent = (raw.get("accent") or "").strip()
    return {
        "speaker": speaker,
        "gender": (raw.get("gender") or "").strip().lower(),
        "accent": accent.title() if accent else "",
        "age": _coerce_age(raw.get("age")),
        "native_speaker": (raw.get("native speaker") or "").strip().lower(),
        "origin": (raw.get("origin") or "").strip(),
        "recordingdate": (raw.get("recordingdate") or "").strip(),
        "recordingroom": (raw.get("recordingroom") or "").strip(),
    }


def load_metadata(path: Path | str = DEFAULT_META_PATH) -> pd.DataFrame:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    rows = [_normalize_record(spk, raw) for spk, raw in data.items()]
    df = pd.DataFrame(rows).sort_values("speaker").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_metadata()
    print(df.head())
    print(f"\n{len(df)} speakers loaded")
    print(df["gender"].value_counts())
