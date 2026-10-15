"""Build the single, aligned label table that every probe reads from.

CRITICAL DISCIPLINE (A): the label table is built from the SAME row order as the
waveform cache (which is row-for-row aligned to the Day-1 manifest). We never
sort or shuffle here. Activations are extracted with shuffle=False over
``indices = arange(N)``, so cache row i == activation row i == label row i.

Columns (one row per utterance, in cache order):
    index        manifest/cache row index (0..N-1)
    speaker_id   contiguous speaker id 0..59 (from the cache)
    speaker      zero-padded speaker string "01".."60"
    digit        spoken digit 0..9 (the model's TASK label)
    sex          "male"/"female"
    accent_raw   accent string from metadata
    accent_bin   "german" / "non_german"
    age_raw      reported age (float; NaN for the corrupt speaker-45 value)
    age_bin      "young"/"older" by median threshold; NaN where age is dropped
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.metadata import load_metadata


def build_label_table(cache: dict, cfg: dict) -> pd.DataFrame:
    """Construct the aligned per-utterance label table in cache row order."""
    n = len(cache["digit"])
    speaker_str = np.asarray(cache["speaker_str"]).astype(str)
    speaker_id = np.asarray(cache["speaker_id"]).astype(np.int64)
    digit = np.asarray(cache["digit"]).astype(np.int64)
    sex = np.asarray(cache["sex"]).astype(str)

    # speaker-level metadata -> map onto utterance rows by speaker string
    meta = load_metadata().set_index("speaker")
    accent_raw = np.array([str(meta.loc[s, "accent"]) for s in speaker_str], dtype=object)
    accent_bin = np.where(
        np.char.lower(accent_raw.astype(str)) == "german", "german", "non_german"
    )

    # age: corrupt / dropped speakers -> NaN so the age probe excludes them
    drop = set(str(s) for s in cfg.get("age", {}).get("drop_speakers", []))
    thr = cfg.get("age", {}).get("median_threshold", 27)
    age_raw = np.array(
        [float(meta.loc[s, "age"]) if s not in drop else np.nan for s in speaker_str],
        dtype=float,
    )
    age_bin = np.where(np.isnan(age_raw), None, np.where(age_raw <= thr, "young", "older"))

    df = pd.DataFrame(
        {
            "index": np.arange(n, dtype=np.int64),
            "speaker_id": speaker_id,
            "speaker": speaker_str,
            "digit": digit,
            "sex": sex,
            "accent_raw": accent_raw,
            "accent_bin": accent_bin,
            "age_raw": age_raw,
            "age_bin": age_bin,
        }
    )
    # Sanity: cache sex must agree with metadata sex (catches a desync at source).
    meta_sex = np.array([str(meta.loc[s, "gender"]) for s in speaker_str], dtype=object)
    mismatch = int((meta_sex != sex).sum())
    assert mismatch == 0, f"cache sex disagrees with metadata for {mismatch} rows"
    return df


# Mapping from attribute name -> the label-table column it reads.
ATTR_COLUMN = {
    "sex": "sex",
    "accent": "accent_bin",
    "age_bin": "age_bin",
    "speaker_id": "speaker_id",
}
