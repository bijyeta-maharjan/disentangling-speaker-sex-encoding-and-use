"""Per-utterance acoustic descriptors used for EDA.

Each utterance gets:
- duration_s: seconds
- rms_mean, rms_std: short-term RMS energy stats
- f0_median_hz, f0_mean_hz, f0_std_hz, voiced_frac: pyin-based pitch
- spectral_centroid_mean, spectral_centroid_std

Audio is loaded mono and resampled to ``ANALYSIS_SR`` (16 kHz) — fine for the
F0/centroid ranges we care about and a lot faster than the native 48 kHz.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

ANALYSIS_SR = 16_000
PYIN_FMIN = 50.0
PYIN_FMAX = 400.0


def compute_features(filepath: str | Path) -> dict:
    y, sr = librosa.load(str(filepath), sr=ANALYSIS_SR, mono=True)
    duration_s = float(len(y) / sr)

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=sr, fill_na=np.nan
    )
    f0_voiced = f0[~np.isnan(f0)]

    return {
        "duration_s": duration_s,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "f0_median_hz": float(np.median(f0_voiced)) if f0_voiced.size else float("nan"),
        "f0_mean_hz": float(np.mean(f0_voiced)) if f0_voiced.size else float("nan"),
        "f0_std_hz": float(np.std(f0_voiced)) if f0_voiced.size else float("nan"),
        "voiced_frac": float(np.mean(voiced_flag)) if voiced_flag.size else float("nan"),
    }
