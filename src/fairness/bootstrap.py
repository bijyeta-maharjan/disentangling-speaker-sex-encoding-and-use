"""Speaker-level bootstrap for group metrics.

SPEAKER is the unit of analysis. A group metric (e.g. accuracy for women) is the
MEAN OF PER-SPEAKER values, and its uncertainty comes from resampling the *set of
test speakers* with replacement (~1000x) and recomputing — never from pooling
utterances as if independent. Use this for every group metric.
"""
from __future__ import annotations

import numpy as np


def bootstrap_speaker_ci(
    per_speaker_values,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    statistic=np.mean,
):
    """CI for an aggregate of per-speaker values via resampling SPEAKERS.

    per_speaker_values : 1-D array, one value per speaker in the group.
    Returns dict(point, lo, hi, se, n_speakers, n_boot).
    """
    vals = np.asarray(per_speaker_values, dtype=float)
    n = len(vals)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n_speakers": 0, "n_boot": n_boot}

    rng = np.random.default_rng(seed)
    point = float(statistic(vals))
    if n == 1:  # no spread to resample
        return {"point": point, "lo": point, "hi": point, "se": 0.0,
                "n_speakers": 1, "n_boot": n_boot}

    # resample speaker indices with replacement
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = statistic(vals[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(np.std(boots, ddof=1)),
        "n_speakers": int(n),
        "n_boot": int(n_boot),
    }
