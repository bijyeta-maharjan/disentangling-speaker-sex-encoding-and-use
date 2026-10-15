"""PyTorch Dataset over the preprocessed cache (waveform OR mel-spectrogram).

Serves ``(features, digit_label, speaker_id)`` for a given list of manifest
indices, reading from the memory-mapped cache so audio is never re-decoded.

Week-2 adds an input-representation flag (all WAVEFORM models share the identical
waveform cache; the spectrogram 2D CNN consumes the row-for-row-aligned mel
cache — a deliberate study axis, not a confound):

  feature="waveform"    -> x = waveform   [1, target_length]   (Conv1d-ready)
  feature="spectrogram" -> x = log-mel    [1, n_mels, T]       (Conv2d-ready)

For the spectrogram case the aligned mel array is passed in as ``cache["spec"]``.
The leading channel dim (1) is added either way.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class AudioMNISTCacheDataset(Dataset):
    def __init__(self, cache: dict, indices, feature: str = "waveform"):
        """cache: dict from preprocess.load_cache (+ optional 'spec'); feature: which input to serve."""
        if feature not in ("waveform", "spectrogram"):
            raise ValueError(f"feature must be 'waveform' or 'spectrogram', got {feature!r}")
        self.feature = feature
        self.wav = cache["wav"]
        self.spec = cache.get("spec")
        if feature == "spectrogram" and self.spec is None:
            raise ValueError("feature='spectrogram' requires cache['spec'] (see data.spectrogram.load_spec_cache)")
        self.digit = np.asarray(cache["digit"])
        self.speaker_id = np.asarray(cache["speaker_id"])
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        # copy out of the read-only memmap into a writable tensor
        src = self.spec if self.feature == "spectrogram" else self.wav
        x = torch.from_numpy(np.array(src[idx], dtype=np.float32)).unsqueeze(0)
        return x, int(self.digit[idx]), int(self.speaker_id[idx])
