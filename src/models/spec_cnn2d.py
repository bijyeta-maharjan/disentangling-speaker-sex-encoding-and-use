"""Spectrogram 2D CNN — Week-2 architecture #4 (tests INPUT REPRESENTATION).

This is the one architecture that deliberately changes the input: instead of the
raw waveform it consumes a (1, n_mels, T) log-mel spectrogram (cached once by
``src/data/spectrogram.py`` from the SAME 8 kHz waveform cache, mel params fixed
in config — a study axis, not a confound). A small 2D CNN over time-frequency
isolates "does giving the model an explicit frequency axis change what
demographic information it encodes and which frequencies it relies on".

Stack: ``Conv2d-BN-ReLU-MaxPool2d`` blocks, then global average pool over the
time-frequency plane -> a channel vector (penultimate), then a dense head -> 10.
Plain ops only, LRP-friendly.

Shared interface (Discipline D): ``forward(x, return_features=True)`` returns
``(logits, {block1..blockK, penultimate})``; blocks are hookable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import FeatureHookMixin


class ConvBlock2D(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class SpecCNN2D(FeatureHookMixin):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        conv_channels=(24, 48, 96, 128),
        kernel_size: int = 3,
        pool_size: int = 2,
        fc_dims=(128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        n = len(conv_channels)
        self.block_names = [f"block{i + 1}" for i in range(n)]
        self.intermediate_names = list(self.block_names)

        in_ch = in_channels
        for name, out_ch in zip(self.block_names, conv_channels):
            setattr(self, name, ConvBlock2D(in_ch, out_ch, kernel_size, pool_size))
            in_ch = out_ch

        self.gap = nn.AdaptiveAvgPool2d(1)

        head: list[nn.Module] = []
        dims = [conv_channels[-1], *fc_dims]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            head += [nn.Linear(d_in, d_out), nn.ReLU(inplace=False), nn.Dropout(dropout)]
        head += [nn.Linear(dims[-1], n_classes)]
        self.classifier = nn.Sequential(*head)

        self._hookable = {name: getattr(self, name) for name in self.block_names}
        self._init_feature_hooks()

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # x: (B, 1, n_mels, T)
        feats: dict[str, torch.Tensor] = {}
        for name in self.block_names:
            x = getattr(self, name)(x)
            if return_features:
                feats[name] = x
        pooled = self.gap(x).flatten(1)
        if return_features:
            feats["penultimate"] = pooled
        logits = self.classifier(pooled)
        if return_features:
            return logits, feats
        return logits


def build_spec_cnn2d(cfg: dict) -> SpecCNN2D:
    m = cfg["model"]
    return SpecCNN2D(
        in_channels=m.get("in_channels", 1),
        n_classes=m.get("n_classes", 10),
        conv_channels=tuple(m.get("conv_channels", (24, 48, 96, 128))),
        kernel_size=m.get("kernel_size", 3),
        pool_size=m.get("pool_size", 2),
        fc_dims=tuple(m.get("fc_dims", (128, 64))),
        dropout=m.get("dropout", 0.3),
    )
