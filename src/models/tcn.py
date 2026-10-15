"""Temporal Convolutional Network (TCN) — Week-2 STRETCH architecture #5.

Dilated causal 1D convolutions give a long receptive field with no recurrence and
fully parallel convs — CPU-friendly (Discipline B). A small strided conv stem
first downsamples 8192 -> a shorter sequence so the dilated stack is cheap; the
exponentially-dilated residual blocks then cover a long context.

Tests the "long-range temporal context via dilation" inductive bias. Plain ops
(Conv1d/BN/ReLU/Linear) — LRP-friendly.

Shared interface (Discipline D): ``forward(x, return_features=True)`` returns
``(logits, {stem, tblock1..tblockK, penultimate})``; stages are hookable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import FeatureHookMixin


class CausalConv1d(nn.Module):
    """Left-padded (causal) dilated Conv1d: output[t] sees only inputs <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))  # pad only on the left
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Two causal dilated convs (BN-ReLU each) + residual 1x1 projection."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.proj = None if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        skip = x if self.proj is None else self.proj(x)
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        return self.relu(h + skip)


class TCN(FeatureHookMixin):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        stem_channels: int = 64,
        stem_kernel: int = 9,
        stem_pool: int = 8,
        channels: int = 64,
        kernel: int = 3,
        dilations=(1, 2, 4, 8, 16, 32),
        fc_dims=(64,),
        dropout: float = 0.3,
    ):
        super().__init__()
        # stem: conv-bn-relu-pool downsample so the dilated stack runs on a short seq
        self.stem_conv = nn.Conv1d(in_channels, stem_channels, stem_kernel, padding=stem_kernel // 2)
        self.stem_bn = nn.BatchNorm1d(stem_channels)
        self.stem_relu = nn.ReLU(inplace=False)
        self.stem_pool = nn.MaxPool1d(stem_pool, stem_pool)

        self.block_names = [f"tblock{i + 1}" for i in range(len(dilations))]
        self.intermediate_names = ["stem", *self.block_names]
        in_ch = stem_channels
        for name, d in zip(self.block_names, dilations):
            setattr(self, name, TemporalBlock(in_ch, channels, kernel, d))
            in_ch = channels

        self.gap = nn.AdaptiveAvgPool1d(1)
        head: list[nn.Module] = []
        dims = [channels, *fc_dims]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            head += [nn.Linear(d_in, d_out), nn.ReLU(inplace=False), nn.Dropout(dropout)]
        head += [nn.Linear(dims[-1], n_classes)]
        self.classifier = nn.Sequential(*head)

        self._hookable = {name: getattr(self, name) for name in self.block_names}
        self._init_feature_hooks()

    def _stem(self, x):
        return self.stem_pool(self.stem_relu(self.stem_bn(self.stem_conv(x))))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats: dict[str, torch.Tensor] = {}
        x = self._stem(x)
        if return_features:
            feats["stem"] = x
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


def build_tcn(cfg: dict) -> TCN:
    m = cfg["model"]
    return TCN(
        in_channels=m.get("in_channels", 1),
        n_classes=m.get("n_classes", 10),
        stem_channels=m.get("stem_channels", 64),
        stem_kernel=m.get("stem_kernel", 9),
        stem_pool=m.get("stem_pool", 8),
        channels=m.get("channels", 64),
        kernel=m.get("kernel", 3),
        dilations=tuple(m.get("dilations", (1, 2, 4, 8, 16, 32))),
        fc_dims=tuple(m.get("fc_dims", (64,))),
        dropout=m.get("dropout", 0.3),
    )
