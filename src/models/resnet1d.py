"""Residual 1D CNN (ResNet-1D) — Week-2 architecture #2.

Independent-variable framing: this is the *same* 5-stage conv stack as the
Week-1 baseline (same channel progression 32->64->128->128->256, same per-stage
temporal downsampling by ``pool_size``), with the ONLY change being a skip
connection added around each stage. It isolates "depth / information
preservation" as the inductive-bias axis: if Week-1 findings hold here, residual
shortcuts (which ease gradient flow and preserve low-level signal) don't change
the demographic-encoding story.

Each stage: ``out = ReLU( pool(BN(conv(x))) + shortcut(x) )`` where the shortcut
is a 1x1 conv projection (channel match) followed by the SAME pooling, so the
residual sum is shape-aligned. Plain ops only — LRP-friendly.

Shared interface (Discipline D): ``forward(x, return_features=True)`` returns
``(logits, {res1..res5, penultimate})``; intermediates are also hookable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import FeatureHookMixin


class ResidualConvBlock(nn.Module):
    """Conv-BN (main) + 1x1-proj shortcut, summed after a shared MaxPool, then ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool)
        # Identity shortcut when channels already match, else a 1x1 projection.
        self.proj = None if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        main = self.pool(self.bn(self.conv(x)))
        skip = x if self.proj is None else self.proj(x)
        skip = self.pool(skip)
        return self.relu(main + skip)


class ResNet1D(FeatureHookMixin):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        conv_channels=(32, 64, 128, 128, 256),
        kernel_sizes=(9, 5, 3, 3, 3),
        pool_size: int = 4,
        fc_dims=(128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes), "channels/kernels length mismatch"
        n = len(conv_channels)
        self.block_names = [f"res{i + 1}" for i in range(n)]
        self.intermediate_names = list(self.block_names)

        in_ch = in_channels
        for name, out_ch, k in zip(self.block_names, conv_channels, kernel_sizes):
            setattr(self, name, ResidualConvBlock(in_ch, out_ch, k, pool_size))
            in_ch = out_ch

        self.gap = nn.AdaptiveAvgPool1d(1)

        head: list[nn.Module] = []
        dims = [conv_channels[-1], *fc_dims]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            head += [nn.Linear(d_in, d_out), nn.ReLU(inplace=False), nn.Dropout(dropout)]
        head += [nn.Linear(dims[-1], n_classes)]
        self.classifier = nn.Sequential(*head)

        self._hookable = {name: getattr(self, name) for name in self.block_names}
        self._init_feature_hooks()

    def forward(self, x: torch.Tensor, return_features: bool = False):
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


def build_resnet1d(cfg: dict) -> ResNet1D:
    m = cfg["model"]
    return ResNet1D(
        in_channels=m.get("in_channels", 1),
        n_classes=m.get("n_classes", 10),
        conv_channels=tuple(m.get("conv_channels", (32, 64, 128, 128, 256))),
        kernel_sizes=tuple(m.get("kernel_sizes", (9, 5, 3, 3, 3))),
        pool_size=m.get("pool_size", 4),
        fc_dims=tuple(m.get("fc_dims", (128, 64))),
        dropout=m.get("dropout", 0.3),
    )
