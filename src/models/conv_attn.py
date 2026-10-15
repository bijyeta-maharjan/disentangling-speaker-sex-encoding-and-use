"""Conv + lightweight self-attention — Week-2 STRETCH architecture #6.

Tests the "global self-attention" inductive bias WITHOUT violating the CPU
discipline (Discipline B): self-attention is O(n^2), so it must NEVER touch the
raw 8192-sample sequence. A conv frontend first downsamples 8192 -> a SHORT
sequence (~128 steps); only that short sequence is fed to 1-2 small
transformer-encoder layers. A learned [CLS]-style query is not used; instead we
mean-pool the encoded sequence as the penultimate vector.

Plain ops + nn.TransformerEncoderLayer (GELU inside is standard; the rest of the
network stays LRP-friendly and attention is confined to the short head).

Shared interface (Discipline D): ``forward(x, return_features=True)`` returns
``(logits, {conv1..convK, penultimate})``; conv frontend stages are hookable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import FeatureHookMixin


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class ConvAttn(FeatureHookMixin):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        conv_channels=(32, 64, 128),
        kernel_sizes=(9, 5, 3),
        pool_size: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        max_len: int = 256,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes), "channels/kernels length mismatch"
        n = len(conv_channels)
        self.block_names = [f"conv{i + 1}" for i in range(n)]
        self.intermediate_names = list(self.block_names)
        self.d_model = d_model

        in_ch = in_channels
        for name, out_ch, k in zip(self.block_names, conv_channels, kernel_sizes):
            setattr(self, name, ConvBlock(in_ch, out_ch, k, pool_size))
            in_ch = out_ch

        self.proj = nn.Linear(conv_channels[-1], d_model) if conv_channels[-1] != d_model else nn.Identity()
        # learned positional embedding over the SHORT pooled sequence
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

        self._hookable = {name: getattr(self, name) for name in self.block_names}
        self._init_feature_hooks()

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats: dict[str, torch.Tensor] = {}
        for name in self.block_names:
            x = getattr(self, name)(x)  # (B, C, T)
            if return_features:
                feats[name] = x
        seq = self.proj(x.transpose(1, 2))  # (B, T, d_model)
        t = seq.size(1)
        assert t <= self.pos.size(1), f"sequence length {t} exceeds max_len {self.pos.size(1)}"
        seq = seq + self.pos[:, :t, :]
        enc = self.encoder(seq)  # attention over the SHORT sequence only
        pooled = enc.mean(dim=1)  # mean-pool the encoded tokens
        if return_features:
            feats["penultimate"] = pooled
        logits = self.classifier(self.dropout(pooled))
        if return_features:
            return logits, feats
        return logits


def build_conv_attn(cfg: dict) -> ConvAttn:
    m = cfg["model"]
    return ConvAttn(
        in_channels=m.get("in_channels", 1),
        n_classes=m.get("n_classes", 10),
        conv_channels=tuple(m.get("conv_channels", (32, 64, 128))),
        kernel_sizes=tuple(m.get("kernel_sizes", (9, 5, 3))),
        pool_size=m.get("pool_size", 4),
        d_model=m.get("d_model", 128),
        nhead=m.get("nhead", 4),
        dim_feedforward=m.get("dim_feedforward", 128),
        n_layers=m.get("n_layers", 2),
        dropout=m.get("dropout", 0.3),
        max_len=m.get("max_len", 256),
    )
