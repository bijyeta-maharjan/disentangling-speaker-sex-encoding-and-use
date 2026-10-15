"""GRU-CRNN — Week-2 architecture #3 (tests the *recurrence* inductive bias).

CPU discipline (Discipline B): a recurrent head must NEVER run over the raw
8192-sample sequence. A small strided conv frontend first downsamples
8192 -> a SHORT sequence (~128 steps) of channel vectors; only then do 1-2 GRU
layers run over that short sequence. GRU (not LSTM) — cheaper on CPU.

frontend: ``n_conv`` Conv-BN-ReLU-MaxPool(pool) blocks. With pool=4 and 3 blocks
the 8192-sample input becomes 8192 / 4^3 = 128 time steps, each a ``conv_channels[-1]``
-dim vector. The sequence is transposed to (B, T, C) and fed to the GRU.

penultimate (Discipline D): the final-layer hidden state (B, hidden). The conv
frontend stages are exposed as intermediates (clear depth sequence) and hookable;
the GRU itself is the recurrent head, not a depth stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import FeatureHookMixin


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d (LRP-friendly, matches baseline)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class GRUCRNN(FeatureHookMixin):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        conv_channels=(32, 64, 128),
        kernel_sizes=(9, 5, 3),
        pool_size: int = 4,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes), "channels/kernels length mismatch"
        n = len(conv_channels)
        self.block_names = [f"conv{i + 1}" for i in range(n)]
        self.intermediate_names = list(self.block_names)
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.bidirectional = bidirectional

        in_ch = in_channels
        for name, out_ch, k in zip(self.block_names, conv_channels, kernel_sizes):
            setattr(self, name, ConvBlock(in_ch, out_ch, k, pool_size))
            in_ch = out_ch

        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        feat_dim = gru_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, n_classes)

        self._hookable = {name: getattr(self, name) for name in self.block_names}
        self._init_feature_hooks()

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats: dict[str, torch.Tensor] = {}
        for name in self.block_names:
            x = getattr(self, name)(x)  # (B, C, T)
            if return_features:
                feats[name] = x
        seq = x.transpose(1, 2)  # (B, T, C) -> recurrence over the SHORT sequence
        out, h_n = self.gru(seq)  # h_n: (num_layers * num_dir, B, H)
        if self.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # both directions, last layer
        else:
            last = h_n[-1]  # final-layer hidden state, (B, H)
        if return_features:
            feats["penultimate"] = last
        logits = self.classifier(self.dropout(last))
        if return_features:
            return logits, feats
        return logits


def build_crnn_gru(cfg: dict) -> GRUCRNN:
    m = cfg["model"]
    return GRUCRNN(
        in_channels=m.get("in_channels", 1),
        n_classes=m.get("n_classes", 10),
        conv_channels=tuple(m.get("conv_channels", (32, 64, 128))),
        kernel_sizes=tuple(m.get("kernel_sizes", (9, 5, 3))),
        pool_size=m.get("pool_size", 4),
        gru_hidden=m.get("gru_hidden", 128),
        gru_layers=m.get("gru_layers", 2),
        bidirectional=m.get("bidirectional", False),
        dropout=m.get("dropout", 0.3),
    )
