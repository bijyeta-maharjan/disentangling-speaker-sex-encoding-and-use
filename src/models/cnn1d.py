"""Raw-waveform 1D CNN for spoken-digit classification.

Deliberately simple and LRP-friendly (plain Conv/BN/ReLU/MaxPool/Linear, no
exotic activations) — the model is an instrument, not the contribution.

Architecture
------------
5 conv blocks: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d(stride pool_size)
    channels 32 -> 64 -> 128 -> 128 -> 256, kernel 9 then 5/3/3/3
AdaptiveAvgPool1d(1) -> flatten -> Linear 256->128->64->10 (dropout), raw logits.

Day-3 access point (built now, unused today)
-------------------------------------------
- forward(x, return_features=True) returns (logits, {block_name: activation}).
- register_feature_hooks() attaches forward hooks that stash each block's output
  in `.features` for LRP / probing later.
Each conv block is a named submodule (conv1..conv5) so layers are addressable.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d. Kept flat for clean LRP."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=False)  # not inplace -> LRP/hook safe
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class CNN1D(nn.Module):
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
        assert len(conv_channels) == len(kernel_sizes) == 5, "expected 5 conv blocks"
        self.block_names = [f"conv{i + 1}" for i in range(5)]
        # Week-2 shared interface: intermediate feature keys (clear depth sequence).
        self.intermediate_names = list(self.block_names)

        in_ch = in_channels
        for name, out_ch, k in zip(self.block_names, conv_channels, kernel_sizes):
            setattr(self, name, ConvBlock(in_ch, out_ch, k, pool_size))
            in_ch = out_ch

        self.gap = nn.AdaptiveAvgPool1d(1)

        head: list[nn.Module] = []
        dims = [conv_channels[-1], *fc_dims]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            head += [nn.Linear(d_in, d_out), nn.ReLU(inplace=False), nn.Dropout(dropout)]
        head += [nn.Linear(dims[-1], n_classes)]
        self.classifier = nn.Sequential(*head)

        self._hook_handles: list = []
        self.features: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats: dict[str, torch.Tensor] = {}
        for name in self.block_names:
            x = getattr(self, name)(x)
            if return_features:
                feats[name] = x
        pooled = self.gap(x).flatten(1)
        if return_features:
            # "penultimate" is the shared cross-architecture probing point (Week-2
            # Discipline D); "gap" kept as an alias for Day-3 backward-compat.
            feats["penultimate"] = pooled
            feats["gap"] = pooled
        logits = self.classifier(pooled)
        if return_features:
            return logits, feats
        return logits

    # --- Day-3 hook access point (not used in Day-2 training) ---------- #
    def register_feature_hooks(self) -> None:
        """Attach forward hooks that stash each conv block's output in .features."""
        self.remove_feature_hooks()

        def make_hook(name):
            def hook(_module, _inp, out):
                self.features[name] = out.detach()

            return hook

        for name in self.block_names:
            self._hook_handles.append(getattr(self, name).register_forward_hook(make_hook(name)))

    def remove_feature_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self.features.clear()


def build_model(cfg: dict) -> CNN1D:
    m = cfg["model"]
    return CNN1D(
        in_channels=m["in_channels"],
        n_classes=m["n_classes"],
        conv_channels=tuple(m["conv_channels"]),
        kernel_sizes=tuple(m["kernel_sizes"]),
        pool_size=m["pool_size"],
        fc_dims=tuple(m["fc_dims"]),
        dropout=m["dropout"],
    )
