"""Architecture factory — the single ``build_model(cfg)`` entry point.

Week-2 makes ARCHITECTURE the independent variable, so every consumer (trainer,
evaluator, probing extractor) builds models through here and dispatches on
``cfg["model"]["arch"]``. Defaulting to ``"cnn1d"`` keeps Week-1 configs and
checkpoints (which have no ``arch`` key) working unchanged.

Each builder reads ITS hyperparameters from ``cfg["model"]`` and returns a model
ending in 10-class logits that exposes the shared feature interface (penultimate
+ intermediates) from ``src/models/base.py``.
"""
from __future__ import annotations

import torch.nn as nn

from .cnn1d import build_model as build_cnn1d
from .conv_attn import build_conv_attn
from .crnn_gru import build_crnn_gru
from .resnet1d import build_resnet1d
from .spec_cnn2d import build_spec_cnn2d
from .tcn import build_tcn

# arch key -> (builder, input representation the Dataset must serve)
REGISTRY = {
    "cnn1d": (build_cnn1d, "waveform"),
    "resnet1d": (build_resnet1d, "waveform"),
    "crnn_gru": (build_crnn_gru, "waveform"),
    "spec_cnn2d": (build_spec_cnn2d, "spectrogram"),
    "tcn": (build_tcn, "waveform"),
    "conv_attn": (build_conv_attn, "waveform"),
}

CORE_ARCHS = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
STRETCH_ARCHS = ["tcn", "conv_attn"]


def arch_name(cfg: dict) -> str:
    return cfg["model"].get("arch", "cnn1d")


def input_kind(cfg: dict) -> str:
    """Return 'waveform' or 'spectrogram' for the configured architecture."""
    name = arch_name(cfg)
    # an explicit cfg["model"]["input"] wins, else the registry default
    return cfg["model"].get("input", REGISTRY[name][1])


def build_model(cfg: dict) -> nn.Module:
    name = arch_name(cfg)
    if name not in REGISTRY:
        raise ValueError(f"unknown arch {name!r}; known: {sorted(REGISTRY)}")
    builder, _ = REGISTRY[name]
    return builder(cfg)
