"""Reproducibility helpers.

`set_seed` seeds python/numpy/torch and forces deterministic torch behaviour.
`seed_worker` is the DataLoader `worker_init_fn` so worker RNGs are reproducible.
We are CPU-only, so there are no CUDA calls here — but the cudnn-style
determinism flags are still set via torch.use_deterministic_algorithms.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we touch and request deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        # CPU-only: no cuda seeding needed. Still ask torch for determinism.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch without warn_only
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive each worker's seed from torch's seed."""
    import torch

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
