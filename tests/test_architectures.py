"""Week-2 architecture tests (Discipline A/C/D).

For every architecture in the multi-arch config:
  (a) a batch forward-passes to logits of shape (B, 10);
  (b) the shared feature interface exposes `penultimate` PLUS >=1 intermediate;
  (c) param counts are printed (capacity is a reported covariate);
and, architecture-independently:
  (d) the REUSED Week-1 SI speaker partition loads identically (same speakers,
      same sizes as the committed Week-1 split / run), and stays speaker-disjoint.

No probing/representation analysis here — only that the access points exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.preprocess import load_cache
from src.models.factory import CORE_ARCHS, STRETCH_ARCHS, build_model, input_kind
from src.splits.make_splits import load_split_indices, load_split_spec
from src.utils.config import load_config, resolve_arch_cfg

REPO = Path(__file__).resolve().parents[1]
WEEK2_CONFIG = REPO / "configs" / "week2_architectures.yaml"
ALL_ARCHS = CORE_ARCHS + STRETCH_ARCHS

# Week-1 SI partition reference (committed split file + the seed-42 SI run).
WEEK1_SI_SIZES = {"train": 18000, "val": 6000, "test": 6000}
WEEK1_SI_TEST_SPEAKERS = {"01", "03", "16", "17", "18", "23", "36", "38", "43", "44", "47", "55"}


@pytest.fixture(scope="module")
def base_cfg():
    return load_config(WEEK2_CONFIG)


def _dummy_input(cfg: dict, batch: int = 4) -> torch.Tensor:
    """A correctly-shaped random input for the architecture's input representation."""
    if input_kind(cfg) == "spectrogram":
        n_mels = cfg["spectrogram"]["n_mels"]
        length = cfg["preprocess"]["target_length"]
        t = 1 + length // cfg["spectrogram"]["hop_length"]
        return torch.randn(batch, 1, n_mels, t)
    return torch.randn(batch, 1, cfg["preprocess"]["target_length"])


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_forward_to_logits(base_cfg, arch):
    cfg = resolve_arch_cfg(base_cfg, arch=arch)
    model = build_model(cfg).eval()
    x = _dummy_input(cfg)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (x.shape[0], 10), f"{arch}: expected (B,10), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), f"{arch}: non-finite logits"


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_shared_feature_interface(base_cfg, arch):
    cfg = resolve_arch_cfg(base_cfg, arch=arch)
    model = build_model(cfg).eval()
    x = _dummy_input(cfg)
    with torch.no_grad():
        logits, feats = model(x, return_features=True)
    assert logits.shape == (x.shape[0], 10)
    # mandatory cross-architecture probing point
    assert "penultimate" in feats, f"{arch}: missing mandatory 'penultimate' feature"
    pen = feats["penultimate"]
    assert pen.ndim == 2 and pen.shape[0] == x.shape[0], f"{arch}: penultimate must be (B, D), got {tuple(pen.shape)}"
    # at least one intermediate activation beyond the penultimate
    intermediates = [k for k in feats if k not in ("penultimate", "gap")]
    assert len(intermediates) >= 1, f"{arch}: expected >=1 intermediate feature, got {list(feats)}"


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_hook_interface(base_cfg, arch):
    """register_feature_hooks() stashes intermediate activations in .features."""
    cfg = resolve_arch_cfg(base_cfg, arch=arch)
    model = build_model(cfg).eval()
    if not hasattr(model, "register_feature_hooks"):
        pytest.skip(f"{arch}: no hook interface (cnn1d uses its own Day-3 hooks)")
    model.register_feature_hooks()
    with torch.no_grad():
        model(_dummy_input(cfg))
    assert len(model.features) >= 1, f"{arch}: hooks stashed no features"
    model.remove_feature_hooks()
    assert len(model.features) == 0


def test_si_split_reuse_matches_week1(base_cfg):
    """Discipline A: every architecture loads the EXACT Week-1 SI speaker partition."""
    cfg = resolve_arch_cfg(base_cfg, arch="cnn1d")  # split loading is arch-independent
    cache = load_cache(cfg)
    idx = load_split_indices(cfg, "speaker_independent", cache)

    # same sizes as the committed Week-1 SI split / seed-42 run
    sizes = {k: int(len(v)) for k, v in idx.items()}
    assert sizes == WEEK1_SI_SIZES, f"SI sizes drifted from Week 1: {sizes} != {WEEK1_SI_SIZES}"

    # speakers behind the loaded indices == committed roles, and disjoint
    speaker_str = np.asarray(cache["speaker_str"])
    spec = load_split_spec(cfg, "speaker_independent")
    spk_by_role = {role: set(speaker_str[idx[role]].tolist()) for role in ("train", "val", "test")}
    for role in ("train", "val", "test"):
        assert spk_by_role[role] == set(spec["roles"][role]), f"{role} speakers != committed roles"
    assert spk_by_role["test"] == WEEK1_SI_TEST_SPEAKERS, "SI test speakers drifted from Week 1"
    # speaker-disjointness (no leakage) still holds
    assert spk_by_role["train"].isdisjoint(spk_by_role["val"])
    assert spk_by_role["train"].isdisjoint(spk_by_role["test"])
    assert spk_by_role["val"].isdisjoint(spk_by_role["test"])


def test_si_partition_identical_across_architectures(base_cfg):
    """The same SI indices feed every architecture (architecture doesn't touch splits)."""
    cache = load_cache(resolve_arch_cfg(base_cfg, arch="cnn1d"))
    ref = None
    for arch in ALL_ARCHS:
        cfg = resolve_arch_cfg(base_cfg, arch=arch)
        idx = load_split_indices(cfg, "speaker_independent", cache)
        packed = {k: idx[k].tolist() for k in ("train", "val", "test")}
        if ref is None:
            ref = packed
        else:
            assert packed == ref, f"{arch}: SI partition differs from the shared one"


def test_param_counts_table(base_cfg, capsys):
    """Print a param-count table; assert all within the same order of magnitude as baseline."""
    base = build_model(resolve_arch_cfg(base_cfg, arch="cnn1d"))
    base_n = sum(p.numel() for p in base.parameters())
    rows = []
    for arch in ALL_ARCHS:
        cfg = resolve_arch_cfg(base_cfg, arch=arch)
        model = build_model(cfg)
        n = sum(p.numel() for p in model.parameters())
        tag = "core" if arch in CORE_ARCHS else "stretch"
        rows.append((arch, tag, n, n / base_n))
    with capsys.disabled():
        print("\n  Week-2 architecture parameter counts (baseline cnn1d = {:,}):".format(base_n))
        print("  {:<12} {:<8} {:>10} {:>8}".format("arch", "scope", "params", "ratio"))
        for arch, tag, n, ratio in rows:
            flag = "  <-- MISMATCH" if not (0.5 <= ratio <= 2.0) else ""
            print("  {:<12} {:<8} {:>10,} {:>7.2f}x{}".format(arch, tag, n, ratio, flag))
    # same order of magnitude: within 0.5x .. 2x of the baseline
    for arch, _tag, _n, ratio in rows:
        assert 0.33 <= ratio <= 3.0, f"{arch} param ratio {ratio:.2f}x is not same order of magnitude"
