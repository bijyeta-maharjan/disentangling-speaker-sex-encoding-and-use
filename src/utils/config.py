"""Config loading + small path helpers.

The repo root is two levels above this file (src/utils/config.py -> repo root).
Relative paths in the YAML are resolved against the repo root so scripts work
regardless of the current working directory.

Three top-level roots can be moved without editing any config or script, by
setting environment variables (useful when the data or outputs live on another
disk):

  AMNIST_DATA_ROOT         replaces the leading ``data/``         (raw audio + caches)
  AMNIST_EXPERIMENTS_ROOT  replaces the leading ``experiments/``  (checkpoints, run logs)
  AMNIST_RESULTS_ROOT      replaces the leading ``results/``      (saved result tables)
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "day2_baseline.yaml"

ROOT_ENV_VARS = {
    "data": "AMNIST_DATA_ROOT",
    "experiments": "AMNIST_EXPERIMENTS_ROOT",
    "results": "AMNIST_RESULTS_ROOT",
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load the YAML config as a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Resolve a (possibly relative) path against the repo root.

    A relative path whose first component is ``data``, ``experiments`` or
    ``results`` is redirected to the matching environment variable when it is set.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in ROOT_ENV_VARS:
        override = os.environ.get(ROOT_ENV_VARS[p.parts[0]])
        if override:
            return Path(override).joinpath(*p.parts[1:])
    return REPO_ROOT / p


def paths(cfg: dict) -> SimpleNamespace:
    """Return repo-root-resolved Paths for every entry under cfg['paths']."""
    return SimpleNamespace(**{k: resolve(v) for k, v in cfg["paths"].items()})


def resolve_arch_cfg(cfg: dict, arch: str | None = None, seed: int | None = None) -> dict:
    """Materialise a single-architecture config from a Week-2 multi-arch config.

    Copies ``cfg['architectures'][arch]`` into ``cfg['model']`` (so the shared
    ``build_model`` / trainer code path is unchanged), and optionally overrides
    the seed. Everything else (preprocess, splits, train protocol, eval) is held
    fixed — that is the whole point of the controlled comparison. A config with
    no ``architectures`` block is returned as-is (Week-1 single-model behaviour).
    """
    import copy

    out = copy.deepcopy(cfg)
    if arch is not None:
        archs = out.get("architectures")
        if not archs or arch not in archs:
            known = sorted(archs) if archs else []
            raise ValueError(f"arch {arch!r} not in config; known: {known}")
        out["model"] = copy.deepcopy(archs[arch])
    if seed is not None:
        out["seed"] = seed
    return out
