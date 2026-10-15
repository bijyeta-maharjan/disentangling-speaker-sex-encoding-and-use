"""Shared cross-architecture probing interface (Week-2 Discipline D).

Every architecture in the Week-2 comparison must expose features the SAME way so
the probing/representation code is architecture-agnostic. The contract:

    logits, feats = model(x, return_features=True)

where ``feats`` is an ordered dict that ALWAYS contains the key
``"penultimate"`` — the pooled feature vector fed into the classifier head, the
common cross-architecture probing point — PLUS any intermediate activations the
architecture exposes (the CNN family and TCN have a clear depth sequence and
expose per-stage features; recurrent/attention heads expose their frontend).

``FeatureHookMixin`` additionally provides ``register_feature_hooks`` /
``remove_feature_hooks`` for LRP-style access. Subclasses populate
``self._hookable = {name: module}`` (an ordered dict of named submodules) and the
mixin stashes each module's output in ``self.features`` during forward. This is
the access point ONLY — nothing is analysed with it today.

These are *instruments*, not the contribution: plain Conv/BN/ReLU/Pool/Linear/GRU
where possible, no exotic activations, kept LRP-friendly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

PENULTIMATE = "penultimate"


class FeatureHookMixin(nn.Module):
    """Mixin giving every architecture the shared hook-based feature access.

    Subclasses must:
      * implement ``forward(x, return_features=False)`` returning ``(logits,
        feats)`` when ``return_features=True``, with ``feats["penultimate"]`` set;
      * set ``self._hookable`` (dict name -> nn.Module) BEFORE calling
        ``_init_feature_hooks()`` in ``__init__``.
    """

    def _init_feature_hooks(self) -> None:
        self._hook_handles: list = []
        self.features: dict[str, torch.Tensor] = {}
        if not hasattr(self, "_hookable"):
            self._hookable: dict[str, nn.Module] = {}

    # --- probing access point (not used during training) -------------------- #
    def register_feature_hooks(self) -> None:
        """Attach forward hooks that stash each registered module's output."""
        self.remove_feature_hooks()

        def make_hook(name):
            def hook(_module, _inp, out):
                self.features[name] = out.detach()

            return hook

        for name, module in self._hookable.items():
            self._hook_handles.append(module.register_forward_hook(make_hook(name)))

    def remove_feature_hooks(self) -> None:
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        self._hook_handles = []
        if hasattr(self, "features"):
            self.features.clear()

    # --- introspection used by tests / reporting ---------------------------- #
    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return int(sum(p.numel() for p in ps if (p.requires_grad or not trainable_only)))


def feature_keys(model: nn.Module) -> list[str]:
    """Run a tiny dummy pass is avoided; rely on declared intermediate names."""
    return list(getattr(model, "intermediate_names", []))
