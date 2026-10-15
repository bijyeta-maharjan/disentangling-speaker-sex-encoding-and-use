"""Layer-wise demographic-decodability curve — the Day-3 money plot.

For one attribute: probe accuracy vs conv-block depth, with THREE lines overlaid
(trained SI, untrained control, raw-input baseline) and the chance line marked.
The figure answers two questions at once:
  - does the representation encode the attribute?  (trained vs chance)
  - did TRAINING create that encoding?             (trained vs untrained)
The raw baseline is depth-independent, so it is drawn as a horizontal reference.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

BLOCK_ORDER = ["conv1", "conv2", "conv3", "conv4", "conv5"]
PRETTY = {
    "sex": "Sex (male vs female)",
    "accent": "Accent (German vs non-German)",
    "age_bin": "Age (≤median vs >median)",
    "speaker_id": "Speaker identity (re-identifiability)",
}


def plot_decodability(df: pd.DataFrame, attr: str, probe: str = "linear",
                      save_path: Path | None = None) -> plt.Figure:
    sub = df[(df["attribute"] == attr) & (df["probe"] == probe)]
    if sub.empty:
        raise ValueError(f"no rows for attribute={attr} probe={probe}")

    trained = sub[sub["model"] == "trained"].set_index("block").reindex(BLOCK_ORDER)
    untrained = sub[sub["model"] == "untrained"].set_index("block").reindex(BLOCK_ORDER)
    raw = sub[sub["model"] == "raw"]
    chance = float(sub["chance"].iloc[0])

    x = range(1, len(BLOCK_ORDER) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))

    # trained SI with speaker-bootstrap CI band
    ax.plot(x, trained["acc"], "-o", color="C0", lw=2.5, label="trained SI")
    ax.fill_between(x, trained["ci_lo"], trained["ci_hi"], color="C0", alpha=0.15)
    # untrained control
    ax.plot(x, untrained["acc"], "--s", color="C1", lw=2, label="untrained control")
    ax.fill_between(x, untrained["ci_lo"], untrained["ci_hi"], color="C1", alpha=0.10)
    # raw-input baseline (depth-independent) + chance
    if not raw.empty:
        ax.axhline(float(raw["acc"].iloc[0]), color="C2", ls=":", lw=2,
                   label="raw-input baseline")
    ax.axhline(chance, color="0.4", ls="-.", lw=1.5, label="chance (majority)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(BLOCK_ORDER)
    ax.set_xlabel("Conv block (depth →)")
    ax.set_ylabel(f"Probe accuracy ({probe})")
    ax.set_ylim(0, 1.02)
    ax.set_title(PRETTY.get(attr, attr))
    ax.legend(loc="best", fontsize=12, framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
