"""Necessity (band occlusion) vs. attribution (|DFT-LRP|) per frequency band, cnn1d + resnet1d.

Redraws paper figure dftlrp_band_share.png from saved results only (no model, no GPU):
  results/freq_occlusion/occlusion_results.csv   (scripts/xai_freq_occlusion.py, seed 42)
  results/dft_lrp/dftlrp_band_relevance.csv      (scripts/xai_dft_lrp.py, seed 42)

Normalisation (matches the published figure): each profile is divided by its own sum
over the 16 bands, so it reads as a share of total reliance. Occlusion drops below
zero (removing the band slightly *raised* accuracy) are clipped to 0 before
normalising. The r in each panel title is the Pearson correlation between the raw
(unclipped, unnormalised) occlusion drops and the band relevance.

Also rewrites results/dft_lrp/present_vs_used_summary.csv: the necessity-vs-attribution
r per architecture and, from the occlusion profiles split by speaker sex, the
female-vs-male profile correlation and the band of largest female-male divergence.

    python figures/plot_dftlrp_band_share.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OCCLUSION_CSV = ROOT / "results" / "freq_occlusion" / "occlusion_results.csv"
DFTLRP_CSV = ROOT / "results" / "dft_lrp" / "dftlrp_band_relevance.csv"
SUMMARY_CSV = ROOT / "results" / "dft_lrp" / "present_vs_used_summary.csv"
OUT_PNG = ROOT / "figures" / "output" / "dftlrp_band_share.png"

DFTLRP_ARCHS = ["cnn1d", "resnet1d"]
ALL_ARCHS = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]


def profile(df: pd.DataFrame, arch: str, group: str, col: str) -> pd.DataFrame:
    sub = df[(df["arch"] == arch) & (df["group"] == group)].sort_values("band_lo")
    assert len(sub) == 16, f"{arch}/{group}: expected 16 bands, got {len(sub)}"
    return sub[["band_lo", "band_hi", col]].reset_index(drop=True)


def main() -> None:
    occ = pd.read_csv(OCCLUSION_CSV)
    att = pd.read_csv(DFTLRP_CSV)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, arch in zip(axes, DFTLRP_ARCHS):
        need = profile(occ, arch, "all", "acc_drop")
        attr = profile(att, arch, "all", "mean_abs_relevance")
        assert np.allclose(need["band_lo"], attr["band_lo"]), "band edges differ between the two files"
        center = (need["band_lo"] + need["band_hi"]) / 2

        drop = need["acc_drop"].to_numpy()
        rel = attr["mean_abs_relevance"].to_numpy()
        r = float(np.corrcoef(drop, rel)[0, 1])
        rows.append({"arch": arch, "analysis": "need_vs_attr_all", "metric": "pearson_r", "value": round(r, 4)})

        drop_clipped = np.clip(drop, 0.0, None)
        ax.plot(center, drop_clipped / drop_clipped.sum(), marker="o", color="tab:blue",
                label="necessity (occlusion drop)")
        ax.plot(center, rel / rel.sum(), marker="s", color="tab:orange",
                label="attribution (|DFT-LRP|)")
        ax.set_title(f"{arch}: necessity vs. attribution (r={r:.2f})")
        ax.set_xlabel("freq band center (Hz)")
        ax.set_ylabel("normalized reliance")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

        share = rel / rel.sum()
        print(f"{arch}: r={r:.4f}  |DFT-LRP| share 0-250 Hz={share[0]:.1%}  below 750 Hz={share[:3].sum():.1%}")

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    # female-vs-male necessity profiles (occlusion only, all four architectures)
    for arch in ALL_ARCHS:
        f = profile(occ, arch, "female", "acc_drop")
        m = profile(occ, arch, "male", "acc_drop")
        r_fm = float(np.corrcoef(f["acc_drop"], m["acc_drop"])[0, 1])
        k = int(np.argmax(np.abs(f["acc_drop"].to_numpy() - m["acc_drop"].to_numpy())))
        center_k = float((f.loc[k, "band_lo"] + f.loc[k, "band_hi"]) / 2)
        rows.append({"arch": arch, "analysis": "necessity_female_vs_male", "metric": "pearson_r",
                     "value": round(r_fm, 4)})
        rows.append({"arch": arch, "analysis": "necessity_female_vs_male", "metric": "max_divergence_band_hz",
                     "value": center_k})
        print(f"{arch}: female-vs-male necessity r={r_fm:.4f}, max divergence at {center_k:.0f} Hz")

    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"wrote {OUT_PNG} and {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
