"""Necessity (band occlusion) vs. attribution (|DFT-LRP|) per frequency band, cnn1d + resnet1d.

Redraws paper figure dftlrp_band_share.png from saved results only (no model, no GPU):
  results/freq_occlusion/occlusion_results.csv   (scripts/xai_freq_occlusion.py, seed 42)
  results/dft_lrp/dftlrp_band_relevance.csv      (scripts/xai_dft_lrp.py, seed 42)

IEEE style (figures/ieee_style.py); written as PDF + 600-dpi PNG into figures/ and
paper/figures/. Normalisation: each profile is divided by its own sum
over the 16 bands, so it reads as a share of total reliance. Occlusion drops below
zero (removing the band slightly *raised* accuracy) are clipped to 0 before
normalising. The Pearson r between the raw (unclipped, unnormalised) occlusion drops
and the band relevance is printed and saved below; it belongs in the caption.

Also rewrites results/dft_lrp/present_vs_used_summary.csv: the necessity-vs-attribution
r per architecture and, from the occlusion profiles split by speaker sex, the
female-vs-male profile correlation and the band of largest female-male divergence.

    python figures/plot_dftlrp_band_share.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from figures import ieee_style as S  # noqa: E402

OCCLUSION_CSV = ROOT / "results" / "freq_occlusion" / "occlusion_results.csv"
DFTLRP_CSV = ROOT / "results" / "dft_lrp" / "dftlrp_band_relevance.csv"
SUMMARY_CSV = ROOT / "results" / "dft_lrp" / "present_vs_used_summary.csv"

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
    fig, axes = S.figure(4.15, 1.7, 1, 2, sharey=True)
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
        S.shade_speech_regions(ax, stagger=True)
        ax.plot(center, drop_clipped / drop_clipped.sum(), marker="o", color=S.NECESSITY,
                label="Necessity (occlusion)")
        ax.plot(center, rel / rel.sum(), marker="s", color=S.ATTRIBUTION, label="Attribution (|DFT-LRP|)")
        ax.set_title(S.MODEL_NAME[arch])
        ax.set_xlim(0, 4000)
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("Frequency band centre (Hz)")

        share = rel / rel.sum()
        print(f"{arch}: r={r:.4f}  |DFT-LRP| share 0-250 Hz={share[0]:.1%}  below 750 Hz={share[:3].sum():.1%}")

    axes[0].set_ylabel("Share of total")
    axes[1].legend(loc="upper right")
    S.save(fig, "dftlrp_band_share")

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
    print(f"wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
