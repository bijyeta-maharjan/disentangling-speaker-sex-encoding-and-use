"""Shared IEEE figure style for every paper figure.

Figures are drawn at their final printed size (IEEE column 3.5 in, page width 7.16 in),
so the font sizes below are the sizes that appear in print:

  axis labels 8 pt, ticks / legends / annotations 7 pt (nothing smaller),
  serif Times-like font (STIXGeneral, mathtext "stix").

Colours follow the Okabe-Ito palette and mean the same thing in every figure.
`save(fig, name)` writes a vector PDF (TrueType fonts) and a 600-dpi PNG with the
paper's base filename into figures/ and paper/figures/.

Usage:
    from figures import ieee_style as S
    S.apply()
    fig, ax = S.figure(3.5, 2.0)
    ...
    S.save(fig, "necessity_all_arch")
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = (ROOT / "figures", ROOT / "paper" / "figures")

COLUMN_WIDTH = 3.5
PAGE_WIDTH = 7.16

# --- colours (Okabe-Ito) -------------------------------------------------------
FEMALE = "#D55E00"
MALE = "#0072B2"
SEX_COLOR = {"female": FEMALE, "male": MALE}
SEX_LABEL = {"female": "Female", "male": "Male"}

MODEL_ORDER = ["cnn1d", "resnet1d", "crnn_gru", "spec_cnn2d"]
MODEL_NAME = {"cnn1d": "1-D CNN", "resnet1d": "Residual CNN", "crnn_gru": "GRU-CRNN",
              "spec_cnn2d": "Spectrogram CNN"}
MODEL_COLOR = {"cnn1d": "#0072B2", "resnet1d": "#E69F00", "crnn_gru": "#009E73",
               "spec_cnn2d": "#CC79A7"}
MODEL_MARKER = {"cnn1d": "o", "resnet1d": "s", "crnn_gru": "^", "spec_cnn2d": "D"}

NECESSITY = "#0072B2"
ATTRIBUTION = "#E69F00"
NULL_GREY = "#BBBBBB"
OBSERVED = "black"
REFERENCE_GREY = "#777777"
NEUTRAL_DARK = "#555555"
NEUTRAL_LIGHT = "#AAAAAA"
GRID = "#DDDDDD"

# trained solid, untrained dashed, raw-input dotted, chance thin grey dash-dot
LINESTYLE = {"trained": "-", "untrained": "--", "raw": ":"}
CHANCE = dict(color="#888888", lw=0.6, ls="-.")

# speech regions
F0_BAND = (0, 250)
F1F2_BAND = (250, 750)

MARKERSIZE = 3.5
ERRORBAR = dict(elinewidth=0.6, capsize=1.5, capthick=0.6)

RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.25,
    "legend.columnspacing": 0.9,
    "lines.linewidth": 1.0,
    "lines.markersize": MARKERSIZE,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 3.0,
    "axes.labelpad": 2.0,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "patch.linewidth": 0.5,
    "errorbar.capsize": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 100,
    "axes.unicode_minus": True,
}


def apply() -> None:
    """Install the IEEE rcParams (call once, before creating figures)."""
    matplotlib.rcParams.update(RC)


def figure(width: float, height: float, nrows: int = 1, ncols: int = 1, **kwargs):
    """plt.subplots at the final printed size, with constrained layout."""
    apply()
    kwargs.setdefault("layout", "constrained")
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    return fig, axes


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.3)
    ax.set_axisbelow(True)


def shade_speech_regions(ax, label: bool = True, stagger: bool = False) -> None:
    """Shade the F0 (0-250 Hz) and F1/F2 (250-750 Hz) regions, labelled at the top.
    stagger=True drops the F1/F2 label one line so the two labels don't touch in narrow panels."""
    ax.axvspan(*F0_BAND, color="#E8E8E8", lw=0, zorder=0)
    ax.axvspan(*F1F2_BAND, color="#DCEBF5", lw=0, zorder=0)
    if label:
        tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for (lo, hi), text, y in ((F0_BAND, "F0", 0.99), (F1F2_BAND, "F1/F2", 0.90 if stagger else 0.99)):
            ax.text((lo + hi) / 2, y, text, transform=tr, ha="center", va="top", fontsize=7,
                    color="#444444")


def model_line(ax, x, y, arch: str, label: str | None = None, **kwargs):
    kwargs.setdefault("ls", "-")
    kwargs.setdefault("ms", 3.0)
    return ax.plot(x, y, color=MODEL_COLOR[arch], marker=MODEL_MARKER[arch],
                   label=MODEL_NAME[arch] if label is None else label, **kwargs)


def save(fig, name: str, out_dirs=OUT_DIRS) -> list[Path]:
    """Write <name>.pdf (vector, TrueType) and <name>.png (600 dpi) into each out dir.
    Each format is rendered once and copied, so the copies are byte-identical."""
    out_dirs = [Path(d) for d in out_dirs]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    first = out_dirs[0]
    fig.savefig(first / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(first / f"{name}.png", dpi=600, metadata={"Software": None})
    written = [first / f"{name}.pdf", first / f"{name}.png"]
    for d in out_dirs[1:]:
        for ext in ("pdf", "png"):
            shutil.copyfile(first / f"{name}.{ext}", d / f"{name}.{ext}")
            written.append(d / f"{name}.{ext}")
    plt.close(fig)
    print(f"[fig] {name}: " + ", ".join(str(p.relative_to(ROOT)) for p in written))
    return written
