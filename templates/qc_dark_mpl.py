"""QC Dark Theme for matplotlib.

This module provides a matplotlib style that matches the QC Dark theme
used in Quarto documents (quarto/qc-dark-theme.scss).
"""

from matplotlib.colors import LinearSegmentedColormap

PRIMARY = "#ffffff"
SECONDARY = "#999999"
TERTIARY = "#666666"
DIM = "#4a4a4a"
BORDER = "#333333"
GUIDE = "#444444"
SURFACE = "#2a2a2a"
PAGE = "#1a1a1a"

DANGER = "#ff4466"
INFO = "#4488ff"
SUCCESS = "#44ff88"
WARNING = "#ffaa44"

DATA_PALETTE = [
    "#89AAC8",
    "#D4949A",
    "#8DC0A3",
    "#D4A87C",
    "#A992BE",
    "#B4B89E",
    "#84BEB8",
]


def qc_dark() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Montserrat", "DejaVu Sans", "Arial", "Helvetica"],
        "font.monospace": [
            "JetBrains Mono",
            "DejaVu Sans Mono",
            "Consolas",
            "Courier New",
        ],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": PRIMARY,
        "axes.labelsize": 10,
        "axes.labelcolor": SECONDARY,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": BORDER,
        "axes.linewidth": 1,
        "axes.grid": True,
        "grid.color": GUIDE,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "axes.xmargin": 0.05,
        "axes.ymargin": 0.05,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "figure.edgecolor": "none",
        "axes.axisbelow": True,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0,
        "patch.linewidth": 1,
        "patch.facecolor": SURFACE,
        "patch.edgecolor": BORDER,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "text.color": TERTIARY,
        "text.usetex": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": SECONDARY,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.5,
        "legend.handlelength": 1.5,
        "legend.markerscale": 1.0,
        "legend.facecolor": PAGE,
        "legend.edgecolor": BORDER,
        "legend.title_fontsize": 9,
        "hatch.color": SECONDARY,
        "hatch.linewidth": 0.5,
    }


def qc_dark_colors() -> dict[str, str]:
    return {
        "primary": PRIMARY,
        "secondary": SECONDARY,
        "tertiary": TERTIARY,
        "dim": DIM,
        "border": BORDER,
        "guide": GUIDE,
        "surface": SURFACE,
        "page": PAGE,
        "danger": DANGER,
        "info": INFO,
        "success": SUCCESS,
        "warning": WARNING,
    }


def qc_dark_palette() -> list[str]:
    return list(DATA_PALETTE)


def qc_dark_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("qc_dark_data", DATA_PALETTE, N=256)


def apply() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(qc_dark())


__all__ = [
    "PRIMARY",
    "SECONDARY",
    "TERTIARY",
    "DIM",
    "BORDER",
    "GUIDE",
    "SURFACE",
    "PAGE",
    "DANGER",
    "INFO",
    "SUCCESS",
    "WARNING",
    "DATA_PALETTE",
    "qc_dark",
    "qc_dark_colors",
    "qc_dark_palette",
    "qc_dark_colormap",
    "apply",
]
