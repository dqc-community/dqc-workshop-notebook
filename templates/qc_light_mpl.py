"""QC Light Theme for matplotlib in marimo notebooks.

This module provides a matplotlib style that matches the QC Light theme
used in marimo notebooks (marimo/qc-theme.css).
"""

from matplotlib.colors import LinearSegmentedColormap  # type: ignore

PRIMARY = "#1a1a1a"
SECONDARY = "#666666"
TERTIARY = "#999999"
DIM = "#aaaaaa"
BORDER = "#cccccc"
GUIDE = "#dddddd"
SURFACE = "#f5f5f5"
PAGE = "#ffffff"

DANGER = "#ff0044"
INFO = "#0055ff"
SUCCESS = "#00ff88"
WARNING = "#ff8800"

DATA_PALETTE = [
    "#89AAC8",
    "#D4949A",
    "#8DC0A3",
    "#D4A87C",
    "#A992BE",
    "#B4B89E",
    "#84BEB8",
]


def qc_light() -> dict[str, object]:
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


def qc_light_colors() -> dict[str, str]:
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


def qc_light_palette() -> list[str]:
    return list(DATA_PALETTE)


def qc_light_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("qc_light_data", DATA_PALETTE, N=256)


def apply() -> None:
    import matplotlib.pyplot as plt  # type: ignore

    plt.rcParams.update(qc_light())


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
    "qc_light",
    "qc_light_colors",
    "qc_light_palette",
    "qc_light_colormap",
    "apply",
]
