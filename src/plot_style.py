"""Shared visual style for publication charts."""
from pathlib import Path

import matplotlib.pyplot as plt

INK = "#202233"
ACCENT = "#c73a2c"
BLUE = "#35628f"
MUTE = "#9295a3"
GRID = "#e7e8ed"
SOFT_RED = "#f6e9e6"


def apply_chart_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.9,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 140,
        "savefig.dpi": 160,
    })


def add_header(fig, title: str, subtitle: str, *, left: float = 0.08,
               title_y: float = 0.965, subtitle_y: float = 0.905,
               title_size: float = 19) -> None:
    fig.text(left, title_y, title, ha="left", va="top", color=INK,
             fontsize=title_size, fontweight="bold")
    fig.text(left, subtitle_y, subtitle, ha="left", va="top", color=MUTE,
             fontsize=10.5)


def style_axes(ax, *, grid_axis: str = "y", hide_left: bool = False) -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def add_footer(fig, text: str, *, left: float = 0.08, y: float = 0.025) -> None:
    fig.text(left, y, text, ha="left", va="bottom", color=MUTE, fontsize=9)


def save_chart(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
