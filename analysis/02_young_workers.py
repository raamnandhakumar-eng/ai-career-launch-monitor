"""
02 - The headline chart: change in young-worker share vs AI exposure.

x = observed AI exposure (real Anthropic)
y = change in age 20-29 employment share, baseline -> recent (from CPS panel)

If the underlying CPS panel is the synthetic plumbing file, the figure is
stamped with a loud SYNTHETIC watermark and the script refuses to write to the
publication figure name. Only a real CPS panel yields figures/young_worker_effect.png.

    python analysis/02_young_workers.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED, FIGURES  # noqa: E402
from src.data_guard import guard_publication  # noqa: E402
from src.plot_style import (  # noqa: E402
    ACCENT, INK, MUTE, add_footer, add_header, apply_chart_style, save_chart,
    style_axes,
)

apply_chart_style()


def main(allow_synthetic: bool = False):
    occ = pd.read_csv(PROCESSED / "panel_occ.csv", dtype={"occ_code": str})
    d = occ.dropna(subset=["young_share_change"]).copy()
    d = d[d["ai_exposure"] > 0]  # focus on occupations with any AI usage
    synth = guard_publication(allow_synthetic)
    if len(d) < 3:
        raise ValueError("Need at least three matched occupations with positive exposure")

    x = d["ai_exposure"].values
    y = d["young_share_change"].values * 100
    fallback = d["employment_recent"].dropna().median()
    w = d["employment_recent"].fillna(fallback).clip(lower=1).values

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    ax.axhline(0, color=MUTE, lw=0.8, zorder=1)
    ax.scatter(x, y, s=np.sqrt(w) / 6, alpha=0.45, color=ACCENT,
               edgecolor="white", linewidth=0.4, zorder=2)

    # employment-weighted linear fit
    b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color=INK, lw=2.2, zorder=3)

    ax.set_xlabel("Observed AI exposure (Anthropic Economic Index)")
    ax.set_ylabel("Change in age 20-29 employment share (percentage points)")
    style_axes(ax, grid_axis="y")
    ax.text(
        0.98, 0.96, f"Weighted slope\n{b:.1f} pp per exposure unit",
        transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=10,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white",
              "edgecolor": "#d8d9df", "linewidth": 0.8},
    )
    add_header(
        fig,
        "More-exposed occupations show a larger decline",
        "Change in age 20-29 employment share, 2022-2025; each circle is one occupation",
        left=0.09,
    )
    add_footer(
        fig,
        "Circle area reflects recent employment. Positive-exposure occupations only.",
        left=0.09,
    )

    if synth:
        ax.text(0.5, 0.5, "SYNTHETIC DATA\nNOT A FINDING", transform=ax.transAxes,
                fontsize=34, color="red", alpha=0.28, ha="center", va="center",
                rotation=18, fontweight="bold", zorder=5)
        out = FIGURES / "young_worker_effect_SYNTHETIC.png"
    else:
        out = FIGURES / "young_worker_effect.png"

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.79)
    save_chart(fig, out)
    print(f"Saved {out}" + ("  [SYNTHETIC -- not a result]" if synth else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-synthetic", action="store_true")
    main(**vars(parser.parse_args()))
