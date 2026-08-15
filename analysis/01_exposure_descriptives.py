"""
01 - Descriptives on the real Anthropic observed-exposure measure.

Produces figures/exposure_by_major_group.png and prints the ranked
occupation tables. 100% real AEI data; no CPS required.

    python analysis/01_exposure_descriptives.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED, FIGURES  # noqa: E402
from src.plot_style import (  # noqa: E402
    ACCENT, INK, MUTE, add_header, apply_chart_style, save_chart, style_axes,
)

apply_chart_style()


def main():
    df = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})

    grp = (df.groupby("soc_major_label")["ai_exposure"]
             .mean().sort_values())
    share_zero = (df["ai_exposure"] == 0).mean()

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(18, 10.5), gridspec_kw={"width_ratios": [1, 1.05]}
    )
    add_header(
        fig,
        "Where observed AI use is concentrated",
        f"Anthropic Economic Index  |  {len(df)} detailed SOC occupations  |  "
        f"{share_zero:.0%} have no observed use",
        left=0.04, title_y=0.97, subtitle_y=0.915, title_size=23,
    )

    # --- Left: mean exposure by SOC major group ---
    colors = [ACCENT if v >= 0.15 else MUTE for v in grp.values]
    axL.barh(grp.index, grp.values, color=colors, height=0.72)
    axL.set_title("Mean exposure by occupation group",
                  fontsize=15, fontweight="bold", color=INK, loc="left", pad=20)
    axL.text(
        0, 1.005,
        "Unweighted mean across detailed occupations in each SOC group",
        transform=axL.transAxes, fontsize=9.5, color=MUTE, va="bottom",
    )
    axL.set_xlabel("Observed exposure (share of occupational tasks seen in Claude use)",
                   labelpad=10)
    axL.set_xlim(0, 0.42)
    axL.tick_params(axis="y", length=0, labelsize=10.5)
    axL.tick_params(axis="x", length=0, labelsize=9.5)
    style_axes(axL, grid_axis="x", hide_left=True)
    for y, v in enumerate(grp.values):
        axL.text(max(v + 0.004, 0.006), y, f"{v:.2f}", va="center",
                 fontsize=9, color=INK)

    # --- Bottom: most-exposed detailed occupations ---
    top = df.nlargest(12, "ai_exposure")[::-1]
    labels = top["title"].map(lambda s: textwrap.fill(s, width=43))
    axR.barh(labels, top["ai_exposure"],
             color=ACCENT, height=0.72)
    axR.set_title("Most-exposed detailed occupations",
                  fontsize=15, fontweight="bold", color=INK, loc="left", pad=20)
    axR.text(
        0, 1.005,
        "Detailed 2018 SOC occupations ranked by observed exposure",
        transform=axR.transAxes, fontsize=9.5, color=MUTE, va="bottom",
    )
    axR.set_xlabel("Observed exposure", labelpad=10)
    axR.set_xlim(0, 0.80)
    axR.tick_params(axis="y", length=0, labelsize=10)
    axR.tick_params(axis="x", length=0, labelsize=9.5)
    style_axes(axR, grid_axis="x", hide_left=True)
    for label in axR.get_yticklabels():
        label.set_linespacing(0.95)
    for y, v in enumerate(top["ai_exposure"].values):
        axR.text(v + 0.009, y, f"{v:.2f}", va="center", fontsize=9, color=INK)

    out = FIGURES / "exposure_by_major_group.png"
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.09, top=0.80, wspace=0.62)
    save_chart(fig, out)
    print(f"Saved {out}")

    print("\nMost exposed:")
    print(df.nlargest(10, "ai_exposure")[["occ_code", "title", "ai_exposure"]]
          .to_string(index=False))
    print("\nExposure concentration:")
    print(f"  mean={df.ai_exposure.mean():.3f}  median={df.ai_exposure.median():.3f}  "
          f"share_zero={share_zero:.1%}")


if __name__ == "__main__":
    main()
