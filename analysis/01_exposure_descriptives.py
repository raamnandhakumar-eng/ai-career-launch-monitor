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

INK = "#1a1a2e"
ACCENT = "#c0392b"
MUTE = "#8a8a99"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "figure.dpi": 130, "savefig.dpi": 150,
})


def main():
    df = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})

    grp = (df.groupby("soc_major_label")["ai_exposure"]
             .mean().sort_values())
    share_zero = (df["ai_exposure"] == 0).mean()

    # A vertical layout keeps every occupation label readable when the image is
    # displayed at GitHub README width.
    fig = plt.figure(figsize=(12, 14.2), layout="constrained")
    fig.set_constrained_layout_pads(h_pad=0.16, hspace=0.08)
    grid = fig.add_gridspec(3, 1, height_ratios=[0.28, 2.05, 1.55])
    header = fig.add_subplot(grid[0])
    axL = fig.add_subplot(grid[1])
    axR = fig.add_subplot(grid[2])

    header.axis("off")
    fig.text(
        0.035, 0.982,
        "Where observed AI use is concentrated across U.S. occupations",
        fontsize=19, fontweight="bold", color=INK, ha="left", va="top",
    )
    fig.text(
        0.035, 0.953,
        f"Anthropic Economic Index  |  {len(df)} SOC occupations  |  "
        f"{share_zero:.0%} show zero observed usage",
        fontsize=11, color=MUTE, ha="left", va="top",
    )

    # --- Left: mean exposure by SOC major group ---
    colors = [ACCENT if v >= 0.15 else MUTE for v in grp.values]
    axL.barh(grp.index, grp.values, color=colors, height=0.72)
    axL.set_title("Mean exposure by occupation group",
                  fontsize=14, fontweight="bold", color=INK, loc="left", pad=18)
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
    axL.xaxis.grid(True, color="#e8e8ee", linewidth=0.8)
    axL.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axL.spines[s].set_visible(False)
    for y, v in enumerate(grp.values):
        axL.text(max(v + 0.004, 0.006), y, f"{v:.2f}", va="center",
                 fontsize=9, color=INK)

    # --- Bottom: most-exposed detailed occupations ---
    top = df.nlargest(12, "ai_exposure")[::-1]
    labels = top["title"].map(lambda s: textwrap.fill(s, width=62))
    axR.barh(labels, top["ai_exposure"],
             color=ACCENT, height=0.72)
    axR.set_title("Most-exposed detailed occupations",
                  fontsize=14, fontweight="bold", color=INK, loc="left", pad=18)
    axR.text(
        0, 1.005,
        "Detailed 2018 SOC occupations ranked by observed exposure",
        transform=axR.transAxes, fontsize=9.5, color=MUTE, va="bottom",
    )
    axR.set_xlabel("Observed exposure", labelpad=10)
    axR.set_xlim(0, 0.80)
    axR.tick_params(axis="y", length=0, labelsize=10)
    axR.tick_params(axis="x", length=0, labelsize=9.5)
    axR.xaxis.grid(True, color="#e8e8ee", linewidth=0.8)
    axR.set_axisbelow(True)
    for label in axR.get_yticklabels():
        label.set_linespacing(0.95)
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)
    for y, v in enumerate(top["ai_exposure"].values):
        axR.text(v + 0.009, y, f"{v:.2f}", va="center", fontsize=9, color=INK)

    out = FIGURES / "exposure_by_major_group.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    print("\nMost exposed:")
    print(df.nlargest(10, "ai_exposure")[["occ_code", "title", "ai_exposure"]]
          .to_string(index=False))
    print("\nExposure concentration:")
    print(f"  mean={df.ai_exposure.mean():.3f}  median={df.ai_exposure.median():.3f}  "
          f"share_zero={share_zero:.1%}")


if __name__ == "__main__":
    main()
