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

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 7.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- Left: mean exposure by SOC major group ---
    colors = [ACCENT if v >= 0.15 else MUTE for v in grp.values]
    axL.barh(grp.index, grp.values, color=colors, height=0.72)
    axL.set_title("Observed AI exposure by occupation group",
                  fontsize=12, fontweight="bold", color=INK, loc="left")
    axL.set_xlabel("Unweighted mean exposure (share of tasks seen in Claude use)")
    axL.tick_params(length=0)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    for y, v in enumerate(grp.values):
        axL.text(v + 0.004, y, f"{v:.2f}", va="center", fontsize=8, color=INK)

    # --- Right: most- and least-exposed detailed occupations ---
    top = df.nlargest(12, "ai_exposure")[::-1]
    labels = top["title"].map(lambda s: textwrap.fill(s, width=35))
    axR.barh(labels, top["ai_exposure"],
             color=ACCENT, height=0.72)
    axR.set_title("Most-exposed detailed occupations",
                  fontsize=12, fontweight="bold", color=INK, loc="left")
    axR.set_xlabel("Observed exposure")
    axR.tick_params(length=0)
    axR.tick_params(axis="y", labelsize=8.5)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    for y, v in enumerate(top["ai_exposure"].values):
        axR.text(v + 0.006, y, f"{v:.2f}", va="center", fontsize=8, color=INK)

    share_zero = (df["ai_exposure"] == 0).mean()
    fig.suptitle("Where observed Claude use maps onto U.S. occupations",
                 x=0.012, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, 0.945,
             f"Anthropic Economic Index, labor-market-impacts release  |  "
             f"{len(df)} SOC occupations  |  {share_zero:.0%} show zero observed usage",
             ha="left", fontsize=9.5, color=MUTE)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = FIGURES / "exposure_by_major_group.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    print("\nMost exposed:")
    print(df.nlargest(10, "ai_exposure")[["occ_code", "title", "ai_exposure"]]
          .to_string(index=False))
    print("\nExposure concentration:")
    print(f"  mean={df.ai_exposure.mean():.3f}  median={df.ai_exposure.median():.3f}  "
          f"share_zero={share_zero:.1%}")


if __name__ == "__main__":
    main()
