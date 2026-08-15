"""Create the headline young-employment trend by AI-exposure group.

The series uses a balanced set of matched SOC occupations and indexes weighted
employment among workers age 20-29 to 100 in 2020. Occupations with zero
observed Anthropic use remain a separate group because they are more than half
of the exposure distribution.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES, PROCESSED  # noqa: E402
from src.data_guard import guard_publication  # noqa: E402

YOUNG = {"20-24", "25-29"}
GROUPS = ["Q0 (none)", "Q1 (low)", "Q2", "Q3", "Q4 (high)"]
COLORS = {
    "Q0 (none)": "#8f919e",
    "Q1 (low)": "#9ab6d3",
    "Q2": "#668fb8",
    "Q3": "#365f8d",
    "Q4 (high)": "#c53b2f",
}


def build_series(long: pd.DataFrame, exposure: pd.DataFrame,
                 base_year: int = 2020) -> pd.DataFrame:
    """Return exposure-group young-employment indices for balanced occupations."""
    years = sorted(long["year"].unique())
    if base_year not in years:
        raise ValueError(f"base year {base_year} is not available: {years}")
    balanced = long.groupby("occ_code")["year"].nunique()
    balanced = balanced[balanced == len(years)].index
    young = long[
        long["occ_code"].isin(balanced) & long["age_band"].isin(YOUNG)
    ].copy()
    young = young.merge(
        exposure[["occ_code", "exposure_q"]], on="occ_code", how="inner"
    )
    out = (young.groupby(["year", "exposure_q"], observed=True, as_index=False)
                .agg(young_employed=("employed", "sum"),
                     occupations=("occ_code", "nunique")))
    base = (out[out["year"] == base_year]
            .set_index("exposure_q")["young_employed"])
    missing = set(out["exposure_q"]) - set(base.index)
    if missing:
        raise ValueError(f"exposure groups missing in base year: {sorted(missing)}")
    out["employment_index"] = out.apply(
        lambda row: 100 * row["young_employed"] / base[row["exposure_q"]], axis=1
    )
    out["exposure_q"] = pd.Categorical(
        out["exposure_q"], categories=GROUPS, ordered=True
    )
    return out.sort_values(["exposure_q", "year"]).reset_index(drop=True)


def main(base_year: int = 2020, allow_synthetic: bool = False) -> None:
    synthetic = guard_publication(allow_synthetic)
    long = pd.read_csv(PROCESSED / "panel_long.csv", dtype={"occ_code": str})
    exposure = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})
    series = build_series(long, exposure, base_year)

    suffix = "_SYNTHETIC" if synthetic else ""
    series.to_csv(PROCESSED / f"headline_series{suffix}.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    for group in GROUPS:
        d = series[series["exposure_q"] == group]
        if d.empty:
            continue
        high = group == "Q4 (high)"
        ax.plot(
            d["year"], d["employment_index"], marker="o",
            linewidth=3.2 if high else 2.0,
            markersize=7 if high else 5,
            color=COLORS[group], label=group, zorder=4 if high else 3,
        )

    years = sorted(series["year"].unique())
    ax.axhline(100, color="#c7c9d1", linewidth=1, zorder=1)
    if max(years) >= 2024:
        ax.axvspan(2023.5, max(years) + 0.35, color="#f7ecea", zorder=0)
        ax.text(2023.62, ax.get_ylim()[1] - 1, "2024-25",
                color="#9b4b43", fontsize=10, va="top")
    ax.set_xticks(years)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Young employment index ({base_year}=100)")
    ax.grid(axis="y", color="#e6e6eb", linewidth=0.8)
    ax.legend(frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Higher-exposure occupation groups lagged after 2023",
        x=0.08, y=0.98, ha="left", fontsize=20, fontweight="bold",
        color="#1b1d2b",
    )
    fig.text(
        0.08, 0.935,
        "CPS ASEC weighted employment, age 20-29, balanced SOC occupations",
        ha="left", fontsize=11, color="#6f7280",
    )
    fig.text(
        0.08, 0.02,
        "Descriptive only. Exposure groups use Anthropic observed use; Q0 has no observed use.",
        ha="left", fontsize=9, color="#6f7280",
    )
    if synthetic:
        ax.text(0.5, 0.5, "SYNTHETIC DATA\nNOT A FINDING",
                transform=ax.transAxes, fontsize=32, color="red", alpha=0.28,
                ha="center", va="center", rotation=18, fontweight="bold")

    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.84)
    out = FIGURES / f"young_employment_by_exposure{suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-year", type=int, default=2020)
    parser.add_argument("--allow-synthetic", action="store_true")
    main(**vars(parser.parse_args()))
