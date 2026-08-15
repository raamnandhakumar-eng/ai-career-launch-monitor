"""Robustness checks for the early-warning design.

Runs age-group heterogeneity, pre-2024 placebo dates, leave-one-occupation-group
out estimates, and a specification with occupation-specific linear trends.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES, PROCESSED  # noqa: E402
from src.data_guard import guard_publication  # noqa: E402

AGE_BANDS = ["20-24", "25-29", "30-39", "40-54", "55+"]
YOUNG = {"20-24", "25-29"}
LEAVE_OUT = [
    "Computer & Mathematical",
    "Business & Financial Operations",
    "Legal",
    "Office & Administrative Support",
    "Sales & Related",
]


def load_young_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    long = pd.read_csv(PROCESSED / "panel_long.csv", dtype={"occ_code": str})
    exposure = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})
    long["young_employed"] = long["employed"].where(
        long["age_band"].isin(YOUNG), 0
    )
    young = (long.groupby(["year", "occ_code"], as_index=False)
             .agg(employed=("employed", "sum"),
                  young_employed=("young_employed", "sum")))
    young["young_share"] = young["young_employed"] / young["employed"]
    young = young.merge(
        exposure[["occ_code", "ai_exposure", "soc_major_label"]],
        on="occ_code", how="inner",
    )
    return long, young


def fit_twfe(df: pd.DataFrame, post_from: int) -> dict:
    from linearmodels.panel import PanelOLS

    work = df.copy()
    work["exp_x_post"] = work["ai_exposure"] * (work["year"] >= post_from)
    work = work.set_index(["occ_code", "year"])
    result = PanelOLS.from_formula(
        "young_share ~ exp_x_post + EntityEffects + TimeEffects", data=work
    ).fit(cov_type="clustered", cluster_entity=True)
    return {
        "coefficient": result.params["exp_x_post"],
        "std_error": result.std_errors["exp_x_post"],
        "p_value": result.pvalues["exp_x_post"],
        "nobs": result.nobs,
    }


def age_heterogeneity(long: pd.DataFrame, exposure: pd.DataFrame,
                      post_from: int) -> pd.DataFrame:
    """Estimate exposure-by-post coefficients for every age-band share."""
    from linearmodels.panel import PanelOLS

    keys = long[["year", "occ_code"]].drop_duplicates()
    bands = pd.DataFrame({"age_band": AGE_BANDS})
    grid = keys.merge(bands, how="cross")
    complete = grid.merge(
        long[["year", "occ_code", "age_band", "employed"]],
        on=["year", "occ_code", "age_band"], how="left",
    )
    complete["employed"] = complete["employed"].fillna(0)
    complete["total"] = complete.groupby(["year", "occ_code"])["employed"].transform("sum")
    complete["age_share"] = complete["employed"] / complete["total"]
    complete = complete.merge(exposure[["occ_code", "ai_exposure"]], on="occ_code")
    rows = []
    for band in AGE_BANDS:
        d = complete[complete["age_band"] == band].copy()
        d["exp_x_post"] = d["ai_exposure"] * (d["year"] >= post_from)
        d = d.set_index(["occ_code", "year"])
        result = PanelOLS.from_formula(
            "age_share ~ exp_x_post + EntityEffects + TimeEffects", data=d
        ).fit(cov_type="clustered", cluster_entity=True)
        rows.append({
            "age_band": band,
            "coefficient": result.params["exp_x_post"],
            "std_error": result.std_errors["exp_x_post"],
            "p_value": result.pvalues["exp_x_post"],
            "nobs": result.nobs,
        })
    return pd.DataFrame(rows)


def occupation_trend_result(young: pd.DataFrame, post_from: int) -> dict:
    import statsmodels.formula.api as smf

    work = young.copy()
    work["exp_x_post"] = work["ai_exposure"] * (work["year"] >= post_from)
    work["trend"] = work["year"] - work["year"].min()
    result = smf.ols(
        "young_share ~ exp_x_post + C(occ_code) + C(year) + C(occ_code):trend",
        data=work,
    ).fit(cov_type="cluster", cov_kwds={"groups": work["occ_code"]})
    return {
        "coefficient": result.params["exp_x_post"],
        "std_error": result.bse["exp_x_post"],
        "p_value": result.pvalues["exp_x_post"],
        "nobs": int(result.nobs),
    }


def main(post_from: int = 2024, allow_synthetic: bool = False) -> None:
    synthetic = guard_publication(allow_synthetic)
    long, young = load_young_panel()
    exposure = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})
    rows = []

    baseline = fit_twfe(young, post_from)
    rows.append({"check": "baseline", "detail": f"post {post_from}", **baseline})

    trend = occupation_trend_result(young, post_from)
    rows.append({"check": "occupation trends", "detail": "linear trend by SOC", **trend})

    pre = young[young["year"] < post_from].copy()
    for fake_year in range(int(pre["year"].min()) + 1, post_from):
        result = fit_twfe(pre, fake_year)
        rows.append({"check": "placebo", "detail": f"fake post {fake_year}", **result})

    for group in LEAVE_OUT:
        result = fit_twfe(young[young["soc_major_label"] != group], post_from)
        rows.append({"check": "leave out", "detail": group, **result})

    age = age_heterogeneity(long, exposure, post_from)
    for row in age.to_dict("records"):
        rows.append({"check": "age group", "detail": row.pop("age_band"), **row})

    summary = pd.DataFrame(rows)
    suffix = "_SYNTHETIC" if synthetic else ""
    summary.to_csv(PROCESSED / f"robustness_summary{suffix}.csv", index=False)
    print(summary.round(4).to_string(index=False))

    plot = age.copy()
    plot["coefficient_pp"] = plot["coefficient"] * 100
    plot["ci"] = 1.96 * plot["std_error"] * 100
    fig, ax = plt.subplots(figsize=(8.8, 5.5), dpi=150)
    colors = ["#c53b2f", "#d9786f", "#668fb8", "#8f919e", "#8f919e"]
    ax.barh(plot["age_band"], plot["coefficient_pp"], xerr=plot["ci"],
            color=colors, alpha=0.9, capsize=4)
    ax.axvline(0, color="#4a4d59", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Exposure × post coefficient (percentage points)")
    ax.set_ylabel("Age band")
    ax.grid(axis="x", color="#e6e6eb", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if synthetic:
        ax.text(0.5, 0.5, "SYNTHETIC DATA\nNOT A FINDING",
                transform=ax.transAxes, fontsize=28, color="red", alpha=0.28,
                ha="center", va="center", rotation=18, fontweight="bold")
    fig.suptitle("Younger coefficients are negative, but imprecise",
                 x=0.13, y=0.98, ha="left", fontsize=17,
                 fontweight="bold", color="#1b1d2b")
    fig.text(0.13, 0.92, "Separate two-way fixed-effects models; 95% CI",
             ha="left", color="#6f7280", fontsize=10)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.82)
    fig.savefig(FIGURES / f"age_heterogeneity{suffix}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-from", type=int, default=2024)
    parser.add_argument("--allow-synthetic", action="store_true")
    main(**vars(parser.parse_args()))
