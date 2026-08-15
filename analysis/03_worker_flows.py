"""Make linked worker flows the headline labor-market outcome.

Runs annual occupation fixed-effects models for entry, entrant composition,
and exit using the Basic Monthly CPS rotation panel. It also estimates
young-worker weekly wage models, runs focused robustness checks, and writes a
joint flow-and-wage occupation table.

    python analysis/03_worker_flows.py --post-from 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    CPS_FLOW_ANNUAL, CPS_FLOW_MONTHLY, CPS_WAGE_PANEL, FIGURES, PROCESSED,
)
from src.plot_style import (  # noqa: E402
    ACCENT, MUTE, SOFT_RED, add_footer, add_header, apply_chart_style,
    save_chart, style_axes,
)

EXPOSURES = {
    "observed": "ai_exposure",
    "automation": "automation_exposure",
    "augmentation": "augmentation_exposure",
    "theoretical": "theoretical_exposure",
}
MIN_FLOW_SAMPLE = 30
MIN_WAGE_SAMPLE = 10
LEAVE_OUT = [
    "Computer & Mathematical",
    "Business & Financial Operations",
    "Legal",
    "Office & Administrative Support",
    "Sales & Related",
]

apply_chart_style()


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        path for path in (CPS_FLOW_MONTHLY, CPS_FLOW_ANNUAL, CPS_WAGE_PANEL)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing linked CPS outputs: " + ", ".join(str(path) for path in missing)
            + ". Run python -m src.build_cps_flows --fetch-ipums first."
        )
    monthly = pd.read_csv(CPS_FLOW_MONTHLY, dtype={"occ_code": str})
    annual = pd.read_csv(CPS_FLOW_ANNUAL, dtype={"occ_code": str})
    wages = pd.read_csv(CPS_WAGE_PANEL, dtype={"occ_code": str})
    exposure = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})
    for name, frame in (("monthly", monthly), ("annual", annual)):
        if frame.get("synthetic", pd.Series(False, index=frame.index)).astype(bool).any():
            raise RuntimeError(f"Synthetic {name} flow data cannot produce findings")
    return monthly, annual, wages, exposure


def fit_twfe(
    data: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    post_from: int,
    weight: str,
    sample_column: str,
    min_sample: int,
    weighted: bool = True,
    occupation_trends: bool = False,
) -> dict:
    columns = list(dict.fromkeys([
        "occ_code", "year", outcome, exposure, weight, sample_column,
    ]))
    work = data[columns].dropna().copy()
    work = work[
        work[weight].gt(0)
        & work[sample_column].ge(min_sample)
    ]
    occupation_count = int(work["occ_code"].nunique())
    work["exposure_x_post"] = work[exposure] * work["year"].ge(post_from)
    if occupation_trends:
        import statsmodels.formula.api as smf

        work["trend"] = work["year"] - work["year"].min()
        formula = (
            f"{outcome} ~ exposure_x_post + C(occ_code) + C(year) "
            "+ C(occ_code):trend"
        )
        if weighted:
            result = smf.wls(formula, data=work, weights=work[weight]).fit(
                cov_type="cluster", cov_kwds={"groups": work["occ_code"]}
            )
        else:
            result = smf.ols(formula, data=work).fit(
                cov_type="cluster", cov_kwds={"groups": work["occ_code"]}
            )
        from scipy.stats import norm

        coefficient = float(result.params["exposure_x_post"])
        variance = float(result.cov_params().loc[
            "exposure_x_post", "exposure_x_post"
        ])
        standard_error = float(np.sqrt(max(variance, 0)))
        p_value = float(
            2 * norm.sf(abs(coefficient / standard_error))
            if standard_error > 0 else np.nan
        )
        nobs = int(result.nobs)
    else:
        from linearmodels.panel import PanelOLS

        work = work.set_index(["occ_code", "year"])
        result = PanelOLS.from_formula(
            f"{outcome} ~ exposure_x_post + EntityEffects + TimeEffects",
            data=work,
            weights=work[weight] if weighted else None,
            drop_absorbed=True,
        ).fit(cov_type="clustered", cluster_entity=True)
        coefficient = float(result.params["exposure_x_post"])
        standard_error = float(result.std_errors["exposure_x_post"])
        p_value = float(result.pvalues["exposure_x_post"])
        nobs = int(result.nobs)
    return {
        "outcome": outcome,
        "exposure": exposure,
        "coefficient": coefficient,
        "std_error": standard_error,
        "coefficient_x100": 100 * coefficient,
        "std_error_x100": 100 * standard_error,
        "p_value": p_value,
        "nobs": nobs,
        "occupations": occupation_count,
        "post_from": post_from,
        "minimum_cell_sample": min_sample,
        "weighted": weighted,
        "occupation_trends": occupation_trends,
    }


def flow_robustness(flow: pd.DataFrame, post_from: int) -> pd.DataFrame:
    """Run targeted specification, timing, sample, and composition checks."""
    rows = []

    def add(check: str, detail: str, data: pd.DataFrame, **kwargs) -> None:
        result = fit_twfe(
            data,
            outcome=kwargs.pop("outcome", "entry_rate"),
            exposure="ai_exposure",
            post_from=kwargs.pop("fake_post", post_from),
            weight=kwargs.pop("weight", "entry_risk"),
            sample_column=kwargs.pop("sample_column", "current_sample_n"),
            min_sample=kwargs.pop("min_sample", MIN_FLOW_SAMPLE),
            **kwargs,
        )
        rows.append({"check": check, "detail": detail, **result})

    add("baseline", "true entry risk set", flow)
    add(
        "alternative outcome", "entrant share of current employment", flow,
        outcome="entrant_share", weight="current_employment",
    )
    add("occupation trends", "linear trend by SOC", flow, occupation_trends=True)
    add("weighting", "unweighted occupation cells", flow, weighted=False)
    add("sample window", "exclude 2020", flow[flow["year"].ge(2021)])

    years = sorted(flow["year"].unique())
    valid = flow[
        flow["current_sample_n"].ge(MIN_FLOW_SAMPLE)
        & flow["entry_rate"].notna()
    ]
    balanced_codes = (
        valid.groupby("occ_code")["year"].nunique()
        .loc[lambda s: s.eq(len(years))]
        .index
    )
    add("sample", "balanced 2020-2025 occupations", flow[
        flow["occ_code"].isin(balanced_codes)
    ])

    for threshold in (15, 50, 100):
        add("minimum sample", f"at least {threshold} current workers", flow,
            min_sample=threshold)

    pre = flow[flow["year"].lt(post_from)].copy()
    for fake_year in range(int(pre["year"].min()) + 1, post_from):
        add("placebo", f"fake post {fake_year}; pre-2024 only", pre,
            fake_post=fake_year)

    for group in LEAVE_OUT:
        add("leave out", group, flow[flow["soc_major_label"].ne(group)])

    out = pd.DataFrame(rows)
    out["coefficient_pp"] = 100 * out["coefficient"]
    out["std_error_pp"] = 100 * out["std_error"]
    return out


def flow_event_study(
    flow: pd.DataFrame,
    *,
    base_year: int = 2023,
    min_sample: int = MIN_FLOW_SAMPLE,
) -> pd.DataFrame:
    """Estimate continuous exposure-by-year entry effects and pre-trend test."""
    from linearmodels.panel import PanelOLS
    from scipy.stats import chi2

    work = flow[[
        "occ_code", "year", "entry_rate", "ai_exposure", "entry_risk",
        "current_sample_n",
    ]].dropna().copy()
    work = work[
        work["entry_risk"].gt(0)
        & work["current_sample_n"].ge(min_sample)
    ]
    years = sorted(work["year"].unique())
    if base_year not in years:
        raise ValueError(f"Event-study base year {base_year} is unavailable")
    names = []
    for year in years:
        if year == base_year:
            continue
        name = f"exposure_x_{year}"
        work[name] = work["ai_exposure"] * work["year"].eq(year)
        names.append(name)
    indexed = work.set_index(["occ_code", "year"])
    formula = "entry_rate ~ " + " + ".join(names) + " + EntityEffects + TimeEffects"
    result = PanelOLS.from_formula(
        formula,
        data=indexed,
        weights=indexed["entry_risk"],
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    pre_names = [f"exposure_x_{year}" for year in years if year < base_year]
    pre = result.params.loc[pre_names].to_numpy()
    pre_cov = result.cov.loc[pre_names, pre_names].to_numpy()
    statistic = float(pre.T @ np.linalg.pinv(pre_cov) @ pre)
    pretrend_p = float(chi2.sf(statistic, len(pre_names)))

    rows = []
    for year in years:
        if year == base_year:
            rows.append({
                "year": year, "coefficient": 0.0, "std_error": 0.0,
                "p_value": np.nan,
            })
            continue
        name = f"exposure_x_{year}"
        rows.append({
            "year": year,
            "coefficient": float(result.params[name]),
            "std_error": float(result.std_errors[name]),
            "p_value": float(result.pvalues[name]),
        })
    out = pd.DataFrame(rows)
    out["coefficient_pp"] = 100 * out["coefficient"]
    out["std_error_pp"] = 100 * out["std_error"]
    out["base_year"] = base_year
    out["joint_pretrend_p"] = pretrend_p
    out["nobs"] = int(result.nobs)
    out["occupations"] = int(work["occ_code"].nunique())
    return out


def joint_flow_wage_table(
    annual: pd.DataFrame,
    wages: pd.DataFrame,
    exposure: pd.DataFrame,
    post_from: int,
) -> pd.DataFrame:
    flow = annual.copy()
    flow["young_employment"] = (
        flow["current_employment"] / flow["months"].replace(0, np.nan)
    )
    flow["period"] = np.where(flow["year"].ge(post_from), "recent", "baseline")
    flow = (
        flow.groupby(["occ_code", "period"], as_index=False)
        .agg(
            entries=("entries", "sum"),
            entry_risk=("entry_risk", "sum"),
            current_employment=("current_employment", "sum"),
            exits=("exits", "sum"),
            exit_at_risk=("exit_at_risk", "sum"),
            current_sample_n=("current_sample_n", "sum"),
            young_employment=("young_employment", "mean"),
        )
    )
    flow["entry_rate"] = flow["entries"] / flow["entry_risk"]
    flow["entrant_share"] = flow["entries"] / flow["current_employment"]
    flow["exit_rate"] = flow["exits"] / flow["exit_at_risk"]
    flow = flow.pivot(index="occ_code", columns="period", values=[
        "entry_rate", "entrant_share", "exit_rate", "current_sample_n",
        "young_employment"
    ])
    flow.columns = [f"{metric}_{period}" for metric, period in flow.columns]
    flow = flow.reset_index()

    wage = wages[wages["wage_sample_n"].ge(MIN_WAGE_SAMPLE)].copy()
    wage["period"] = np.where(wage["year"].ge(post_from), "recent", "baseline")
    wage["weighted_wage"] = wage["median_weekly_wage"] * wage["wage_sample_n"]
    wage = (
        wage.groupby(["occ_code", "period"], as_index=False)
        .agg(
            weighted_wage=("weighted_wage", "sum"),
            wage_sample_n=("wage_sample_n", "sum"),
        )
    )
    wage["median_weekly_wage"] = wage["weighted_wage"] / wage["wage_sample_n"]
    wage = wage.pivot(
        index="occ_code", columns="period", values=["median_weekly_wage", "wage_sample_n"]
    )
    wage.columns = [f"{metric}_{period}" for metric, period in wage.columns]
    wage = wage.reset_index()

    out = flow.merge(wage, on="occ_code", how="outer").merge(
        exposure[[
            "occ_code", "title", "soc_major_label", "ai_exposure",
            "automation_exposure", "augmentation_exposure",
            "theoretical_exposure",
        ]],
        on="occ_code",
        how="left",
    )
    out["entry_rate_change_pp"] = 100 * (
        out["entry_rate_recent"] - out["entry_rate_baseline"]
    )
    out["exit_rate_change_pp"] = 100 * (
        out["exit_rate_recent"] - out["exit_rate_baseline"]
    )
    out["wage_growth"] = (
        out["median_weekly_wage_recent"] / out["median_weekly_wage_baseline"] - 1
    )
    out["young_employment_growth"] = (
        out["young_employment_recent"] / out["young_employment_baseline"] - 1
    )
    out["employment_wage_interaction"] = (
        out["young_employment_growth"] * out["wage_growth"]
    )
    sufficient = (
        out["current_sample_n_baseline"].ge(MIN_FLOW_SAMPLE)
        & out["current_sample_n_recent"].ge(MIN_FLOW_SAMPLE)
        & out["wage_sample_n_baseline"].ge(MIN_WAGE_SAMPLE)
        & out["wage_sample_n_recent"].ge(MIN_WAGE_SAMPLE)
        & out["young_employment_growth"].notna()
        & out["wage_growth"].notna()
    )
    fewer_juniors = out["young_employment_growth"].lt(0)
    lower_wage = out["wage_growth"].lt(0)
    out["joint_pattern"] = np.select(
        [
            sufficient & fewer_juniors & lower_wage,
            sufficient & fewer_juniors & ~lower_wage,
            sufficient & ~fewer_juniors & lower_wage,
            sufficient & ~fewer_juniors & ~lower_wage,
        ],
        [
            "fewer juniors + lower wages",
            "fewer juniors + higher wages",
            "more juniors + lower wages",
            "more juniors + higher wages",
        ],
        default="insufficient data",
    )
    return out.sort_values("ai_exposure", ascending=False).reset_index(drop=True)


def make_headline(annual: pd.DataFrame, exposure: pd.DataFrame) -> None:
    work = annual.merge(
        exposure[["occ_code", "exposure_q"]], on="occ_code", how="inner"
    )
    work = work[work["exposure_q"].isin(["Q0 (none)", "Q4 (high)"])]
    series = (
        work.groupby(["year", "exposure_q"], as_index=False)
        .agg(
            entries=("entries", "sum"),
            entry_risk=("entry_risk", "sum"),
            exits=("exits", "sum"),
            exit_at_risk=("exit_at_risk", "sum"),
        )
    )
    series["entry_rate"] = 100 * series["entries"] / series["entry_risk"]
    series["exit_rate"] = 100 * series["exits"] / series["exit_at_risk"]
    for outcome in ("entry_rate", "exit_rate"):
        base = series[series["year"].eq(2023)].set_index("exposure_q")[outcome]
        series[f"{outcome}_index"] = 100 * series[outcome] / series["exposure_q"].map(base)
    series.to_csv(PROCESSED / "flow_headline_series.csv", index=False)

    colors = {"Q0 (none)": MUTE, "Q4 (high)": ACCENT}
    labels = {
        "Q0 (none)": "No observed exposure",
        "Q4 (high)": "Highest-exposure quartile",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharex=True)
    for ax, outcome, title in zip(
        axes,
        ["entry_rate_index", "exit_rate_index"],
        ["Entry probability", "Exit probability"],
    ):
        for group in ["Q0 (none)", "Q4 (high)"]:
            selected = series[series["exposure_q"].eq(group)]
            ax.plot(
                selected["year"], selected[outcome], marker="o",
                linewidth=3 if group == "Q4 (high)" else 2.2,
                markersize=7 if group == "Q4 (high)" else 6,
                color=colors[group], label=labels[group],
            )
        if series["year"].max() >= 2024:
            ax.axvspan(2023.5, series["year"].max() + 0.35, color=SOFT_RED, zorder=0)
        ax.axhline(100, color="#4a4d59", linewidth=1, zorder=0)
        ax.set_title(title, loc="left", pad=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Index (2023 = 100)")
        ax.set_xticks(sorted(series["year"].unique()))
        style_axes(ax, grid_axis="y")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, frameon=False, ncol=2, loc="upper left",
        bbox_to_anchor=(0.075, 0.84), fontsize=9.5,
    )
    add_header(
        fig,
        "Are AI-exposed occupations seeing fewer young workers enter?",
        "Linked Basic Monthly CPS workers age 20-29; top quartile versus zero observed exposure",
        left=0.07,
    )
    add_footer(
        fig,
        "Indexed to 2023. Entry risk set: linked age-20-29 respondents not in the occupation last month.",
        left=0.07,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.74, wspace=0.23)
    save_chart(fig, FIGURES / "worker_flows_by_exposure.png")


def main(post_from: int = 2024) -> None:
    _, annual, wages, exposure = _load()
    flow = annual.merge(exposure, on="occ_code", how="inner")
    wage = wages.merge(exposure, on="occ_code", how="inner")
    wage["log_weekly_wage"] = np.log(wage["median_weekly_wage"])

    rows = []
    for label, column in EXPOSURES.items():
        for outcome, weight, sample_column in (
            ("entry_rate", "entry_risk", "current_sample_n"),
            ("entrant_share", "current_employment", "current_sample_n"),
            ("exit_rate", "exit_at_risk", "exit_sample_n"),
        ):
            result = fit_twfe(
                flow,
                outcome=outcome,
                exposure=column,
                post_from=post_from,
                weight=weight,
                sample_column=sample_column,
                min_sample=MIN_FLOW_SAMPLE,
            )
            rows.append({"measure": label, **result})
        result = fit_twfe(
            wage,
            outcome="log_weekly_wage",
            exposure=column,
            post_from=post_from,
            weight="wage_sample_n",
            sample_column="wage_sample_n",
            min_sample=MIN_WAGE_SAMPLE,
        )
        rows.append({"measure": label, **result})

    results = pd.DataFrame(rows)
    results.to_csv(PROCESSED / "flow_wage_regressions.csv", index=False)
    robustness = flow_robustness(flow, post_from)
    robustness.to_csv(PROCESSED / "flow_robustness.csv", index=False)
    event_study = flow_event_study(flow, base_year=post_from - 1)
    event_study.to_csv(PROCESSED / "flow_event_study.csv", index=False)
    joint = joint_flow_wage_table(annual, wages, exposure, post_from)
    joint.to_csv(PROCESSED / "flow_wage_joint.csv", index=False)
    make_headline(annual, exposure)

    display = results[[
        "outcome", "measure", "coefficient_x100", "std_error_x100",
        "p_value", "nobs", "occupations",
    ]]
    print(display.round(4).to_string(index=False))
    print("\nEntry-rate robustness:")
    print(robustness[[
        "check", "detail", "coefficient_pp", "std_error_pp", "p_value",
        "nobs", "occupations",
    ]].round(4).to_string(index=False))
    print("\nEntry-rate event study:")
    print(event_study[[
        "year", "coefficient_pp", "std_error_pp", "p_value",
        "joint_pretrend_p",
    ]].round(4).to_string(index=False))
    print(f"Saved {FIGURES / 'worker_flows_by_exposure.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-from", type=int, default=2024)
    main(**vars(parser.parse_args()))
