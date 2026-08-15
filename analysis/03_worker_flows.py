"""Make linked worker flows the headline labor-market outcome.

Runs annual occupation fixed-effects models for entry and exit rates using the
Basic Monthly CPS rotation panel. It also estimates young-worker weekly wage
models and writes a joint entry-and-wage occupation table.

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
) -> dict:
    from linearmodels.panel import PanelOLS

    columns = list(dict.fromkeys([
        "occ_code", "year", outcome, exposure, weight, sample_column,
    ]))
    work = data[columns].dropna().copy()
    work = work[
        work[weight].gt(0)
        & work[sample_column].ge(min_sample)
    ]
    work["exposure_x_post"] = work[exposure] * work["year"].ge(post_from)
    work = work.set_index(["occ_code", "year"])
    result = PanelOLS.from_formula(
        f"{outcome} ~ exposure_x_post + EntityEffects + TimeEffects",
        data=work,
        weights=work[weight],
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True)
    coefficient = float(result.params["exposure_x_post"])
    standard_error = float(result.std_errors["exposure_x_post"])
    return {
        "outcome": outcome,
        "exposure": exposure,
        "coefficient": coefficient,
        "std_error": standard_error,
        "coefficient_x100": 100 * coefficient,
        "std_error_x100": 100 * standard_error,
        "p_value": float(result.pvalues["exposure_x_post"]),
        "nobs": int(result.nobs),
        "occupations": int(work.index.get_level_values("occ_code").nunique()),
        "post_from": post_from,
        "minimum_cell_sample": min_sample,
    }


def joint_flow_wage_table(
    annual: pd.DataFrame,
    wages: pd.DataFrame,
    exposure: pd.DataFrame,
    post_from: int,
) -> pd.DataFrame:
    flow = annual.copy()
    flow["young_employment"] = flow["entry_at_risk"] / flow["months"].replace(0, np.nan)
    flow["period"] = np.where(flow["year"].ge(post_from), "recent", "baseline")
    flow = (
        flow.groupby(["occ_code", "period"], as_index=False)
        .agg(
            entries=("entries", "sum"),
            entry_at_risk=("entry_at_risk", "sum"),
            exits=("exits", "sum"),
            exit_at_risk=("exit_at_risk", "sum"),
            entry_sample_n=("entry_sample_n", "sum"),
            young_employment=("young_employment", "mean"),
        )
    )
    flow["entry_rate"] = flow["entries"] / flow["entry_at_risk"]
    flow["exit_rate"] = flow["exits"] / flow["exit_at_risk"]
    flow = flow.pivot(index="occ_code", columns="period", values=[
        "entry_rate", "exit_rate", "entry_sample_n", "young_employment"
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
        out["entry_sample_n_baseline"].ge(MIN_FLOW_SAMPLE)
        & out["entry_sample_n_recent"].ge(MIN_FLOW_SAMPLE)
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
            "fewer entrants + lower wages",
            "fewer entrants + higher wages",
            "more entrants + lower wages",
            "more entrants + higher wages",
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
            entry_at_risk=("entry_at_risk", "sum"),
            exits=("exits", "sum"),
            exit_at_risk=("exit_at_risk", "sum"),
        )
    )
    series["entry_rate"] = 100 * series["entries"] / series["entry_at_risk"]
    series["exit_rate"] = 100 * series["exits"] / series["exit_at_risk"]
    series.to_csv(PROCESSED / "flow_headline_series.csv", index=False)

    colors = {"Q0 (none)": MUTE, "Q4 (high)": ACCENT}
    labels = {
        "Q0 (none)": "No observed exposure",
        "Q4 (high)": "Highest-exposure quartile",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharex=True)
    for ax, outcome, title in zip(
        axes,
        ["entry_rate", "exit_rate"],
        ["Entry rate", "Exit rate"],
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
        ax.set_title(title, loc="left", pad=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent of linked young workers")
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
        "Descriptive worker-month transition rates. CPSIDV links; PANLWT weights.",
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
            ("entry_rate", "entry_at_risk", "entry_sample_n"),
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
    joint = joint_flow_wage_table(annual, wages, exposure, post_from)
    joint.to_csv(PROCESSED / "flow_wage_joint.csv", index=False)
    make_headline(annual, exposure)

    display = results[[
        "outcome", "measure", "coefficient_x100", "std_error_x100",
        "p_value", "nobs", "occupations",
    ]]
    print(display.round(4).to_string(index=False))
    print(f"Saved {FIGURES / 'worker_flows_by_exposure.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-from", type=int, default=2024)
    main(**vars(parser.parse_args()))
