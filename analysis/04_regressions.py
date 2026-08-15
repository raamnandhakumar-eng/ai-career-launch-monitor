"""
04 - Early-warning panel design.

Tests whether the young-worker employment share fell faster in more AI-exposed
occupations, net of occupation and year fixed effects.

Baseline (difference-in-differences flavour):
    young_share_{o,t} = beta * (AIexposure_o x Post_t)
                        + occ FE + year FE + e_{o,t}
    SEs clustered by occupation.

Event study:
    young_share_{o,t} = sum_k gamma_k * (AIexposure_o x 1[t=k]) + occ FE + year FE
    with the pre-AI base year omitted, to check pre-trends.

CAUSAL CAUTION: AI adoption has no single clean onset and broad macro trends can
confound this. Read beta as an association under an early-warning design, not a
treatment effect. Growth in remote work, post-pandemic hiring normalisation, and
interest-rate-driven tech-hiring cycles are live confounders; the event-study
pre-trend is the first check, not a proof.

    python analysis/04_regressions.py --post-from 2024
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIGURES, PROCESSED  # noqa: E402
from src.data_guard import guard_publication  # noqa: E402
from src.plot_style import (  # noqa: E402
    ACCENT, MUTE, SOFT_RED, add_footer, add_header, apply_chart_style,
    save_chart, style_axes,
)

apply_chart_style()

YOUNG = {"20-24", "25-29"}


def _young_panel() -> pd.DataFrame:
    long = pd.read_csv(PROCESSED / "panel_long.csv", dtype={"occ_code": str})
    exp = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})
    long["young_employed"] = long["employed"].where(
        long["age_band"].isin(YOUNG), 0
    )
    y = (long.groupby(["year", "occ_code"], as_index=False)
             .agg(employed=("employed", "sum"),
                  young_employed=("young_employed", "sum")))
    if (y["employed"] <= 0).any():
        raise ValueError("All occupation-year cells must have positive employment")
    y["young_share"] = y["young_employed"] / y["employed"]
    y["young_per_100k"] = (
        100_000 * y["young_employed"]
        / y.groupby("year")["young_employed"].transform("sum")
    )
    return y.merge(exp[["occ_code", "ai_exposure", "soc_major_label"]], on="occ_code")


def run(post_from: int, base_year: int | None = None,
        allow_synthetic: bool = False):
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        sys.exit("pip install linearmodels  (needed for absorbing FE)")

    synthetic = guard_publication(allow_synthetic)
    df = _young_panel()
    years = sorted(df["year"].unique())
    if post_from not in years:
        raise ValueError(f"post-from year {post_from} is not in the panel: {years}")
    base_year = post_from - 1 if base_year is None else base_year
    if base_year not in years:
        raise ValueError(f"base year {base_year} is not in the panel: {years}")
    df["post"] = (df["year"] >= post_from).astype(int)
    df["exp_x_post"] = df["ai_exposure"] * df["post"]

    baseline_rows = []
    outcomes = {
        "young_share": "Age 20-29 share within occupation",
        "young_per_100k": "Young workers in occupation per 100,000 young workers",
    }
    for outcome, label in outcomes.items():
        indexed = df.set_index(["occ_code", "year"])
        model = PanelOLS.from_formula(
            f"{outcome} ~ exp_x_post + EntityEffects + TimeEffects", data=indexed
        )
        res = model.fit(cov_type="clustered", cluster_entity=True)
        print("=" * 72)
        print(f"BASELINE: {label}")
        print("=" * 72)
        print(f"coef = {res.params['exp_x_post']:.4f}   "
              f"SE = {res.std_errors['exp_x_post']:.4f}   "
              f"p = {res.pvalues['exp_x_post']:.4f}   n = {res.nobs}")
        baseline_rows.append({
            "outcome": outcome,
            "term": "exposure_x_post",
            "coefficient": res.params["exp_x_post"],
            "std_error": res.std_errors["exp_x_post"],
            "t_stat": res.tstats["exp_x_post"],
            "p_value": res.pvalues["exp_x_post"],
            "nobs": res.nobs,
            "post_from": post_from,
        })

    suffix = "_SYNTHETIC" if synthetic else ""
    pd.DataFrame(baseline_rows).to_csv(
        PROCESSED / f"regression_baseline{suffix}.csv", index=False
    )

    # --- event study ---
    event_rows = []
    for outcome, label in outcomes.items():
        d = df.copy()
        terms = []
        for yr in sorted(d["year"].unique()):
            if yr == base_year:
                continue
            term = f"exp_x_{yr}"
            d[term] = d["ai_exposure"] * (d["year"] == yr).astype(int)
            terms.append(term)
        d = d.set_index(["occ_code", "year"])
        es = PanelOLS.from_formula(
            f"{outcome} ~ " + " + ".join(terms)
            + " + EntityEffects + TimeEffects", data=d
        ).fit(cov_type="clustered", cluster_entity=True)
        print("\n" + "=" * 72)
        print(f"EVENT STUDY: {label} (base year {base_year})")
        print("=" * 72)
        for term in terms:
            event_rows.append({
                "outcome": outcome,
                "year": int(term.replace("exp_x_", "")),
                "coefficient": es.params[term],
                "std_error": es.std_errors[term],
                "ci_low": es.params[term] - 1.96 * es.std_errors[term],
                "ci_high": es.params[term] + 1.96 * es.std_errors[term],
                "base_year": base_year,
            })
        print(pd.DataFrame(event_rows).query("outcome == @outcome")
              [["year", "coefficient", "std_error"]].round(4).to_string(index=False))

    event = pd.DataFrame(event_rows)
    event.to_csv(PROCESSED / f"event_study{suffix}.csv", index=False)

    # The continuous event study is the main design diagnostic.
    plot = event[event["outcome"] == "young_share"].copy()
    base = pd.DataFrame({
        "year": [base_year], "coefficient": [0.0],
        "ci_low": [0.0], "ci_high": [0.0],
    })
    plot = pd.concat([plot, base], ignore_index=True).sort_values("year")
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=150)
    coef = plot["coefficient"] * 100
    low = plot["ci_low"] * 100
    high = plot["ci_high"] * 100
    ax.errorbar(plot["year"], coef, yerr=[coef - low, high - coef],
                fmt="o-", color=ACCENT, linewidth=2.2, capsize=4,
                markersize=6)
    ax.axhline(0, color=MUTE, linewidth=1)
    if max(years) >= 2024:
        ax.axvspan(2023.5, max(years) + 0.35, color=SOFT_RED, zorder=0)
    ax.text(base_year, 0.97, f"{base_year} reference",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            color=MUTE, fontsize=9)
    ax.set_xticks(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient (percentage points)")
    style_axes(ax, grid_axis="y")
    if synthetic:
        ax.text(0.5, 0.5, "SYNTHETIC DATA\nNOT A FINDING",
                transform=ax.transAxes, fontsize=28, color="red", alpha=0.28,
                ha="center", va="center", rotation=18, fontweight="bold")
    add_header(
        fig,
        "No clean post-2023 break",
        "Continuous exposure event study; occupation and year fixed effects",
        left=0.1,
    )
    add_footer(
        fig,
        "95% confidence intervals; standard errors clustered by occupation.",
        left=0.1,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.79)
    save_chart(fig, FIGURES / f"event_study{suffix}.png")
    print("\nA flat/near-zero pre-period (years before AI ramp) supports the "
          "design; a sloped pre-period warns of confounding trends.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-from", type=int, default=2024,
                    help="first year treated as the AI-adoption era")
    ap.add_argument("--base-year", type=int, default=None,
                    help="omitted event-study year (default: post-from minus one)")
    ap.add_argument("--allow-synthetic", action="store_true")
    run(**vars(ap.parse_args()))
