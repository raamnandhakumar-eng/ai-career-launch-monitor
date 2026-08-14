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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED  # noqa: E402
from src.data_guard import guard_publication  # noqa: E402

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
    df = df.set_index(["occ_code", "year"])

    # --- baseline DiD ---
    m = PanelOLS.from_formula(
        "young_share ~ exp_x_post + EntityEffects + TimeEffects", data=df)
    res = m.fit(cov_type="clustered", cluster_entity=True)
    print("=" * 66)
    print("BASELINE: young_share ~ exposure x post + occ FE + year FE")
    print("=" * 66)
    print(res.params[["exp_x_post"]].round(4).to_string())
    print(res.std_errors[["exp_x_post"]].round(4).to_string())
    print(f"tstat(exp_x_post) = {res.tstats['exp_x_post']:.2f}   "
          f"pval = {res.pvalues['exp_x_post']:.4f}   n = {res.nobs}")
    suffix = "_SYNTHETIC" if synthetic else ""
    pd.DataFrame({
        "term": ["exposure_x_post"],
        "coefficient": [res.params["exp_x_post"]],
        "std_error": [res.std_errors["exp_x_post"]],
        "t_stat": [res.tstats["exp_x_post"]],
        "p_value": [res.pvalues["exp_x_post"]],
        "nobs": [res.nobs],
        "post_from": [post_from],
    }).to_csv(PROCESSED / f"regression_baseline{suffix}.csv", index=False)

    # --- event study ---
    d = df.reset_index()
    for yr in sorted(d["year"].unique()):
        if yr == base_year:
            continue
        d[f"exp_x_{yr}"] = d["ai_exposure"] * (d["year"] == yr).astype(int)
    d = d.set_index(["occ_code", "year"])
    terms = [c for c in d.columns if c.startswith("exp_x_") and c != "exp_x_post"]
    es = PanelOLS.from_formula(
        "young_share ~ " + " + ".join(terms) + " + EntityEffects + TimeEffects",
        data=d).fit(cov_type="clustered", cluster_entity=True)
    print("\n" + "=" * 66)
    print(f"EVENT STUDY (base year {base_year} omitted): exposure x year_k")
    print("=" * 66)
    out = pd.DataFrame({"coef": es.params[terms], "se": es.std_errors[terms]})
    out.index = [t.replace("exp_x_", "") for t in out.index]
    print(out.round(4).to_string())
    out.rename_axis("year").reset_index().to_csv(
        PROCESSED / f"event_study{suffix}.csv", index=False)
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
