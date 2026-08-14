"""
Build the enriched occupation-level exposure table.

Input : data/raw/anthropic/job_exposure.csv   (real Anthropic AEI data)
Output: data/processed/exposure.csv

Adds the SOC major-group code/label and an exposure quartile so downstream
steps (panel build, ELSI, regressions) have a clean occupation spine.

    python -m src.build_exposure
"""
import json

import pandas as pd

from src.config import RAW_ANTHROPIC, PROCESSED, soc_major, soc_major_label


def build() -> pd.DataFrame:
    df = pd.read_csv(RAW_ANTHROPIC / "job_exposure.csv", dtype={"occ_code": str})
    required = {"occ_code", "title", "observed_exposure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"job_exposure.csv missing columns: {sorted(missing)}")
    if df["occ_code"].duplicated().any():
        raise ValueError("job_exposure.csv contains duplicate occupation codes")
    if not df["occ_code"].str.fullmatch(r"\d{2}-\d{4}").all():
        raise ValueError("occ_code must use detailed SOC form NN-NNNN")
    if not df["observed_exposure"].between(0, 1).all():
        raise ValueError("observed_exposure must be between 0 and 1")

    df = df.rename(columns={"observed_exposure": "ai_exposure"})
    df["soc_major"] = df["occ_code"].map(soc_major)
    df["soc_major_label"] = df["occ_code"].map(soc_major_label)

    # Exposure quartile among occupations with any observed usage. Occupations
    # with exactly zero observed exposure form their own "Q0 (none)" bucket so
    # the ~54% physical-world tail doesn't swamp the quartile cuts.
    nonzero = df["ai_exposure"] > 0
    df["exposure_q"] = "Q0 (none)"
    # Rank first so tied exposure values cannot collapse quantile boundaries.
    df.loc[nonzero, "exposure_q"] = pd.qcut(
        df.loc[nonzero, "ai_exposure"].rank(method="first"), 4,
        labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
    ).astype(str)

    # A simple high-exposure flag (top quartile of the full distribution) used
    # by the early-warning design and the ELSI.
    df["high_exposure"] = (df["ai_exposure"] >= df["ai_exposure"].quantile(0.75)).astype(int)

    out = PROCESSED / "exposure.csv"
    df.to_csv(out, index=False)
    summary = {
        "occupations": int(len(df)),
        "mean_exposure": float(df["ai_exposure"].mean()),
        "median_exposure": float(df["ai_exposure"].median()),
        "share_zero_exposure": float((df["ai_exposure"] == 0).mean()),
        "high_exposure_cutoff": float(df["ai_exposure"].quantile(0.75)),
    }
    (PROCESSED / "exposure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}  ({len(df)} occupations)")
    return df


if __name__ == "__main__":
    build()
