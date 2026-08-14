"""
Build the analysis panel.

Combines the enriched exposure table with the CPS occupation x age panel to
produce, per occupation:
  - the age composition of employment in a baseline vs. recent window
  - the change in the young-worker (age 20-29) employment share
  - AI exposure and high-exposure flag

Outputs:
  data/processed/panel_long.csv   (occ_code x year x age_band, with shares)
  data/processed/panel_occ.csv    (one row per occupation; feeds ELSI + regressions)

    python -m src.build_panel --baseline 2022 --recent 2025

Requires data/raw/bls/cps_panel.csv (see download_bls.py). If it is missing the
script errors with instructions rather than inventing numbers.
"""
import argparse
import json
import sys

import pandas as pd

from src.config import PROCESSED, CPS_PANEL_EXPECTED
from src.data_guard import is_synthetic_panel

YOUNG = {"20-24", "25-29"}
AGE_BANDS = YOUNG | {"30-39", "40-54", "55+"}


def _load_cps() -> pd.DataFrame:
    if not CPS_PANEL_EXPECTED.exists():
        sys.exit(
            f"\nMissing {CPS_PANEL_EXPECTED}.\n"
            "Build the CPS occupation x age panel first "
            "(see src/download_bls.py, or run tools/make_synthetic_cps.py to "
            "smoke-test the pipeline with clearly-labelled FAKE data).\n")
    cps = pd.read_csv(CPS_PANEL_EXPECTED, dtype={"occ_code": str})
    need = {"year", "occ_code", "age_band", "employed"}
    missing = need - set(cps.columns)
    if missing:
        sys.exit(f"cps_panel.csv missing columns: {missing}")
    if cps.duplicated(["year", "occ_code", "age_band"]).any():
        sys.exit("cps_panel.csv has duplicate year/occupation/age-band rows")
    unexpected = set(cps["age_band"].dropna().unique()) - AGE_BANDS
    if unexpected:
        sys.exit(f"cps_panel.csv has unexpected age bands: {sorted(unexpected)}")
    cps["year"] = pd.to_numeric(cps["year"], errors="raise").astype(int)
    cps["employed"] = pd.to_numeric(cps["employed"], errors="raise")
    if cps["employed"].isna().any() or (cps["employed"] < 0).any():
        sys.exit("cps_panel.csv employed must be non-negative and non-missing")
    return cps


def young_worker_panel(cps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate age-band employment to occupation-year young-worker shares."""
    work = cps.copy()
    work["young_employed"] = work["employed"].where(
        work["age_band"].isin(YOUNG), 0
    )
    out = (work.groupby(["year", "occ_code"], as_index=False)
               .agg(employed=("employed", "sum"),
                    young_employed=("young_employed", "sum")))
    if (out["employed"] <= 0).any():
        bad = out.loc[out["employed"] <= 0, ["year", "occ_code"]]
        raise ValueError(f"occupation-year cells must have positive employment: {bad.head().to_dict('records')}")
    out["young_share"] = out["young_employed"] / out["employed"]
    return out


def _check_partial_year(recent: int, allow_partial_year: bool) -> None:
    metadata_path = CPS_PANEL_EXPECTED.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    partial = set(metadata.get("partial_years", []))
    if recent in partial and not allow_partial_year:
        raise ValueError(
            f"Recent year {recent} has incomplete month coverage. Use a complete "
            "year or pass --allow-partial-year and disclose the seasonal-comparison risk."
        )


def build(baseline: int, recent: int, allow_partial_year: bool = False) -> pd.DataFrame:
    exp = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})
    cps = _load_cps()
    _check_partial_year(recent, allow_partial_year)
    available_years = set(cps["year"].unique())
    missing_years = {baseline, recent} - available_years
    if missing_years:
        raise ValueError(
            f"Requested years not present in CPS panel: {sorted(missing_years)}"
        )

    # age shares within occupation-year
    tot = cps.groupby(["year", "occ_code"])["employed"].transform("sum")
    cps["age_share"] = cps["employed"] / tot
    cps.to_csv(PROCESSED / "panel_long.csv", index=False)

    # young-worker share per occupation-year
    young = young_worker_panel(cps)

    b = young[young.year == baseline][["occ_code", "young_share"]].rename(
        columns={"young_share": "young_share_base"})
    r = young[young.year == recent][["occ_code", "young_share"]].rename(
        columns={"young_share": "young_share_recent"})
    occ = (exp.merge(b, on="occ_code", how="left")
              .merge(r, on="occ_code", how="left"))
    occ["young_share_change"] = occ["young_share_recent"] - occ["young_share_base"]

    # occupation size (recent total employment) for weighting
    size = (cps[cps.year == recent].groupby("occ_code")["employed"].sum()
            .rename("employment_recent").reset_index())
    occ = occ.merge(size, on="occ_code", how="left")

    out = PROCESSED / "panel_occ.csv"
    occ.to_csv(out, index=False)
    metadata = {
        "baseline_year": baseline,
        "recent_year": recent,
        "young_age_definition": "20-29",
        "synthetic": is_synthetic_panel(),
        "matched_occupations": int(occ["young_share_change"].notna().sum()),
        "exposure_occupations": int(len(exp)),
    }
    (PROCESSED / "panel_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}  ({occ['young_share_change'].notna().sum()} occ with change)")
    return occ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=int, default=2022)
    ap.add_argument("--recent", type=int, default=2025)
    ap.add_argument("--allow-partial-year", action="store_true")
    build(**vars(ap.parse_args()))
