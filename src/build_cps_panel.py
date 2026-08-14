"""Build the occupation-by-age CPS panel from an IPUMS CPS extract.

The script accepts CSV, compressed CSV, or Stata input. It supports either a
direct SOC column or a Census occupation column plus a crosswalk.

Example for basic-monthly CPS data::

    python -m src.build_cps_panel data/raw/bls/cps_extract.csv.gz \
        --census-occ-column OCC2010 --frequency monthly

Expected default columns are YEAR, MONTH, AGE, EMPSTAT, WTFINL, and OCC2010.
EMPSTAT 10 and 12 are treated as employed. Override every field from the CLI
when an extract uses different names or codes.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CPS_PANEL_EXPECTED
from src.occupation_crosswalk import LOCAL_XWALK


AGE_LABELS = ["20-24", "25-29", "30-39", "40-54", "55+"]


def age_band(age: pd.Series) -> pd.Series:
    """Map numeric ages 20+ into the five analysis bands."""
    return pd.cut(
        pd.to_numeric(age, errors="coerce"),
        bins=[19, 24, 29, 39, 54, np.inf],
        labels=AGE_LABELS,
    )


def normalize_soc(value: object) -> str | None:
    """Normalize values such as 151252, 15-1252.00, or 15-1252 to NN-NNNN."""
    if pd.isna(value):
        return None
    raw = str(value).strip()
    digits = re.sub(r"\D", "", raw.split(".")[0])
    if not digits:
        return None
    if len(digits) < 6:
        digits = digits.zfill(6)
    if len(digits) != 6:
        return None
    return f"{digits[:2]}-{digits[2:]}"


def normalize_census_occ(value: object) -> str | None:
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value).split(".")[0])
    return digits.zfill(4) if digits else None


def _read_extract(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".dta"):
        return pd.read_stata(path, convert_categoricals=False)
    if suffixes.endswith((".csv", ".csv.gz", ".csv.zip", ".txt", ".txt.gz")):
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Input must be CSV, CSV.GZ, CSV.ZIP, TXT, TXT.GZ, or DTA")


def _crosswalk_map(path: Path) -> dict[str, str]:
    x = pd.read_csv(path, dtype=str)
    required = {"occ_census", "occ_code"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"Crosswalk missing columns: {sorted(missing)}")
    x = x.assign(
        occ_census=x["occ_census"].map(normalize_census_occ),
        occ_code=x["occ_code"].map(normalize_soc),
    ).dropna(subset=["occ_census", "occ_code"])
    # A source crosswalk can repeat mappings. The modal detailed SOC is a
    # transparent first-pass collapse; the limitation is recorded in metadata.
    return (x.groupby("occ_census")["occ_code"]
             .agg(lambda s: s.value_counts().sort_index().idxmax())
             .to_dict())


def transform(
    raw: pd.DataFrame,
    *,
    year_column: str = "YEAR",
    month_column: str = "MONTH",
    age_column: str = "AGE",
    employment_column: str = "EMPSTAT",
    weight_column: str = "WTFINL",
    employed_codes: set[int] | None = None,
    census_occ_column: str | None = "OCC2010",
    soc_column: str | None = None,
    frequency: str = "monthly",
    crosswalk_path: Path = LOCAL_XWALK,
    min_match_rate: float = 0.80,
) -> tuple[pd.DataFrame, dict]:
    """Validate, map, and aggregate CPS person records."""
    employed_codes = employed_codes or {10, 12}
    raw = raw.rename(columns={c: str(c).upper() for c in raw.columns})
    year_column, month_column = year_column.upper(), month_column.upper()
    age_column, employment_column = age_column.upper(), employment_column.upper()
    weight_column = weight_column.upper()
    census_occ_column = census_occ_column.upper() if census_occ_column else None
    soc_column = soc_column.upper() if soc_column else None

    if bool(soc_column) == bool(census_occ_column):
        raise ValueError("Set exactly one of soc_column or census_occ_column")

    required = {year_column, age_column, employment_column, weight_column}
    if frequency == "monthly":
        required.add(month_column)
    required.add(soc_column or census_occ_column or "")
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CPS extract missing columns: {sorted(missing)}")
    work = raw.loc[
        pd.to_numeric(raw[employment_column], errors="coerce").isin(employed_codes)
    ].copy()
    work["year"] = pd.to_numeric(work[year_column], errors="coerce")
    work["weight"] = pd.to_numeric(work[weight_column], errors="coerce")
    work["age_band"] = age_band(work[age_column])
    work = work.dropna(subset=["year", "weight", "age_band"])
    work = work[work["weight"] > 0]
    work["year"] = work["year"].astype(int)

    if soc_column:
        work["occ_code"] = work[soc_column].map(normalize_soc)
        mapping_method = "direct SOC normalization"
    else:
        mapping = _crosswalk_map(crosswalk_path)
        work["occ_code"] = work[census_occ_column].map(normalize_census_occ).map(mapping)
        mapping_method = "modal detailed SOC per Census occupation code"

    match_rate = float(work["occ_code"].notna().mean()) if len(work) else 0.0
    if match_rate < min_match_rate:
        raise ValueError(
            f"Occupation match rate {match_rate:.1%} is below the required "
            f"{min_match_rate:.1%}. Check the occupation vintage/crosswalk."
        )
    work = work.dropna(subset=["occ_code"])

    coverage: dict[str, list[int]] = {}
    if frequency == "monthly":
        work["month"] = pd.to_numeric(work[month_column], errors="coerce")
        work = work.dropna(subset=["month"])
        work["month"] = work["month"].astype(int)
        monthly = (work.groupby(["year", "month", "occ_code", "age_band"],
                                observed=True, as_index=False)["weight"].sum())
        months_per_year = work.groupby("year")["month"].nunique()
        panel = (monthly.groupby(["year", "occ_code", "age_band"],
                                 observed=True, as_index=False)["weight"].sum())
        panel["weight"] = panel["weight"] / panel["year"].map(months_per_year)
        coverage = {
            str(int(year)): sorted(group["month"].unique().astype(int).tolist())
            for year, group in work.groupby("year")
        }
    elif frequency == "annual":
        panel = (work.groupby(["year", "occ_code", "age_band"],
                              observed=True, as_index=False)["weight"].sum())
    else:
        raise ValueError("frequency must be 'monthly' or 'annual'")

    panel = panel.rename(columns={"weight": "employed"})
    panel["age_band"] = panel["age_band"].astype(str)
    panel["synthetic"] = False
    panel = panel.sort_values(["year", "occ_code", "age_band"]).reset_index(drop=True)
    metadata = {
        "synthetic": False,
        "created": date.today().isoformat(),
        "frequency": frequency,
        "employment_codes": sorted(employed_codes),
        "occupation_mapping": mapping_method,
        "occupation_match_rate": match_rate,
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "month_coverage": coverage,
        "partial_years": [
            int(year) for year, months in coverage.items() if len(months) < 12
        ],
        "note": "Monthly files are converted to annual-average employment by averaging monthly weighted stocks.",
    }
    return panel, metadata


def build(input_path: Path, output_path: Path = CPS_PANEL_EXPECTED, **kwargs) -> pd.DataFrame:
    raw = _read_extract(input_path)
    panel, metadata = transform(raw, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    metadata["source_file"] = input_path.name
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({len(panel):,} rows)")
    print(f"Occupation match rate: {metadata['occupation_match_rate']:.1%}")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-path", type=Path, default=CPS_PANEL_EXPECTED)
    parser.add_argument("--year-column", default="YEAR")
    parser.add_argument("--month-column", default="MONTH")
    parser.add_argument("--age-column", default="AGE")
    parser.add_argument("--employment-column", default="EMPSTAT")
    parser.add_argument("--weight-column", default="WTFINL")
    parser.add_argument("--employed-codes", default="10,12")
    parser.add_argument("--census-occ-column", default="OCC2010")
    parser.add_argument("--soc-column", default=None)
    parser.add_argument("--frequency", choices=["monthly", "annual"], default="monthly")
    parser.add_argument("--crosswalk-path", type=Path, default=LOCAL_XWALK)
    parser.add_argument("--min-match-rate", type=float, default=0.80)
    args = vars(parser.parse_args())
    args["employed_codes"] = {int(x) for x in args["employed_codes"].split(",")}
    if args["soc_column"]:
        args["census_occ_column"] = None
    input_path = args.pop("input_path")
    output_path = args.pop("output_path")
    build(input_path, output_path, **args)


if __name__ == "__main__":
    main()
