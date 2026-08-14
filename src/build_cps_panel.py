"""Build the real occupation-by-age CPS panel.

Two reproducible routes are supported:

1. Submit and download an IPUMS CPS extract through the API::

       python -m src.build_cps_panel --fetch-ipums \
           --sample asec --start 2020 --end 2025

2. Convert an existing IPUMS CSV, compressed CSV, or Stata extract::

       python -m src.build_cps_panel data/raw/bls/cps_extract.csv.gz \
           --census-occ-column OCC --frequency monthly

The automated route uses contemporary ``OCC`` codes and is restricted to
2020 onward so they can be mapped through the repo's 2018 Census OCC-to-SOC
crosswalk. EMPSTAT 10 and 12 are treated as employed. ASEC extracts use
ASECWT; basic-monthly extracts use WTFINL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Collection, Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CPS_PANEL_EXPECTED
from src.occupation_crosswalk import LOCAL_XWALK


AGE_LABELS = ["20-24", "25-29", "30-39", "40-54", "55+"]
EMPLOYED_CODES = {10, 12}
IPUMS_COLLECTION = "cps"
IPUMS_BASE_VARIABLES = ["YEAR", "MONTH", "AGE", "OCC", "EMPSTAT"]


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


def candidate_sample_ids(start: int, end: int, sample: str) -> list[str]:
    """Return the IPUMS CPS sample IDs implied by an analysis window."""
    if start > end:
        raise ValueError("start must not be later than end")
    if start < 2020:
        raise ValueError(
            "The automated OCC-to-2018-SOC route requires 2020 or later. "
            "Use a vintage-matched occupation variable and crosswalk for earlier years."
        )
    if sample == "asec":
        return [f"cps{year}_03s" for year in range(start, end + 1)]
    if sample == "basic":
        return [
            f"cps{year}_{month:02d}b"
            for year in range(start, end + 1)
            for month in range(1, 13)
        ]
    raise ValueError("sample must be 'asec' or 'basic'")


def available_sample_ids(
    start: int,
    end: int,
    sample: str,
    available: Collection[str] | Mapping[str, object],
) -> list[str]:
    """Filter requested IDs against the samples currently published by IPUMS."""
    published = set(available.keys() if isinstance(available, Mapping) else available)
    requested = candidate_sample_ids(start, end, sample)
    selected = [sample_id for sample_id in requested if sample_id in published]
    missing = [sample_id for sample_id in requested if sample_id not in published]

    if sample == "asec" and missing:
        raise ValueError(
            "Requested ASEC samples are not published: " + ", ".join(missing)
        )
    if sample == "basic":
        for year in range(start, end + 1):
            prefix = f"cps{year}_"
            if not any(sample_id.startswith(prefix) for sample_id in selected):
                raise ValueError(f"No published basic-monthly sample found for {year}")
    if not selected:
        raise ValueError("No requested IPUMS CPS samples are currently published")
    return selected


def ipums_extract_spec(start: int, end: int, sample: str) -> dict:
    """Return a non-sensitive, serializable extract specification."""
    samples = candidate_sample_ids(start, end, sample)
    weight = "ASECWT" if sample == "asec" else "WTFINL"
    return {
        "collection": IPUMS_COLLECTION,
        "sample_type": sample,
        "start": start,
        "end": end,
        "samples": samples,
        "variables": [*IPUMS_BASE_VARIABLES, weight],
        "weight": weight,
        "frequency": "annual" if sample == "asec" else "monthly",
        "occupation_variable": "OCC",
    }


def _fetch_ipums(start: int, end: int, sample: str) -> tuple[pd.DataFrame, dict]:
    """Submit, wait for, download, and read a current IPUMS CPS extract."""
    try:
        from ipumspy import IpumsApiClient, MicrodataExtract, readers
    except ImportError as exc:
        raise RuntimeError(
            "Install project requirements (including ipumspy) before fetching."
        ) from exc

    api_key = os.environ.get("IPUMS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set IPUMS_API_KEY to an API key from https://account.ipums.org."
        )

    spec = ipums_extract_spec(start, end, sample)
    client = IpumsApiClient(api_key)
    published = client.get_all_sample_info(IPUMS_COLLECTION)
    selected = available_sample_ids(start, end, sample, published)
    missing = [sample_id for sample_id in spec["samples"] if sample_id not in selected]
    spec["samples"] = selected
    spec["unavailable_samples"] = missing

    extract = MicrodataExtract(
        collection=IPUMS_COLLECTION,
        samples=selected,
        variables=spec["variables"],
        description=f"ai-career-launch-monitor {sample} {start}-{end}",
    )
    print(
        f"Submitting IPUMS CPS {sample.upper()} extract "
        f"({len(selected)} published sample(s))"
    )
    if missing:
        print("Skipping unavailable samples: " + ", ".join(missing))
    client.submit_extract(extract)
    client.wait_for_extract(extract)

    with tempfile.TemporaryDirectory() as temp_dir:
        client.download_extract(extract, download_dir=temp_dir)
        files = list(Path(temp_dir).rglob("*"))
        ddi_paths = [path for path in files if path.suffix.lower() == ".xml"]
        data_paths = [
            path for path in files
            if "".join(path.suffixes).lower().endswith((".dat.gz", ".csv.gz"))
        ]
        if len(ddi_paths) != 1 or len(data_paths) != 1:
            raise RuntimeError(
                "Expected one IPUMS DDI file and one microdata file; "
                f"found {len(ddi_paths)} DDI and {len(data_paths)} data files."
            )
        ddi = readers.read_ipums_ddi(ddi_paths[0])
        raw = readers.read_microdata(ddi, data_paths[0])
    return raw, spec


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
    census_occ_column: str | None = "OCC",
    soc_column: str | None = None,
    frequency: str = "monthly",
    crosswalk_path: Path = LOCAL_XWALK,
    min_match_rate: float = 0.80,
) -> tuple[pd.DataFrame, dict]:
    """Validate, map, and aggregate CPS person records."""
    employed_codes = employed_codes or EMPLOYED_CODES
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

    using_repo_crosswalk = Path(crosswalk_path).resolve() == LOCAL_XWALK.resolve()
    if census_occ_column and using_repo_crosswalk:
        if census_occ_column == "OCC2010":
            raise ValueError(
                "OCC2010 is a 2010-basis variable and cannot use the repo's "
                "2018 Census crosswalk. Use OCC for 2020+ or supply a matched crosswalk."
            )
        if census_occ_column == "OCC" and (work["year"] < 2020).any():
            raise ValueError(
                "The repo's 2018 Census crosswalk cannot map pre-2020 OCC values. "
                "Restrict to 2020+ or supply a vintage-matched crosswalk."
            )

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
        "age_universe": "employed people age 20 or older",
        "note": (
            "Monthly files are converted to annual-average employment by averaging "
            "monthly weighted stocks."
            if frequency == "monthly"
            else "ASEC person weights are summed within occupation, year, and age band."
        ),
    }
    return panel, metadata


def _write_panel(panel: pd.DataFrame, metadata: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({len(panel):,} rows)")
    print(f"Occupation match rate: {metadata['occupation_match_rate']:.1%}")


def build(input_path: Path, output_path: Path = CPS_PANEL_EXPECTED, **kwargs) -> pd.DataFrame:
    raw = _read_extract(input_path)
    panel, metadata = transform(raw, **kwargs)
    metadata["source_file"] = input_path.name
    _write_panel(panel, metadata, output_path)
    return panel


def build_from_ipums(
    start: int,
    end: int,
    sample: str,
    *,
    output_path: Path = CPS_PANEL_EXPECTED,
    crosswalk_path: Path = LOCAL_XWALK,
    min_match_rate: float = 0.80,
) -> pd.DataFrame:
    """Fetch a real IPUMS extract and build the validated analysis panel."""
    if not crosswalk_path.exists():
        raise FileNotFoundError(
            f"{crosswalk_path} is missing. Run python -m src.occupation_crosswalk "
            "before submitting the extract."
        )
    raw, spec = _fetch_ipums(start, end, sample)
    panel, metadata = transform(
        raw,
        weight_column=spec["weight"],
        census_occ_column="OCC",
        frequency=spec["frequency"],
        crosswalk_path=crosswalk_path,
        min_match_rate=min_match_rate,
    )
    metadata["source"] = "IPUMS CPS API"
    metadata["ipums_extract"] = spec
    _write_panel(panel, metadata, output_path)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_path", type=Path, nargs="?")
    parser.add_argument(
        "--fetch-ipums",
        action="store_true",
        help="submit and download the extract through the IPUMS API",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the API extract specification without a key or network call",
    )
    parser.add_argument("--sample", choices=["asec", "basic"], default="asec")
    parser.add_argument("--start", type=int, default=2020)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--output-path", type=Path, default=CPS_PANEL_EXPECTED)
    parser.add_argument("--year-column", default="YEAR")
    parser.add_argument("--month-column", default="MONTH")
    parser.add_argument("--age-column", default="AGE")
    parser.add_argument("--employment-column", default="EMPSTAT")
    parser.add_argument("--weight-column", default="WTFINL")
    parser.add_argument("--employed-codes", default="10,12")
    parser.add_argument("--census-occ-column", default="OCC")
    parser.add_argument("--soc-column", default=None)
    parser.add_argument("--frequency", choices=["monthly", "annual"], default="monthly")
    parser.add_argument("--crosswalk-path", type=Path, default=LOCAL_XWALK)
    parser.add_argument("--min-match-rate", type=float, default=0.80)
    args = parser.parse_args()

    if args.dry_run:
        if args.input_path:
            parser.error("input_path cannot be combined with --dry-run")
        try:
            spec = ipums_extract_spec(args.start, args.end, args.sample)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(spec, indent=2))
        return

    if args.fetch_ipums:
        if args.input_path:
            parser.error("input_path cannot be combined with --fetch-ipums")
        try:
            build_from_ipums(
                args.start,
                args.end,
                args.sample,
                output_path=args.output_path,
                crosswalk_path=args.crosswalk_path,
                min_match_rate=args.min_match_rate,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        return

    if args.input_path is None:
        parser.error("provide input_path, --fetch-ipums, or --dry-run")
    employed_codes = {int(value) for value in args.employed_codes.split(",")}
    census_occ_column = None if args.soc_column else args.census_occ_column
    build(
        args.input_path,
        args.output_path,
        year_column=args.year_column,
        month_column=args.month_column,
        age_column=args.age_column,
        employment_column=args.employment_column,
        weight_column=args.weight_column,
        employed_codes=employed_codes,
        census_occ_column=census_occ_column,
        soc_column=args.soc_column,
        frequency=args.frequency,
        crosswalk_path=args.crosswalk_path,
        min_match_rate=args.min_match_rate,
    )


if __name__ == "__main__":
    main()
