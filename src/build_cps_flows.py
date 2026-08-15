"""Build linked young-worker flows and wages from Basic Monthly CPS data.

The same validated person is linked across adjacent Basic Monthly CPS samples
with ``CPSIDV``. ``PANLWT`` weights the linked pair. For each occupation:

* entry: not employed in that occupation last month, employed in it now;
* retention: employed in that occupation last month and still there now;
* exit: employed in that occupation last month and not in it now.

Entry is measured for workers age 20-29 in the current month. Retention and
exit are measured for workers age 20-29 in the previous month. ``entry_rate``
uses all linked young people who were not in the occupation last month as its
risk set. ``entrant_share`` uses entrants plus stayers and therefore describes
the composition of current occupation employment. Weekly earnings come from
outgoing rotation groups (MISH 4 and 8) and use ``EARNWT``.

    python -m src.build_cps_flows --fetch-ipums --start 2020 --end 2025
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_cps_panel import (
    EMPLOYED_CODES,
    IPUMS_COLLECTION,
    _crosswalk_map,
    _read_extract,
    available_sample_ids,
    candidate_sample_ids,
    normalize_census_occ,
)
from src.config import CPS_FLOW_ANNUAL, CPS_FLOW_MONTHLY, CPS_WAGE_PANEL
from src.occupation_crosswalk import LOCAL_XWALK

FLOW_VARIABLES = [
    "YEAR", "MONTH", "CPSIDV", "MISH", "AGE", "OCC", "EMPSTAT",
    "PANLWT", "EARNWEEK", "EARNWEEK2", "EARNWT",
]
YOUNG_MIN = 20
YOUNG_MAX = 29


def flow_extract_spec(start: int, end: int) -> dict:
    return {
        "collection": IPUMS_COLLECTION,
        "sample_type": "basic",
        "start": start,
        "end": end,
        "samples": candidate_sample_ids(start, end, "basic"),
        "variables": FLOW_VARIABLES,
        "person_identifier": "CPSIDV",
        "flow_weight": "PANLWT",
        "earnings_weight": "EARNWT",
        "occupation_variable": "OCC",
    }


def _fetch_ipums(start: int, end: int) -> tuple[pd.DataFrame, dict]:
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

    spec = flow_extract_spec(start, end)
    client = IpumsApiClient(api_key)
    published = client.get_all_sample_info(IPUMS_COLLECTION)
    selected = available_sample_ids(start, end, "basic", published)
    missing = [
        month_key for month_key in spec["samples"]
        if not any(sample.startswith(month_key) for sample in selected)
    ]
    spec["samples"] = selected
    spec["unavailable_samples"] = missing

    extract = MicrodataExtract(
        collection=IPUMS_COLLECTION,
        samples=selected,
        variables=FLOW_VARIABLES,
        description=f"ai-career-launch-monitor worker flows {start}-{end}",
    )
    print(
        f"Submitting IPUMS Basic Monthly CPS extract "
        f"({len(selected)} published samples)"
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
                "Expected one IPUMS DDI and one microdata file; found "
                f"{len(ddi_paths)} DDI and {len(data_paths)} data files."
            )
        ddi = readers.read_ipums_ddi(ddi_paths[0])
        raw = readers.read_microdata(ddi, data_paths[0])
    return raw, spec


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return float("nan")
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2)])


def _event_table(
    linked: pd.DataFrame,
    mask: pd.Series,
    occ_column: str,
    value_name: str,
) -> pd.DataFrame:
    selected = linked.loc[
        mask, ["year", "month", occ_column, "pair_weight"]
    ].rename(columns={occ_column: "occ_code"})
    if selected.empty:
        return pd.DataFrame(columns=["year", "month", "occ_code", value_name,
                                     f"{value_name}_n"])
    return (
        selected.groupby(["year", "month", "occ_code"], as_index=False)
        .agg(**{
            value_name: ("pair_weight", "sum"),
            f"{value_name}_n": ("pair_weight", "size"),
        })
    )


def _build_wages(work: pd.DataFrame) -> pd.DataFrame:
    wages = work[
        work["employed"]
        & work["age"].between(YOUNG_MIN, YOUNG_MAX)
        & work["mish"].isin([4, 8])
        & work["occ_code"].notna()
    ].copy()
    legacy = pd.to_numeric(wages["EARNWEEK"], errors="coerce")
    modern = pd.to_numeric(wages["EARNWEEK2"], errors="coerce")
    legacy = legacy.where(legacy.gt(0) & legacy.lt(9999.99))
    modern = modern.where(modern.gt(0) & modern.lt(9999.99))
    wages["weekly_wage"] = modern.fillna(legacy)
    wages["earn_weight"] = pd.to_numeric(wages["EARNWT"], errors="coerce")
    wages = wages[
        wages["weekly_wage"].gt(0)
        & wages["weekly_wage"].lt(9999.99)
        & wages["earn_weight"].gt(0)
    ]
    rows = []
    for (year, occ_code), group in wages.groupby(["year", "occ_code"]):
        rows.append({
            "year": int(year),
            "occ_code": occ_code,
            "median_weekly_wage": weighted_median(
                group["weekly_wage"], group["earn_weight"]
            ),
            "mean_weekly_wage": float(
                np.average(group["weekly_wage"], weights=group["earn_weight"])
            ),
            "wage_sample_n": int(len(group)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "year", "occ_code", "median_weekly_wage", "mean_weekly_wage",
            "wage_sample_n", "wage_percentile", "wage_growth",
            "log_wage_growth",
        ])
    out = out.sort_values(["occ_code", "year"]).reset_index(drop=True)
    out["wage_percentile"] = out.groupby("year")["median_weekly_wage"].rank(
        method="average", pct=True
    )
    previous = out.groupby("occ_code")["median_weekly_wage"].shift()
    out["wage_growth"] = out["median_weekly_wage"] / previous - 1
    out["log_wage_growth"] = np.log(out["median_weekly_wage"]) - np.log(previous)
    return out


def transform_flows(
    raw: pd.DataFrame,
    *,
    crosswalk_path: Path = LOCAL_XWALK,
    min_match_rate: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Link adjacent months and return monthly flows, annual flows, and wages."""
    raw = raw.rename(columns={column: str(column).upper() for column in raw.columns})
    missing = set(FLOW_VARIABLES) - set(raw.columns)
    if missing:
        raise ValueError(f"CPS flow extract missing columns: {sorted(missing)}")

    work = raw[FLOW_VARIABLES].copy()
    for column in ("YEAR", "MONTH", "MISH", "AGE", "EMPSTAT", "PANLWT"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["YEAR", "MONTH", "MISH", "AGE", "CPSIDV"])
    work = work[work["AGE"].between(YOUNG_MIN - 1, YOUNG_MAX + 1)]
    work = work[work["CPSIDV"].astype(str).ne("0")].copy()
    work["year"] = work["YEAR"].astype(int)
    work["month"] = work["MONTH"].astype(int)
    work["mish"] = work["MISH"].astype(int)
    work["age"] = work["AGE"].astype(int)
    work["period"] = work["year"] * 12 + work["month"]
    work["person_id"] = work["CPSIDV"].astype(str)
    work["employed"] = work["EMPSTAT"].isin(EMPLOYED_CODES)

    mapping = _crosswalk_map(crosswalk_path)
    work["occ_code"] = work["OCC"].map(normalize_census_occ).map(mapping)
    employed = work["employed"]
    match_rate = float(work.loc[employed, "occ_code"].notna().mean())
    if match_rate < min_match_rate:
        raise ValueError(
            f"Occupation match rate {match_rate:.1%} is below the required "
            f"{min_match_rate:.1%}. Check OCC vintage and crosswalk."
        )
    work.loc[~work["employed"], "occ_code"] = None

    duplicate = work.duplicated(["person_id", "period"], keep=False)
    if duplicate.any():
        raise ValueError("CPSIDV is not unique within a person-month")

    wages = _build_wages(work)

    work = work.sort_values(["person_id", "period"]).reset_index(drop=True)
    prior_columns = ["period", "mish", "age", "occ_code"]
    for column in prior_columns:
        work[f"previous_{column}"] = work.groupby("person_id", sort=False)[column].shift()
    linked = work[
        work["period"].sub(work["previous_period"]).eq(1)
        & work["mish"].sub(work["previous_mish"]).eq(1)
    ].copy()
    linked["pair_weight"] = pd.to_numeric(linked["PANLWT"], errors="coerce")
    linked = linked[linked["pair_weight"].gt(0)]
    if linked.empty:
        raise ValueError("No adjacent-month CPS links with positive PANLWT")

    same_occ = (
        linked["occ_code"].notna()
        & linked["previous_occ_code"].notna()
        & linked["occ_code"].eq(linked["previous_occ_code"])
    )
    young_now = linked["age"].between(YOUNG_MIN, YOUNG_MAX)
    young_previous = linked["previous_age"].between(YOUNG_MIN, YOUNG_MAX)

    entry_population = (
        linked.loc[young_now, ["year", "month", "pair_weight"]]
        .groupby(["year", "month"], as_index=False)
        .agg(
            entry_population=("pair_weight", "sum"),
            entry_population_n=("pair_weight", "size"),
        )
    )
    tables = [
        _event_table(
            linked,
            young_now & linked["occ_code"].notna() & ~same_occ,
            "occ_code",
            "entries",
        ),
        _event_table(
            linked,
            young_now & same_occ,
            "occ_code",
            "current_stayers",
        ),
        _event_table(
            linked,
            young_previous & same_occ,
            "previous_occ_code",
            "retained",
        ),
        _event_table(
            linked,
            young_previous & linked["previous_occ_code"].notna() & ~same_occ,
            "previous_occ_code",
            "exits",
        ),
        _event_table(
            linked,
            young_now & linked["previous_occ_code"].notna(),
            "previous_occ_code",
            "previous_members_current_age",
        ),
    ]
    keys = ["year", "month", "occ_code"]
    months = linked[["year", "month"]].drop_duplicates()
    occupations = pd.DataFrame({"occ_code": sorted(set(mapping.values()))})
    monthly = months.merge(occupations, how="cross").merge(
        entry_population, on=["year", "month"], how="left",
    )
    for table in tables:
        monthly = monthly.merge(table, on=keys, how="outer")
    count_columns = [
        "entries", "entries_n", "current_stayers", "current_stayers_n",
        "retained", "retained_n", "exits", "exits_n",
        "previous_members_current_age", "previous_members_current_age_n",
    ]
    for column in count_columns:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce").fillna(0)
    monthly["entry_risk"] = (
        monthly["entry_population"] - monthly["previous_members_current_age"]
    ).clip(lower=0)
    monthly["entry_risk_n"] = (
        monthly["entry_population_n"] - monthly["previous_members_current_age_n"]
    ).clip(lower=0)
    monthly["current_employment"] = monthly["entries"] + monthly["current_stayers"]
    monthly["current_sample_n"] = monthly["entries_n"] + monthly["current_stayers_n"]
    monthly["exit_at_risk"] = monthly["exits"] + monthly["retained"]
    monthly["entry_rate"] = monthly["entries"] / monthly["entry_risk"].replace(0, np.nan)
    monthly["entrant_share"] = (
        monthly["entries"] / monthly["current_employment"].replace(0, np.nan)
    )
    monthly["retention_rate"] = monthly["retained"] / monthly["exit_at_risk"].replace(0, np.nan)
    monthly["exit_rate"] = monthly["exits"] / monthly["exit_at_risk"].replace(0, np.nan)
    monthly["synthetic"] = False
    monthly = monthly.sort_values(keys).reset_index(drop=True)

    annual = (
        monthly.groupby(["year", "occ_code"], as_index=False)
        .agg(
            entries=("entries", "sum"),
            entries_n=("entries_n", "sum"),
            current_stayers=("current_stayers", "sum"),
            current_stayers_n=("current_stayers_n", "sum"),
            retained=("retained", "sum"),
            retained_n=("retained_n", "sum"),
            exits=("exits", "sum"),
            exits_n=("exits_n", "sum"),
            entry_population=("entry_population", "sum"),
            entry_population_n=("entry_population_n", "sum"),
            previous_members_current_age=("previous_members_current_age", "sum"),
            previous_members_current_age_n=("previous_members_current_age_n", "sum"),
            months=("month", "nunique"),
        )
    )
    annual["entry_risk"] = (
        annual["entry_population"] - annual["previous_members_current_age"]
    ).clip(lower=0)
    annual["entry_risk_n"] = (
        annual["entry_population_n"] - annual["previous_members_current_age_n"]
    ).clip(lower=0)
    annual["current_employment"] = annual["entries"] + annual["current_stayers"]
    annual["current_sample_n"] = annual["entries_n"] + annual["current_stayers_n"]
    annual["exit_at_risk"] = annual["exits"] + annual["retained"]
    annual["exit_sample_n"] = annual["exits_n"] + annual["retained_n"]
    annual["entry_rate"] = annual["entries"] / annual["entry_risk"].replace(0, np.nan)
    annual["entrant_share"] = (
        annual["entries"] / annual["current_employment"].replace(0, np.nan)
    )
    annual["retention_rate"] = annual["retained"] / annual["exit_at_risk"].replace(0, np.nan)
    annual["exit_rate"] = annual["exits"] / annual["exit_at_risk"].replace(0, np.nan)
    annual["synthetic"] = False

    coverage = {
        str(int(year)): sorted(group["month"].unique().astype(int).tolist())
        for year, group in linked.groupby("year")
    }
    metadata = {
        "synthetic": False,
        "created": date.today().isoformat(),
        "person_identifier": "CPSIDV",
        "flow_weight": "PANLWT",
        "earnings_weight": "EARNWT",
        "young_age_definition": "20-29",
        "occupation_match_rate": match_rate,
        "linked_person_months": int(len(linked)),
        "linked_month_coverage": coverage,
        "entry_definition": (
            "Age 20-29 now; not employed in the current occupation in the prior month."
        ),
        "retention_definition": (
            "Age 20-29 in the prior month; employed in the same occupation now."
        ),
        "exit_definition": (
            "Age 20-29 in the prior month; no longer employed in that occupation now."
        ),
        "rate_denominators": {
            "entry_rate": (
                "All linked people age 20-29 now who were not in the occupation "
                "in the prior month."
            ),
            "entrant_share": (
                "Current linked workers age 20-29 in the occupation: entrants "
                "plus stayers."
            ),
            "exit_rate_and_retention_rate": (
                "Prior linked workers age 20-29 in the occupation: exits plus "
                "retained workers."
            ),
        },
        "wage_definition": (
            "Current-dollar weekly earnings for employed age 20-29 outgoing-rotation "
            "respondents (MISH 4 or 8), using EARNWEEK2 where available and legacy "
            "EARNWEEK otherwise, weighted by EARNWT."
        ),
    }
    return monthly, annual, wages, metadata


def _write_outputs(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    wages: pd.DataFrame,
    metadata: dict,
) -> None:
    for path in (CPS_FLOW_MONTHLY, CPS_FLOW_ANNUAL, CPS_WAGE_PANEL):
        path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(CPS_FLOW_MONTHLY, index=False)
    annual.to_csv(CPS_FLOW_ANNUAL, index=False)
    wages.to_csv(CPS_WAGE_PANEL, index=False)
    CPS_FLOW_MONTHLY.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {CPS_FLOW_MONTHLY} ({len(monthly):,} rows)")
    print(f"Wrote {CPS_FLOW_ANNUAL} ({len(annual):,} rows)")
    print(f"Wrote {CPS_WAGE_PANEL} ({len(wages):,} rows)")


def build(input_path: Path, *, crosswalk_path: Path = LOCAL_XWALK,
          min_match_rate: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _read_extract(input_path)
    monthly, annual, wages, metadata = transform_flows(
        raw, crosswalk_path=crosswalk_path, min_match_rate=min_match_rate
    )
    metadata["source_file"] = input_path.name
    _write_outputs(monthly, annual, wages, metadata)
    return monthly, annual, wages


def build_from_ipums(start: int, end: int, *, crosswalk_path: Path = LOCAL_XWALK,
                     min_match_rate: float = 0.80):
    if not crosswalk_path.exists():
        raise FileNotFoundError(
            f"{crosswalk_path} is missing. Run python -m src.occupation_crosswalk first."
        )
    raw, spec = _fetch_ipums(start, end)
    monthly, annual, wages, metadata = transform_flows(
        raw, crosswalk_path=crosswalk_path, min_match_rate=min_match_rate
    )
    metadata["source"] = "IPUMS CPS API"
    metadata["ipums_extract"] = spec
    _write_outputs(monthly, annual, wages, metadata)
    return monthly, annual, wages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_path", type=Path, nargs="?")
    parser.add_argument("--fetch-ipums", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=2020)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--crosswalk-path", type=Path, default=LOCAL_XWALK)
    parser.add_argument("--min-match-rate", type=float, default=0.80)
    args = parser.parse_args()

    if args.dry_run:
        if args.input_path or args.fetch_ipums:
            parser.error("--dry-run cannot be combined with input_path or --fetch-ipums")
        print(json.dumps(flow_extract_spec(args.start, args.end), indent=2))
        return
    if args.fetch_ipums:
        if args.input_path:
            parser.error("input_path cannot be combined with --fetch-ipums")
        try:
            build_from_ipums(
                args.start,
                args.end,
                crosswalk_path=args.crosswalk_path,
                min_match_rate=args.min_match_rate,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        return
    if args.input_path is None:
        parser.error("provide input_path, --fetch-ipums, or --dry-run")
    build(
        args.input_path,
        crosswalk_path=args.crosswalk_path,
        min_match_rate=args.min_match_rate,
    )


if __name__ == "__main__":
    main()
