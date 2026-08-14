"""
Census OCC <-> SOC crosswalk.

CPS microdata code occupation with Census occupation codes, not SOC. This
repo's crosswalk is for the contemporary OCC values in 2020+ samples (the 2018
Census scheme); it must not be used with the harmonized OCC2010 variable. The
AEI exposure measure and OEWS are SOC-coded. This module loads the official
Census OCC->SOC crosswalk and exposes a mapping helper.

The crosswalk file itself (an Excel from census.gov) is not bundled. Download it
once via `fetch()` on a networked machine, or drop it at
data/raw/bls/census_occ_soc_crosswalk.csv with columns: occ_census, occ_code.

Because multiple Census OCC codes can map to one SOC (and vice versa), the
default `to_soc` collapses to the modal SOC per Census code; for share-based
panel work this is adequate. For wage-weighted work, carry the full many-to-many
map instead.
"""
import urllib.request
import re
from functools import lru_cache

import pandas as pd

from src.config import RAW_BLS

# 2018 Census OCC -> 2018 SOC crosswalk (public).
CENSUS_XWALK_URL = (
    "https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/"
    "2018-occupation-code-list-and-crosswalk.xlsx"
)
LOCAL_XWALK = RAW_BLS / "census_occ_soc_crosswalk.csv"


def fetch():
    """Download the Census crosswalk and normalise to (occ_census, occ_code)."""
    print(f"Fetching Census OCC<->SOC crosswalk ...")
    req = urllib.request.Request(CENSUS_XWALK_URL,
                                 headers={"User-Agent": "ai-career-launch-monitor"})
    dst = RAW_BLS / "census_crosswalk.xlsx"
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        f.write(r.read())
    # The workbook has a sheet mapping Census codes to 2018 SOC. Column names
    # vary by vintage, so we locate them heuristically.
    xls = pd.read_excel(dst, sheet_name=None, dtype=str, header=None)
    for sheet in xls.values():
        for header_row in range(min(30, len(sheet))):
            labels = sheet.iloc[header_row].fillna("").astype(str).str.lower()
            occ_candidates = [i for i, label in labels.items()
                              if "census" in label and "occupation" in label and "code" in label]
            soc_candidates = [i for i, label in labels.items()
                              if "soc" in label and "code" in label]
            if not occ_candidates or not soc_candidates:
                continue
            occ_c, soc_c = occ_candidates[0], soc_candidates[0]
            rows = sheet.iloc[header_row + 1:, [occ_c, soc_c]].copy()
            rows.columns = ["occ_census", "soc_raw"]
            records = []
            for _, row in rows.dropna(how="all").iterrows():
                census_match = re.search(r"\b(\d{4})\b", str(row["occ_census"]))
                soc_matches = re.findall(r"\b\d{2}-\d{4}\b", str(row["soc_raw"]))
                if census_match:
                    records.extend((census_match.group(1), soc) for soc in soc_matches)
            if records:
                out = pd.DataFrame(records, columns=["occ_census", "occ_code"])
                out = out.drop_duplicates().sort_values(["occ_census", "occ_code"])
                out.to_csv(LOCAL_XWALK, index=False)
                print(f"  wrote {LOCAL_XWALK} ({len(out)} rows)")
                return
    raise RuntimeError("Could not locate OCC/SOC columns in crosswalk workbook.")


@lru_cache(maxsize=1)
def _map() -> dict:
    if not LOCAL_XWALK.exists():
        raise FileNotFoundError(
            f"{LOCAL_XWALK} missing. Run occupation_crosswalk.fetch() first.")
    x = pd.read_csv(LOCAL_XWALK, dtype=str)
    x["occ_census"] = x["occ_census"].str.extract(r"(\d+)", expand=False).str.zfill(4)
    x["occ_code"] = x["occ_code"].str.strip()
    # modal SOC per census code
    modal = (x.groupby("occ_census")["occ_code"]
               .agg(lambda s: s.value_counts().index[0]))
    return modal.to_dict()


def to_soc(occ_census) -> str | None:
    digits = re.sub(r"\D", "", str(occ_census).split(".")[0]).zfill(4)
    return _map().get(digits)


if __name__ == "__main__":
    fetch()
