"""
Download BLS inputs.

1. May 2025 OEWS national employment + wages by SOC occupation (public).
2. Guidance for the CPS occupation x age panel, which drives the young-worker
   analysis. CPS microdata are large and governed by IPUMS terms, so we do NOT
   bundle them. Two supported routes are documented below and in data/README.md.

    python -m src.download_bls          # fetches OEWS national zip + extracts

The CPS panel is expected at data/raw/bls/cps_panel.csv with columns:
    year, occ_code, age_band, employed        (age_band in {20-24,25-29,30-39,40-54,55+})
Build it once with either route, then re-run build_panel.py.
"""
import io
import urllib.request
import zipfile

from src.config import BLS_OEWS_NATIONAL, RAW_BLS


def download_oews():
    print(f"Downloading OEWS national: {BLS_OEWS_NATIONAL}")
    req = urllib.request.Request(BLS_OEWS_NATIONAL,
                                 headers={"User-Agent": "ai-career-launch-monitor"})
    with urllib.request.urlopen(req) as r:
        blob = r.read()
    destination = RAW_BLS / "oews"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            archive.extract(member, destination)
    print(f"  extracted to {destination}")


CPS_INSTRUCTIONS = """
CPS occupation x age panel -- build once, then it feeds build_panel.py.

Route A (recommended, reproducible): automated IPUMS CPS API extract
    - Register at cps.ipums.org and set IPUMS_API_KEY.
    - Run `python -m src.occupation_crosswalk` once.
    - Inspect the request without a key or API call:
      python -m src.build_cps_panel --dry-run \\
          --sample asec --start 2020 --end 2025
    - Submit, download, validate, aggregate, and write the panel:
      python -m src.build_cps_panel --fetch-ipums \\
          --sample asec --start 2020 --end 2025
    - ASEC uses ASECWT. `--sample basic` uses WTFINL, checks the published
      sample catalog, skips unavailable months, and records exact coverage.
    - The automated OCC->2018 SOC route deliberately rejects pre-2020 years.

Route A2: convert an existing extract
    - Request YEAR, MONTH, AGE, OCC, EMPSTAT, and WTFINL for 2020+ basic-monthly
      files, then run:
      python -m src.build_cps_panel data/raw/bls/cps_extract.csv.gz \\
          --census-occ-column OCC --frequency monthly
    - Do not map harmonized OCC2010 through the repo's 2018 Census crosswalk.

Route B (no key): BLS CPS public tables
    - Employment by detailed occupation and age is available in the annual CPS
      tables (e.g. Table 11b) at bls.gov/cps/tables.htm, but only for broad age
      groups and a coarse occupation list. Lower resolution; fine for a first pass.

Either way the required output schema is:
    year, occ_code, age_band, employed
"""


def main():
    download_oews()
    print(CPS_INSTRUCTIONS)


if __name__ == "__main__":
    main()
