# Data

The exposure input is public and vendored. Larger files are fetched or supplied
locally. IPUMS users must follow the source's access and redistribution terms.

## `raw/anthropic/job_exposure.csv`  — vendored, real
Anthropic Economic Index, labor-market-impacts release (CC-BY 4.0).
- `occ_code` — 2018 SOC detailed occupation code
- `title` — SOC occupation title
- `observed_exposure` — share of the occupation's tasks observed in Claude
  usage, 0–1 (renamed `ai_exposure` downstream)
- 756 occupations; ~54% show zero observed usage.
- Refresh / add `task_penetration.csv` with `python -m src.download_anthropic`.
- Source: https://huggingface.co/datasets/Anthropic/EconomicIndex

## `raw/bls/` — fetched
- `oews/` — OEWS national employment + wages by SOC (`python -m src.download_bls`).
- `census_occ_soc_crosswalk.csv` — Census OCC→SOC map
  (`python -m src.occupation_crosswalk`).
- `cps_panel.csv` — **you build this locally.** Occupation × age employment over time.
  Schema: `year, occ_code, age_band, employed` with
  `age_band ∈ {20-24, 25-29, 30-39, 40-54, 55+}`.
  Recommended route: run `python -m src.occupation_crosswalk`, set
  `IPUMS_API_KEY`, then run `python -m src.build_cps_panel --fetch-ipums
  --sample asec --start 2020 --end 2025`. Use `--dry-run` first to inspect the
  extract specification without a key or network call. The API download is
  temporary; the aggregated panel and metadata are retained locally.
  A previously downloaded extract can instead be converted with
  `python -m src.build_cps_panel <extract.csv.gz> --census-occ-column OCC`.
  This repo's crosswalk expects contemporary `OCC` codes from 2020 onward, not
  harmonized `OCC2010` codes. See `src/download_bls.py` and the main README.
- `cps_panel.metadata.json` — source filename, year/month coverage, mapping
  method, match rate, partial-year flags, and (for API builds) the non-sensitive
  extract specification and unavailable-sample list.

## `raw/onet/` — fetched
O*NET Job Zones and Education/Training files (`python -m src.download_onet`),
used for entry-level dependence and education-pathway context.
Source: https://www.onetcenter.org/database.html

## `processed/` — generated
- `exposure.csv` — exposure + SOC major group + quartiles (`build_exposure.py`)
- `panel_long.csv` — occ × year × age_band shares (`build_panel.py`)
- `panel_occ.csv` — one row per occupation, incl. young-share change
- `elsi.csv` — Entry-Level Squeeze Index rankings (`elsi.py`, real input only)
- Files ending in `_SYNTHETIC` are plumbing outputs and are not findings.

## Provenance / integrity note
`job_exposure.csv` here was pulled from the Hugging Face
`labor_market_impacts/job_exposure.csv` resolve endpoint. Its checksum and
license are recorded in `data/provenance.json`. Re-download with
`src.download_anthropic` and verify the checksum before publishing an update.
