# Data

Public exposure inputs are vendored or reproducibly derived. CPS microdata and
generated flow panels remain local under IPUMS access and redistribution terms.

## `raw/anthropic/`

### `job_exposure.csv`

Real Anthropic Economic Index labor-market-impacts data (CC-BY 4.0).

- `occ_code`: 2018 detailed SOC code
- `title`: occupation title
- `observed_exposure`: share of occupation tasks observed in Claude usage
- 756 occupations; about 54% have zero observed usage

### `exposure_components.csv`

Reproducibly derived from public source files by
`python -m src.build_exposure_components`.

- `automation_exposure`: observed exposure allocated to directive and
  feedback-loop interactions
- `augmentation_exposure`: observed exposure allocated to task iteration,
  validation, and learning
- `theoretical_exposure`: Eloundou et al. GPT-4 gamma rating (`E1 + E2`)
- `theoretical_exposure_conservative`: GPT-4 beta rating (`E1`)
- `automation_share`, `augmentation_share`: classified interaction shares
- `classified_task_n`: matched O*NET task count

The split uses Anthropic March 2025 task labels and 2026 task penetration, then
maps 2010 SOC task statements to 2018 SOC with the official BLS crosswalk.
Coverage and limitations are recorded in `exposure_components.metadata.json`.

## `raw/bls/` — local generated files

### Basic Monthly CPS worker flows

Run:

```bash
python -m src.occupation_crosswalk
export IPUMS_API_KEY=your_key_from_account.ipums.org
python -m src.build_cps_flows --fetch-ipums --start 2020 --end 2025
```

The extract requests `YEAR`, `MONTH`, `CPSIDV`, `MISH`, `AGE`, `OCC`,
`EMPSTAT`, `PANLWT`, `EARNWEEK`, `EARNWEEK2`, and `EARNWT`. Downloaded
microdata are read from a temporary directory and are not retained.

Generated files:

- `cps_flows_monthly.csv`: occupation × transition month entry, retention,
  exit, denominators, weighted rates, and unweighted counts
- `cps_flows_annual.csv`: occupation × year sums and rates
- `cps_young_wages.csv`: age-20–29 occupation × year weekly wage statistics,
  wage growth, percentile, and sample size
- `cps_flows_monthly.metadata.json`: identifier, weights, definitions, match
  rate, link count, exact month coverage, and non-sensitive extract spec

Monthly flow schema:

```text
year, month, occ_code,
entries, entries_n, current_stayers, current_stayers_n,
retained, retained_n, exits, exits_n,
entry_population, previous_members_current_age,
entry_risk, entry_risk_n, current_employment, current_sample_n,
exit_at_risk, entry_rate, entrant_share, retention_rate, exit_rate, synthetic
```

`entry_rate` divides entries by all linked age-20–29 respondents who were not
in the occupation last month. `entrant_share` divides entries by current
occupation employment and preserves the older composition measure under the
correct name. The builder creates a complete occupation × month grid so valid
zero-entry cells remain in the panel.

Wage schema:

```text
year, occ_code, median_weekly_wage, mean_weekly_wage,
wage_sample_n, wage_percentile, wage_growth, log_wage_growth
```

The real 2020–2025 build used 71 published Basic Monthly samples, produced
608,893 adjacent-month links, and matched 93.45% of employed occupation records.
October 2025 was unavailable.

### Supporting ASEC panel

`cps_panel.csv` is the older occupation × age stock panel. Build it with:

```bash
python -m src.build_cps_panel --fetch-ipums \
  --sample asec --start 2020 --end 2025
```

Schema: `year, occ_code, age_band, employed, sample_n`, where `age_band` is one
of `20-24`, `25-29`, `30-39`, `40-54`, or `55+`.

### Other BLS inputs

- `census_occ_soc_crosswalk.csv`: contemporary Census OCC → 2018 SOC
- `oews/`: May 2025 national OEWS employment and wage files

## `raw/onet/`

O*NET Job Zones and Education/Training files, fetched by
`python -m src.download_onet`.

## `processed/`

- `exposure.csv`: observed, automation, augmentation, and theoretical exposure
  plus SOC labels and exposure groups
- `flow_wage_regressions.csv`: fixed-effects entry, exit, and wage estimates
- `flow_robustness.csv`: trends, placebos, weighting, balanced-panel, sample
  thresholds, and leave-group-out checks for the corrected entry rate
- `flow_event_study.csv`: exposure-by-year entry coefficients relative to 2023
  and a joint pre-trend test
- `flow_headline_series.csv`: weighted annual rates behind the main figure
- `flow_wage_joint.csv`: occupation changes in entry, exit, young employment,
  wages, and their descriptive joint pattern
- `panel_long.csv`, `panel_occ.csv`: supporting ASEC stock outcomes
- `headline_series.csv`, `robustness_summary.csv`: supporting stock results

Synthetic data are never used for worker-flow findings. Supporting outputs with
synthetic input are visibly watermarked and use `_SYNTHETIC` filenames.

## Provenance

Checksums and upstream URLs for committed inputs are in `provenance.json`.
Rebuild scripts validate schemas and record coverage before writing outputs.
