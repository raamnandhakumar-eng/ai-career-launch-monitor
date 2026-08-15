# AI Career Launch Monitor

## Question

**Are highly AI-exposed occupations seeing fewer young workers enter, rather
than more workers exit?**

**Main finding.** Not in the first real worker-flow estimates. In linked Basic
Monthly CPS data for 2020–2025, observed exposure has a positive but imprecise
post-2024 entry coefficient: **+1.02 percentage points** (SE 1.14, p=0.372).
The exit estimate is **−0.03 points** (SE 1.02, p=0.977).

**Robustness.** Automation, augmentation, and theoretical exposure also do not
predict weaker entry. The older young-worker-share relationship weakens when
occupation-specific trends are added.

**Interpretation.** The more direct flow measure does not support an early
entry squeeze in this window. This is descriptive evidence, not a causal
estimate or proof of no effect.

![Young-worker entry and exit rates by AI exposure](figures/worker_flows_by_exposure.png)

The chart shows weighted worker-month transition rates for the highest observed
exposure quartile and occupations with zero observed exposure. The fixed-effects
estimates below use continuous occupation exposure and all qualifying cells.

## What v2 measures

The repository now links the same CPS respondent across adjacent survey months
with `CPSIDV` and applies the IPUMS longitudinal weight `PANLWT`.

| Flow | Definition for workers age 20–29 |
|---|---|
| Entry | Not in the occupation last month → in it now |
| Retention | In the occupation last month → still in it now |
| Exit | In the occupation last month → no longer in it now |

Job-to-job moves count as an exit from the old occupation and an entry into the
new one. The extract contains **608,893 linked person-months** across **71 Basic
Monthly CPS samples**. October 2025 was unavailable, so exact month coverage is
recorded in the local metadata.

## Main estimates

The model is an occupation and year fixed-effects specification with standard
errors clustered by occupation. Coefficients are for a full-unit change in the
listed exposure measure; most occupations are far below one.

| Exposure | Entry rate × post | Exit rate × post | Young weekly wage × post |
|---|---:|---:|---:|
| Observed | +1.02 pp (p=0.372) | −0.03 pp (p=0.977) | −3.11% (p=0.363) |
| Automation | +2.87 pp (p=0.273) | −0.31 pp (p=0.905) | −5.02% (p=0.562) |
| Augmentation | +1.72 pp (p=0.359) | −0.33 pp (p=0.838) | −6.16% (p=0.251) |
| Theoretical | +0.31 pp (p=0.633) | +0.74 pp (p=0.219) | −1.14% (p=0.545) |

None of these estimates is statistically precise. In particular, the data do
not show automation-heavy exposure predicting weaker entry while augmentation
does not.

## Exposure measures

| Measure | Construction |
|---|---|
| Observed | Share of occupation tasks observed in Claude usage |
| Automation | Observed exposure allocated to directive and feedback-loop interactions |
| Augmentation | Observed exposure allocated to task iteration, validation, and learning |
| Theoretical | GPT-4 task exposure from Eloundou et al. (`E1 + E2`) |

The automation/augmentation split combines Anthropic's March 2025 task
interaction labels with its 2026 task-penetration release. It covers 89.0% of
occupations with positive observed exposure. It is a mixed-vintage descriptive
split, not an occupation-level probability of automation.

## Wages and joint outcomes

Young-worker weekly earnings come from outgoing rotation groups (`MISH` 4 and
8), using `EARNWEEK2` where available, legacy `EARNWEEK` otherwise, and
`EARNWT`. The pipeline writes:

- weighted median and mean weekly wages;
- annual wage growth and within-year wage percentile;
- young-employment growth; and
- an employment × wage-growth interaction plus four descriptive patterns:
  fewer/more juniors combined with lower/higher wages.

Wages are current-dollar and detailed occupation cells can be noisy. The main
wage models require at least 10 unweighted observations per cell.

## Reproduce v2

Requires Python 3.10 or newer and an IPUMS account.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Refresh the four exposure measures, then build the occupation spine
python -m src.build_exposure_components
python -m src.build_exposure

# Build the Census OCC → 2018 SOC crosswalk and real worker-flow extract
python -m src.occupation_crosswalk
export IPUMS_API_KEY=your_key_from_account.ipums.org
python -m src.build_cps_flows --fetch-ipums --start 2020 --end 2025

# Main flow, exposure, and wage analysis
python analysis/03_worker_flows.py --post-from 2024
make test
```

The API download is temporary. IPUMS microdata and generated CPS flow panels
stay outside Git; the repository commits the extraction code, aggregate result
tables, figure, and provenance.

## Supporting stock analysis

The earlier ASEC analysis remains as a diagnostic rather than the headline. It
measures young-worker share and young workers per 100,000 employed young
workers. Its baseline association is negative, but it disappears with
occupation-specific trends. Those outcomes measure occupation composition, not
actual worker entry.

```bash
python -m src.build_cps_panel --fetch-ipums \
  --sample asec --start 2020 --end 2025
make research
```

## Caveats

- The design has no clean AI-adoption date and is not causal.
- Flows are adjacent-month transitions, not a direct employer hiring record.
- A job-to-job switch contributes to both an occupation exit and an entry.
- Detailed occupation cells are noisy; flow models require at least 30 linked
  observations per occupation-year.
- October 2025 CPS was not collected, leaving ten linked transition months in
  2025.
- `CPSIDV` validates links but cannot eliminate all survey attrition or coding
  error.
- The modal Census-to-SOC crosswalk loses detail.
- Observed Claude use is a lower bound on total AI use.
- The automation split combines sources from different Anthropic releases.
- Weekly wages are current-dollar and affected by CPS top-code conventions.

## Repository map

```text
src/build_cps_flows.py                Basic Monthly CPS links, flows, and wages
src/build_exposure_components.py      Automation, augmentation, and theory
analysis/03_worker_flows.py           Main flow and wage estimates + figure
src/build_cps_panel.py                Supporting ASEC occupation-age panel
analysis/04_regressions.py            Supporting stock-outcome event study
analysis/05_robustness.py             Trends, placebos, leave-outs, age groups
tests/test_pipeline.py                Deterministic pipeline tests
```

## Data and license

- [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex): CC-BY 4.0
- [IPUMS CPS](https://cps.ipums.org/cps/): IPUMS terms apply
- [BLS CPS](https://www.bls.gov/cps/): U.S. government data
- [GPTs are GPTs](https://github.com/openai/GPTs-are-GPTs): theoretical exposure
- Repository code: MIT

See [`CITATION.cff`](CITATION.cff), [`data/README.md`](data/README.md), and
[`data/provenance.json`](data/provenance.json).
