"""
Shared configuration for the AI Career Launch Monitor.

Central place for file paths, canonical data-source URLs, and the SOC
major-group crosswalk used throughout the pipeline.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "figures"

RAW_ANTHROPIC = RAW / "anthropic"
RAW_BLS = RAW / "bls"
RAW_ONET = RAW / "onet"
RAW_OPENAI = RAW / "openai"

for _p in (RAW_ANTHROPIC, RAW_BLS, RAW_ONET, RAW_OPENAI, PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical data sources (public, documented)
# ---------------------------------------------------------------------------
# Anthropic Economic Index -- Labor Market Impacts release (CC-BY).
# Source of the observed AI-exposure measure at the SOC-occupation level.
AEI_BASE = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main"
AEI_FILES = {
    "job_exposure.csv": f"{AEI_BASE}/labor_market_impacts/job_exposure.csv?download=true",
    "task_penetration.csv": f"{AEI_BASE}/labor_market_impacts/task_penetration.csv?download=true",
}

# Inputs used to split observed exposure into automation and augmentation and
# to add the Eloundou et al. theoretical capability measure.
AEI_AUTOMATION_TASKS = (
    f"{AEI_BASE}/release_2025_03_27/"
    "automation_vs_augmentation_by_task.csv?download=true"
)
AEI_ONET_TASKS = (
    f"{AEI_BASE}/release_2025_03_27/onet_task_statements.csv?download=true"
)
OPENAI_THEORETICAL_EXPOSURE = (
    "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/"
    "main/data/occ_level.csv"
)
BLS_SOC_2010_2018_CROSSWALK = (
    "https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx"
)

# BLS OEWS (Occupational Employment and Wage Statistics) -- national, by SOC.
# Provides employment levels and wages by occupation. Used as denominators and
# controls. May 2025 is the latest published annual table as of August 2026.
BLS_OEWS_NATIONAL = "https://www.bls.gov/oes/special-requests/oesm25nat.zip"

# O*NET database (individual files). Task, skill, and education/zone data.
ONET_DB_VERSION = "30_3"  # current production release as of May 2026
ONET_JOBZONES = f"https://www.onetcenter.org/dl_files/database/db_{ONET_DB_VERSION}_text/Job%20Zones.txt"
ONET_EDU = f"https://www.onetcenter.org/dl_files/database/db_{ONET_DB_VERSION}_text/Education%2C%20Training%2C%20and%20Experience.txt"

# CPS young-worker panel. Occupation x age employment shares over time come from
# the CPS. These are NOT bundled here (large microdata / IPUMS terms). See
# src/download_bls.py and data/README.md for how to build data/raw/bls/cps_panel.csv.
CPS_PANEL_EXPECTED = RAW_BLS / "cps_panel.csv"
CPS_FLOW_MONTHLY = RAW_BLS / "cps_flows_monthly.csv"
CPS_FLOW_ANNUAL = RAW_BLS / "cps_flows_annual.csv"
CPS_WAGE_PANEL = RAW_BLS / "cps_young_wages.csv"

# ---------------------------------------------------------------------------
# SOC 2018 major-group labels (public taxonomy, keyed by the 2-digit prefix)
# ---------------------------------------------------------------------------
SOC_MAJOR_GROUPS = {
    "11": "Management",
    "13": "Business & Financial Operations",
    "15": "Computer & Mathematical",
    "17": "Architecture & Engineering",
    "19": "Life, Physical, & Social Science",
    "21": "Community & Social Service",
    "23": "Legal",
    "25": "Educational Instruction & Library",
    "27": "Arts, Design, Entertainment, Sports, & Media",
    "29": "Healthcare Practitioners & Technical",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Preparation & Serving",
    "37": "Building & Grounds Cleaning & Maintenance",
    "39": "Personal Care & Service",
    "41": "Sales & Related",
    "43": "Office & Administrative Support",
    "45": "Farming, Fishing, & Forestry",
    "47": "Construction & Extraction",
    "49": "Installation, Maintenance, & Repair",
    "51": "Production",
    "53": "Transportation & Material Moving",
    "55": "Military-Specific",
}


def soc_major(occ_code: str) -> str:
    """Return the 2-digit SOC major-group code from a detailed SOC code."""
    return str(occ_code)[:2]


def soc_major_label(occ_code: str) -> str:
    """Return the human-readable SOC major-group label for a detailed SOC code."""
    return SOC_MAJOR_GROUPS.get(soc_major(occ_code), "Unknown")
