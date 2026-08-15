"""Build occupation-level AI exposure components from public source data.

The output adds three measures to the observed Anthropic occupation spine:

* automation exposure: observed exposure allocated to directive and feedback-
  loop interactions;
* augmentation exposure: observed exposure allocated to task iteration,
  validation, and learning interactions;
* theoretical exposure: Eloundou et al. GPT-4 gamma rating (E1 + E2).

The automation split combines the March 2025 Anthropic task interaction labels
with the 2026 labor-market task-penetration file. The March task statements use
2010 SOC codes, so the official BLS 2010-to-2018 SOC crosswalk is applied before
merging to the 2018 SOC occupation spine. The resulting split is descriptive
and mixed-vintage; it is not a causal automation probability.

    python -m src.build_exposure_components
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    AEI_AUTOMATION_TASKS,
    AEI_FILES,
    AEI_ONET_TASKS,
    BLS_SOC_2010_2018_CROSSWALK,
    OPENAI_THEORETICAL_EXPOSURE,
    RAW_ANTHROPIC,
)

OUTPUT = RAW_ANTHROPIC / "exposure_components.csv"
METADATA = RAW_ANTHROPIC / "exposure_components.metadata.json"


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "ai-career-launch-monitor"}
    )
    with urllib.request.urlopen(request) as response:
        return response.read()


def _download_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_download(url)))


def normalize_task(values: pd.Series) -> pd.Series:
    """Normalize task strings without fuzzy or semantic matching."""
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _soc6(values: pd.Series) -> pd.Series:
    return values.astype(str).str.extract(r"(\d{2}-\d{4})", expand=False)


def parse_soc_crosswalk(blob: bytes) -> pd.DataFrame:
    """Return the official 2010-to-2018 detailed SOC pairs."""
    raw = pd.read_excel(io.BytesIO(blob), header=8, dtype=str)
    columns = {str(column).strip(): column for column in raw.columns}
    required = {"2010 SOC Code", "2018 SOC Code"}
    if not required.issubset(columns):
        raise ValueError("BLS SOC crosswalk has unexpected columns")
    out = pd.DataFrame({
        "soc2010": _soc6(raw[columns["2010 SOC Code"]]),
        "occ_code": _soc6(raw[columns["2018 SOC Code"]]),
    }).dropna().drop_duplicates()
    if out.empty:
        raise ValueError("BLS SOC crosswalk contains no detailed SOC pairs")
    return out


def build_components(
    jobs: pd.DataFrame,
    penetration: pd.DataFrame,
    interactions: pd.DataFrame,
    task_statements: pd.DataFrame,
    theoretical: pd.DataFrame,
    soc_crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Construct the occupation exposure components from loaded inputs."""
    job_required = {"occ_code", "observed_exposure"}
    if not job_required.issubset(jobs):
        raise ValueError(f"job exposure missing columns: {sorted(job_required - set(jobs))}")

    pen = penetration[["task", "penetration"]].copy()
    pen["task_key"] = normalize_task(pen["task"])
    pen["penetration"] = pd.to_numeric(pen["penetration"], errors="coerce")
    pen = pen.groupby("task_key", as_index=False)["penetration"].mean()

    interaction_columns = [
        "feedback_loop", "directive", "task_iteration", "validation", "learning"
    ]
    needed = {"task_name", *interaction_columns}
    if not needed.issubset(interactions):
        raise ValueError(
            f"automation task data missing columns: {sorted(needed - set(interactions))}"
        )
    modes = interactions[["task_name", *interaction_columns]].copy()
    modes["task_key"] = normalize_task(modes["task_name"])
    for column in interaction_columns:
        modes[column] = pd.to_numeric(modes[column], errors="coerce")
    modes = modes.drop_duplicates("task_key")

    tasks = task_statements[["O*NET-SOC Code", "Task"]].copy()
    tasks["soc2010"] = _soc6(tasks["O*NET-SOC Code"])
    tasks["task_key"] = normalize_task(tasks["Task"])
    tasks = tasks.dropna(subset=["soc2010"]).drop_duplicates(["soc2010", "task_key"])

    classified = (
        tasks[["soc2010", "task_key"]]
        .merge(pen, on="task_key", how="inner")
        .merge(modes.drop(columns="task_name"), on="task_key", how="inner")
        .merge(soc_crosswalk[["soc2010", "occ_code"]], on="soc2010", how="inner")
    )
    classified["automation_mass"] = classified["penetration"] * (
        classified["directive"] + classified["feedback_loop"]
    )
    classified["augmentation_mass"] = classified["penetration"] * (
        classified["task_iteration"]
        + classified["validation"]
        + classified["learning"]
    )
    split = (
        classified.groupby("occ_code", as_index=False)
        .agg(
            automation_mass=("automation_mass", "sum"),
            augmentation_mass=("augmentation_mass", "sum"),
            classified_task_n=("task_key", "nunique"),
        )
    )
    split["classified_mass"] = split["automation_mass"] + split["augmentation_mass"]
    split["automation_share"] = split["automation_mass"] / split["classified_mass"]
    split["augmentation_share"] = split["augmentation_mass"] / split["classified_mass"]
    split.loc[split["classified_mass"] <= 0, ["automation_share", "augmentation_share"]] = np.nan

    theory = theoretical.copy()
    theory["occ_code"] = _soc6(theory["O*NET-SOC Code"])
    theory["dv_rating_gamma"] = pd.to_numeric(
        theory["dv_rating_gamma"], errors="coerce"
    )
    theory["dv_rating_beta"] = pd.to_numeric(
        theory["dv_rating_beta"], errors="coerce"
    )
    theory = (
        theory.groupby("occ_code", as_index=False)
        .agg(
            theoretical_exposure=("dv_rating_gamma", "mean"),
            theoretical_exposure_conservative=("dv_rating_beta", "mean"),
        )
    )

    out = (
        jobs[["occ_code", "observed_exposure"]]
        .merge(split, on="occ_code", how="left")
        .merge(theory, on="occ_code", how="left")
    )
    out["automation_exposure"] = out["observed_exposure"] * out["automation_share"]
    out["augmentation_exposure"] = out["observed_exposure"] * out["augmentation_share"]
    zero = out["observed_exposure"].eq(0)
    out.loc[zero, ["automation_exposure", "augmentation_exposure"]] = 0.0

    positive_match_rate = float(
        out.loc[out["observed_exposure"].gt(0), "automation_exposure"].notna().mean()
    )
    keep = [
        "occ_code", "automation_exposure", "augmentation_exposure",
        "theoretical_exposure", "theoretical_exposure_conservative",
        "automation_share", "augmentation_share", "classified_task_n",
    ]
    out = out[keep].sort_values("occ_code").reset_index(drop=True)
    metadata = {
        "created": date.today().isoformat(),
        "occupations": int(len(out)),
        "theoretical_match_rate": float(out["theoretical_exposure"].notna().mean()),
        "automation_match_rate_positive_observed": positive_match_rate,
        "automation_definition": (
            "Observed exposure multiplied by the penetration-weighted share of "
            "directive plus feedback-loop interactions among classified tasks."
        ),
        "augmentation_definition": (
            "Observed exposure multiplied by the penetration-weighted share of task "
            "iteration, validation, and learning interactions among classified tasks."
        ),
        "theoretical_definition": "GPT-4 gamma rating (E1 + E2) from Eloundou et al.",
        "sources": {
            "anthropic_task_penetration": AEI_FILES["task_penetration.csv"],
            "anthropic_interaction_labels": AEI_AUTOMATION_TASKS,
            "anthropic_onet_tasks": AEI_ONET_TASKS,
            "openai_theoretical_exposure": OPENAI_THEORETICAL_EXPOSURE,
            "bls_soc_crosswalk": BLS_SOC_2010_2018_CROSSWALK,
        },
        "limitation": (
            "The automation split combines March 2025 interaction labels with the "
            "2026 task-penetration release and is unavailable where no task can be "
            "classified. It is a descriptive mixed-vintage split."
        ),
    }
    return out, metadata


def build() -> pd.DataFrame:
    jobs = pd.read_csv(RAW_ANTHROPIC / "job_exposure.csv", dtype={"occ_code": str})
    print("Downloading public exposure component inputs...")
    penetration = _download_csv(AEI_FILES["task_penetration.csv"])
    interactions = _download_csv(AEI_AUTOMATION_TASKS)
    task_statements = _download_csv(AEI_ONET_TASKS)
    theoretical = _download_csv(OPENAI_THEORETICAL_EXPOSURE)
    crosswalk = parse_soc_crosswalk(_download(BLS_SOC_2010_2018_CROSSWALK))
    out, metadata = build_components(
        jobs, penetration, interactions, task_statements, theoretical, crosswalk
    )
    out.to_csv(OUTPUT, index=False)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(out)} occupations)")
    print(
        "Automation split coverage among positive-observed occupations: "
        f"{metadata['automation_match_rate_positive_observed']:.1%}"
    )
    return out


if __name__ == "__main__":
    build()
