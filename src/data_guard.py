"""Shared safeguards that prevent synthetic results from looking publishable."""

from pathlib import Path

import pandas as pd

from src.config import CPS_PANEL_EXPECTED


TRUE_VALUES = {"1", "true", "t", "yes", "y"}


def is_synthetic_panel(path: Path = CPS_PANEL_EXPECTED) -> bool:
    """Return whether a CPS panel explicitly identifies itself as synthetic."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build a real CPS panel or run the synthetic smoke test."
        )
    head = pd.read_csv(path, usecols=lambda c: c == "synthetic", nrows=100)
    if "synthetic" not in head.columns:
        return False
    values = head["synthetic"].dropna().astype(str).str.lower().str.strip()
    return bool(values.isin(TRUE_VALUES).any())


def guard_publication(allow_synthetic: bool, path: Path = CPS_PANEL_EXPECTED) -> bool:
    """Refuse synthetic analysis unless the caller explicitly opts into a test run."""
    synthetic = is_synthetic_panel(path)
    if synthetic and not allow_synthetic:
        raise RuntimeError(
            "Synthetic CPS data detected. Re-run with --allow-synthetic only for a "
            "pipeline smoke test. Synthetic outputs are not findings."
        )
    return synthetic
