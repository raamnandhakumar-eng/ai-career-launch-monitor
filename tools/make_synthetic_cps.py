"""
=============================================================================
  !!!  SYNTHETIC / FAKE DATA  --  FOR PIPELINE PLUMBING TESTS ONLY  !!!
=============================================================================
This writes a data/raw/bls/cps_panel.csv with the CORRECT SCHEMA but MADE-UP
numbers, so build_panel.py / elsi.py / 04_regressions.py can be run end-to-end
before you have real CPS data.

Outputs produced from this file are NOT research findings and MUST NOT be
reported. Replace cps_panel.csv with a real CPS panel (see src/download_bls.py)
before drawing any conclusion. Every run prints a warning; the file is stamped
with a `synthetic` column so downstream code / reviewers can detect it.

    python tools/make_synthetic_cps.py
=============================================================================
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED, RAW_BLS  # noqa: E402

AGE_BANDS = ["20-24", "25-29", "30-39", "40-54", "55+"]
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]


def main(seed: int = 7):
    rng = np.random.default_rng(seed)
    exp = pd.read_csv(PROCESSED / "exposure.csv", dtype={"occ_code": str})

    rows = []
    for _, o in exp.iterrows():
        base_emp = rng.integers(5_000, 500_000)
        # baseline age mix (arbitrary but plausible-shaped)
        base_mix = np.array([0.12, 0.15, 0.26, 0.30, 0.17])
        base_mix = base_mix * rng.uniform(0.8, 1.2, size=5)
        base_mix /= base_mix.sum()
        for yr in YEARS:
            t = (yr - 2022)
            # FAKE embedded pattern: in post-2022 years, higher-exposure occs
            # see the young bands shrink. This is injected on purpose so the
            # plumbing produces a visible signal -- it is NOT evidence.
            drift = 0.010 * o["ai_exposure"] * max(t, 0)
            mix = base_mix.copy()
            mix[0] -= drift * 0.6
            mix[1] -= drift * 0.4
            mix[3] += drift * 0.6
            mix[4] += drift * 0.4
            mix = np.clip(mix, 0.01, None)
            mix /= mix.sum()
            emp = base_emp * rng.uniform(0.95, 1.05)
            for band, share in zip(AGE_BANDS, mix):
                rows.append((yr, o["occ_code"], band,
                             int(emp * share)))

    cps = pd.DataFrame(rows, columns=["year", "occ_code", "age_band", "employed"])
    cps["synthetic"] = True
    RAW_BLS.mkdir(parents=True, exist_ok=True)
    dst = RAW_BLS / "cps_panel.csv"
    cps.to_csv(dst, index=False)
    print("*" * 70)
    print("WROTE SYNTHETIC (FAKE) CPS PANEL:", dst)
    print("Numbers are invented for plumbing tests. DO NOT report results")
    print("derived from this file. Replace with real CPS data before analysis.")
    print("*" * 70)


if __name__ == "__main__":
    main()
