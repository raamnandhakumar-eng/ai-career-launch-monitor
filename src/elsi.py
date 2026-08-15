"""
Entry-Level Squeeze Index (ELSI), a descriptive screening index.

An occupation-level composite flagging where AI exposure and a shrinking
young-worker share coincide in jobs that historically depend on junior entrants.

    ELSI = f(exposure) * f(young-share decline) * f(entry-level dependence)

Each factor is min-max scaled to [0, 1] across occupations so the product is
comparable and bounded. Rationale:
  - exposure                -> real Anthropic observed exposure (the AI signal)
  - young-share decline     -> max(0, -change in age 20-29 employment share)
                               (only declines count; growth is not a squeeze)
  - entry-level dependence  -> baseline young-worker share (how reliant the
                               occupation is on junior workers to begin with)

A high ELSI is an *early-warning* flag, not a causal claim. See the README on
interpretation and confounders.

    python -m src.elsi            # real data: writes elsi.csv + prints top 20
"""
import argparse

import pandas as pd

from src.config import PROCESSED
from src.data_guard import guard_publication


def _unit(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def calculate(occ: pd.DataFrame) -> pd.DataFrame:
    """Calculate ELSI components and tiers from an occupation-level panel."""
    d = occ.dropna(subset=["young_share_change", "young_share_base"]).copy()
    if d.empty:
        raise ValueError("No occupations have baseline and recent young-worker shares")

    d["decline"] = (-d["young_share_change"]).clip(lower=0)   # only declines
    d["f_exposure"] = _unit(d["ai_exposure"])
    d["f_decline"] = _unit(d["decline"])
    d["f_entry_dep"] = _unit(d["young_share_base"])

    d["elsi"] = d["f_exposure"] * d["f_decline"] * d["f_entry_dep"]
    d["elsi"] = _unit(d["elsi"])  # rescale product to [0,1] for readability

    # New CPS builds include unweighted record counts. Older aggregate panels
    # cannot recover these counts, so they are flagged instead of receiving a
    # false precision label.
    d["precision_flag"] = "sample unavailable"
    if "young_sample_n_recent" in d:
        d["precision_flag"] = "standard"
        d.loc[d["young_sample_n_recent"] < 100, "precision_flag"] = "low precision"

    # Tiers by percentile rank among positive ELSI values. This stays stable
    # when ties would make quantile cut points non-unique.
    pos = d["elsi"] > 0
    d["risk_tier"] = "low"
    if pos.any():
        pct = d.loc[pos, "elsi"].rank(method="average", pct=True)
        d.loc[pos & (pct.reindex(d.index).fillna(0) <= 1 / 3), "risk_tier"] = "watch"
        d.loc[pos & (pct.reindex(d.index).fillna(0).between(1 / 3, 2 / 3, inclusive="right")),
              "risk_tier"] = "elevated"
        d.loc[pos & (pct.reindex(d.index).fillna(0) > 2 / 3), "risk_tier"] = "high"

    return d.sort_values("elsi", ascending=False)


def compute(allow_synthetic: bool = False) -> pd.DataFrame:
    synthetic = guard_publication(allow_synthetic)
    occ = pd.read_csv(PROCESSED / "panel_occ.csv", dtype={"occ_code": str})
    d = calculate(occ)

    suffix = "_SYNTHETIC" if synthetic else ""
    out = PROCESSED / f"elsi{suffix}.csv"
    cols = ["occ_code", "title", "soc_major_label", "ai_exposure",
            "young_share_base", "young_share_change", "decline",
            "f_exposure", "f_decline", "f_entry_dep", "elsi", "risk_tier",
            "precision_flag", "employment_recent"]
    if "young_sample_n_recent" in d:
        cols.append("young_sample_n_recent")
    d[cols].to_csv(out, index=False)
    print(f"Wrote {out}")
    print("\nTop 20 Entry-Level Squeeze occupations:")
    show = d.head(20)[["occ_code", "title", "ai_exposure",
                       "young_share_change", "elsi", "risk_tier"]].copy()
    show["young_share_change"] = (show["young_share_change"] * 100).round(1)
    show["ai_exposure"] = show["ai_exposure"].round(2)
    show["elsi"] = show["elsi"].round(3)
    print(show.to_string(index=False))
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-synthetic", action="store_true")
    compute(**vars(parser.parse_args()))
