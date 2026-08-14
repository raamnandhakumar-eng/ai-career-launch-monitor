import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.build_cps_panel import (
    age_band,
    available_sample_ids,
    candidate_sample_ids,
    ipums_extract_spec,
    normalize_soc,
    transform,
)
from src.build_panel import young_worker_panel
from src.data_guard import is_synthetic_panel
from src.elsi import calculate


class CpsBuilderTests(unittest.TestCase):
    def test_age_bands(self):
        values = pd.Series([19, 20, 24, 25, 29, 30, 39, 40, 54, 55, 80])
        got = age_band(values).astype(object).where(lambda s: s.notna(), None).tolist()
        self.assertEqual(
            got,
            [None, "20-24", "20-24", "25-29", "25-29", "30-39",
             "30-39", "40-54", "40-54", "55+", "55+"],
        )

    def test_soc_normalization(self):
        self.assertEqual(normalize_soc("15-1252.00"), "15-1252")
        self.assertEqual(normalize_soc(151252), "15-1252")
        self.assertIsNone(normalize_soc("not-a-code"))

    def test_asec_extract_spec_uses_current_occ_and_weight(self):
        spec = ipums_extract_spec(2020, 2022, "asec")
        self.assertEqual(
            spec["samples"],
            ["cps2020_03s", "cps2021_03s", "cps2022_03s"],
        )
        self.assertIn("OCC", spec["variables"])
        self.assertNotIn("OCC2010", spec["variables"])
        self.assertEqual(spec["weight"], "ASECWT")
        self.assertEqual(spec["frequency"], "annual")

    def test_basic_sample_selection_skips_unpublished_months(self):
        published = {
            sample_id: {}
            for sample_id in candidate_sample_ids(2025, 2025, "basic")
            if sample_id != "cps2025_10b"
        }
        selected = available_sample_ids(2025, 2025, "basic", published)
        self.assertEqual(len(selected), 11)
        self.assertNotIn("cps2025_10b", selected)

    def test_automated_crosswalk_route_rejects_pre_2020(self):
        with self.assertRaisesRegex(ValueError, "2020 or later"):
            candidate_sample_ids(2019, 2020, "asec")

    def test_repo_crosswalk_rejects_occ2010(self):
        raw = pd.DataFrame({
            "YEAR": [2024],
            "MONTH": [1],
            "AGE": [24],
            "EMPSTAT": [10],
            "WTFINL": [100],
            "OCC2010": [4700],
        })
        with self.assertRaisesRegex(ValueError, "2010-basis"):
            transform(raw, census_occ_column="OCC2010")

    def test_monthly_weights_become_annual_average(self):
        raw = pd.DataFrame({
            "YEAR": [2024] * 4,
            "MONTH": [1, 1, 2, 2],
            "AGE": [22, 32, 22, 32],
            "EMPSTAT": [10, 10, 12, 10],
            "WTFINL": [100, 300, 120, 280],
            "OCCSOC": ["15-1251"] * 4,
        })
        panel, metadata = transform(
            raw,
            soc_column="OCCSOC",
            census_occ_column=None,
            min_match_rate=1,
        )
        by_band = panel.set_index("age_band")["employed"].to_dict()
        self.assertEqual(by_band["20-24"], 110)
        self.assertEqual(by_band["30-39"], 290)
        self.assertEqual(metadata["month_coverage"], {"2024": [1, 2]})

    def test_contemporary_occ_maps_to_dashed_soc(self):
        raw = pd.DataFrame({
            "YEAR": [2024],
            "MONTH": [1],
            "AGE": [24],
            "EMPSTAT": [10],
            "WTFINL": [100],
            "OCC": [4700],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            crosswalk = Path(temp_dir) / "crosswalk.csv"
            pd.DataFrame({
                "occ_census": ["4700"],
                "occ_code": ["41-2031"],
            }).to_csv(crosswalk, index=False)
            panel, _ = transform(raw, crosswalk_path=crosswalk, min_match_rate=1)
        self.assertEqual(panel.loc[0, "occ_code"], "41-2031")


class PanelTests(unittest.TestCase):
    def test_young_share(self):
        cps = pd.DataFrame({
            "year": [2024] * 3,
            "occ_code": ["15-1251"] * 3,
            "age_band": ["20-24", "25-29", "30-39"],
            "employed": [10, 20, 70],
        })
        out = young_worker_panel(cps)
        self.assertAlmostEqual(out.loc[0, "young_share"], 0.30)

    def test_synthetic_marker_uses_values_not_column_presence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "panel.csv"
            pd.DataFrame({"synthetic": [False]}).to_csv(path, index=False)
            self.assertFalse(is_synthetic_panel(path))
            pd.DataFrame({"synthetic": [True]}).to_csv(path, index=False)
            self.assertTrue(is_synthetic_panel(path))


class ElsiTests(unittest.TestCase):
    def test_index_is_bounded_and_declines_only(self):
        occ = pd.DataFrame({
            "occ_code": ["a", "b", "c", "d"],
            "title": ["A", "B", "C", "D"],
            "soc_major_label": ["X"] * 4,
            "ai_exposure": [0.8, 0.4, 0.2, 0.1],
            "young_share_base": [0.4, 0.3, 0.2, 0.1],
            "young_share_change": [-0.10, -0.05, 0.02, 0.00],
            "employment_recent": [100] * 4,
        })
        out = calculate(occ)
        self.assertTrue(out["elsi"].between(0, 1).all())
        self.assertEqual(out.loc[out["occ_code"] == "c", "elsi"].iloc[0], 0)
        self.assertEqual(out.iloc[0]["occ_code"], "a")


if __name__ == "__main__":
    unittest.main()
