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
from src.occupation_crosswalk import _parse_sheets


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
        sample_n = panel.set_index("age_band")["sample_n"].to_dict()
        self.assertEqual(sample_n["20-24"], 2)
        self.assertEqual(sample_n["30-39"], 2)
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

    def test_official_crosswalk_headers_and_ranges(self):
        sheet = pd.DataFrame([
            [None, "2018 Census Title", "2018 Census Code", "2018 SOC Code"],
            [None, "Category heading", "0010-3550", "11-0000 - 29-0000"],
            [None, "Chief executives", "0010", "11-1011"],
            [None, "General and operations managers", "0020", "11-1021"],
        ])
        out = _parse_sheets({"2018 Census Occ Code List": sheet})
        self.assertEqual(
            out.to_dict("records"),
            [
                {"occ_census": "0010", "occ_code": "11-1011"},
                {"occ_census": "0020", "occ_code": "11-1021"},
            ],
        )


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
        self.assertAlmostEqual(out.loc[0, "young_per_100k"], 100000)

    def test_entry_proxy_sums_to_100k_each_year(self):
        cps = pd.DataFrame({
            "year": [2024] * 4,
            "occ_code": ["15-1251", "15-1251", "41-2031", "41-2031"],
            "age_band": ["20-24", "30-39", "20-24", "30-39"],
            "employed": [25, 75, 75, 25],
        })
        out = young_worker_panel(cps)
        self.assertAlmostEqual(out["young_per_100k"].sum(), 100000)

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
        self.assertTrue((out["precision_flag"] == "sample unavailable").all())

    def test_low_precision_flag_uses_unweighted_young_sample(self):
        occ = pd.DataFrame({
            "occ_code": ["a", "b"],
            "title": ["A", "B"],
            "soc_major_label": ["X", "X"],
            "ai_exposure": [0.8, 0.4],
            "young_share_base": [0.4, 0.3],
            "young_share_change": [-0.10, -0.05],
            "young_sample_n_recent": [99, 100],
            "employment_recent": [1000, 1000],
        })
        out = calculate(occ).set_index("occ_code")
        self.assertEqual(out.loc["a", "precision_flag"], "low precision")
        self.assertEqual(out.loc["b", "precision_flag"], "standard")


if __name__ == "__main__":
    unittest.main()
