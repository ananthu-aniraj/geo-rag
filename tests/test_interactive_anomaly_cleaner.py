import os
import tempfile
import unittest

import h3
import pandas as pd
from fastapi.testclient import TestClient

from src.processing.interactive_anomaly_cleaner import (
    ContinentDatasetIndex,
    create_app,
    execute_parquet_purge,
)


class TestInteractiveAnomalyCleaner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.parquet_path = os.path.join(self.tmpdir.name, "test_dataset.parquet")
        self.rules_path = os.path.join(self.tmpdir.name, "rules.json")

        # Create mock data with 2 distinct H3 res 11 cells in Africa
        # We pick lat/lons in Africa
        c1_res11 = h3.latlng_to_cell(1.29, 36.82, 11)  # Nairobi, Kenya
        c2_res11 = h3.latlng_to_cell(0.34, 32.58, 11)  # Kampala, Uganda

        # Add a cell in Europe that should be filtered out when continent is Africa
        c3_res11 = h3.latlng_to_cell(48.85, 2.35, 11)  # Paris, France

        df = pd.DataFrame(
            {
                "Photo_ID": ["101", "102", "103", "104", "105"],
                "Platform": [
                    "mapillary",
                    "flickr",
                    "mapillary",
                    "mapillary",
                    "mapillary",
                ],
                "Latitude": [1.29, 1.29, 0.34, 0.34, 48.85],
                "Longitude": [36.82, 36.82, 32.58, 32.58, 2.35],
                "Image_URL": [
                    "mapillary://101",
                    "https://flickr.com/102.jpg",
                    "mapillary://103",
                    "mapillary://104",
                    "mapillary://105",
                ],
                "Captured_At": [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                ],
                "H3_Cell": [c1_res11, c1_res11, c2_res11, c2_res11, c3_res11],
                "continent": ["Africa", "Africa", "Africa", "Africa", "Europe"],
                "country": ["Kenya", "Kenya", "Uganda", "Uganda", "France"],
                "Koppen_Code": ["Cfb", "Cfb", "Af", "Af", "Cfb"],
                "Koppen_Desc": [
                    "Marine West Coast",
                    "Marine West Coast",
                    "Tropical Rainforest",
                    "Tropical Rainforest",
                    "Marine West Coast",
                ],
            }
        )
        df.to_parquet(self.parquet_path, index=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_continent_indexing(self):
        index = ContinentDatasetIndex(
            parquet_path=self.parquet_path,
            continent="Africa",
            h3_res=4,
            rules_path=self.rules_path,
        )
        # Should only contain the 4 African rows, not the 1 European row
        self.assertEqual(index.total_records, 4)
        self.assertGreater(len(index.cell_stats), 0)

        # Retrieve a cell and verify sample data
        sample_cell = list(index.cell_stats.keys())[0]
        details = index.get_cell_samples(sample_cell)
        self.assertIn("samples", details)
        self.assertIn("countries", details)
        self.assertIn("mapillary", details)

    def test_purge_execution(self):
        index = ContinentDatasetIndex(
            parquet_path=self.parquet_path,
            continent="Africa",
            h3_res=4,
            rules_path=self.rules_path,
        )
        sample_cell = list(index.cell_stats.keys())[0]

        # Purge only Mapillary from sample_cell
        rules = [
            {"cell": sample_cell, "platform": "mapillary", "expand_sequence": False}
        ]
        output_parquet = os.path.join(self.tmpdir.name, "cleaned.parquet")

        res = execute_parquet_purge(
            self.parquet_path, output_parquet, rules, df_index=index
        )
        self.assertGreater(res["removed_rows"], 0)

        # Check output parquet
        cleaned_df = pd.read_parquet(output_parquet)
        self.assertEqual(len(cleaned_df), 5 - res["removed_rows"])
        # Europe row should still be intact
        self.assertIn("Europe", cleaned_df["continent"].values)

    def test_fastapi_endpoints(self):
        index = ContinentDatasetIndex(
            parquet_path=self.parquet_path,
            continent="Africa",
            h3_res=4,
            rules_path=self.rules_path,
        )
        output_parquet = os.path.join(self.tmpdir.name, "cleaned.parquet")
        app = create_app(index, output_parquet)
        client = TestClient(app)

        # 1. Dashboard HTML
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Geo-RAG Anomaly Cleaner", r.text)

        # 2. GeoJSON cells
        r = client.get("/api/cells")
        self.assertEqual(r.status_code, 200)
        geojson = r.json()
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertGreater(len(geojson["features"]), 0)

        # 3. Cell details
        sample_cell = geojson["features"][0]["properties"]["cell"]
        r = client.get(f"/api/cell/{sample_cell}")
        self.assertEqual(r.status_code, 200)
        cell_info = r.json()
        self.assertEqual(cell_info["cell"], sample_cell)

        # 4. Rules add and remove
        r = client.post(
            "/api/rules/add",
            json={
                "cell": sample_cell,
                "platform": "mapillary",
                "expand_sequence": False,
            },
        )
        self.assertEqual(r.status_code, 200)

        r = client.get("/api/rules")
        self.assertEqual(r.status_code, 200)
        rules = r.json()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["cell"], sample_cell)

        r = client.post("/api/rules/remove", json={"cell": sample_cell})
        self.assertEqual(r.status_code, 200)

        r = client.get("/api/rules")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 0)


if __name__ == "__main__":
    unittest.main()
