import os
import tempfile
import threading
import unittest

import h3
import pandas as pd
import requests

from src.processing.interactive_anomaly_cleaner import (
    ContinentDatasetIndex,
    create_server,
    execute_parquet_purge,
)


class TestInteractiveAnomalyCleaner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.parquet_path = os.path.join(self.tmpdir.name, "test_dataset.parquet")
        self.rules_path = os.path.join(self.tmpdir.name, "rules.json")

        # Create mock data with 2 distinct H3 res 11 cells in Africa
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
                "Season": ["Summer", "Summer", "Spring", "Spring", "Winter"],
                "Time_Of_Day": [
                    "Afternoon",
                    "Morning",
                    "Afternoon",
                    "Evening",
                    "Morning",
                ],
                "License": ["CC-BY-SA", "CC-BY", "CC-BY-SA", "CC-BY-SA", "CC0"],
                "Region": [
                    "Nairobi County",
                    "Nairobi County",
                    "Central Region",
                    "Central Region",
                    "Île-de-France",
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
        self.assertEqual(index.dataset_summary["total_records"], 4)
        self.assertEqual(index.dataset_summary["unique_countries"], 2)

        # Retrieve a cell and verify sample data & per-image location metadata
        sample_cell = list(index.cell_stats.keys())[0]
        details = index.get_cell_samples(sample_cell)
        self.assertIn("samples", details)
        self.assertIn("countries", details)
        self.assertIn("mapillary", details)
        self.assertIn("bbox", details)
        self.assertIn("platforms", details)
        self.assertGreater(len(details["samples"]), 0)

        first_sample = details["samples"][0]
        self.assertIn(first_sample["country"], ("Kenya", "Uganda"))
        self.assertEqual(first_sample["continent"], "Africa")
        self.assertIn(first_sample["region"], ("Nairobi County", "Central Region"))
        self.assertIn(first_sample["country"], first_sample["location_label"])
        self.assertIn("h3_cell", first_sample)
        self.assertIn("koppen_code", first_sample)
        self.assertIn("koppen_desc", first_sample)
        self.assertIn("season", first_sample)
        self.assertIn("time_of_day", first_sample)
        self.assertIn("license", first_sample)
        self.assertIn("osm_url", first_sample)
        self.assertIn("gmaps_url", first_sample)

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

    def test_http_server_endpoints(self):
        index = ContinentDatasetIndex(
            parquet_path=self.parquet_path,
            continent="Africa",
            h3_res=4,
            rules_path=self.rules_path,
        )
        output_parquet = os.path.join(self.tmpdir.name, "cleaned.parquet")

        # Start server on an ephemeral OS-assigned port
        server = create_server(index, output_parquet, host="127.0.0.1", port=0)
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            # 1. Dashboard HTML
            r = requests.get(f"{base_url}/")
            self.assertEqual(r.status_code, 200)
            self.assertIn("Geo-RAG Anomaly Cleaner", r.text)

            # 1b. Dataset summary endpoint
            r = requests.get(f"{base_url}/api/summary")
            self.assertEqual(r.status_code, 200)
            summary = r.json()
            self.assertEqual(summary["total_records"], 4)
            self.assertEqual(summary["unique_countries"], 2)

            # 2. GeoJSON cells
            r = requests.get(f"{base_url}/api/cells")
            self.assertEqual(r.status_code, 200)
            geojson = r.json()
            self.assertEqual(geojson["type"], "FeatureCollection")
            self.assertGreater(len(geojson["features"]), 0)
            first_props = geojson["features"][0]["properties"]
            self.assertEqual(first_props["count"], first_props["total"])
            self.assertIn("centroid", first_props)

            # 3. Cell details
            sample_cell = first_props["cell"]
            r = requests.get(f"{base_url}/api/cell/{sample_cell}")
            self.assertEqual(r.status_code, 200)
            cell_info = r.json()
            self.assertEqual(cell_info["cell"], sample_cell)
            self.assertGreater(len(cell_info["samples"]), 0)
            self.assertIn("location_label", cell_info["samples"][0])
            self.assertIn("country", cell_info["samples"][0])

            # 4. Rules add and remove
            r = requests.post(
                f"{base_url}/api/rules/add",
                json={
                    "cell": sample_cell,
                    "platform": "mapillary",
                    "expand_sequence": False,
                },
            )
            self.assertEqual(r.status_code, 200)

            r = requests.get(f"{base_url}/api/rules")
            self.assertEqual(r.status_code, 200)
            rules = r.json()
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0]["cell"], sample_cell)

            r = requests.post(
                f"{base_url}/api/rules/remove", json={"cell": sample_cell}
            )
            self.assertEqual(r.status_code, 200)

            r = requests.get(f"{base_url}/api/rules")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(len(r.json()), 0)

            # 5. Sequence track endpoint
            r = requests.get(f"{base_url}/api/sequence/track/101")
            self.assertEqual(r.status_code, 200)
            track_data = r.json()
            self.assertIn("is_valid", track_data)
            self.assertIn("reason", track_data)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
