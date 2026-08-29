import os
import shutil
import sys
import unittest
from unittest.mock import patch

import h3
import pandas as pd

from src.indexing.build_spatial_semantic_index import main
from src.utils.io import load_dataframe, save_dataframe


class TestBuildSpatialSemanticIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch_index"
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_parquet = os.path.join(self.test_dir, "clustered.parquet")
        self.output_parquet = os.path.join(self.test_dir, "aggregated_index.parquet")

        # 1. Create a dummy dataset (10 samples)
        # We specify coordinates and map them to H3 cells at resolution 11
        latitudes = [
            40.7128,
            48.8566,
            35.6762,
            -33.8688,
            51.5074,
            40.7306,
            48.8647,
            35.6895,
            -33.8599,
            51.5154,
        ]
        longitudes = [
            -74.0060,
            2.3522,
            139.6503,
            151.2093,
            -0.1278,
            -73.9352,
            2.3490,
            139.6917,
            151.2111,
            -0.1419,
        ]
        # Define compatibility wrapper for H3 version 3 vs 4
        if hasattr(h3, "latlng_to_cell"):
            h3_cells = [
                h3.latlng_to_cell(lat, lon, 11)
                for lat, lon in zip(latitudes, longitudes)
            ]
        else:
            h3_cells = [
                h3.geo_to_h3(lat, lon, 11) for lat, lon in zip(latitudes, longitudes)
            ]

        df = pd.DataFrame(
            {
                "Photo_ID": [str(i) for i in range(10)],
                "Platform": ["flickr"] * 10,
                "Latitude": latitudes,
                "Longitude": longitudes,
                "H3_Cell": h3_cells,
                "Captured_At": ["2024-01-01"] * 10,
                "Image_URL": [f"url_{i}" for i in range(10)],
                "cluster_id": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                "cluster_label": [f"Label {i%5}" for i in range(10)],
                "cluster_description": [f"Description {i%5}" for i in range(10)],
                "parent_cluster_id": [0, 0, 1, 1, 2, 0, 0, 1, 1, 2],
                "parent_cluster_label": [f"Parent {i%3}" for i in range(10)],
                "Season": [
                    "Winter",
                    "Summer",
                    "Spring",
                    "Autumn",
                    "Winter",
                    "Winter",
                    "Summer",
                    "Spring",
                    "Autumn",
                    "Winter",
                ],
                "Time_Of_Day": [
                    "Day",
                    "Night",
                    "Morning",
                    "Evening",
                    "Day",
                    "Day",
                    "Night",
                    "Morning",
                    "Evening",
                    "Day",
                ],
                "Koppen_Code": [
                    "Cfa",
                    "Cfb",
                    "Cfa",
                    "Cfb",
                    "Cfb",
                    "Cfa",
                    "Cfb",
                    "Cfa",
                    "Cfb",
                    "Cfb",
                ],
                "Koppen_Desc": [
                    "Humid subtropical",
                    "Oceanic",
                    "Humid subtropical",
                    "Oceanic",
                    "Oceanic",
                    "Humid subtropical",
                    "Oceanic",
                    "Humid subtropical",
                    "Oceanic",
                    "Oceanic",
                ],
                "country": [
                    "USA",
                    "France",
                    "Japan",
                    "Australia",
                    "UK",
                    "USA",
                    "France",
                    "Japan",
                    "Australia",
                    "UK",
                ],
                "continent": [
                    "North America",
                    "Europe",
                    "Asia",
                    "Oceania",
                    "Europe",
                    "North America",
                    "Europe",
                    "Asia",
                    "Oceania",
                    "Europe",
                ],
            }
        )

        save_dataframe(df, self.input_parquet)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_end_to_end_build_index(self):
        test_args = [
            "build_spatial_semantic_index.py",
            "--input",
            self.input_parquet,
            "--output",
            self.output_parquet,
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Verify output exists
        self.assertTrue(os.path.exists(self.output_parquet))

        # Load output Parquet and check schema and aggregated rows
        df_index = load_dataframe(self.output_parquet)

        # Check required index columns are present
        required_cols = [
            "query_cell",
            "resolution",
            "Season",
            "Time_Of_Day",
            "Koppen_Code",
            "country",
            "continent",
            "cluster_id",
            "parent_cluster_id",
            "image_count",
            "cluster_label",
            "cluster_description",
            "parent_cluster_label",
        ]
        for col in required_cols:
            self.assertIn(col, df_index.columns)

        # Output resolution should cover 1 through 11
        unique_res = sorted(df_index["resolution"].unique())
        self.assertEqual(unique_res, list(range(1, 12)))

        # Verify that image_count is summed correctly
        self.assertTrue((df_index["image_count"] >= 1).all())


if __name__ == "__main__":
    unittest.main()
