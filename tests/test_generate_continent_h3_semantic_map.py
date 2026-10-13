import os
import shutil
import tempfile
import unittest

import h3
import pandas as pd

from src.visualization.generate_continent_h3_semantic_map import (
    generate_continent_map,
    normalize_continent_name,
)


class TestGenerateContinentH3SemanticMap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normalize_continent_name(self):
        self.assertEqual(normalize_continent_name("africa"), "Africa")
        self.assertEqual(normalize_continent_name("EUROPE"), "Europe")
        self.assertEqual(normalize_continent_name("north america"), "North America")

    def test_generate_continent_map(self):
        # 1. Create dummy H3 index for Africa at resolution 4
        # Latitude ~0.0, Longitude ~20.0 is in central Africa
        cell_11 = h3.latlng_to_cell(0.0, 20.0, 11)
        cell_4 = h3.cell_to_parent(cell_11, 4)

        df_index = pd.DataFrame(
            [
                {
                    "resolution": 4,
                    "query_cell": cell_4,
                    "Season": "Summer",
                    "Time_Of_Day": "Day",
                    "Koppen_Code": "Af",
                    "Koppen_Desc": "Tropical rainforest",
                    "country": "DR Congo",
                    "continent": "Africa",
                    "cluster_id": 1,
                    "cluster_label": "Tropical Forest",
                    "cluster_description": "Dense equatorial rainforest canopy",
                    "parent_cluster_id": 10,
                    "parent_cluster_label": "Forest",
                    "image_count": 25,
                }
            ]
        )
        index_path = os.path.join(self.temp_dir, "test_h3_index.parquet")
        df_index.to_parquet(index_path)

        # 2. Create dummy cleaned photo dataset (without cluster labels or embeddings)
        df_data = pd.DataFrame(
            [
                {
                    "Photo_ID": f"photo_{i}",
                    "Platform": "Mapillary" if i % 2 == 0 else "Flickr",
                    "Image_URL": f"https://example.com/photo_{i}.jpg",
                    "H3_Cell": cell_11,
                    "Latitude": 0.0 + i * 0.001,
                    "Longitude": 20.0 + i * 0.001,
                    "Captured_At": "2023-06-15",
                    "continent": "Africa",
                }
                for i in range(10)
            ]
        )
        dataset_path = os.path.join(self.temp_dir, "test_dataset_cleaned.parquet")
        df_data.to_parquet(dataset_path)

        out_html = os.path.join(self.temp_dir, "africa_res4_map.html")

        # 3. Generate map
        generate_continent_map(
            continent="Africa",
            index_path=index_path,
            dataset_path=dataset_path,
            output_html=out_html,
            h3_res=4,
            min_count=5,
            samples_per_cell=4,
            replace_prefix="http://localhost:8000/",
        )

        self.assertTrue(os.path.exists(out_html))
        with open(out_html, "r", encoding="utf-8") as f:
            html = f.read()

        # Check essential components
        self.assertIn("Africa", html)
        self.assertIn(cell_4, html)
        self.assertIn("Tropical Forest", html)
        # Check SSH tunneling / prefix-remapping control injection
        self.assertIn("findPrefixInput", html)
        self.assertIn("replacePrefixInput", html)
        self.assertIn("Offline Image Server Mapping", html)
        self.assertIn("http://localhost:8000/", html)

    def test_cli_parsing(self):
        from unittest.mock import patch

        from src.visualization.generate_continent_h3_semantic_map import main

        # Create dummy index and dataset files
        idx_path = os.path.join(self.temp_dir, "cli_idx.parquet")
        data_path = os.path.join(self.temp_dir, "cli_data.parquet")
        pd.DataFrame({"resolution": [4]}).to_parquet(idx_path)
        pd.DataFrame({"continent": ["Africa"]}).to_parquet(data_path)

        out_html = os.path.join(self.temp_dir, "cli_map.html")

        test_args = [
            "generate_continent_h3_semantic_map.py",
            "--continent",
            "Africa",
            "--index",
            idx_path,
            "--dataset",
            data_path,
            "--output",
            out_html,
        ]

        with patch("sys.argv", test_args):
            with patch(
                "src.visualization.generate_continent_h3_semantic_map.generate_continent_map"
            ) as mock_gen:
                main()
                mock_gen.assert_called_once()
                args, kwargs = mock_gen.call_args
                self.assertEqual(kwargs["continent"], "Africa")
                self.assertEqual(kwargs["index_path"], idx_path)
                self.assertEqual(kwargs["dataset_path"], data_path)
                self.assertEqual(kwargs["output_html"], out_html)
