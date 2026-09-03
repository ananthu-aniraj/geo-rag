import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image

from src.indexing.relabel_failed_clusters import main, save_dataset
from src.utils.io import load_dataframe, save_dataframe


class TestRelabelFailedClusters(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch_relabel"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir)

    def test_save_dataset_direct(self):
        # Directly test save_dataset on a DataFrame with cluster_id and parent_cluster_id
        df = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3", "4"],
                "cluster_id": [0, 0, 1, 1],
                "parent_cluster_id": [0, 0, 0, 0],
                "cluster_label": [
                    "Error Labeling",
                    "Error Labeling",
                    "Unlabeled",
                    "Unlabeled",
                ],
                "cluster_description": ["", "", "", ""],
                "parent_cluster_label": ["Error Labeling"] * 4,
            }
        )
        out_parquet = os.path.join(self.test_dir, "test_save_dataset.parquet")

        final_results = {
            0: (
                "Broadleaved forest",
                "Dense forest with deciduous trees.",
                "Forest visual description",
            ),
            1: (
                "Herbaceous cropland",
                "Cultivated farm field with crops.",
                "Cropland visual description",
            ),
        }
        parent_results = {
            0: (
                "Natural vegetation",
                "Broad ecological vegetation.",
                "Natural visual description",
            ),
        }

        save_dataset(df, final_results, parent_results, out_parquet)

        df_loaded = load_dataframe(out_parquet)
        self.assertEqual(len(df_loaded), 4)
        self.assertEqual(
            df_loaded.loc[df_loaded["cluster_id"] == 0, "cluster_label"].iloc[0],
            "Broadleaved forest",
        )
        self.assertEqual(
            df_loaded.loc[df_loaded["cluster_id"] == 1, "cluster_label"].iloc[0],
            "Herbaceous cropland",
        )
        self.assertEqual(
            df_loaded.loc[
                df_loaded["parent_cluster_id"] == 0, "parent_cluster_label"
            ].iloc[0],
            "Natural vegetation",
        )

    def test_end_to_end_relabel_parquet(self):
        input_parquet = os.path.join(self.test_dir, "relabel_input.parquet")
        output_parquet = os.path.join(self.test_dir, "relabel_output.parquet")

        df = pd.DataFrame(
            {
                "Photo_ID": [str(i) for i in range(6)],
                "Platform": ["flickr"] * 6,
                "Latitude": [40.0, 40.1, 40.2, 50.0, 50.1, 50.2],
                "Longitude": [-74.0, -74.1, -74.2, 2.0, 2.1, 2.2],
                "Image_URL": [f"http://example.com/{i}.jpg" for i in range(6)],
                "cluster_id": [0, 0, 0, 1, 1, 1],
                "parent_cluster_id": [0, 0, 0, 0, 0, 0],
                "cluster_label": [
                    "Error Labeling",
                    "Error Labeling",
                    "Error Labeling",
                    "ExistingLabel",
                    "ExistingLabel",
                    "ExistingLabel",
                ],
                "parent_cluster_label": ["Error Labeling"] * 6,
            }
        )
        embs = np.zeros((6, 4), dtype=np.float32)
        embs[:3, 0] = 1.0
        embs[3:, 1] = 1.0
        df["embedding"] = list(embs)
        save_dataframe(df, input_parquet, representation_type="cls")

        mock_img = Image.new("RGB", (100, 100), color="green")
        mock_url_resp = MagicMock()
        mock_url_resp.status = 200
        mock_url_resp.__enter__.return_value = mock_url_resp

        with (
            patch("urllib.request.urlopen", return_value=mock_url_resp),
            patch(
                "src.indexing.relabel_failed_clusters.load_image", return_value=mock_img
            ),
            patch(
                "src.indexing.relabel_failed_clusters.query_vlm_openai_api",
                side_effect=[
                    # Child cluster 0 step 1 & 2
                    "Visual desc child0",
                    "LABEL: Broadleaved forest\nDESCRIPTION: Re-labeled child forest description.",
                    # Parent cluster 0 step 1 & 2
                    "Visual desc parent0",
                    "LABEL: Natural grassland\nDESCRIPTION: Re-labeled parent description.",
                ],
            ),
        ):
            test_args = [
                "relabel_failed_clusters.py",
                "--file",
                input_parquet,
                "--out",
                output_parquet,
                "--mllm_model",
                "mock-model",
                "--mllm_endpoint",
                "http://localhost:8000",
                "--num_medoids",
                "2",
                "--save_interval",
                "1",
            ]
            with patch.object(sys, "argv", test_args):
                main()

        df_out = load_dataframe(output_parquet)
        self.assertEqual(len(df_out), 6)
        # Cluster 0 should now be re-labeled
        self.assertEqual(
            df_out.loc[df_out["cluster_id"] == 0, "cluster_label"].iloc[0],
            "Broadleaved forest",
        )
        # Cluster 1 was already labeled, so it stays unchanged
        self.assertEqual(
            df_out.loc[df_out["cluster_id"] == 1, "cluster_label"].iloc[0],
            "ExistingLabel",
        )
        # Parent cluster should now be re-labeled
        self.assertEqual(
            df_out.loc[df_out["parent_cluster_id"] == 0, "parent_cluster_label"].iloc[
                0
            ],
            "Natural grassland",
        )


if __name__ == "__main__":
    unittest.main()
