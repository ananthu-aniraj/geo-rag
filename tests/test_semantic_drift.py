import os
import shutil
import sys
import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.utils.check_semantic_drift import main
from src.utils.io import save_dataframe


class TestSemanticDrift(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch_drift"
        os.makedirs(self.test_dir, exist_ok=True)
        self.centroids_parquet = os.path.join(self.test_dir, "old_centroids.parquet")
        self.new_input_parquet = os.path.join(self.test_dir, "new_input.parquet")

        # 1. Create a dummy old clustered database
        df_old = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3"],
                "Platform": ["flickr", "flickr", "mapillary"],
                "cluster_id": [0, 1, 2],
            }
        )
        # Embeddings: standard unit vectors
        emb_old = np.eye(3, 768, dtype=np.float32)
        df_old["embedding"] = list(emb_old)
        save_dataframe(df_old, self.centroids_parquet)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_drift_detector_assign(self):
        # Create new input dataset where the new image embedding is identical to centroid 0 (similarity = 1.0)
        df_new = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3", "4"],  # "4" is new
                "Platform": ["flickr", "flickr", "mapillary", "flickr"],
            }
        )
        # Image 4 has embedding identical to centroid 0
        emb_new = np.zeros((4, 768), dtype=np.float32)
        emb_new[3, 0] = 1.0  # unit vector matching centroid 0
        df_new["embedding"] = list(emb_new)
        save_dataframe(df_new, self.new_input_parquet)

        test_args = [
            "check_semantic_drift",
            "--input",
            self.new_input_parquet,
            "--centroids_parquet",
            self.centroids_parquet,
            "--k_clusters",
            "3",
            "--no_gpu",
        ]

        # Capture print output
        captured_output = StringIO()
        sys.stdout = captured_output

        with patch.object(sys, "argv", test_args):
            main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue().strip()
        last_line = output.splitlines()[-1] if output else ""
        self.assertEqual(last_line, "assign")

    def test_drift_detector_fit(self):
        # Create new input dataset where the new image embedding is orthogonal/outlier (similarity = 0.0)
        df_new = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3", "4"],  # "4" is new
                "Platform": ["flickr", "flickr", "mapillary", "flickr"],
            }
        )
        # Image 4 has embedding orthogonal to all old unit vectors (e.g. index 10 is 1.0)
        emb_new = np.zeros((4, 768), dtype=np.float32)
        emb_new[3, 10] = 1.0
        df_new["embedding"] = list(emb_new)
        save_dataframe(df_new, self.new_input_parquet)

        test_args = [
            "check_semantic_drift",
            "--input",
            self.new_input_parquet,
            "--centroids_parquet",
            self.centroids_parquet,
            "--k_clusters",
            "3",
            "--no_gpu",
        ]

        captured_output = StringIO()
        sys.stdout = captured_output

        with patch.object(sys, "argv", test_args):
            main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue().strip()
        last_line = output.splitlines()[-1] if output else ""
        self.assertEqual(last_line, "fit")

    def test_pruning_cleaned_images_with_new_data(self):
        # Image "2" was pruned by the cleaning step (not in new_input)
        # Image "4" is newly added and identical to centroid 0
        df_new = pd.DataFrame(
            {
                "Photo_ID": ["1", "3", "4"],
                "Platform": ["flickr", "mapillary", "flickr"],
            }
        )
        emb_new = np.zeros((3, 768), dtype=np.float32)
        emb_new[0, 0] = 1.0  # Image 1
        emb_new[1, 2] = 1.0  # Image 3
        emb_new[2, 0] = 1.0  # Image 4 (identical to centroid 0)
        df_new["embedding"] = list(emb_new)
        save_dataframe(df_new, self.new_input_parquet)

        test_args = [
            "check_semantic_drift",
            "--input",
            self.new_input_parquet,
            "--centroids_parquet",
            self.centroids_parquet,
            "--k_clusters",
            "3",
            "--no_gpu",
        ]

        captured_output = StringIO()
        sys.stdout = captured_output

        with patch.object(sys, "argv", test_args):
            main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue().strip()
        last_line = output.splitlines()[-1] if output else ""
        self.assertEqual(last_line, "assign")

        # Verify image "2" was pruned from centroids_parquet on disk
        df_centroids_after = pd.read_parquet(self.centroids_parquet)
        self.assertEqual(len(df_centroids_after), 2)
        self.assertListEqual(
            list(df_centroids_after["Photo_ID"].astype(str)), ["1", "3"]
        )

    def test_pruning_cleaned_images_without_new_data(self):
        # Image "2" was pruned by the cleaning step, but NO new images were added
        df_new = pd.DataFrame(
            {
                "Photo_ID": ["1", "3"],
                "Platform": ["flickr", "mapillary"],
            }
        )
        emb_new = np.zeros((2, 768), dtype=np.float32)
        emb_new[0, 0] = 1.0  # Image 1
        emb_new[1, 2] = 1.0  # Image 3
        df_new["embedding"] = list(emb_new)
        save_dataframe(df_new, self.new_input_parquet)

        test_args = [
            "check_semantic_drift",
            "--input",
            self.new_input_parquet,
            "--centroids_parquet",
            self.centroids_parquet,
            "--k_clusters",
            "3",
            "--no_gpu",
        ]

        captured_output = StringIO()
        sys.stdout = captured_output

        with patch.object(sys, "argv", test_args):
            main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue().strip()
        last_line = output.splitlines()[-1] if output else ""
        self.assertEqual(last_line, "assign")

        # Verify image "2" was pruned from centroids_parquet on disk
        df_centroids_after = pd.read_parquet(self.centroids_parquet)
        self.assertEqual(len(df_centroids_after), 2)
        self.assertListEqual(
            list(df_centroids_after["Photo_ID"].astype(str)), ["1", "3"]
        )


if __name__ == "__main__":
    unittest.main()
