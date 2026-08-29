import os
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.indexing.cluster_images_global import (
    cluster_data,
    main,
    map_resampled_parents_to_children,
    sample_closest_points,
)
from src.utils.io import load_dataframe, load_embeddings, save_dataframe


class TestClusterImagesGlobal(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch_cluster"
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_parquet = os.path.join(self.test_dir, "test_db_cleaned.parquet")
        self.output_parquet = os.path.join(self.test_dir, "test_db_clustered.parquet")

        # 1. Create a dummy dataset (10 samples, 4 dimensions)
        df = pd.DataFrame(
            {
                "Photo_ID": [str(i) for i in range(10)],
                "Platform": ["flickr"] * 10,
                "Latitude": np.linspace(40.0, 41.0, 10),
                "Longitude": np.linspace(-74.0, -73.0, 10),
                "Captured_At": ["2024-01-01"] * 10,
                "Image_URL": [f"url_{i}" for i in range(10)],
            }
        )
        # embeddings filled with predictable clusters
        # First 5 points clustered around [1,0,0,0], next 5 around [0,1,0,0]
        embs = np.zeros((10, 4), dtype=np.float32)
        embs[:5, 0] = 1.0
        embs[5:, 1] = 1.0
        # Add slight noise
        embs += np.random.normal(scale=0.01, size=embs.shape).astype(np.float32)

        df["embedding"] = list(embs)
        save_dataframe(df, self.input_parquet, representation_type="cls")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cluster_data_cpu(self):
        # Generate random dummy embeddings (20 points, 8 dimensions)
        embeddings = np.random.rand(20, 8).astype(np.float32)

        # Test standard KMeans
        cluster_ids, centroids = cluster_data(
            embeddings, k=3, gpu_enabled=False, minibatch_enabled=False
        )
        self.assertEqual(len(cluster_ids), 20)
        self.assertEqual(centroids.shape, (3, 8))
        self.assertTrue(all(0 <= cid < 3 for cid in cluster_ids))

        # Test MiniBatchKMeans
        cluster_ids_mb, centroids_mb = cluster_data(
            embeddings, k=2, gpu_enabled=False, minibatch_enabled=True
        )
        self.assertEqual(len(cluster_ids_mb), 20)
        self.assertEqual(centroids_mb.shape, (2, 8))
        self.assertTrue(all(0 <= cid < 2 for cid in cluster_ids_mb))

    def test_sample_closest_points(self):
        # 4 points in 2D
        embeddings = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
        )
        cluster_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Sample 1 closest point per centroid
        sampled = sample_closest_points(embeddings, cluster_ids, centroids, n_samples=1)
        # Centroid 0 is closest to index 0, Centroid 1 is closest to index 2
        self.assertEqual(len(sampled), 2)
        self.assertIn(0, sampled)
        self.assertIn(2, sampled)

    def test_map_resampled_parents_to_children(self):
        # 4 resampled points
        sampled_indices = np.array([0, 1, 2, 3], dtype=np.int64)
        resampled_parent_ids = np.array([10, 10, 20, 20], dtype=np.int32)

        # Mapping resampled points to child clusters (indices 0 and 1 belong to child 0, indices 2 and 3 belong to child 1)
        child_cluster_ids = np.array([0, 0, 1, 1], dtype=np.int32)

        mapping = map_resampled_parents_to_children(
            sampled_indices=sampled_indices,
            resampled_parent_ids=resampled_parent_ids,
            child_cluster_ids=child_cluster_ids,
            k_child=2,
            k_parent=2,
        )
        # Child 0 should be mapped to parent 10, Child 1 should be mapped to parent 20
        self.assertEqual(mapping[0], 10)
        self.assertEqual(mapping[1], 20)

    def test_end_to_end_fit_mode_cpu(self):
        # Simulating sys.argv to run main() in CPU fit mode
        test_args = [
            "cluster_images_global.py",
            "--pkl",
            self.input_parquet,
            "--k",
            "4",
            "--k_parents",
            "2",
            "--no_gpu",
            "--out",
            self.output_parquet,
            "--clustering_mode",
            "fit",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Verify output files exist
        self.assertTrue(os.path.exists(self.output_parquet))

        # Parquet file should contain cluster columns
        df_out = load_dataframe(self.output_parquet)
        self.assertIn("cluster_id", df_out.columns)
        self.assertIn("parent_cluster_id", df_out.columns)
        self.assertTrue((df_out["cluster_id"] >= 0).all())
        self.assertTrue((df_out["parent_cluster_id"] >= 0).all())

        # Verify that load_embeddings resolves back to the correct underlying source embeddings
        loaded_embs = load_embeddings(self.output_parquet, representation_type="cls")
        self.assertEqual(len(loaded_embs), 10)


if __name__ == "__main__":
    unittest.main()
