import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.indexing.cluster_images_global import main as cluster_main
from src.utils.io import (
    load_dataset_with_clusters,
    load_embeddings,
    save_dataframe,
)


class TestModelNamespacedClustering(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_dataset_with_clusters_and_embeddings_namespaced(self):
        # 1. Create a dummy base dataframe with embeddings
        num_rows = 20
        dim = 64
        rng = np.random.RandomState(42)

        df_base = pd.DataFrame(
            {
                "Platform": ["flickr"] * num_rows,
                "Photo_ID": [str(i) for i in range(num_rows)],
                "Latitude": [45.0 + i * 0.1 for i in range(num_rows)],
                "Longitude": [9.0 + i * 0.1 for i in range(num_rows)],
                "embedding": [
                    rng.randn(dim).astype(np.float32) for _ in range(num_rows)
                ],
            }
        )

        base_path = os.path.join(self.temp_dir, "geo_space_offline.parquet")
        save_dataframe(
            df_base,
            base_path,
            representation_type="cls",
            precision="float16",
            model_name="facebook/dinov2-base",
        )

        # Verify companion npy was created
        expected_npy = os.path.join(
            self.temp_dir, "geo_space_facebook_dinov2-base_cls_embeddings.npy"
        )
        self.assertTrue(
            os.path.exists(expected_npy), f"Expected npy not found: {expected_npy}"
        )

        # 2. Create namespaced sidecar in a subfolder vis_10_facebook_dinov2-base_cls_float16
        vis_dir = os.path.join(self.temp_dir, "vis_10_facebook_dinov2-base_cls_float16")
        os.makedirs(vis_dir, exist_ok=True)

        sidecar_path = os.path.join(
            vis_dir,
            "geo_space_offline_facebook_dinov2-base_cls_float16_clustered_k_10.parquet",
        )
        df_sidecar = pd.DataFrame(
            {
                "Platform": ["flickr"] * num_rows,
                "Photo_ID": [str(i) for i in range(num_rows)],
                "cluster_id": [i % 10 for i in range(num_rows)],
                "parent_cluster_id": [0] * num_rows,
                "cluster_label": [f"Cluster {i % 10}" for i in range(num_rows)],
                "model_name": ["facebook/dinov2-base"] * num_rows,
                "representation_type": ["cls"] * num_rows,
                "precision": ["float16"] * num_rows,
            }
        )
        save_dataframe(df_sidecar, sidecar_path)

        # 3. Test loading via base_path with model parameters
        # Also place sidecar in root to test root resolution
        root_sidecar = os.path.join(
            self.temp_dir,
            "geo_space_offline_facebook_dinov2-base_cls_float16_clustered_k_10.parquet",
        )
        shutil.copy(sidecar_path, root_sidecar)

        merged_from_base = load_dataset_with_clusters(
            base_path,
            k_clusters=10,
            model_name="facebook/dinov2-base",
            representation_type="cls",
            precision="float16",
        )
        self.assertIn("cluster_id", merged_from_base.columns)
        self.assertIn("model_name", merged_from_base.columns)
        self.assertEqual(merged_from_base["model_name"].iloc[0], "facebook/dinov2-base")

        # 4. Test loading directly from sidecar path located inside subfolder
        merged_from_subfolder_sidecar = load_dataset_with_clusters(sidecar_path)
        self.assertIn("Latitude", merged_from_subfolder_sidecar.columns)
        self.assertIn("cluster_id", merged_from_subfolder_sidecar.columns)
        self.assertEqual(len(merged_from_subfolder_sidecar), num_rows)

        # 5. Test loading embeddings directly from sidecar path in subfolder
        embs = load_embeddings(
            sidecar_path,
            representation_type="cls",
            model_name="facebook/dinov2-base",
            precision="float16",
        )
        self.assertEqual(embs.shape, (num_rows, dim))

    def test_cluster_images_global_saves_provenance(self):
        num_rows = 15
        dim = 32
        rng = np.random.RandomState(99)

        df_base = pd.DataFrame(
            {
                "Platform": ["flickr"] * num_rows,
                "Photo_ID": [str(i) for i in range(num_rows)],
                "Latitude": [50.0] * num_rows,
                "Longitude": [10.0] * num_rows,
                "embedding": [
                    rng.randn(dim).astype(np.float32) for _ in range(num_rows)
                ],
            }
        )

        base_path = os.path.join(self.temp_dir, "geo_space_test.parquet")
        save_dataframe(df_base, base_path, representation_type="cls")

        out_sidecar = os.path.join(
            self.temp_dir, "geo_space_test_clustered_k_3.parquet"
        )

        test_args = [
            "cluster_images_global",
            "--pkl",
            base_path,
            "--out",
            out_sidecar,
            "--k",
            "3",
            "--no_gpu",
            "--model_name",
            "google/tipsv2-b14",
            "--representation_type",
            "cls",
            "--precision",
            "float32",
        ]
        with patch.object(sys, "argv", test_args):
            cluster_main()

        self.assertTrue(os.path.exists(out_sidecar))
        df_result = pd.read_parquet(out_sidecar)
        self.assertIn("cluster_id", df_result.columns)
        self.assertIn("model_name", df_result.columns)
        self.assertIn("representation_type", df_result.columns)
        self.assertIn("precision", df_result.columns)
        self.assertEqual(df_result["model_name"].iloc[0], "google/tipsv2-b14")
        self.assertEqual(df_result["representation_type"].iloc[0], "cls")
        self.assertEqual(df_result["precision"].iloc[0], "float32")


if __name__ == "__main__":
    unittest.main()
