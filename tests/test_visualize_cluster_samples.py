import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.utils.io import save_dataframe
from src.visualization.visualize_cluster_samples import create_sample_grid


class TestVisualizeClusterSamples(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_sample_grid_with_mmap(self):
        # 1. Create a dummy clustered dataframe with companion embeddings
        n_samples = 20
        dim = 16
        rng = np.random.RandomState(42)

        df = pd.DataFrame(
            {
                "Photo_ID": [f"photo_{i}" for i in range(n_samples)],
                "Platform": ["Flickr"] * n_samples,
                "photo_key": [f"flickr_photo_{i}" for i in range(n_samples)],
                "Latitude": rng.uniform(40.0, 50.0, size=n_samples),
                "Longitude": rng.uniform(2.0, 10.0, size=n_samples),
                "H3_Cell": ["881fb46625fffff"] * n_samples,
                "cluster_id": [i % 2 for i in range(n_samples)],  # 2 clusters
                "cluster_label": [f"Cluster_{i % 2}" for i in range(n_samples)],
                "cluster_description": [
                    f"Description for cluster {i % 2}" for i in range(n_samples)
                ],
                "parent_cluster_id": [0] * n_samples,
                "parent_cluster_label": ["Parent Cluster"] * n_samples,
                "visual_description": ["A scenic view"] * n_samples,
                "License": ["CC-BY"] * n_samples,
                "Season": ["Summer"] * n_samples,
                "Time_Of_Day": ["Day"] * n_samples,
                "Image_URL": [
                    f"http://example.com/img_{i}.jpg" for i in range(n_samples)
                ],
                "embedding": [
                    rng.randn(dim).astype(np.float32) for _ in range(n_samples)
                ],
            }
        )

        base_path = os.path.join(self.temp_dir, "geo_dummy_k_2.parquet")
        save_dataframe(
            df,
            base_path,
            representation_type="cls",
            precision="float32",
            model_name="google/tipsv2-b14",
        )

        out_html = os.path.join(self.temp_dir, "samples.html")

        # 2. Run create_sample_grid with mmap=True
        create_sample_grid(
            base_path,
            out_html,
            top_n=3,
            representation_type="cls",
            model_name="google/tipsv2-b14",
            precision="float32",
            mmap=True,
        )

        self.assertTrue(os.path.exists(out_html))
        out_js = os.path.join(self.temp_dir, "samples_data.js")
        self.assertTrue(os.path.exists(out_js))

        with open(out_js, "r", encoding="utf-8") as f:
            js_content = f.read()

        self.assertIn("Cluster_0", js_content)
        self.assertIn("Cluster_1", js_content)
        self.assertIn("Centroid Image", js_content)
