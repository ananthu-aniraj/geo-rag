import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.utils.io import save_dataframe
from src.utils.sync_offline_dataset import (
    discover_clustered_sidecar_files,
    discover_companion_embedding_files,
    sync_offline_dataset,
)


class TestSyncOfflineDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # 5 total items:
        # online will keep item_1, item_2, item_3
        # online has purged item_4, item_5 (e.g. Mapillary glitches)
        self.all_keys = [f"mapillary_{i}" for i in range(1, 6)]
        self.all_pids = [f"{i}" for i in range(1, 6)]
        self.online_keys = [f"mapillary_{i}" for i in range(1, 4)]

        # Create dummy image files
        self.image_paths = []
        for pid in self.all_pids:
            img_rel = f"mapillary/{pid}.jpg"
            img_abs = os.path.join(self.image_dir, img_rel)
            os.makedirs(os.path.dirname(img_abs), exist_ok=True)
            with open(img_abs, "wb") as f:
                f.write(b"dummy_image_bytes_" + pid.encode())
            self.image_paths.append(img_rel)

        # 1. Master Online dataset (3 items)
        self.df_online = pd.DataFrame(
            {
                "Photo_ID": [f"{i}" for i in range(1, 4)],
                "Platform": ["mapillary"] * 3,
                "photo_key": self.online_keys,
                "Latitude": [10.0, 11.0, 12.0],
                "Longitude": [20.0, 21.0, 22.0],
                "Image_URL": [f"https://online.url/{i}" for i in range(1, 4)],
            }
        )
        self.online_path = os.path.join(self.test_dir, "geo_space_cleaned.parquet")
        save_dataframe(self.df_online, self.online_path)

        # 2. Target Offline dataset (5 items)
        self.df_offline = pd.DataFrame(
            {
                "Photo_ID": self.all_pids,
                "Platform": ["mapillary"] * 5,
                "photo_key": self.all_keys,
                "Latitude": [10.0, 11.0, 12.0, 13.0, 14.0],
                "Longitude": [20.0, 21.0, 22.0, 23.0, 24.0],
                "Image_Location": self.image_paths,
            }
        )
        self.offline_path = os.path.join(
            self.test_dir, "geo_space_cleaned_offline.parquet"
        )
        save_dataframe(self.df_offline, self.offline_path)

        # 3. Companion embeddings for Model A (google_tipsv2-b14_cls, dim=8)
        self.emb_model_a = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
        self.model_a_npy = os.path.join(
            self.test_dir,
            "geo_space_cleaned_offline_google_tipsv2-b14_cls_embeddings.npy",
        )
        self.model_a_keys = os.path.join(
            self.test_dir,
            "geo_space_cleaned_offline_google_tipsv2-b14_cls_embeddings.keys.parquet",
        )
        np.save(self.model_a_npy, self.emb_model_a)
        pd.DataFrame({"photo_key": self.all_keys}).to_parquet(self.model_a_keys)

        # 4. Companion embeddings for Model B (facebook_dinov2-base_cls_avg_patch, dim=16)
        self.emb_model_b = np.arange(5 * 16, dtype=np.float16).reshape(5, 16)
        self.model_b_npy = os.path.join(
            self.test_dir,
            "geo_space_cleaned_offline_facebook_dinov2-base_cls_avg_patch_embeddings.npy",
        )
        self.model_b_keys = os.path.join(
            self.test_dir,
            "geo_space_cleaned_offline_facebook_dinov2-base_cls_avg_patch_embeddings.keys.parquet",
        )
        np.save(self.model_b_npy, self.emb_model_b)
        pd.DataFrame({"photo_key": self.all_keys}).to_parquet(self.model_b_keys)

        # 5. Clustered sidecar in same directory (photo_key schema)
        self.sidecar_path = os.path.join(
            self.test_dir, "geo_space_cleaned_offline_clustered_k_2.parquet"
        )
        self.df_sidecar = pd.DataFrame(
            {
                "photo_key": self.all_keys,
                "cluster_id": [0, 1, 0, 1, 0],
                "cluster_label": ["A", "B", "A", "B", "A"],
            }
        )
        save_dataframe(self.df_sidecar, self.sidecar_path)

        # 6. Clustered sidecar in extra output directory (Platform/Photo_ID schema)
        self.extra_dir = os.path.join(self.test_dir, "extra_output")
        os.makedirs(self.extra_dir, exist_ok=True)
        self.extra_sidecar_path = os.path.join(
            self.extra_dir, "geo_space_clustered_k_5.parquet"
        )
        self.df_extra_sidecar = pd.DataFrame(
            {
                "Platform": ["mapillary"] * 5,
                "Photo_ID": self.all_pids,
                "cluster_id": [1, 2, 0, 1, 2],
            }
        )
        save_dataframe(self.df_extra_sidecar, self.extra_sidecar_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_discover_companion_embedding_files(self):
        discovered = discover_companion_embedding_files(self.offline_path)
        self.assertEqual(len(discovered), 2)
        npy_basenames = [os.path.basename(n) for n, _ in discovered]
        self.assertIn(
            "geo_space_cleaned_offline_google_tipsv2-b14_cls_embeddings.npy",
            npy_basenames,
        )
        self.assertIn(
            "geo_space_cleaned_offline_facebook_dinov2-base_cls_avg_patch_embeddings.npy",
            npy_basenames,
        )

    def test_discover_clustered_sidecar_files(self):
        # Without extra_dirs: only self.sidecar_path in self.test_dir
        discovered = discover_clustered_sidecar_files(self.offline_path)
        self.assertEqual(len(discovered), 1)
        self.assertEqual(
            os.path.basename(discovered[0]),
            "geo_space_cleaned_offline_clustered_k_2.parquet",
        )

        # With extra_dirs: both sidecars found
        discovered_all = discover_clustered_sidecar_files(
            self.offline_path, extra_dirs=[self.extra_dir]
        )
        self.assertEqual(len(discovered_all), 2)
        basenames = [os.path.basename(f) for f in discovered_all]
        self.assertIn("geo_space_cleaned_offline_clustered_k_2.parquet", basenames)
        self.assertIn("geo_space_clustered_k_5.parquet", basenames)

    def test_sync_dry_run_does_not_modify_files(self):
        summary = sync_offline_dataset(
            online_path=self.online_path,
            offline_path=self.offline_path,
            image_dir=self.image_dir,
            extra_dirs=[self.extra_dir],
            delete_images=True,
            dry_run=True,
        )
        self.assertEqual(summary["total_offline"], 5)
        self.assertEqual(summary["surviving_rows"], 3)
        self.assertEqual(summary["purged_rows"], 2)
        self.assertEqual(summary["clustered_sidecars_synced"], 2)
        self.assertEqual(summary["purged_sidecar_rows_total"], 4)

        # Files should still have original lengths
        df_off = pd.read_parquet(self.offline_path)
        self.assertEqual(len(df_off), 5)
        arr_a = np.load(self.model_a_npy)
        self.assertEqual(len(arr_a), 5)
        df_sc = pd.read_parquet(self.sidecar_path)
        self.assertEqual(len(df_sc), 5)
        df_extra_sc = pd.read_parquet(self.extra_sidecar_path)
        self.assertEqual(len(df_extra_sc), 5)
        # Images on disk should still exist
        for img_rel in self.image_paths:
            self.assertTrue(os.path.exists(os.path.join(self.image_dir, img_rel)))

    def test_sync_pruning_metadata_and_multimodel_embeddings(self):
        summary = sync_offline_dataset(
            online_path=self.online_path,
            offline_path=self.offline_path,
            image_dir=self.image_dir,
            extra_dirs=[self.extra_dir],
            delete_images=True,
            dry_run=False,
        )
        self.assertEqual(summary["total_offline"], 5)
        self.assertEqual(summary["surviving_rows"], 3)
        self.assertEqual(summary["purged_rows"], 2)
        self.assertEqual(summary["companion_models_synced"], 2)
        self.assertEqual(summary["purged_vectors_total"], 4)  # 2 vectors x 2 models
        self.assertEqual(summary["clustered_sidecars_synced"], 2)
        self.assertEqual(summary["purged_sidecar_rows_total"], 4)  # 2 rows x 2 sidecars
        self.assertEqual(summary["deleted_images_count"], 2)

        # 1. Verify offline metadata
        df_off = pd.read_parquet(self.offline_path)
        self.assertEqual(len(df_off), 3)
        self.assertEqual(list(df_off["photo_key"]), self.online_keys)

        # 2. Verify Model A embeddings & keys
        arr_a = np.load(self.model_a_npy)
        keys_a = pd.read_parquet(self.model_a_keys)["photo_key"].tolist()
        self.assertEqual(len(arr_a), 3)
        self.assertEqual(keys_a, self.online_keys)
        # Vector values for first 3 rows should be strictly preserved
        np.testing.assert_array_equal(arr_a, self.emb_model_a[:3])

        # 3. Verify Model B embeddings & keys
        arr_b = np.load(self.model_b_npy)
        keys_b = pd.read_parquet(self.model_b_keys)["photo_key"].tolist()
        self.assertEqual(len(arr_b), 3)
        self.assertEqual(keys_b, self.online_keys)
        np.testing.assert_array_equal(arr_b, self.emb_model_b[:3])

        # 4. Verify Clustered Sidecars
        df_sc = pd.read_parquet(self.sidecar_path)
        self.assertEqual(len(df_sc), 3)
        self.assertEqual(list(df_sc["photo_key"]), self.online_keys)

        df_extra_sc = pd.read_parquet(self.extra_sidecar_path)
        self.assertEqual(len(df_extra_sc), 3)
        self.assertEqual(list(df_extra_sc["Photo_ID"]), ["1", "2", "3"])

        # 5. Verify physical image files:
        # items 1, 2, 3 should exist, items 4, 5 should be deleted
        for i in range(1, 4):
            self.assertTrue(
                os.path.exists(os.path.join(self.image_dir, f"mapillary/{i}.jpg"))
            )
        for i in range(4, 6):
            self.assertFalse(
                os.path.exists(os.path.join(self.image_dir, f"mapillary/{i}.jpg"))
            )

    def test_sync_no_op_when_already_in_sync(self):
        # First sync
        sync_offline_dataset(
            online_path=self.online_path,
            offline_path=self.offline_path,
            extra_dirs=[self.extra_dir],
            dry_run=False,
        )
        # Second sync: should be 0 purged
        summary = sync_offline_dataset(
            online_path=self.online_path,
            offline_path=self.offline_path,
            extra_dirs=[self.extra_dir],
            dry_run=False,
        )
        self.assertEqual(summary["purged_rows"], 0)
        self.assertEqual(summary["surviving_rows"], 3)
        self.assertEqual(summary["purged_sidecar_rows_total"], 0)


if __name__ == "__main__":
    unittest.main()
