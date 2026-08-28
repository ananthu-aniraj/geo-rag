import os
import shutil
import unittest

import numpy as np
import pandas as pd

from src.processing.process_scraped_data import save_checkpoint, stream_update_parquet
from src.utils.io import load_embeddings, save_dataframe


class TestStreamUpdate(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch"
        os.makedirs(self.test_dir, exist_ok=True)

        self.input_parquet = os.path.join(self.test_dir, "test_db.parquet")

        # 1. Create a mock dataset with string and large_string columns
        df = pd.DataFrame(
            {
                "Photo_ID": ["123", "456", "789"],
                "Platform": ["flickr", "mapillary", "flickr"],
                "H3_Cell": ["841f8f3ffffffff", "841f8f3ffffffff", "841f8f5ffffffff"],
                "Latitude": [40.0, 41.0, 42.0],
                "Longitude": [-74.0, -73.0, -72.0],
                "Image_URL": ["url1", "url2", "url3"],
                "Captured_At": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "License": ["CC-BY", "CC-BY-SA", "CC0"],
            }
        )

        save_dataframe(df, self.input_parquet)

        # 2. Create mock companion embeddings
        emb_matrix = np.random.rand(3, 768).astype(np.float32)
        npy_path = self.input_parquet.replace(".parquet", "_cls_embeddings.npy")
        np.save(npy_path, emb_matrix)

        # Create companion keys index
        keys_df = pd.DataFrame(
            {"photo_key": ["flickr_123", "mapillary_456", "flickr_789"]}
        )
        keys_path = self.input_parquet.replace(
            ".parquet", "_cls_embeddings.keys.parquet"
        )
        keys_df.to_parquet(keys_path)

        self.output_parquet = os.path.join(self.test_dir, "test_output.parquet")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_stream_update_success(self):
        # Define new/updated batch data
        df_new = pd.DataFrame(
            {
                "Photo_ID": ["999"],
                "Platform": ["mapillary"],
                "H3_Cell": ["841f8f3ffffffff"],
                "Latitude": [40.5],
                "Longitude": [-73.5],
                "Image_URL": ["url999"],
                "Captured_At": ["2026-02-01"],
                "License": ["CC-BY"],
                "photo_key": ["mapillary_999"],
            }
        )
        new_embs = np.random.rand(1, 768).astype(np.float32)
        df_new["embedding"] = list(new_embs)

        # Active cells to replace (all items in '841f8f3ffffffff' will be overwritten by df_new)
        active_cells = {"841f8f3ffffffff"}

        # Run streaming update
        try:
            stream_update_parquet(
                input_path=self.input_parquet,
                output_path=self.output_parquet,
                df_new=df_new,
                active_cells=active_cells,
                representation_type="cls",
            )
            success = True
        except Exception as e:
            self.fail(f"stream_update_parquet failed with exception: {e}")
            success = False

        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.output_parquet))

        # Verify the output parquet content
        df_out = pd.read_parquet(self.output_parquet)

        # The old rows in cell '841f8f3ffffffff' (ids 123, 456) should be filtered out
        # The new row (999) should be present
        # The old row in cell '841f8f5ffffffff' should be preserved
        self.assertNotIn("123", df_out["Photo_ID"].values)
        self.assertNotIn("456", df_out["Photo_ID"].values)
        self.assertIn("789", df_out["Photo_ID"].values)
        self.assertIn("999", df_out["Photo_ID"].values)

        # Verify companion embeddings were saved correctly
        out_npy_path = self.output_parquet.replace(".parquet", "_cls_embeddings.npy")
        self.assertTrue(os.path.exists(out_npy_path))
        out_embs = np.load(out_npy_path)
        self.assertEqual(len(out_embs), len(df_out))

    def test_checkpoint_resume_and_save_success(self):
        # 1. Save an initial checkpoint file from first run state
        initial_data = [
            {
                "Photo_ID": "999",
                "Platform": "mapillary",
                "H3_Cell": "841f8f3ffffffff",
                "Latitude": 40.5,
                "Longitude": -73.5,
                "Image_URL": "url999",
                "Captured_At": "2026-02-01",
                "License": "CC-BY",
                "photo_key": "mapillary_999",
                "embedding": np.random.rand(768).astype(np.float32),
            }
        ]
        ckpt_path = os.path.join(self.test_dir, "db_checkpoint.parquet")
        ckpt_meta_path = ckpt_path.replace(".parquet", "_meta.pkl")

        save_checkpoint(
            initial_data,
            {"841f8f3ffffffff"},
            ckpt_path,
            ckpt_meta_path,
            resume_from=self.input_parquet,
            active_cells={"841f8f3ffffffff"},
            representation_type="cls",
        )

        self.assertTrue(os.path.exists(ckpt_path))

        # 2. Simulate resuming from the checkpoint: load Parquet AND load companion embeddings
        from src.utils.io import load_dataframe

        df_ckpt = load_dataframe(ckpt_path)

        # Verify it has no embedding column by default (since it is decoupled)
        self.assertNotIn("embedding", df_ckpt.columns)

        # Load companion embeddings and reconstruct df
        ckpt_embs = load_embeddings(ckpt_path, representation_type="cls")
        df_ckpt["embedding"] = list(ckpt_embs)

        # Filter and retrieve final_data dicts
        df_ckpt_active = df_ckpt[df_ckpt["H3_Cell"].isin({"841f8f3ffffffff"})]
        final_data = df_ckpt_active.to_dict("records")

        # Verify embeddings are successfully restored
        self.assertIn("embedding", final_data[0])
        self.assertIsInstance(final_data[0]["embedding"], np.ndarray)

        # 3. Save second checkpoint using the restored final_data (should succeed without KeyError)
        ckpt_path_2 = os.path.join(self.test_dir, "db_checkpoint_2.parquet")
        ckpt_meta_path_2 = ckpt_path_2.replace(".parquet", "_meta.pkl")

        try:
            save_checkpoint(
                final_data,
                {"841f8f3ffffffff"},
                ckpt_path_2,
                ckpt_meta_path_2,
                resume_from=self.input_parquet,
                active_cells={"841f8f3ffffffff"},
                representation_type="cls",
            )
            second_save_success = True
        except Exception as e:
            self.fail(f"save_checkpoint after resume failed with exception: {e}")
            second_save_success = False

        self.assertTrue(second_save_success)
        self.assertTrue(os.path.exists(ckpt_path_2))

    def test_load_embeddings_no_checkpoint_mixup(self):
        # 1. Create a base database file with 1.0-filled embeddings
        base_db_path = os.path.join(self.test_dir, "test_mixup.parquet")
        df_base = pd.DataFrame(
            {
                "Photo_ID": ["1", "2"],
                "Platform": ["flickr", "flickr"],
                "H3_Cell": ["841f8f3ffffffff", "841f8f3ffffffff"],
                "Latitude": [40.0, 40.1],
                "Longitude": [-74.0, -74.1],
                "Image_URL": ["url1", "url2"],
                "Captured_At": ["2024-01-01", "2024-01-02"],
            }
        )
        df_base["embedding"] = list(np.ones((2, 768), dtype=np.float32))
        save_dataframe(df_base, base_db_path, representation_type="cls")

        # 2. Create a checkpoint file in the same directory with 2.0-filled embeddings
        ckpt_db_path = os.path.join(self.test_dir, "test_mixup_checkpoint.parquet")
        df_ckpt = pd.DataFrame(
            {
                "Photo_ID": ["1", "2"],
                "Platform": ["flickr", "flickr"],
                "H3_Cell": ["841f8f3ffffffff", "841f8f3ffffffff"],
                "Latitude": [40.0, 40.1],
                "Longitude": [-74.0, -74.1],
                "Image_URL": ["url1", "url2"],
                "Captured_At": ["2024-01-01", "2024-01-02"],
            }
        )
        df_ckpt["embedding"] = list(np.ones((2, 768), dtype=np.float32) * 2.0)
        save_dataframe(df_ckpt, ckpt_db_path, representation_type="cls")

        # 3. Load embeddings for the base database path
        loaded_embs = load_embeddings(base_db_path, representation_type="cls")

        # 4. Verify that we resolved the base embeddings (all 1.0) and NOT the checkpoint embeddings (all 2.0)
        np.testing.assert_array_equal(loaded_embs, np.ones((2, 768), dtype=np.float32))

    def test_save_dataframe_with_model_name(self):
        db_path = os.path.join(self.test_dir, "test_model_db.parquet")
        df = pd.DataFrame(
            {
                "Photo_ID": ["1", "2"],
                "Platform": ["flickr", "flickr"],
                "H3_Cell": ["841f8f3ffffffff", "841f8f3ffffffff"],
                "Latitude": [40.0, 40.1],
                "Longitude": [-74.0, -74.1],
                "Image_URL": ["url1", "url2"],
                "Captured_At": ["2024-01-01", "2024-01-02"],
            }
        )
        # Use a distinctive embedding filled with 3.5
        df["embedding"] = list(np.ones((2, 768), dtype=np.float32) * 3.5)

        # Save with a specific model name
        model_name = "google/tipsv2"
        save_dataframe(df, db_path, representation_type="cls", model_name=model_name)

        # Expected file paths
        expected_npy = os.path.join(
            self.test_dir, "test_model_db_google_tipsv2_cls_embeddings.npy"
        )
        expected_keys = os.path.join(
            self.test_dir, "test_model_db_google_tipsv2_cls_embeddings.keys.parquet"
        )

        self.assertTrue(os.path.exists(expected_npy))
        self.assertTrue(os.path.exists(expected_keys))

        # Test loading it back using load_embeddings with model_name
        loaded_embs = load_embeddings(
            db_path, model_name=model_name, representation_type="cls"
        )
        np.testing.assert_array_equal(
            loaded_embs, np.ones((2, 768), dtype=np.float32) * 3.5
        )


if __name__ == "__main__":
    unittest.main()
