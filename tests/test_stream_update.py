import os
import shutil
import unittest

import numpy as np
import pandas as pd

from src.processing.process_scraped_data import stream_update_parquet
from src.utils.io import save_dataframe


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


if __name__ == "__main__":
    unittest.main()
