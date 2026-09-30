import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.processing.backfill_embeddings import (
    find_companion_files,
    main,
    resolve_checkpoint_paths,
    save_checkpoint_companion,
)
from src.utils.io import load_dataframe, load_embeddings, save_dataframe


def _dummy_transform(img):
    return torch.zeros((3, 224, 224), dtype=torch.float32)


class TestBackfillEmbeddingsResume(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_backfill_")
        self.input_parquet = os.path.join(self.test_dir, "geo_space_offline.parquet")
        self.output_parquet = os.path.join(self.test_dir, "geo_space_output.parquet")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_find_companion_files(self):
        dim = 16
        df = pd.DataFrame(
            {
                "Photo_ID": ["1", "2"],
                "Platform": ["flickr", "flickr"],
                "Latitude": [40.0, 41.0],
                "Longitude": [-74.0, -73.0],
                "Image_URL": ["url_1", "url_2"],
                "embedding": [np.ones(dim, dtype=np.float32) for _ in range(2)],
            }
        )
        save_dataframe(
            df,
            self.input_parquet,
            representation_type="cls",
            model_name="facebook/dinov2-base",
        )

        npy_path, keys_path = find_companion_files(
            self.input_parquet,
            model_name="facebook/dinov2-base",
            representation_type="cls",
        )
        self.assertIsNotNone(npy_path)
        self.assertIsNotNone(keys_path)
        self.assertTrue(os.path.exists(npy_path))
        self.assertTrue(os.path.exists(keys_path))

    def test_backfill_embeddings_all_already_computed(self):
        # Create dataset with pre-existing companion embeddings
        dim = 16
        df = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3"],
                "Platform": ["flickr", "flickr", "flickr"],
                "Latitude": [40.0, 41.0, 42.0],
                "Longitude": [-74.0, -73.0, -72.0],
                "Image_URL": ["url_1", "url_2", "url_3"],
                "embedding": [
                    np.full(dim, float(i), dtype=np.float32) for i in range(3)
                ],
            }
        )
        save_dataframe(
            df,
            self.input_parquet,
            representation_type="cls",
            precision="float32",
            model_name="google/tipsv2-b14",
        )

        # Run backfill_embeddings with all images already having embeddings
        test_args = [
            "backfill_embeddings.py",
            "--input",
            self.input_parquet,
            "--model_name",
            "google/tipsv2-b14",
            "--representation_type",
            "cls",
            "--precision",
            "float32",
        ]
        with patch.object(sys, "argv", test_args):
            with patch(
                "src.processing.backfill_embeddings.load_vision_model"
            ) as mock_load_model:
                main()
                # Since all images already had embeddings, load_vision_model should NOT be called
                mock_load_model.assert_not_called()

        # Check loaded embeddings match original
        loaded_embs = load_embeddings(
            self.input_parquet,
            representation_type="cls",
            model_name="google/tipsv2-b14",
        )
        self.assertEqual(len(loaded_embs), 3)
        np.testing.assert_allclose(loaded_embs[1], np.full(dim, 1.0, dtype=np.float32))

    def test_backfill_embeddings_incremental_resume(self):
        # 1. Create base dataset with 3 rows and save companion embeddings
        dim = 16
        df_base = pd.DataFrame(
            {
                "Photo_ID": ["10", "20", "30"],
                "Platform": ["flickr", "flickr", "flickr"],
                "Latitude": [10.0, 20.0, 30.0],
                "Longitude": [1.0, 2.0, 3.0],
                "Image_URL": ["url_10", "url_20", "url_30"],
                "embedding": [
                    np.full(dim, float(i + 1), dtype=np.float32) for i in range(3)
                ],
            }
        )
        save_dataframe(
            df_base,
            self.input_parquet,
            representation_type="cls",
            precision="float32",
            model_name="facebook/dinov2-base",
        )

        # 2. Append 2 new rows to the dataset (total 5 rows)
        df_expanded = pd.DataFrame(
            {
                "Photo_ID": ["10", "20", "30", "40", "50"],
                "Platform": ["flickr", "flickr", "flickr", "flickr", "flickr"],
                "Latitude": [10.0, 20.0, 30.0, 40.0, 50.0],
                "Longitude": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Image_URL": ["url_10", "url_20", "url_30", "url_40", "url_50"],
            }
        )
        # Save expanded dataset metadata (without embedding columns)
        df_expanded.to_parquet(self.input_parquet)

        # 3. Mock vision model and download for the 2 new images
        dummy_model = MagicMock()
        dummy_img = Image.new("RGB", (10, 10), color="blue")

        test_args = [
            "backfill_embeddings.py",
            "--input",
            self.input_parquet,
            "--model_name",
            "facebook/dinov2-base",
            "--representation_type",
            "cls",
            "--precision",
            "float32",
            "--resume",
        ]

        def mock_extract(
            model, batch_tensors, representation_type="cls", is_local=False
        ):
            batch_count = batch_tensors.shape[0]
            # Return predictable vectors for new images: 99.0
            return np.full((batch_count, dim), 99.0, dtype=np.float32)

        with patch.object(sys, "argv", test_args):
            with patch(
                "src.processing.backfill_embeddings.load_vision_model",
                return_value=(dummy_model, _dummy_transform, 224),
            ):
                with patch(
                    "src.processing.backfill_embeddings.download_image",
                    return_value=dummy_img,
                ):
                    with patch(
                        "src.processing.backfill_embeddings.extract_regular_embeddings",
                        side_effect=mock_extract,
                    ) as mock_feat:
                        main()
                        # Verify extract_regular_embeddings was called for exactly 2 new images
                        self.assertEqual(mock_feat.call_count, 1)

        # 4. Verify results
        df_result = load_dataframe(self.input_parquet)
        self.assertEqual(len(df_result), 5)

        loaded_embs = load_embeddings(
            self.input_parquet,
            representation_type="cls",
            model_name="facebook/dinov2-base",
            precision="float32",
        )
        self.assertEqual(len(loaded_embs), 5)
        # Old rows (first 3) should have their original embeddings preserved
        np.testing.assert_allclose(loaded_embs[0], np.full(dim, 1.0, dtype=np.float32))
        np.testing.assert_allclose(loaded_embs[1], np.full(dim, 2.0, dtype=np.float32))
        np.testing.assert_allclose(loaded_embs[2], np.full(dim, 3.0, dtype=np.float32))
        # New rows (last 2) should have newly computed embeddings: 99.0
        np.testing.assert_allclose(loaded_embs[3], np.full(dim, 99.0, dtype=np.float32))
        np.testing.assert_allclose(loaded_embs[4], np.full(dim, 99.0, dtype=np.float32))

    def test_backfill_embeddings_checkpoint_resume(self):
        dim = 16
        # Dataset has 4 rows: 1, 2 in checkpoint, 3, 4 missing
        df = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3", "4"],
                "Platform": ["flickr", "flickr", "flickr", "flickr"],
                "Latitude": [1.0, 2.0, 3.0, 4.0],
                "Longitude": [1.0, 2.0, 3.0, 4.0],
                "Image_URL": ["url_1", "url_2", "url_3", "url_4"],
            }
        )
        df.to_parquet(self.input_parquet)

        # Manually create a checkpoint companion file with vectors for rows 1 and 2
        ckpt_npy, ckpt_keys = resolve_checkpoint_paths(
            self.input_parquet, "facebook/dinov2-base", "cls"
        )
        ckpt_keys_list = ["flickr_1", "flickr_2"]
        ckpt_embs = np.array(
            [
                np.full(dim, 77.0, dtype=np.float32),
                np.full(dim, 88.0, dtype=np.float32),
            ]
        )
        save_checkpoint_companion(
            ckpt_npy, ckpt_keys, ckpt_keys_list, ckpt_embs, np.float32
        )
        self.assertTrue(os.path.exists(ckpt_npy))
        self.assertTrue(os.path.exists(ckpt_keys))

        dummy_model = MagicMock()
        dummy_img = Image.new("RGB", (10, 10), color="red")

        test_args = [
            "backfill_embeddings.py",
            "--input",
            self.input_parquet,
            "--model_name",
            "facebook/dinov2-base",
            "--representation_type",
            "cls",
            "--precision",
            "float32",
            "--resume",
        ]

        def mock_extract(
            model, batch_tensors, representation_type="cls", is_local=False
        ):
            batch_count = batch_tensors.shape[0]
            return np.full((batch_count, dim), 55.0, dtype=np.float32)

        with patch.object(sys, "argv", test_args):
            with patch(
                "src.processing.backfill_embeddings.load_vision_model",
                return_value=(dummy_model, _dummy_transform, 224),
            ):
                with patch(
                    "src.processing.backfill_embeddings.download_image",
                    return_value=dummy_img,
                ):
                    with patch(
                        "src.processing.backfill_embeddings.extract_regular_embeddings",
                        side_effect=mock_extract,
                    ) as mock_feat:
                        main()
                        # Should only process 2 remaining images (3 and 4)
                        self.assertEqual(mock_feat.call_count, 1)

        # Checkpoint files should be cleaned up on completion
        self.assertFalse(os.path.exists(ckpt_npy))
        self.assertFalse(os.path.exists(ckpt_keys))

        # Check final embeddings
        loaded_embs = load_embeddings(
            self.input_parquet,
            representation_type="cls",
            model_name="facebook/dinov2-base",
            precision="float32",
        )
        self.assertEqual(len(loaded_embs), 4)
        np.testing.assert_allclose(loaded_embs[0], np.full(dim, 77.0, dtype=np.float32))
        np.testing.assert_allclose(loaded_embs[1], np.full(dim, 88.0, dtype=np.float32))
        np.testing.assert_allclose(loaded_embs[2], np.full(dim, 55.0, dtype=np.float32))
        np.testing.assert_allclose(loaded_embs[3], np.full(dim, 55.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
