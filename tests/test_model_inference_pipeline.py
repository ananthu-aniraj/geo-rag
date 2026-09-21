import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.processing.process_scraped_data import (
    get_image_embeddings,
    get_tips_embeddings,
    process_cell,
    save_checkpoint,
    stream_update_parquet,
)
from src.utils.io import load_embeddings, save_dataframe


class TestModelInferencePipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch_models"
        os.makedirs(self.test_dir, exist_ok=True)
        self.device = torch.device("cpu")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_extract_regular_embeddings_representations(self):
        """Test extract_regular_embeddings across HF, local, and timm architectures."""
        from src.models.vision_model_inference import extract_regular_embeddings

        B = 2
        dim = 64
        num_patches = 16
        dummy_tensors = torch.zeros((B, 3, 32, 32))

        # 1. HuggingFace style model with encode_image
        hf_model = MagicMock()
        mock_hf_out = MagicMock()
        mock_hf_out.cls_token = torch.ones((B, dim), dtype=torch.float32)
        mock_hf_out.patch_tokens = (
            torch.ones((B, num_patches, dim), dtype=torch.float32) * 2.0
        )
        hf_model.encode_image.return_value = mock_hf_out

        embs_cls = extract_regular_embeddings(
            hf_model, dummy_tensors, representation_type="cls"
        )
        self.assertEqual(embs_cls.shape, (B, dim))
        np.testing.assert_allclose(embs_cls, 1.0)

        embs_patch = extract_regular_embeddings(
            hf_model, dummy_tensors, representation_type="avg_patch"
        )
        self.assertEqual(embs_patch.shape, (B, dim))
        np.testing.assert_allclose(embs_patch, 2.0)

        embs_concat = extract_regular_embeddings(
            hf_model, dummy_tensors, representation_type="cls_avg_patch"
        )
        self.assertEqual(embs_concat.shape, (B, 2 * dim))
        np.testing.assert_allclose(embs_concat[:, :dim], 1.0)
        np.testing.assert_allclose(embs_concat[:, dim:], 2.0)

        # 2. Local TIPSv2 checkpoint model (returns tuple: (cls1, cls2, patch_tokens))
        local_model = MagicMock()
        local_model.return_value = (
            torch.ones((B, dim), dtype=torch.float32),
            torch.ones((B, dim), dtype=torch.float32),
            torch.ones((B, num_patches, dim), dtype=torch.float32) * 2.0,
        )
        embs_local_cls = extract_regular_embeddings(
            local_model, dummy_tensors, representation_type="cls", is_local=True
        )
        self.assertEqual(embs_local_cls.shape, (B, dim))
        np.testing.assert_allclose(embs_local_cls, 1.0)

        embs_local_patch = extract_regular_embeddings(
            local_model, dummy_tensors, representation_type="avg_patch", is_local=True
        )
        self.assertEqual(embs_local_patch.shape, (B, dim))
        np.testing.assert_allclose(embs_local_patch, 2.0)

        # 3. timm style ViT model with forward_features
        timm_model = MagicMock()
        del timm_model.encode_image
        timm_tokens = torch.cat(
            [
                torch.ones((B, 1, dim), dtype=torch.float32),
                torch.ones((B, num_patches, dim), dtype=torch.float32) * 2.0,
            ],
            dim=1,
        )
        timm_model.forward_features.return_value = timm_tokens
        timm_model.has_cls_token = True
        timm_model.num_prefix_tokens = 1

        embs_timm_cls = extract_regular_embeddings(
            timm_model, dummy_tensors, representation_type="cls"
        )
        self.assertEqual(embs_timm_cls.shape, (B, dim))
        np.testing.assert_allclose(embs_timm_cls, 1.0)

        embs_timm_patch = extract_regular_embeddings(
            timm_model, dummy_tensors, representation_type="avg_patch"
        )
        self.assertEqual(embs_timm_patch.shape, (B, dim))
        np.testing.assert_allclose(embs_timm_patch, 2.0)

    def test_get_image_embeddings_representations(self):
        """Test get_image_embeddings with batched image processing."""
        # Create 3 dummy PIL images
        images = [
            Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50)) for i in range(3)
        ]
        dim = 64
        num_patches = 16

        mock_model = MagicMock()

        def mock_encode(batch_tensors):
            b = batch_tensors.shape[0]
            out = MagicMock()
            out.cls_token = torch.ones((b, dim), dtype=torch.float32)
            out.patch_tokens = (
                torch.ones((b, num_patches, dim), dtype=torch.float32) * 2.0
            )
            return out

        mock_model.encode_image.side_effect = mock_encode

        # Test CLS
        embs_cls = get_image_embeddings(
            images, mock_model, self.device, batch_size=2, representation_type="cls"
        )
        self.assertEqual(embs_cls.shape, (3, dim))
        np.testing.assert_allclose(embs_cls, 1.0)

        # Test avg_patch
        embs_patch = get_image_embeddings(
            images,
            mock_model,
            self.device,
            batch_size=2,
            representation_type="avg_patch",
        )
        self.assertEqual(embs_patch.shape, (3, dim))
        np.testing.assert_allclose(embs_patch, 2.0)

        # Test cls_avg_patch
        embs_concat = get_image_embeddings(
            images,
            mock_model,
            self.device,
            batch_size=2,
            representation_type="cls_avg_patch",
        )
        self.assertEqual(embs_concat.shape, (3, 2 * dim))

        # Test backwards-compatible get_tips_embeddings wrapper
        embs_tips = get_tips_embeddings(
            images, mock_model, self.device, batch_size=2, representation_type="cls"
        )
        self.assertEqual(embs_tips.shape, (3, dim))

    def test_process_cell_without_text_filter(self):
        """Test process_cell deduplication when zero-shot text filtering is disabled/unavailable."""
        mock_model = MagicMock()
        images = [Image.new("RGB", (32, 32)) for _ in range(2)]

        metadata_list = [
            {
                "Photo_ID": "p1",
                "Platform": "flickr",
                "Image_URL": "url1",
                "H3_Cell": "cell_1",
                "Latitude": 10.0,
                "Longitude": 20.0,
            },
            {
                "Photo_ID": "p2",
                "Platform": "inaturalist",
                "Image_URL": "url2",
                "H3_Cell": "cell_1",
                "Latitude": 10.0,
                "Longitude": 20.0,
            },
        ]

        # Return identical embeddings for both images to test deduplication
        with (
            patch(
                "src.processing.process_scraped_data.get_image_embeddings",
                return_value=np.ones((2, 64), dtype=np.float32),
            ),
            patch(
                "src.processing.process_scraped_data.download_image",
                side_effect=lambda url, **kwargs: Image.new("RGB", (32, 32)),
            ),
        ):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as executor:
                deduped = process_cell(
                    cell_id="cell_1",
                    metadata_list=metadata_list,
                    model=mock_model,
                    device=self.device,
                    sim_threshold=0.90,  # Cosine similarity is 1.0 > 0.90, so p2 will be dropped
                    executor=executor,
                    text_features=None,  # No text filter!
                )

            # Only 1 unique image should be kept
            self.assertEqual(len(deduped), 1)
            self.assertEqual(deduped[0]["Photo_ID"], "p1")

    def test_stream_update_and_save_with_model_name(self):
        """Test stream_update_parquet and save_dataframe with custom model_name."""
        input_parquet = os.path.join(self.test_dir, "base.parquet")
        output_parquet = os.path.join(self.test_dir, "updated.parquet")
        model_name = "timm/vit_base_patch16_224"

        df_base = pd.DataFrame(
            {
                "Photo_ID": ["10", "20"],
                "Platform": ["flickr", "inaturalist"],
                "H3_Cell": ["cell_a", "cell_b"],
                "Latitude": [1.0, 2.0],
                "Longitude": [3.0, 4.0],
                "Image_URL": ["u1", "u2"],
                "embedding": [
                    np.ones(64, dtype=np.float32),
                    np.ones(64, dtype=np.float32) * 2,
                ],
            }
        )
        save_dataframe(
            df_base,
            input_parquet,
            representation_type="cls",
            precision="float32",
            model_name=model_name,
        )

        expected_model_suffix = "_" + model_name.replace("/", "_")
        expected_base_npy = os.path.join(
            self.test_dir, f"base{expected_model_suffix}_cls_embeddings.npy"
        )
        self.assertTrue(os.path.exists(expected_base_npy))

        # Test loading embeddings with model_name
        loaded = load_embeddings(
            input_parquet, representation_type="cls", model_name=model_name
        )
        self.assertEqual(loaded.shape, (2, 64))

        # Test stream_update_parquet with model_name
        df_new = pd.DataFrame(
            {
                "Photo_ID": ["30"],
                "Platform": ["flickr"],
                "H3_Cell": ["cell_a"],  # Replaces cell_a
                "Latitude": [1.5],
                "Longitude": [3.5],
                "Image_URL": ["u3"],
                "embedding": [np.ones(64, dtype=np.float32) * 3],
            }
        )

        stream_update_parquet(
            input_path=input_parquet,
            output_path=output_parquet,
            df_new=df_new,
            active_cells={"cell_a"},
            representation_type="cls",
            precision="float32",
            model_name=model_name,
        )

        expected_updated_npy = os.path.join(
            self.test_dir, f"updated{expected_model_suffix}_cls_embeddings.npy"
        )
        self.assertTrue(os.path.exists(expected_updated_npy))

        # Verify updated embeddings: row 20 from base + row 30 from df_new
        loaded_updated = load_embeddings(
            output_parquet, representation_type="cls", model_name=model_name
        )
        self.assertEqual(loaded_updated.shape, (2, 64))
        np.testing.assert_allclose(loaded_updated[0], 2.0)  # row 20
        np.testing.assert_allclose(loaded_updated[1], 3.0)  # row 30

    def test_save_checkpoint_with_model_name(self):
        """Test save_checkpoint with model_name passes model_name to companion files."""
        # 1. Test TIPSv2 saves without model suffix (simplifying assumption)
        ckpt_parquet_tips = os.path.join(self.test_dir, "test_ckpt_tips.parquet")
        ckpt_meta_tips = os.path.join(self.test_dir, "test_ckpt_tips_meta.pkl")
        model_name_tips = "google/tipsv2-b14"

        data = [
            {
                "Photo_ID": "1",
                "Platform": "flickr",
                "Latitude": 10.0,
                "Longitude": 20.0,
                "Image_URL": "url",
                "H3_Cell": "cell",
                "embedding": np.ones(64, dtype=np.float32),
            }
        ]

        save_checkpoint(
            final_data=data,
            processed_cells={"cell"},
            checkpoint_path=ckpt_parquet_tips,
            checkpoint_meta_path=ckpt_meta_tips,
            representation_type="cls",
            precision="float32",
            model_name=model_name_tips,
        )

        self.assertTrue(os.path.exists(ckpt_parquet_tips))
        self.assertTrue(os.path.exists(ckpt_meta_tips))
        expected_tips_npy = os.path.join(
            self.test_dir, "test_ckpt_tips_cls_embeddings.npy"
        )
        self.assertTrue(os.path.exists(expected_tips_npy))

        # 2. Test custom model saves with model suffix
        ckpt_parquet_custom = os.path.join(self.test_dir, "test_ckpt_custom.parquet")
        ckpt_meta_custom = os.path.join(self.test_dir, "test_ckpt_custom_meta.pkl")
        model_name_custom = "timm/vit_base_patch16_224"

        save_checkpoint(
            final_data=data,
            processed_cells={"cell"},
            checkpoint_path=ckpt_parquet_custom,
            checkpoint_meta_path=ckpt_meta_custom,
            representation_type="cls",
            precision="float32",
            model_name=model_name_custom,
        )

        self.assertTrue(os.path.exists(ckpt_parquet_custom))
        self.assertTrue(os.path.exists(ckpt_meta_custom))
        expected_custom_suf = "_" + model_name_custom.replace("/", "_")
        expected_custom_npy = os.path.join(
            self.test_dir, f"test_ckpt_custom{expected_custom_suf}_cls_embeddings.npy"
        )
        self.assertTrue(os.path.exists(expected_custom_npy))

    def test_load_embeddings_missing_backfill_error(self):
        """Test load_embeddings raises FileNotFoundError with backfill instructions when missing."""
        parquet_path = os.path.join(self.test_dir, "sample.parquet")
        df = pd.DataFrame(
            {
                "Photo_ID": ["1"],
                "H3_Cell": ["cell"],
                "embedding": [np.ones(64, dtype=np.float32)],
            }
        )
        # Save legacy TIPSv2 CLS embeddings (no model name suffix)
        save_dataframe(
            df,
            parquet_path,
            representation_type="cls",
            precision="float32",
            model_name="google/tipsv2-b14",
        )

        # 1. Non-TIPSv2 model should NOT fall back to legacy TIPSv2 files and raise backfill guidance
        with self.assertRaises(FileNotFoundError) as ctx:
            load_embeddings(
                parquet_path,
                representation_type="cls",
                model_name="facebook/dinov2-base",
            )
        self.assertIn("backfill_embeddings", str(ctx.exception))
        self.assertIn("--model_name facebook/dinov2-base", str(ctx.exception))

        # 2. Missing TIPSv2 representation (e.g. avg_patch) should raise backfill guidance
        with self.assertRaises(FileNotFoundError) as ctx:
            load_embeddings(
                parquet_path,
                representation_type="avg_patch",
                model_name="google/tipsv2-b14",
            )
        self.assertIn("backfill_embeddings", str(ctx.exception))
        self.assertIn("--representation_type avg_patch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
