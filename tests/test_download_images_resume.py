import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.download_images import main, save_stream_checkpoint
from src.utils.io import load_dataframe, load_embeddings, save_dataframe


class TestDownloadImagesResume(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_download_resume_")
        self.input_parquet = os.path.join(self.test_dir, "input_dataset.parquet")
        self.output_parquet = os.path.join(self.test_dir, "output_dataset.parquet")
        self.output_images_dir = os.path.join(self.test_dir, "output_images")
        self.offline_images_dir = os.path.join(self.test_dir, "offline_images")
        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.offline_images_dir, exist_ok=True)

        # Create 4 mock 10x10 images
        self.image_paths = []
        for i in range(4):
            img_path = os.path.join(self.offline_images_dir, f"photo_{i}.jpg")
            img = Image.new("RGB", (10, 10), color=(i * 40, i * 40, i * 40))
            img.save(img_path, format="JPEG")
            self.image_paths.append(img_path)

        # Create input metadata
        self.df_input = pd.DataFrame(
            {
                "Photo_ID": ["100", "101", "102", "103"],
                "Platform": ["test_plat", "test_plat", "test_plat", "test_plat"],
                "Latitude": [10.0, 11.0, 12.0, 13.0],
                "Longitude": [20.0, 21.0, 22.0, 23.0],
                "Image_URL": [
                    "https://example.com/100.jpg",
                    "https://example.com/101.jpg",
                    "https://example.com/102.jpg",
                    "https://example.com/103.jpg",
                ],
                "Captured_At": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                "License": ["CC-BY", "CC-BY", "CC0", "CC0"],
            }
        )
        self.input_embs = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
        self.df_input["embedding"] = list(self.input_embs)
        save_dataframe(self.df_input, self.input_parquet, representation_type="cls")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_stream_checkpoint(self):
        df_sub = self.df_input.iloc[:2].copy()
        df_sub["photo_key"] = ["test_plat_100", "test_plat_101"]
        embs_sub = self.input_embs[:2]

        save_stream_checkpoint(
            out_metadata=self.output_parquet,
            df_combined=df_sub,
            embeddings_combined=embs_sub,
            representation_type="cls",
            precision="float32",
        )

        self.assertTrue(os.path.exists(self.output_parquet))
        loaded_df = load_dataframe(self.output_parquet)
        self.assertEqual(len(loaded_df), 2)
        loaded_embs = load_embeddings(self.output_parquet, representation_type="cls")
        self.assertEqual(loaded_embs.shape, (2, 8))
        np.testing.assert_allclose(loaded_embs, embs_sub)

    def test_resume_skips_existing_images(self):
        # 1. Pre-populate output_parquet with the first 2 images
        # Place images in output_images_dir so they pass disk validation
        plat_dir = os.path.join(self.output_images_dir, "test_plat")
        os.makedirs(plat_dir, exist_ok=True)
        shutil.copy2(self.image_paths[0], os.path.join(plat_dir, "100.jpg"))
        shutil.copy2(self.image_paths[1], os.path.join(plat_dir, "101.jpg"))

        df_first2 = self.df_input.iloc[:2].copy()
        df_first2["photo_key"] = ["test_plat_100", "test_plat_101"]
        df_first2["file_name"] = ["100.jpg", "101.jpg"]
        df_first2["Image_Location"] = [
            "./output_images/test_plat/100.jpg",
            "./output_images/test_plat/101.jpg",
        ]
        df_first2["Image_URL"] = df_first2["Image_Location"]
        save_stream_checkpoint(
            out_metadata=self.output_parquet,
            df_combined=df_first2,
            embeddings_combined=self.input_embs[:2],
            representation_type="cls",
            precision="float32",
        )

        # 2. Mock download_image so that it only downloads the remaining 2 images (102, 103)
        downloaded_ids = []

        def mock_download(url, output_path, photo_id, platform, timeout=10):
            downloaded_ids.append(str(photo_id))
            # Create dummy image at output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img = Image.new("RGB", (10, 10), color="blue")
            img.save(output_path, format="JPEG")
            return True

        test_args = [
            "download_images.py",
            "--input",
            self.input_parquet,
            "--output",
            self.output_parquet,
            "--output_dir",
            self.output_images_dir,
            "--resume",
            "--threads",
            "2",
        ]

        with (
            patch("sys.argv", test_args),
            patch(
                "src.utils.download_images.download_image", side_effect=mock_download
            ),
        ):
            main()

        # Check that download_image was ONLY called for 102 and 103
        self.assertEqual(sorted(downloaded_ids), ["102", "103"])

        # Check that the output dataset now contains all 4 rows and 4 embeddings
        df_res = load_dataframe(self.output_parquet)
        self.assertEqual(len(df_res), 4)
        embs_res = load_embeddings(self.output_parquet, representation_type="cls")
        self.assertEqual(len(embs_res), 4)
        self.assertEqual(list(df_res["Photo_ID"]), ["100", "101", "102", "103"])

    def test_copy_offline_images(self):
        # Point image_root_dirs to offline_images_dir where photo_0..photo_3 reside
        # Map input metadata Photo_IDs to photo_0..photo_3 via Image_Location
        df_offline = self.df_input.copy()
        df_offline["Image_Location"] = [f"photo_{i}.jpg" for i in range(4)]
        save_dataframe(df_offline, self.input_parquet, representation_type="cls")

        test_args = [
            "download_images.py",
            "--input",
            self.input_parquet,
            "--output",
            self.output_parquet,
            "--output_dir",
            self.output_images_dir,
            "--image_root_dirs",
            self.offline_images_dir,
            "--copy_offline_images",
            "--threads",
            "2",
        ]

        with patch("sys.argv", test_args):
            main()

        # Check that all 4 images were copied to output_images_dir/test_plat/<Photo_ID>.jpg
        for pid in ["100", "101", "102", "103"]:
            target_img = os.path.join(self.output_images_dir, "test_plat", f"{pid}.jpg")
            self.assertTrue(
                os.path.exists(target_img), f"Missing copied image: {target_img}"
            )

        # Check output parquet metadata
        df_res = load_dataframe(self.output_parquet)
        self.assertEqual(len(df_res), 4)
        for loc in df_res["Image_Location"]:
            self.assertTrue(loc.startswith("./output_images/test_plat/"))
        # Verify original remote Image_URL was preserved
        self.assertEqual(
            list(df_res["Image_URL"]),
            [
                "https://example.com/100.jpg",
                "https://example.com/101.jpg",
                "https://example.com/102.jpg",
                "https://example.com/103.jpg",
            ],
        )

    def test_copy_offline_images_nested_folders(self):
        # Create nested folders in offline_images_dir:
        # site_A/cam_1/100.jpg
        # deep/level2/level3/101.jpg
        # custom_sub/102.jpg
        # 103.jpg (flat)
        nested_dir1 = os.path.join(self.offline_images_dir, "site_A", "cam_1")
        nested_dir2 = os.path.join(self.offline_images_dir, "deep", "level2", "level3")
        nested_dir3 = os.path.join(self.offline_images_dir, "custom_sub")
        os.makedirs(nested_dir1, exist_ok=True)
        os.makedirs(nested_dir2, exist_ok=True)
        os.makedirs(nested_dir3, exist_ok=True)

        shutil.copy2(self.image_paths[0], os.path.join(nested_dir1, "100.jpg"))
        shutil.copy2(self.image_paths[1], os.path.join(nested_dir2, "101.jpg"))
        shutil.copy2(self.image_paths[2], os.path.join(nested_dir3, "102.jpg"))
        shutil.copy2(
            self.image_paths[3], os.path.join(self.offline_images_dir, "103.jpg")
        )

        df_nested = self.df_input.copy()
        # Even without Image_Location or with generic names, lookup by Photo_ID succeeds
        df_nested["Image_Location"] = [
            "site_A/cam_1/100.jpg",
            "101.jpg",
            "different_name.jpg",  # Photo_ID is 102
            "103.jpg",
        ]
        save_dataframe(df_nested, self.input_parquet, representation_type="cls")

        test_args = [
            "download_images.py",
            "--input",
            self.input_parquet,
            "--output",
            self.output_parquet,
            "--output_dir",
            self.output_images_dir,
            "--image_root_dirs",
            self.offline_images_dir,
            "--copy_offline_images",
            "--threads",
            "2",
        ]

        with patch("sys.argv", test_args):
            main()

        # Check all 4 images were discovered from nested folders and unified in output_dir
        for pid in ["100", "101", "102", "103"]:
            target_img = os.path.join(self.output_images_dir, "test_plat", f"{pid}.jpg")
            self.assertTrue(
                os.path.exists(target_img),
                f"Missing copied image from nested folder: {target_img}",
            )

        df_res = load_dataframe(self.output_parquet)
        self.assertEqual(len(df_res), 4)
        for loc in df_res["Image_Location"]:
            self.assertTrue(loc.startswith("./output_images/test_plat/"))

    def test_resume_verify_existing_redownloads_missing(self):
        # 1. Pre-populate output_parquet with first 2 images
        plat_dir = os.path.join(self.output_images_dir, "test_plat")
        os.makedirs(plat_dir, exist_ok=True)
        img_100 = os.path.join(plat_dir, "100.jpg")
        img_101 = os.path.join(plat_dir, "101.jpg")
        shutil.copy2(self.image_paths[0], img_100)
        shutil.copy2(self.image_paths[1], img_101)

        df_first2 = self.df_input.iloc[:2].copy()
        df_first2["photo_key"] = ["test_plat_100", "test_plat_101"]
        df_first2["file_name"] = ["100.jpg", "101.jpg"]
        df_first2["Image_Location"] = [
            "./output_images/test_plat/100.jpg",
            "./output_images/test_plat/101.jpg",
        ]
        df_first2["Image_URL"] = df_first2["Image_Location"]
        save_stream_checkpoint(
            out_metadata=self.output_parquet,
            df_combined=df_first2,
            embeddings_combined=self.input_embs[:2],
            representation_type="cls",
            precision="float32",
        )

        # 2. Delete 100.jpg from disk to simulate missing file
        os.remove(img_100)

        # 3. Resume WITH --verify_existing
        downloaded_ids = []

        def mock_download(url, output_path, photo_id, platform, timeout=10):
            downloaded_ids.append(str(photo_id))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img = Image.new("RGB", (10, 10), color="green")
            img.save(output_path, format="JPEG")
            return True

        test_args = [
            "download_images.py",
            "--input",
            self.input_parquet,
            "--output",
            self.output_parquet,
            "--output_dir",
            self.output_images_dir,
            "--resume",
            "--verify_existing",
            "--threads",
            "2",
        ]

        with (
            patch("sys.argv", test_args),
            patch(
                "src.utils.download_images.download_image", side_effect=mock_download
            ),
        ):
            main()

        # 100 was missing on disk, 102 and 103 were missing from parquet -> all 3 re-downloaded
        # 101 was valid on disk -> skipped
        self.assertEqual(sorted(downloaded_ids), ["100", "102", "103"])
        df_res = load_dataframe(self.output_parquet)
        self.assertEqual(len(df_res), 4)
        embs_res = load_embeddings(self.output_parquet, representation_type="cls")
        self.assertEqual(len(embs_res), 4)

    def test_fast_resume_skips_disk_verification(self):
        # 1. Pre-populate output_parquet with first 2 images
        plat_dir = os.path.join(self.output_images_dir, "test_plat")
        os.makedirs(plat_dir, exist_ok=True)
        img_100 = os.path.join(plat_dir, "100.jpg")
        img_101 = os.path.join(plat_dir, "101.jpg")
        shutil.copy2(self.image_paths[0], img_100)
        shutil.copy2(self.image_paths[1], img_101)

        df_first2 = self.df_input.iloc[:2].copy()
        df_first2["photo_key"] = ["test_plat_100", "test_plat_101"]
        df_first2["file_name"] = ["100.jpg", "101.jpg"]
        df_first2["Image_Location"] = [
            "./output_images/test_plat/100.jpg",
            "./output_images/test_plat/101.jpg",
        ]
        df_first2["Image_URL"] = df_first2["Image_Location"]
        save_stream_checkpoint(
            out_metadata=self.output_parquet,
            df_combined=df_first2,
            embeddings_combined=self.input_embs[:2],
            representation_type="cls",
            precision="float32",
        )

        # 2. Delete 100.jpg from disk to simulate missing file
        os.remove(img_100)

        # 3. Resume WITHOUT --verify_existing (default fast resume)
        downloaded_ids = []

        def mock_download(url, output_path, photo_id, platform, timeout=10):
            downloaded_ids.append(str(photo_id))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img = Image.new("RGB", (10, 10), color="green")
            img.save(output_path, format="JPEG")
            return True

        test_args = [
            "download_images.py",
            "--input",
            self.input_parquet,
            "--output",
            self.output_parquet,
            "--output_dir",
            self.output_images_dir,
            "--resume",
            "--threads",
            "2",
        ]

        with (
            patch("sys.argv", test_args),
            patch(
                "src.utils.download_images.download_image", side_effect=mock_download
            ),
        ):
            main()

        # Fast resume trusts output_parquet: only 102 and 103 are downloaded, 100 is skipped
        self.assertEqual(sorted(downloaded_ids), ["102", "103"])

    def test_resume_copy_offline_images_unifies_existing_records(self):
        # 1. Place offline images in offline_images_dir
        offline_img_100 = os.path.join(self.offline_images_dir, "100.jpg")
        offline_img_101 = os.path.join(self.offline_images_dir, "101.jpg")
        shutil.copy2(self.image_paths[0], offline_img_100)
        shutil.copy2(self.image_paths[1], offline_img_101)

        # 2. Output parquet has 100 and 101 recorded, but files are NOT yet in output_images_dir
        df_first2 = self.df_input.iloc[:2].copy()
        df_first2["photo_key"] = ["test_plat_100", "test_plat_101"]
        df_first2["file_name"] = ["old_100.jpg", "old_101.jpg"]
        df_first2["Image_Location"] = [
            offline_img_100,
            offline_img_101,
        ]
        df_first2["Image_URL"] = [
            "https://example.com/orig_100.jpg",
            "https://example.com/orig_101.jpg",
        ]
        save_stream_checkpoint(
            out_metadata=self.output_parquet,
            df_combined=df_first2,
            embeddings_combined=self.input_embs[:2],
            representation_type="cls",
            precision="float32",
        )

        downloaded_ids = []

        def mock_download(url, output_path, photo_id, platform, timeout=10):
            downloaded_ids.append(str(photo_id))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img = Image.new("RGB", (10, 10), color="blue")
            img.save(output_path, format="JPEG")
            return True

        test_args = [
            "download_images.py",
            "--input",
            self.input_parquet,
            "--output",
            self.output_parquet,
            "--output_dir",
            self.output_images_dir,
            "--image_root_dirs",
            self.offline_images_dir,
            "--copy_offline_images",
            "--resume",
            "--threads",
            "2",
        ]

        with (
            patch("sys.argv", test_args),
            patch(
                "src.utils.download_images.download_image",
                side_effect=mock_download,
            ),
        ):
            main()

        # Existing offline images 100 and 101 must be copied to output_images_dir
        target_100 = os.path.join(self.output_images_dir, "test_plat", "100.jpg")
        target_101 = os.path.join(self.output_images_dir, "test_plat", "101.jpg")
        self.assertTrue(os.path.exists(target_100))
        self.assertTrue(os.path.exists(target_101))

        # Remaining images 102 and 103 were downloaded
        self.assertEqual(sorted(downloaded_ids), ["102", "103"])

        # Output parquet metadata must be unified with rel paths
        df_res = load_dataframe(self.output_parquet)
        self.assertEqual(len(df_res), 4)
        for loc in df_res["Image_Location"][:2]:
            self.assertTrue(loc.startswith("./output_images/test_plat/"))
        # Original Image_URL preserved without --overwrite_image_url
        self.assertEqual(
            df_res["Image_URL"].iloc[0], "https://example.com/orig_100.jpg"
        )
