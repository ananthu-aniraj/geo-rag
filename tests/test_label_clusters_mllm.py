import os
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.indexing.label_clusters_mllm import (
    build_prompt_templates,
    label_clusters_zeroshot,
    load_image,
    resize_image_aspect,
)


class TestLabelClustersMLLM(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch_mllm"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a dummy image
        self.dummy_image_path = os.path.join(self.test_dir, "dummy.jpg")
        img = Image.new("RGB", (800, 600), color="blue")
        img.save(self.dummy_image_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir)

    def test_resize_image_aspect(self):
        # 1. Image larger than target_max (800x600 -> 448 max)
        img = Image.new("RGB", (800, 600))
        resized = resize_image_aspect(img, target_max=448)
        self.assertEqual(resized.size, (448, 336))

        # 2. Image smaller than target_max (200x150)
        img_small = Image.new("RGB", (200, 150))
        resized_small = resize_image_aspect(img_small, target_max=448)
        self.assertEqual(resized_small.size, (200, 150))

        # 3. Square image (600x600)
        img_square = Image.new("RGB", (600, 600))
        resized_square = resize_image_aspect(img_square, target_max=300)
        self.assertEqual(resized_square.size, (300, 300))

    def test_load_image_local(self):
        img = load_image(self.dummy_image_path, target_max=300)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (300, 225))

    def test_build_prompt_templates(self):
        # Build template with standard parameters
        rep_item = {
            "Latitude": 40.7128,
            "Longitude": -74.0060,
            "country": "USA",
            "continent": "North America",
            "Season": "Winter",
            "Time_Of_Day": "Day",
            "Koppen_Code": "Cfa",
            "Koppen_Desc": "Humid subtropical",
        }
        step1_template = "This is step 1."
        step2_template = "Location: {location}, Country: {country}, Continent: {continent}, Season: {season}, Time: {time_of_day}, Koppen: {koppen_code} ({koppen_desc}), Visual: {visual_description}, LULC: {lulc_list}"
        lulc_list_str = "Forest, City"

        system_prompt, user_prompt_template = build_prompt_templates(
            representative_item=rep_item,
            prompt_step1_template=step1_template,
            prompt_step2_template=step2_template,
            lulc_list_str=lulc_list_str,
        )
        self.assertEqual(system_prompt, step1_template)
        self.assertIn("USA", user_prompt_template)
        self.assertIn("Forest, City", user_prompt_template)

    def test_label_clusters_zeroshot(self):
        # 2 centroids, 4 dimensions
        centroids = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )

        # 3 text categories represented by target feature vectors
        text_features = np.array(
            [
                [0.99, 0.01, 0.0, 0.0],  # Matches centroid 0
                [0.02, 0.98, 0.0, 0.0],  # Matches centroid 1
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        categories = ["Forest", "City", "Ocean"]

        predictions = label_clusters_zeroshot(
            centroids=centroids,
            text_features=text_features,
            categories=categories,
            top_k=2,
        )

        self.assertEqual(len(predictions), 2)
        # Centroid 0 top label should be Forest
        self.assertTrue(predictions[0].startswith("Forest"))
        # Centroid 1 top label should be City
        self.assertTrue(predictions[1].startswith("City"))

    def test_create_letterboxed_cell(self):
        from src.indexing.multi_medoid_utils import create_letterboxed_cell

        img = Image.new("RGB", (100, 100), color="blue")
        cell = create_letterboxed_cell(img, target_w=512, target_h=256)
        self.assertEqual(cell.size, (512, 256))
        # Verify background border color (should be dark-gray at coordinate 0,0)
        border_pixel = cell.getpixel((0, 0))
        self.assertEqual(border_pixel, (40, 40, 40))

    def test_stitch_cells_vertically(self):
        from src.indexing.multi_medoid_utils import stitch_cells_vertically

        cells = [
            Image.new("RGB", (512, 256), color="red"),
            Image.new("RGB", (512, 256), color="green"),
            Image.new("RGB", (512, 256), color="blue"),
        ]
        collage = stitch_cells_vertically(cells, target_w=512, target_h=256)
        self.assertEqual(collage.size, (512, 768))

    def test_sample_diverse_medoids(self):
        from src.indexing.multi_medoid_utils import sample_diverse_medoids

        # 4 indices, 2 dimensions
        embeddings_norm = np.array(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
        )
        indices = np.array([0, 1, 2, 3])
        centroid_norm = np.array([1.0, 0.0], dtype=np.float32)

        df = pd.DataFrame(
            {
                "Latitude": [40.0, 40.0, 50.0, 50.0],
                "Longitude": [-74.0, -74.0, 2.0, 2.0],
                "Image_URL": ["url0", "url0", "url1", "url2"],
            }
        )

        # When we sample diverse medoids, it should select index 0 (closest),
        # then index 2 (geographically and URL-wise diverse), and then fill the rest.
        selected = sample_diverse_medoids(
            embeddings_norm, indices, centroid_norm, df, n_medoids=2
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[1], 3)

    def test_aggregate_medoid_metadata(self):
        from src.indexing.multi_medoid_utils import aggregate_medoid_metadata

        df = pd.DataFrame(
            {
                "Latitude": [40.0, 41.0],
                "Longitude": [-74.0, -73.0],
                "country": ["USA", "USA"],
                "continent": ["North America", "North America"],
                "Season": ["Winter", "Spring"],
                "Time_Of_Day": ["Day", "Night"],
                "Koppen_Code": ["Cfa", "Cfb"],
                "Koppen_Desc": ["Humid", "Oceanic"],
            }
        )

        meta = aggregate_medoid_metadata([0, 1], df)
        self.assertEqual(meta["country"], "USA")
        self.assertEqual(meta["season"], "Winter, Spring")
        self.assertEqual(meta["time_of_day"], "Day, Night")
        self.assertIn("Bounding Box", meta["location"])
        self.assertEqual(meta["koppen_code"], "Cfa, Cfb")

    @unittest.mock.patch("src.indexing.label_clusters_mllm.load_image")
    @unittest.mock.patch("src.indexing.label_clusters_mllm.query_vlm_openai_api")
    def test_label_clusters_mllm_batched_partial_download_failure(
        self, mock_query, mock_load
    ):
        from src.indexing.label_clusters_mllm import label_clusters_mllm_batched

        # Mock load_image: first returns a PIL Image, second returns None, third returns PIL Image, fourth returns None
        mock_img = Image.new("RGB", (100, 100), color="blue")
        mock_load.side_effect = [mock_img, None, mock_img, None]

        # Mock query_vlm_openai_api to return valid description and label
        mock_query.side_effect = [
            "This is a description.",
            "LABEL: Forest\nDESCRIPTION: This is a description.",
        ]

        tasks = [
            {
                "cid": 0,
                "prompt_step1": "Prompt 1",
                "prompt_step2_template": "Prompt 2: {visual_description}",
                "medoids": [
                    {"img_url": "success1", "photo_id": "1", "platform": "flickr"},
                    {"img_url": "fail1", "photo_id": "2", "platform": "flickr"},
                    {"img_url": "success2", "photo_id": "3", "platform": "flickr"},
                    {"img_url": "fail2", "photo_id": "4", "platform": "flickr"},
                ],
            }
        ]

        results = label_clusters_mllm_batched(
            tasks=tasks,
            model_name="gemma",
            endpoint_url="http://localhost:11434",
            chunk_size=1,
            img_max_dim=100,
        )

        # The labeling should succeed because 2 of the 4 images successfully downloaded!
        self.assertIn(0, results)
        label, desc, desc_vis = results[0]
        self.assertEqual(label, "Forest")
        self.assertEqual(desc, "This is a description.")

        # Also let's check that load_image was called 4 times
        self.assertEqual(mock_load.call_count, 4)

    def test_dataframe_row_wrapper(self):
        import pandas as pd

        from src.indexing.multi_medoid_utils import DataFrameRowWrapper

        df = pd.DataFrame(
            {
                "Latitude": [45.0, None, 46.0],
                "Longitude": [-122.0, -123.0, None],
                "Image_URL": ["url1", "url2", "url3"],
                "Photo_ID": ["1", "2.0", "3"],
            }
        )

        wrapper = DataFrameRowWrapper(df)
        self.assertEqual(len(wrapper), 3)

        # Test individual row retrieval
        row0 = wrapper[0]
        self.assertEqual(row0["Image_URL"], "url1")
        self.assertEqual(row0.get("Latitude"), 45.0)
        self.assertEqual(row0.get("Longitude"), -122.0)

        # Test null handle
        self.assertEqual(row0.get("Nonexistent", "default"), "default")
        row1 = wrapper[1]
        self.assertIsNone(row1.get("Latitude"))

        # Test iteration
        rows = list(wrapper)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["Image_URL"], "url3")

    def test_diverse_medoids_pre_sliced(self):
        import pandas as pd

        from src.indexing.multi_medoid_utils import sample_diverse_medoids

        df = pd.DataFrame(
            {
                "Latitude": [45.0, 45.0, 46.0, 47.0],
                "Longitude": [-122.0, -122.0, -123.0, -124.0],
                "Image_URL": ["url1", "url1", "url3", "url4"],
            }
        )

        # If we pass pre-sliced cluster embeddings (shape matching len(indices)):
        cluster_embs_norm = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
        centroid_norm = np.array([1.0, 0.0])
        indices = np.array([0, 1, 2, 3])

        selected = sample_diverse_medoids(
            cluster_embs_norm, indices, centroid_norm, df, n_medoids=3
        )

        # index 0 and 1 are identical (lat/lon/url), so one should be pruned for diversity
        self.assertIn(0, selected)
        self.assertNotIn(1, selected)
        self.assertIn(2, selected)
        self.assertIn(3, selected)

    def test_end_to_end_mllm_labeling_and_parquet_saving(self):
        import sys
        from unittest.mock import MagicMock, patch

        from src.indexing.label_clusters_mllm import main
        from src.utils.io import load_dataframe, save_dataframe

        input_parquet = os.path.join(self.test_dir, "test_input_mllm.parquet")
        output_parquet = os.path.join(self.test_dir, "test_output_mllm.parquet")

        df = pd.DataFrame(
            {
                "Photo_ID": [str(i) for i in range(6)],
                "Platform": ["flickr"] * 6,
                "Latitude": [40.0, 40.1, 40.2, 50.0, 50.1, 50.2],
                "Longitude": [-74.0, -74.1, -74.2, 2.0, 2.1, 2.2],
                "Image_URL": [f"http://example.com/{i}.jpg" for i in range(6)],
                "cluster_id": [0, 0, 0, 1, 1, 1],
                "parent_cluster_id": [0, 0, 0, 0, 0, 0],
            }
        )
        embs = np.zeros((6, 4), dtype=np.float32)
        embs[:3, 0] = 1.0
        embs[3:, 1] = 1.0
        df["embedding"] = list(embs)
        save_dataframe(df, input_parquet, representation_type="cls")

        mock_img = Image.new("RGB", (100, 100), color="green")
        mock_url_resp = MagicMock()
        mock_url_resp.status = 200
        mock_url_resp.__enter__.return_value = mock_url_resp

        with (
            patch("urllib.request.urlopen", return_value=mock_url_resp),
            patch("src.indexing.label_clusters_mllm.load_image", return_value=mock_img),
            patch(
                "src.indexing.label_clusters_mllm.query_vlm_openai_api",
                side_effect=[
                    "Visual desc parent",
                    "LABEL: ParentForest\nDESCRIPTION: Parent forest description.",
                    "Visual desc child0",
                    "LABEL: Forest\nDESCRIPTION: Child forest description.",
                    "Visual desc child1",
                    "LABEL: Agriculture\nDESCRIPTION: Child agriculture description.",
                ],
            ),
        ):
            test_args = [
                "label_clusters_mllm.py",
                "--in",
                input_parquet,
                "--out",
                output_parquet,
                "--label_method",
                "mllm",
                "--mllm_model",
                "mock-model",
                "--mllm_endpoint",
                "http://localhost:8000",
                "--num_medoids",
                "2",
                "--chunk_size",
                "2",
            ]
            with patch.object(sys, "argv", test_args):
                main()

        df_out = load_dataframe(output_parquet)
        self.assertEqual(len(df_out), 6)
        self.assertIn("cluster_label", df_out.columns)
        self.assertIn("cluster_description", df_out.columns)
        self.assertIn("visual_description", df_out.columns)
        self.assertIn("parent_cluster_label", df_out.columns)
        self.assertIn("parent_cluster_description", df_out.columns)
        self.assertIn("parent_visual_description", df_out.columns)

        self.assertEqual(df_out["cluster_label"].iloc[0], "Forest")
        self.assertEqual(df_out["cluster_label"].iloc[3], "Agriculture")
        self.assertEqual(df_out["parent_cluster_label"].iloc[0], "ParentForest")
        self.assertEqual(df_out["visual_description"].iloc[0], "Visual desc child0")
        self.assertEqual(df_out["visual_description"].iloc[3], "Visual desc child1")
        self.assertEqual(
            df_out["parent_visual_description"].iloc[0], "Visual desc parent"
        )

    def test_end_to_end_single_medoid_in_place_saving(self):
        import sys
        from unittest.mock import MagicMock, patch

        from src.indexing.label_clusters_mllm import main
        from src.utils.io import load_dataframe, save_dataframe

        input_parquet = os.path.join(self.test_dir, "test_input_single.parquet")

        df = pd.DataFrame(
            {
                "Photo_ID": [str(i) for i in range(4)],
                "Platform": ["flickr"] * 4,
                "Latitude": [40.0, 40.1, 50.0, 50.1],
                "Longitude": [-74.0, -74.1, 2.0, 2.1],
                "Image_URL": [f"http://example.com/{i}.jpg" for i in range(4)],
                "cluster_id": [0, 0, 1, 1],
                "parent_cluster_id": [0, 0, 0, 0],
            }
        )
        embs = np.zeros((4, 4), dtype=np.float32)
        embs[:2, 0] = 1.0
        embs[2:, 1] = 1.0
        df["embedding"] = list(embs)
        save_dataframe(df, input_parquet, representation_type="cls")

        mock_img = Image.new("RGB", (100, 100), color="green")
        mock_url_resp = MagicMock()
        mock_url_resp.status = 200
        mock_url_resp.__enter__.return_value = mock_url_resp

        with (
            patch("urllib.request.urlopen", return_value=mock_url_resp),
            patch("src.indexing.label_clusters_mllm.load_image", return_value=mock_img),
            patch(
                "src.indexing.label_clusters_mllm.query_vlm_openai_api",
                side_effect=[
                    "Visual desc parent",
                    "LABEL: ParentForest\nDESCRIPTION: Parent forest description.",
                    "Visual desc child0",
                    "LABEL: Forest\nDESCRIPTION: Child forest description.",
                    "Visual desc child1",
                    "LABEL: Agriculture\nDESCRIPTION: Child agriculture description.",
                ],
            ),
        ):
            test_args = [
                "label_clusters_mllm.py",
                "--in",
                input_parquet,
                "--num_medoids",
                "1",
                "--chunk_size",
                "2",
            ]
            with patch.object(sys, "argv", test_args):
                main()

        df_in_place = load_dataframe(input_parquet)
        self.assertEqual(len(df_in_place), 4)
        self.assertEqual(df_in_place["cluster_label"].iloc[0], "Forest")
        self.assertEqual(df_in_place["cluster_label"].iloc[2], "Agriculture")

    def test_end_to_end_pickle_saving(self):
        import pickle
        import sys
        from unittest.mock import MagicMock, patch

        from src.indexing.label_clusters_mllm import main

        pkl_input = os.path.join(self.test_dir, "test_input.pkl")
        pkl_output = os.path.join(self.test_dir, "test_output.pkl")

        df = pd.DataFrame(
            {
                "Photo_ID": [str(i) for i in range(4)],
                "Platform": ["flickr"] * 4,
                "Latitude": [40.0, 40.1, 50.0, 50.1],
                "Longitude": [-74.0, -74.1, 2.0, 2.1],
                "Image_URL": [f"http://example.com/{i}.jpg" for i in range(4)],
                "cluster_id": [0, 0, 1, 1],
                "parent_cluster_id": [0, 0, 0, 0],
            }
        )
        embs = np.zeros((4, 4), dtype=np.float32)
        embs[:2, 0] = 1.0
        embs[2:, 1] = 1.0
        df["embedding"] = list(embs)

        with open(pkl_input, "wb") as f:
            pickle.dump(df.to_dict("records"), f)
        np.save(
            os.path.join(self.test_dir, "test_input_cls_embeddings.npy"),
            embs,
        )

        mock_img = Image.new("RGB", (100, 100), color="green")
        mock_url_resp = MagicMock()
        mock_url_resp.status = 200
        mock_url_resp.__enter__.return_value = mock_url_resp

        with (
            patch("urllib.request.urlopen", return_value=mock_url_resp),
            patch("src.indexing.label_clusters_mllm.load_image", return_value=mock_img),
            patch(
                "src.indexing.label_clusters_mllm.query_vlm_openai_api",
                side_effect=[
                    "Visual desc parent",
                    "LABEL: ParentForest\nDESCRIPTION: Parent forest description.",
                    "Visual desc child0",
                    "LABEL: Forest\nDESCRIPTION: Child forest description.",
                    "Visual desc child1",
                    "LABEL: Agriculture\nDESCRIPTION: Child agriculture description.",
                ],
            ),
        ):
            test_args = [
                "label_clusters_mllm.py",
                "--in",
                pkl_input,
                "--out",
                pkl_output,
                "--num_medoids",
                "2",
                "--chunk_size",
                "2",
            ]
            with patch.object(sys, "argv", test_args):
                main()

        with open(pkl_output, "rb") as f:
            saved_data = pickle.load(f)
        self.assertEqual(len(saved_data), 4)
        self.assertEqual(saved_data[0]["cluster_label"], "Forest")
        self.assertEqual(saved_data[2]["cluster_label"], "Agriculture")

    def test_end_to_end_zeroshot_labeling_and_parquet_saving(self):
        import sys
        from unittest.mock import MagicMock, patch

        from src.indexing.label_clusters_mllm import main
        from src.utils.io import load_dataframe, save_dataframe

        input_parquet = os.path.join(self.test_dir, "test_input_zeroshot.parquet")
        output_parquet = os.path.join(self.test_dir, "test_output_zeroshot.parquet")

        df = pd.DataFrame(
            {
                "Photo_ID": [str(i) for i in range(4)],
                "Platform": ["flickr"] * 4,
                "Latitude": [40.0, 40.1, 50.0, 50.1],
                "Longitude": [-74.0, -74.1, 2.0, 2.1],
                "Image_URL": [f"http://example.com/{i}.jpg" for i in range(4)],
                "cluster_id": [0, 0, 1, 1],
                "parent_cluster_id": [0, 0, 0, 0],
            }
        )
        embs = np.zeros((4, 4), dtype=np.float32)
        embs[:2, 0] = 1.0
        embs[2:, 1] = 1.0
        df["embedding"] = list(embs)
        save_dataframe(df, input_parquet, representation_type="cls")

        # Mock model for zero-shot text encoder
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (40, 1)
        )
        mock_model.encode_text.return_value = mock_tensor

        with patch(
            "src.indexing.label_clusters_mllm.AutoModel.from_pretrained",
            return_value=mock_model,
        ):
            test_args = [
                "label_clusters_mllm.py",
                "--in",
                input_parquet,
                "--out",
                output_parquet,
                "--label_method",
                "zeroshot",
            ]
            with patch.object(sys, "argv", test_args):
                main()

        df_out = load_dataframe(output_parquet)
        self.assertEqual(len(df_out), 4)
        self.assertIn("cluster_label", df_out.columns)
        self.assertIn("parent_cluster_label", df_out.columns)
        self.assertTrue(len(df_out["cluster_label"].iloc[0]) > 0)


if __name__ == "__main__":
    unittest.main()
