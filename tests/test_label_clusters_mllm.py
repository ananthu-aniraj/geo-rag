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


if __name__ == "__main__":
    unittest.main()
