import os
import unittest

import numpy as np
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


if __name__ == "__main__":
    unittest.main()
