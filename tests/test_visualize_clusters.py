import os
import pickle
import tempfile
import unittest

from src.visualization.visualize_clusters import create_map


class TestVisualizeClusters(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_pkl(self, items):
        pkl_path = os.path.join(self.temp_dir, "test_clusters.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(items, f)
        return pkl_path

    def test_create_map_with_offline_images(self):
        sample_data = [
            {
                "Latitude": 48.8566,
                "Longitude": 2.3522,
                "H3_Cell": "881fb46625fffff",
                "cluster_id": 1,
                "cluster_label": "Urban Architecture",
                "cluster_description": "Dense `urban` scene with buildings",
                "parent_cluster_id": 10,
                "parent_cluster_label": "Built-up",
                "Platform": "Flickr",
                "Captured_At": "2023-05-10T12:00:00",
                "Image_URL": "file:///data/datasets/wildlife/img_001.jpg",
                "Photo_ID": "1001",
                "Koppen_Code": "Cfb",
                "Koppen_Desc": "Marine west coast",
                "Season": "Spring",
            }
        ]
        pkl_path = self._create_sample_pkl(sample_data)
        out_html = os.path.join(self.temp_dir, "map.html")

        create_map(pkl_path, out_html)

        self.assertTrue(os.path.exists(out_html))
        with open(out_html, "r", encoding="utf-8") as f:
            html = f.read()

        # Check offline mapping control presence
        self.assertIn("findPrefixInput", html)
        self.assertIn("replacePrefixInput", html)
        self.assertIn("Offline Image Server Mapping", html)

        # Check auto-detection of base directory
        self.assertIn("file:///data/datasets/wildlife/", html)

        # Check popup attributes
        self.assertIn(
            'data-orig-src="file:///data/datasets/wildlife/img_001.jpg"', html
        )
        self.assertIn(
            'data-orig-href="file:///data/datasets/wildlife/img_001.jpg"', html
        )

        # Check backtick escaping
        self.assertIn("&#96;urban&#96;", html)

        # Check contrasting dark basemap default and layer switcher
        self.assertIn("dark_all", html)
        self.assertIn("CartoDB Positron (Light)", html)
        self.assertIn("CartoDB Voyager (Color)", html)
        self.assertIn("OpenStreetMap", html)

        # Check marker tooltip on hover
        self.assertIn("Cluster 1", html)

        # Check high contrast CSS styling
        self.assertIn("#0f172a", html)
        self.assertIn("#f8fafc", html)
        self.assertIn("#38bdf8", html)

    def test_create_map_with_explicit_prefixes(self):
        sample_data = [
            {
                "Latitude": 10.0,
                "Longitude": 20.0,
                "cluster_id": 2,
                "Platform": "Mapillary",
                "Image_URL": "mapillary://999",
                "Photo_ID": "999",
            }
        ]
        pkl_path = self._create_sample_pkl(sample_data)
        out_html = os.path.join(self.temp_dir, "map_custom.html")

        create_map(
            pkl_path,
            out_html,
            find_prefix="custom_find_path/",
            replace_prefix="http://localhost:8080/tunnel/",
        )

        with open(out_html, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn("custom_find_path/", html)
        self.assertIn("http://localhost:8080/tunnel/", html)

    def test_create_map_with_image_root_dir_fallback(self):
        sample_data = [
            {
                "Latitude": -33.8688,
                "Longitude": 151.2093,
                "cluster_id": 3,
                "Platform": "Flickr",
                "Image_URL": "relative/path/photo.jpg",
                "Photo_ID": "2002",
            }
        ]
        pkl_path = self._create_sample_pkl(sample_data)
        out_html = os.path.join(self.temp_dir, "map_root.html")

        mock_root = "/mnt/storage/images"
        create_map(pkl_path, out_html, image_root_dir=[mock_root])

        with open(out_html, "r", encoding="utf-8") as f:
            html = f.read()

        # Should fall back to image_root_dir for auto-detection
        self.assertIn("file:///mnt/storage/images/", html)

    def test_url_replacement_js_logic(self):
        # Verify getDisplayUrl logic from image_error_handler.js directly
        js_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "visualization",
            "templates",
            "image_error_handler.js",
        )
        self.assertTrue(os.path.exists(js_path))
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()

        self.assertIn("function getDisplayUrl(url)", js)
        self.assertIn("function applyPrefixReplacement(popupElement)", js)
        self.assertIn("function updateAllPopupUrls()", js)
        self.assertIn("function initOfflineMapping()", js)


if __name__ == "__main__":
    unittest.main()
