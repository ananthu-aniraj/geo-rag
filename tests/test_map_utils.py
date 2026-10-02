import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import folium

from src.visualization.map_utils import (
    CARTO_ATTRIBUTION,
    create_carto_tile_layer,
    create_folium_map,
    get_carto_api_key,
    get_carto_key_param,
    get_carto_tile_config,
    is_carto_tiles,
)


class TestMapUtils(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.get("CARTO_API_KEY")
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["CARTO_API_KEY"] = self.orig_env
        else:
            os.environ.pop("CARTO_API_KEY", None)

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_carto_api_key_from_env(self):
        os.environ["CARTO_API_KEY"] = "test_key_12345"
        self.assertEqual(get_carto_api_key(), "test_key_12345")

    def test_get_carto_api_key_strips_quotes(self):
        os.environ["CARTO_API_KEY"] = '"test_key_quoted"'
        self.assertEqual(get_carto_api_key(), "test_key_quoted")

        os.environ["CARTO_API_KEY"] = "'test_key_single_quoted'"
        self.assertEqual(get_carto_api_key(), "test_key_single_quoted")

    def test_get_carto_api_key_from_env_file(self):
        os.environ.pop("CARTO_API_KEY", None)
        env_file = os.path.join(self.test_dir, ".env")
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("# Comment\nCARTO_API_KEY=from_env_file_987\n")

        key = get_carto_api_key(search_dir=self.test_dir)
        self.assertEqual(key, "from_env_file_987")

    def test_get_carto_key_param(self):
        self.assertEqual(get_carto_key_param("my_key"), "?key=my_key")
        self.assertEqual(get_carto_key_param(""), "")
        self.assertEqual(
            get_carto_key_param(None),
            f"?key={get_carto_api_key()}" if get_carto_api_key() else "",
        )

    def test_get_carto_tile_config_variants(self):
        # Positron
        url, attr = get_carto_tile_config("CartoDB Positron", api_key="test_key")
        self.assertIn("light_all", url)
        self.assertIn("?key=test_key", url)
        self.assertEqual(attr, CARTO_ATTRIBUTION)

        # Dark Matter
        url, _ = get_carto_tile_config("CartoDB Dark_Matter", api_key="test_key")
        self.assertIn("dark_all", url)
        self.assertIn("?key=test_key", url)

        # Voyager
        url, _ = get_carto_tile_config("CartoDB Voyager", api_key="test_key")
        self.assertIn("rastertiles/voyager", url)
        self.assertIn("?key=test_key", url)

        # Without API key
        url, _ = get_carto_tile_config("CartoDB Positron", api_key="")
        self.assertIn("light_all", url)
        self.assertNotIn("?key=", url)

    def test_is_carto_tiles(self):
        self.assertTrue(is_carto_tiles("CartoDB Positron"))
        self.assertTrue(is_carto_tiles("cartodbpositron"))
        self.assertTrue(is_carto_tiles("CartoDB Dark_Matter"))
        self.assertTrue(is_carto_tiles("voyager"))
        self.assertTrue(
            is_carto_tiles(
                "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            )
        )
        self.assertFalse(is_carto_tiles("OpenStreetMap"))
        self.assertFalse(is_carto_tiles(None))

    def test_create_folium_map_with_key(self):
        m = create_folium_map(
            location=[20, 0],
            zoom_start=3,
            tiles="CartoDB Positron",
            carto_api_key="custom_test_key_xyz",
        )
        self.assertIsInstance(m, folium.Map)
        html = m._repr_html_()
        self.assertIn("custom_test_key_xyz", html)
        self.assertIn("cartocdn.com/light_all", html)

    def test_create_folium_map_without_key(self):
        os.environ.pop("CARTO_API_KEY", None)
        with patch("src.visualization.map_utils.get_carto_api_key", return_value=""):
            m = create_folium_map(
                location=[20, 0], zoom_start=2, tiles="CartoDB Positron"
            )
            self.assertIsInstance(m, folium.Map)
            html = m._repr_html_()
            self.assertIn("cartocdn.com/light_all", html)
            self.assertNotIn("?key=", html)

    def test_create_folium_map_non_carto(self):
        m = create_folium_map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")
        self.assertIsInstance(m, folium.Map)
        html = m._repr_html_()
        self.assertIn("openstreetmap.org", html)

    def test_create_carto_tile_layer(self):
        layer = create_carto_tile_layer(variant="light_all", api_key="layer_key_123")
        self.assertIsInstance(layer, folium.TileLayer)
        m = folium.Map(tiles=None)
        layer.add_to(m)
        html = m._repr_html_()
        self.assertIn("layer_key_123", html)

    def test_cluster_dashboard_template_integration(self):
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "visualization",
            "templates",
            "cluster_dashboard.html",
        )
        self.assertTrue(os.path.exists(template_path))
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        self.assertIn("{{CARTO_KEY_PARAM}}", template)
        rendered = template.replace("{{CARTO_KEY_PARAM}}", "?key=test_rendered_key")
        self.assertIn("cartocdn.com/light_all/{z}/{x}/{y}{r}.png${cartoKey}", rendered)
        self.assertIn("const CARTO_KEY_PARAM = '?key=test_rendered_key';", rendered)

    def test_geopandas_explore_carto_tiles(self):
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            self.skipTest("geopandas not installed")

        import importlib.util

        if not importlib.util.find_spec("mapclassify"):
            self.skipTest("mapclassify not installed")

        gdf = gpd.GeoDataFrame([{"geometry": Point(0, 0)}], crs="EPSG:4326")
        url, attr = get_carto_tile_config("CartoDB Positron", api_key="shp_key_xyz")
        try:
            m = gdf.explore(tiles=url, attr=attr)
        except ImportError as e:
            self.skipTest(f"explore() dependencies missing: {e}")

        html = m._repr_html_()
        self.assertIn("shp_key_xyz", html)
        self.assertIn("cartocdn.com/light_all", html)


if __name__ == "__main__":
    unittest.main()
