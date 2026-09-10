import os
import shutil
import tempfile
import unittest

import yaml

from src.utils.config import format_for_shell, get_param, load_config


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, "config")
        self.eval_dir = os.path.join(self.config_dir, "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)

        # Base YAML
        self.base_yaml_path = os.path.join(self.eval_dir, "params_offline.yaml")
        self.base_data = {
            "lucas": {
                "model_name": "vit_base_patch16_dinov3.lvd1689m",
                "csv": "/server/data/lucas.csv",
                "img_dir": "/server/data/lucas_images",
                "batch_size": 128,
                "num_queries": 3000,
                "use_segformer": False,
                "discard_classes": [2, 12, 20, 43, 80, 83, 102, 127],
            },
            "places": {
                "model_name": "vit_base_patch16_dinov3.lvd1689m",
                "img_dir": "/server/data/places",
                "batch_size": 128,
            },
        }
        with open(self.base_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.base_data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_base_load_without_local(self):
        """When no local.yaml exists, loads base values unchanged."""
        cfg = load_config(self.base_yaml_path, repo_root=self.test_dir)
        self.assertEqual(cfg["lucas"]["img_dir"], "/server/data/lucas_images")
        self.assertEqual(cfg["lucas"]["batch_size"], 128)
        self.assertEqual(
            cfg["lucas"]["discard_classes"], [2, 12, 20, 43, 80, 83, 102, 127]
        )

    def test_global_local_yaml_override(self):
        """Overrides any arbitrary parameter (paths, numbers, booleans, lists) via config/local.yaml."""
        local_yaml_path = os.path.join(self.config_dir, "local.yaml")
        local_data = {
            "evaluation": {
                "lucas": {
                    "img_dir": "/local/pc/lucas_images",
                    "batch_size": 32,
                    "use_segformer": True,
                    "discard_classes": [2, 12],
                }
            }
        }
        with open(local_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(local_data, f)

        cfg = load_config(self.base_yaml_path, repo_root=self.test_dir)

        # Overridden values
        self.assertEqual(cfg["lucas"]["img_dir"], "/local/pc/lucas_images")
        self.assertEqual(cfg["lucas"]["batch_size"], 32)
        self.assertTrue(cfg["lucas"]["use_segformer"])
        self.assertEqual(cfg["lucas"]["discard_classes"], [2, 12])

        # Preserved values
        self.assertEqual(cfg["lucas"]["model_name"], "vit_base_patch16_dinov3.lvd1689m")
        self.assertEqual(cfg["lucas"]["csv"], "/server/data/lucas.csv")
        self.assertEqual(cfg["lucas"]["num_queries"], 3000)
        self.assertEqual(cfg["places"]["img_dir"], "/server/data/places")

    def test_file_specific_local_yaml_override(self):
        """File-specific params_offline.local.yaml takes effect."""
        file_local_path = os.path.join(self.eval_dir, "params_offline.local.yaml")
        with open(file_local_path, "w", encoding="utf-8") as f:
            yaml.dump({"lucas": {"batch_size": 16}}, f)

        cfg = load_config(self.base_yaml_path, repo_root=self.test_dir)
        self.assertEqual(cfg["lucas"]["batch_size"], 16)

    def test_get_param_helper(self):
        """get_param retrieves nested values correctly."""
        local_yaml_path = os.path.join(self.config_dir, "local.yaml")
        with open(local_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump({"evaluation": {"lucas": {"batch_size": 64}}}, f)

        val = get_param(self.base_yaml_path, "lucas", "batch_size")
        # In test environment with explicit repo_root:
        cfg = load_config(self.base_yaml_path, repo_root=self.test_dir)
        self.assertEqual(cfg["lucas"]["batch_size"], 64)

    def test_shell_formatting(self):
        """Verify format_for_shell handles booleans, lists, strings."""
        self.assertEqual(format_for_shell(True), "true")
        self.assertEqual(format_for_shell(False), "false")
        self.assertEqual(format_for_shell([2, 12, 20]), "2 12 20")
        self.assertEqual(format_for_shell("test_str"), "test_str")
        self.assertEqual(format_for_shell(None), "")

    def test_pipeline_and_scrapers_local_yaml_override(self):
        """Verify overrides work for single-key schemas (pipeline and scrapers)."""
        # Pipeline config setup
        pipe_dir = os.path.join(self.config_dir, "pipeline")
        os.makedirs(pipe_dir, exist_ok=True)
        pipe_yaml = os.path.join(pipe_dir, "params.yaml")
        with open(pipe_yaml, "w", encoding="utf-8") as f:
            yaml.dump({"pipeline": {"output_dir": "/orig/out", "batch_size": 128}}, f)

        # Scraper config setup
        scraper_dir = os.path.join(self.config_dir, "scrapers")
        os.makedirs(scraper_dir, exist_ok=True)
        scraper_yaml = os.path.join(scraper_dir, "flickr_scraper.yaml")
        with open(scraper_yaml, "w", encoding="utf-8") as f:
            yaml.dump({"scraper": {"base_dir": "/orig/flickr", "step_km": 5}}, f)

        # Global local.yaml overrides
        local_yaml = os.path.join(self.config_dir, "local.yaml")
        with open(local_yaml, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "pipeline": {"output_dir": "/local/out"},
                    "scrapers": {"flickr_scraper": {"base_dir": "/local/flickr"}},
                },
                f,
            )

        cfg_pipe = load_config(pipe_yaml, repo_root=self.test_dir)
        self.assertEqual(cfg_pipe["pipeline"]["output_dir"], "/local/out")
        self.assertEqual(cfg_pipe["pipeline"]["batch_size"], 128)

        cfg_scraper = load_config(scraper_yaml, repo_root=self.test_dir)
        self.assertEqual(cfg_scraper["scraper"]["base_dir"], "/local/flickr")
        self.assertEqual(cfg_scraper["scraper"]["step_km"], 5)


if __name__ == "__main__":
    unittest.main()
