import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.utils.credentials import (
    clean_credential_value,
    find_env_file,
    get_carto_api_key,
    get_credential,
    get_flickr_api_key,
    get_hf_token,
    get_mapillary_token,
    get_wildlife_insights_credentials,
    get_wildobs_api_key,
)


class TestCredentials(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.env_keys = [
            "MAPILLARY_TOKEN",
            "CARTO_API_KEY",
            "FLICKR_API_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "WILDOBS_API_KEY",
            "WILDOBSR_API_KEY",
            "WILDLIFE_INSIGHTS_COOKIE",
            "WILDLIFE_INSIGHTS_TOKEN",
            "TEST_CUSTOM_KEY",
        ]
        self.orig_env = {k: os.environ.get(k) for k in self.env_keys}
        for k in self.env_keys:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.orig_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_clean_credential_value(self):
        self.assertEqual(clean_credential_value(None), "")
        self.assertEqual(clean_credential_value(""), "")
        self.assertEqual(clean_credential_value("  abc  "), "abc")
        self.assertEqual(clean_credential_value('"quoted"'), "quoted")
        self.assertEqual(clean_credential_value("'single_quoted'"), "single_quoted")
        self.assertEqual(clean_credential_value('  "nested_space"  '), "nested_space")

    def test_get_credential_from_environ(self):
        os.environ["TEST_CUSTOM_KEY"] = "custom_secret_123"
        self.assertEqual(
            get_credential("TEST_CUSTOM_KEY", search_dir=self.test_dir),
            "custom_secret_123",
        )

    def test_get_credential_strips_quotes(self):
        os.environ["TEST_CUSTOM_KEY"] = '"double_quote_val"'
        self.assertEqual(
            get_credential("TEST_CUSTOM_KEY", search_dir=self.test_dir),
            "double_quote_val",
        )

    def test_get_credential_from_env_file(self):
        env_path = os.path.join(self.test_dir, ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("# Sample comment\nTEST_CUSTOM_KEY=loaded_from_file\n")

        val = get_credential("TEST_CUSTOM_KEY", search_dir=self.test_dir)
        self.assertEqual(val, "loaded_from_file")

    def test_get_mapillary_token(self):
        os.environ["MAPILLARY_TOKEN"] = "mapillary_tok_xyz"
        self.assertEqual(
            get_mapillary_token(search_dir=self.test_dir), "mapillary_tok_xyz"
        )

        os.environ.pop("MAPILLARY_TOKEN", None)
        env_path = os.path.join(self.test_dir, ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("MAPILLARY_TOKEN='mly_from_env_file'\n")
        self.assertEqual(
            get_mapillary_token(search_dir=self.test_dir), "mly_from_env_file"
        )

    def test_get_carto_api_key(self):
        os.environ["CARTO_API_KEY"] = "carto_key_999"
        self.assertEqual(get_carto_api_key(search_dir=self.test_dir), "carto_key_999")

    def test_get_flickr_api_key(self):
        os.environ["FLICKR_API_KEY"] = "flickr_abc_123"
        self.assertEqual(get_flickr_api_key(search_dir=self.test_dir), "flickr_abc_123")

    def test_get_hf_token_fallbacks(self):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = "hub_tok_fallback"
        self.assertEqual(get_hf_token(search_dir=self.test_dir), "hub_tok_fallback")

        os.environ["HF_TOKEN"] = "primary_hf_token"
        self.assertEqual(get_hf_token(search_dir=self.test_dir), "primary_hf_token")

    def test_get_wildobs_api_key_fallbacks(self):
        os.environ["WILDOBSR_API_KEY"] = "wildobsr_fallback"
        self.assertEqual(
            get_wildobs_api_key(search_dir=self.test_dir), "wildobsr_fallback"
        )

        os.environ["WILDOBS_API_KEY"] = "wildobs_primary"
        self.assertEqual(
            get_wildobs_api_key(search_dir=self.test_dir), "wildobs_primary"
        )

    def test_get_wildlife_insights_credentials(self):
        os.environ["WILDLIFE_INSIGHTS_COOKIE"] = "connect.sid=cookie123"
        os.environ["WILDLIFE_INSIGHTS_TOKEN"] = "jwt_token_456"
        cookie, token = get_wildlife_insights_credentials(search_dir=self.test_dir)
        self.assertEqual(cookie, "connect.sid=cookie123")
        self.assertEqual(token, "jwt_token_456")

    def test_find_env_file_walkup(self):
        # Create nested directory tree: test_dir / sub1 / sub2
        sub_dir = Path(self.test_dir) / "sub1" / "sub2"
        sub_dir.mkdir(parents=True)
        env_path = Path(self.test_dir) / ".env"
        env_path.write_text("TEST_CUSTOM_KEY=nested_found\n", encoding="utf-8")

        found = find_env_file(search_dir=sub_dir)
        self.assertIsNotNone(found)
        self.assertEqual(found.resolve(), env_path.resolve())

        val = get_credential("TEST_CUSTOM_KEY", search_dir=sub_dir)
        self.assertEqual(val, "nested_found")


if __name__ == "__main__":
    unittest.main()
