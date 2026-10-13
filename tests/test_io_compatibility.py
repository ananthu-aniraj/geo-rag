"""Offline characterization of the legacy ``src.utils.io`` surface."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from src.utils import credentials, io as io_module


class IOCompatibilityCharacterizationTest(unittest.TestCase):
    def test_legacy_names_remain_importable(self):
        functions = (
            "get_core_base_name",
            "load_dataframe",
            "save_dataframe",
            "get_parquet_writer",
            "load_dataset_with_clusters",
            "load_embeddings",
            "is_valid_image_file",
            "build_offline_image_index",
            "resolve_offline_image_path",
            "_get_http_session",
            "get_wildlife_insights_token",
            "resolve_wildlife_insights_url",
            "download_image",
        )
        for name in functions:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(io_module, name)))
        self.assertIn(
            "6da31bd91c242aa98de0ddae4ad89e3c", io_module.KNOWN_PLACEHOLDER_MD5_HASHES
        )

    def test_legacy_cluster_loader_uses_patchable_io_dataframe_loader(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "clusters.parquet")
            pq.write_table(
                pa.table(
                    {
                        "Latitude": [1.0],
                        "Longitude": [2.0],
                        "cluster_id": [3],
                    }
                ),
                path,
            )
            expected = object()
            with patch.object(
                io_module, "load_dataframe", return_value=expected
            ) as load:
                result = io_module.load_dataset_with_clusters(path)

        self.assertIs(result, expected)
        load.assert_called_once_with(path, columns=None)

    def test_legacy_download_image_uses_patchable_offline_resolver(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "resolved.jpg")
            Image.new("RGB", (11, 7), color=(5, 10, 15)).save(image_path)
            with (
                patch.object(
                    io_module, "resolve_offline_image_path", return_value=image_path
                ) as resolve,
                patch.object(io_module, "_get_http_session") as get_session,
            ):
                image = io_module.download_image(
                    "https://example.invalid/original.jpg",
                    mapillary_token="unused",
                    photo_id="42",
                    platform="flickr",
                    offline_dirs=[root],
                    image_size=5,
                    max_retries=1,
                )

        self.assertEqual(image.size, (5, 5))
        resolve.assert_called_once_with(
            "https://example.invalid/original.jpg", [root], "42", "flickr"
        )
        get_session.assert_not_called()

    def test_legacy_http_session_is_lazy_and_keeps_retry_configuration(self):
        session = Mock()
        with (
            patch.object(io_module, "_http_session", None),
            patch.object(
                io_module.requests, "Session", return_value=session
            ) as factory,
            patch.object(io_module, "HTTPAdapter") as adapter_class,
            patch.object(io_module, "Retry") as retry_class,
        ):
            first = io_module._get_http_session()
            second = io_module._get_http_session()

        self.assertIs(first, session)
        self.assertIs(second, session)
        factory.assert_called_once_with()
        retry_class.assert_called_once_with(
            total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
        )
        adapter_class.assert_called_once_with(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=retry_class.return_value,
        )
        session.mount.assert_has_calls(
            [
                call("https://", adapter_class.return_value),
                call("http://", adapter_class.return_value),
            ]
        )

    def test_offline_lookup_uses_env_roots_and_relative_url_path(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "images", "camera", "photo.jpg")
            os.makedirs(os.path.dirname(image_path))
            Path(image_path).write_bytes(b"local fixture")
            with patch.dict(
                os.environ,
                {"IMAGE_ROOT_DIR": root, "OFFLINE_DATASET_DIRS": ""},
            ):
                result = io_module.resolve_offline_image_path(
                    "https://example.invalid/images/camera/photo.jpg",
                    image_root_dirs=None,
                )

        self.assertEqual(result, os.path.abspath(image_path))

    def test_offline_lookup_parses_multiple_env_roots(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "first")
            second = os.path.join(root, "second")
            os.makedirs(first)
            os.makedirs(second)
            image_path = os.path.join(second, "9.jpg")
            Path(image_path).write_bytes(b"local fixture")
            with patch.dict(
                os.environ,
                {"IMAGE_ROOT_DIR": "", "OFFLINE_DATASET_DIRS": f"{first} {second}"},
            ):
                result = io_module.resolve_offline_image_path(
                    "https://example.invalid/remote.jpg",
                    image_root_dirs=None,
                    photo_id=9,
                )

        self.assertEqual(result, os.path.abspath(image_path))

    def test_credentials_refresh_uses_cookie_and_search_dir_through_io_alias(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, ".env").write_text(
                "WILDLIFE_INSIGHTS_COOKIE='connect.sid=fixture-cookie'\n"
                "WILDLIFE_INSIGHTS_TOKEN='stale-token'\n",
                encoding="utf-8",
            )
            html = (
                '<script id="__NEXT_DATA__" type="application/json">'
                '{"props":{"initialState":{"auth":{"token":"fresh-token"}}}}'
                "</script>"
            )
            response = Mock(text=html)
            with (
                patch.dict(
                    os.environ,
                    {"WILDLIFE_INSIGHTS_COOKIE": "", "WILDLIFE_INSIGHTS_TOKEN": ""},
                ),
                patch.object(credentials, "_wi_token", "cached-token"),
                patch(
                    "src.utils.credentials.requests.get", return_value=response
                ) as get,
            ):
                result = io_module.get_wildlife_insights_token(
                    refresh=True, search_dir=root
                )

        self.assertEqual(result, "fresh-token")
        get.assert_called_once_with(
            "https://app.wildlifeinsights.org/explore",
            headers={
                "Cookie": "connect.sid=fixture-cookie",
                "User-Agent": "Mozilla/5.0",
            },
            timeout=10,
        )

    def test_wildlife_url_refreshes_through_patchable_io_token(self):
        token = Mock(side_effect=["stale-token", "fresh-token"])
        unauthorized = Mock(status_code=401)
        authorized = Mock(status_code=200)
        authorized.json.return_value = {
            "data": {
                "getDataFilePublicDownloadUrl": {
                    "url": "https://storage.invalid/photo.jpg"
                }
            }
        }
        url = "https://app.wildlifeinsights.org/download/42/project/7/data-files/photo-uuid"

        with (
            patch.object(io_module, "get_wildlife_insights_token", token),
            patch.object(
                io_module.requests, "post", side_effect=[unauthorized, authorized]
            ) as post,
        ):
            result = io_module.resolve_wildlife_insights_url(url)

        self.assertEqual(result, "https://storage.invalid/photo.jpg")
        self.assertEqual(post.call_count, 2)
        token.assert_has_calls([call(None), call(None, refresh=True)])
        self.assertEqual(
            post.call_args_list[1].kwargs["headers"]["Authorization"],
            "Bearer fresh-token",
        )


class EmptyAndMalformedEmbeddingCharacterizationTest(unittest.TestCase):
    def test_empty_inline_layout_raises_index_error(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "empty.parquet")
            pq.write_table(
                pa.table({"embedding": pa.array([], type=pa.list_(pa.float32()))}),
                path,
            )
            with self.assertRaises(IndexError):
                io_module.load_embeddings(path)

    def test_empty_companion_layout_preserves_zero_row_shape_and_dtype(self):
        with tempfile.TemporaryDirectory() as root:
            np.save(
                os.path.join(root, "empty_cls_embeddings.npy"),
                np.empty((0, 4), dtype=np.float32),
            )
            path = os.path.join(root, "empty.parquet")
            pq.write_table(pa.table({"city": pa.array([], type=pa.string())}), path)

            result = io_module.load_embeddings(path)

        self.assertEqual(result.shape, (0, 4))
        self.assertEqual(result.dtype, np.float32)

    def test_ragged_inline_layout_raises_value_error(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "ragged.parquet")
            pq.write_table(
                pa.table(
                    {
                        "embedding": pa.array(
                            [[1.0, 2.0], [3.0]], type=pa.list_(pa.float32())
                        )
                    }
                ),
                path,
            )
            with self.assertRaises(ValueError):
                io_module.load_embeddings(path)

    def test_corrupt_companion_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "broken_cls_embeddings.npy").write_bytes(b"not a numpy file")
            path = os.path.join(root, "broken.parquet")
            pq.write_table(pa.table({"city": ["fixture"]}), path)

            with self.assertRaises(ValueError):
                io_module.load_embeddings(path)


if __name__ == "__main__":
    unittest.main()
