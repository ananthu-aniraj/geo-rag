"""Characterization tests for load_embeddings and save_dataframe (src/utils/io.py).

These pin the CURRENT behaviour of the embedding loader and of the companion-file
writer so that a refactor of src/utils/io.py can be shown to be behaviour-preserving.
Where the current behaviour looks wrong, the test still asserts it and says so in a
CHARACTERIZATION comment: it is not a statement of desired behaviour.
"""

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.io import load_embeddings, save_dataframe


def _list_column(matrix, value_type=pa.float32()):
    """Builds an Arrow list column holding one embedding per row of ``matrix``."""
    return pa.array(matrix.tolist(), type=pa.list_(value_type))


def _run_quiet(fn, *args, **kwargs):
    """Runs ``fn`` with stdout captured and returns ``(result, stdout_text)``."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


class _CharacterizationCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Data lives one level below the temp root, so the loader's parent-directory
        # scan only ever sees this folder and never unrelated files.
        self.root = os.path.join(tmp.name, "emb")
        os.makedirs(self.root)
        self.rng = np.random.default_rng(0)

    def path(self, name):
        return os.path.join(self.root, name)

    def write_parquet(self, name, columns):
        pq.write_table(pa.table(columns), self.path(name))
        return self.path(name)

    def write_npy(self, name, arr):
        np.save(self.path(name), arr)

    def write_keys(self, name, keys):
        pq.write_table(pa.table({"photo_key": keys}), self.path(name))


class LoadEmbeddingsCharacterizationTest(_CharacterizationCase):
    def test_a_inline_embedding_column_returns_float32_matrix(self):
        emb = self.rng.random((3, 4), dtype=np.float32)
        path = self.write_parquet("inline.parquet", {"embedding": _list_column(emb)})

        result = load_embeddings(path)

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_equal(result, emb)

        # float64 inline data is cast to float32 on the way out.
        emb64 = self.rng.random((2, 4))
        path64 = self.write_parquet(
            "inline64.parquet", {"embedding": _list_column(emb64, pa.float64())}
        )
        result64 = load_embeddings(path64)
        self.assertEqual(result64.dtype, np.float32)
        np.testing.assert_array_equal(result64, emb64.astype(np.float32))

    def test_b_companion_keys_reorder_rows_to_parquet_order(self):
        # npy rows follow the keys-file order k0, k1, k2; parquet order is K2, K0, K1.
        emb = self.rng.random((3, 4), dtype=np.float32)
        self.write_npy("geo_cls_embeddings.npy", emb)
        self.write_keys("geo_cls_embeddings.keys.parquet", ["k0", "k1", "k2"])
        path = self.write_parquet("geo.parquet", {"photo_key": ["K2", "K0", "K1"]})

        result, out = _run_quiet(load_embeddings, path)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, emb[[2, 0, 1]])
        # CHARACTERIZATION: this progress line is printed to stdout on every keyed load,
        # not only when something goes wrong.
        self.assertIn("Resolving embeddings via companion keys index", out)

    def test_c_platform_and_photo_id_build_keys(self):
        emb = self.rng.random((3, 4), dtype=np.float32)
        self.write_npy("geo_cls_embeddings.npy", emb)
        self.write_keys(
            "geo_cls_embeddings.keys.parquet",
            ["flickr_456", "mapillary_9", "flickr_123"],
        )
        path = self.write_parquet(
            "geo.parquet",
            {"Platform": ["Flickr", "Mapillary", "Flickr"], "Photo_ID": [123, 9, 456]},
        )

        result, out = _run_quiet(load_embeddings, path)

        np.testing.assert_array_equal(result, emb[[2, 1, 0]])
        self.assertNotIn("Warning", out)

    def test_c2_platform_path_does_not_lowercase_photo_id(self):
        # CHARACTERIZATION (suspected bug): the Platform/Photo_ID key is lowercased on
        # the Platform side only, while the keys file is lowercased as a whole. A
        # mixed-case Photo_ID therefore never matches and is silently zero-filled
        # (with a warning).
        emb = self.rng.random((2, 4), dtype=np.float32)
        self.write_npy("geo_cls_embeddings.npy", emb)
        self.write_keys("geo_cls_embeddings.keys.parquet", ["Flickr_AbC", "Flickr_xyz"])
        path = self.write_parquet(
            "geo.parquet",
            {"Platform": ["Flickr", "Flickr"], "Photo_ID": ["AbC", "xyz"]},
        )

        result, out = _run_quiet(load_embeddings, path)

        expected = np.zeros((2, 4), dtype=np.float32)
        expected[1] = emb[1]
        np.testing.assert_array_equal(result, expected)
        self.assertIn("Warning: Found 1 missing keys", out)
        self.assertIn(" -> Sample missing keys: ['flickr_AbC']", out)

    def test_d_duplicate_keys_first_occurrence_wins(self):
        emb = self.rng.random((4, 4), dtype=np.float32)
        self.write_npy("geo_cls_embeddings.npy", emb)
        self.write_keys("geo_cls_embeddings.keys.parquet", ["a", "b", "a", "c"])
        path = self.write_parquet("geo.parquet", {"photo_key": ["a", "b", "c"]})

        result, out = _run_quiet(load_embeddings, path)

        # Key "a" resolves to row 0, not the duplicate at row 2.
        np.testing.assert_array_equal(result, emb[[0, 1, 3]])
        self.assertNotIn("Warning", out)

    def test_e_missing_keys_are_zero_filled_with_warning(self):
        emb = self.rng.random((2, 4), dtype=np.float32)
        self.write_npy("geo_cls_embeddings.npy", emb)
        self.write_keys("geo_cls_embeddings.keys.parquet", ["k0", "k1"])
        path = self.write_parquet(
            "geo.parquet", {"photo_key": ["K1", "MISSING_X", "k0", "MISSING_Y"]}
        )

        result, out = _run_quiet(load_embeddings, path)

        expected = np.zeros((4, 4), dtype=np.float32)
        expected[0] = emb[1]
        expected[2] = emb[0]
        np.testing.assert_array_equal(result, expected)
        self.assertIn(
            "Warning: Found 2 missing keys in embeddings keys index. Zero-filling...",
            out,
        )
        self.assertIn(" -> Sample missing keys: ['missing_x', 'missing_y']", out)

    def test_f_embedding_idx_selects_rows(self):
        emb = self.rng.random((3, 4), dtype=np.float32)
        self.write_npy("geo_cls_embeddings.npy", emb)
        path = self.write_parquet("geo.parquet", {"embedding_idx": [2, 0, 1]})

        result = load_embeddings(path)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, emb[[2, 0, 1]])

    def test_g_out_of_bounds_embedding_idx_clamped_and_zero_filled(self):
        # CHARACTERIZATION: a negative index is treated as out of bounds and zero-filled,
        # not wrapped around Python-style.
        emb = self.rng.random((3, 4), dtype=np.float32)
        self.write_npy("geo_cls_embeddings.npy", emb)
        path = self.write_parquet("geo.parquet", {"embedding_idx": [1, 7, -1, 0]})

        result, out = _run_quiet(load_embeddings, path)

        expected = np.zeros((4, 4), dtype=np.float32)
        expected[0] = emb[1]
        expected[3] = emb[0]
        np.testing.assert_array_equal(result, expected)
        self.assertIn(
            "Warning: Found 2 out-of-bounds indices in embedding_idx. "
            "Clamping and zero-filling...",
            out,
        )

    def test_h_companion_without_keys_or_idx_returns_whole_matrix(self):
        emb = self.rng.random((3, 4))  # float64 on disk
        self.write_npy("geo_cls_embeddings.npy", emb)
        path = self.write_parquet("geo.parquet", {"city": ["a", "b", "c"]})

        result = load_embeddings(path)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, emb.astype(np.float32))

    def test_i_tipsv2_legacy_name_without_model_suffix(self):
        emb = self.rng.random((2, 4), dtype=np.float32)
        self.write_npy("geo_space_cls_embeddings.npy", emb)
        # Decoy with the model-specific name: the legacy name wins when model_name is None.
        self.write_npy(
            "geo_space_google_tipsv2-b14_cls_embeddings.npy",
            self.rng.random((2, 4), dtype=np.float32),
        )
        path = self.write_parquet("geo_space.parquet", {"city": ["a", "b"]})

        result = load_embeddings(path)

        np.testing.assert_array_equal(result, emb)

    def test_j_model_specific_name_when_model_name_given(self):
        emb = self.rng.random((2, 4), dtype=np.float32)
        self.write_npy("geo_space_org_other-model_cls_embeddings.npy", emb)
        self.write_npy(
            "geo_space_cls_embeddings.npy", self.rng.random((2, 4), dtype=np.float32)
        )
        path = self.write_parquet("geo_space.parquet", {"city": ["a", "b"]})

        result = load_embeddings(path, model_name="org/other-model")

        np.testing.assert_array_equal(result, emb)

    def test_k_cleaned_base_falls_back_to_deduplicated_npy(self):
        emb = self.rng.random((3, 4), dtype=np.float32)
        self.write_npy("geo_deduplicated_cls_embeddings.npy", emb)
        path = self.write_parquet("geo_cleaned.parquet", {"city": ["a", "b", "c"]})

        result = load_embeddings(path)

        np.testing.assert_array_equal(result, emb)

    def test_k2_cleaned_base_falls_back_to_legacy_deduplicated_name(self):
        emb = self.rng.random((3, 4), dtype=np.float32)
        self.write_npy("geo_deduplicated.npy", emb)
        path = self.write_parquet("geo_cleaned.parquet", {"city": ["a", "b", "c"]})

        result = load_embeddings(path)

        np.testing.assert_array_equal(result, emb)

    def test_l_wildcard_glob_fallback_picks_matching_file(self):
        emb = self.rng.random((2, 4), dtype=np.float32)
        self.write_npy("geo_space_cls_embeddings_v2.npy", emb)
        # Decoy: the avg_patch file is removed by the representation filter.
        self.write_npy(
            "geo_space_avg_patch_embeddings.npy",
            self.rng.random((2, 4), dtype=np.float32),
        )
        path = self.write_parquet("geo_space.parquet", {"city": ["a", "b"]})

        result = load_embeddings(path)

        np.testing.assert_array_equal(result, emb)

    def test_m_wildcard_excludes_checkpoint_files(self):
        self.write_npy(
            "geo_space_cls_embeddings_checkpoint.npy",
            self.rng.random((2, 4), dtype=np.float32),
        )
        path = self.write_parquet("geo_space.parquet", {"city": ["a", "b"]})

        with self.assertRaisesRegex(FileNotFoundError, "Please run"):
            load_embeddings(path)

    def test_m2_wildcard_keeps_checkpoint_files_when_base_is_checkpoint(self):
        emb = self.rng.random((2, 4), dtype=np.float32)
        self.write_npy("geo_checkpoint_partial_embeddings.npy", emb)
        path = self.write_parquet("geo_checkpoint.parquet", {"city": ["a", "b"]})

        result = load_embeddings(path)

        np.testing.assert_array_equal(result, emb)

    def test_n_non_tipsv2_model_without_file_raises(self):
        # The legacy model-agnostic file must NOT be used for a non-TIPSv2 model.
        self.write_npy(
            "geo_space_cls_embeddings.npy", self.rng.random((2, 4), dtype=np.float32)
        )
        path = self.write_parquet("geo_space.parquet", {"city": ["a", "b"]})

        with self.assertRaises(FileNotFoundError) as ctx:
            load_embeddings(path, model_name="org/other-model")

        msg = str(ctx.exception)
        self.assertIn("Embeddings for model 'org/other-model'", msg)
        self.assertIn("backfill_embeddings", msg)
        self.assertIn("--model_name org/other-model", msg)

    def test_o_tipsv2_without_file_raises_please_run(self):
        path = self.write_parquet("geo_space.parquet", {"city": ["a", "b"]})

        with self.assertRaises(FileNotFoundError) as ctx:
            load_embeddings(path)

        msg = str(ctx.exception)
        self.assertIn("Embeddings for model 'google/tipsv2-b14'", msg)
        self.assertIn("Please run", msg)
        self.assertIn("backfill_embeddings", msg)

    def test_p_missing_parquet_raises_file_not_found(self):
        with self.assertRaisesRegex(FileNotFoundError, "File not found: "):
            load_embeddings(self.path("does_not_exist.parquet"))

    def test_q_sidecar_npy_found_through_keys_sibling(self):
        # Sidecar = cluster_id present, no Latitude/Longitude. The npy name matches no
        # base-derived pattern, so only the keys-sibling scan can find it.
        emb = self.rng.random((2, 4), dtype=np.float32)  # rows follow keys order a, b
        self.write_npy("embeddings_store_cls_embeddings.npy", emb)
        self.write_keys("embeddings_store_cls_embeddings.keys.parquet", ["a", "b"])
        path = self.write_parquet(
            "clusters_view.parquet",
            {"photo_key": ["B", "A"], "cluster_id": [1, 2]},
        )

        result, _ = _run_quiet(load_embeddings, path)

        np.testing.assert_array_equal(result, emb[[1, 0]])

    def test_r_model_rep_and_precision_autodetected_from_columns(self):
        emb = self.rng.random((2, 4), dtype=np.float32)
        self.write_npy("geo_org_other-model_avg_patch_embeddings.npy", emb)
        self.write_npy(
            "geo_cls_embeddings.npy", self.rng.random((2, 4), dtype=np.float32)
        )
        path = self.write_parquet(
            "geo.parquet",
            {
                "city": ["a", "a"],
                "model_name": ["org/other-model", "org/other-model"],
                "representation_type": ["avg_patch", "avg_patch"],
                "precision": ["float16", "float16"],
            },
        )

        # Nothing passed: model and representation come from the parquet columns.
        # CHARACTERIZATION: precision is read but has no observable effect on the
        # returned matrix; it only feeds tag stripping, which gives the same base.
        result = load_embeddings(path)

        np.testing.assert_array_equal(result, emb)

    def test_r2_explicit_representation_type_is_not_overridden_by_column(self):
        # The column only overrides when the caller passes None or the default "cls".
        emb = self.rng.random((2, 4), dtype=np.float32)
        self.write_npy("geo_avg_patch_embeddings.npy", emb)
        self.write_npy(
            "geo_cls_embeddings.npy", self.rng.random((2, 4), dtype=np.float32)
        )
        path = self.write_parquet(
            "geo.parquet",
            {"city": ["a", "b"], "representation_type": ["cls", "cls"]},
        )

        result = load_embeddings(path, representation_type="avg_patch")

        np.testing.assert_array_equal(result, emb)

    def test_s_roundtrip_save_then_load_returns_stacked_embeddings(self):
        emb = self.rng.random((3, 4), dtype=np.float32)
        df = pd.DataFrame(
            {
                "Platform": ["Flickr", "Mapillary", "Flickr"],
                "Photo_ID": [1, 2, 3],
                "embedding": list(emb),
            }
        )
        path = self.path("geo.parquet")
        _run_quiet(save_dataframe, df, path)

        self.assertEqual(
            sorted(os.listdir(self.root)),
            [
                "geo.parquet",
                "geo_cls_embeddings.keys.parquet",
                "geo_cls_embeddings.npy",
            ],
        )
        self.assertNotIn("embedding", pq.ParquetFile(path).schema_arrow.names)

        result, _ = _run_quiet(load_embeddings, path)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, emb)

    def test_s2_saver_1536_dim_default_name_not_found_by_default_loader(self):
        # CHARACTERIZATION (suspected bug): a 1536-dim save with no representation_type
        # is written as *_cls_avg_patch_*, but the loader's default representation_type
        # "cls" rejects that name, so a default load raises.
        emb = self.rng.random((2, 1536), dtype=np.float32)
        path = self.path("big.parquet")
        _run_quiet(save_dataframe, pd.DataFrame({"embedding": list(emb)}), path)
        self.assertTrue(os.path.exists(self.path("big_cls_avg_patch_embeddings.npy")))

        with self.assertRaises(FileNotFoundError) as ctx:
            load_embeddings(path)
        self.assertIn("Please run", str(ctx.exception))

        result, _ = _run_quiet(
            load_embeddings, path, representation_type="cls_avg_patch"
        )
        np.testing.assert_array_equal(result, emb)

    def test_t_companion_found_in_parent_directory(self):
        # Pinned: the loader also searches the parent of the parquet's folder.
        emb = self.rng.random((2, 4), dtype=np.float32)
        np.save(os.path.join(os.path.dirname(self.root), "geo_cls_embeddings.npy"), emb)
        path = self.write_parquet("geo.parquet", {"city": ["a", "b"]})

        result = load_embeddings(path)

        np.testing.assert_array_equal(result, emb)


class SaveDataframeCompanionCharacterizationTest(_CharacterizationCase):
    def _emb_frame(self, n=2, dim=4):
        return pd.DataFrame(
            {"embedding": list(self.rng.random((n, dim), dtype=np.float32))}
        )

    def test_save_idx_keys_and_drops_embedding_columns(self):
        emb = self.rng.random((2, 4), dtype=np.float32)
        df = pd.DataFrame(
            {"city": ["x", "y"], "embedding_idx": [0, 1], "embedding": list(emb)}
        )
        path = self.path("geo.parquet")

        _, out = _run_quiet(save_dataframe, df, path)

        np.testing.assert_array_equal(np.load(self.path("geo_cls_embeddings.npy")), emb)
        keys = pq.read_table(self.path("geo_cls_embeddings.keys.parquet"))
        self.assertEqual(keys["photo_key"].to_pylist(), ["idx_0", "idx_1"])
        # embedding and embedding_idx are dropped; photo_key is appended.
        self.assertEqual(pq.ParquetFile(path).schema_arrow.names, ["city", "photo_key"])
        self.assertIn("(dtype=float32)", out)

    def test_save_precision_float16_writes_half_companion(self):
        emb = self.rng.random((3, 4), dtype=np.float32)
        df = pd.DataFrame(
            {
                "Platform": ["Flickr", "Mapillary", "Flickr"],
                "Photo_ID": [1, 2, 3],
                "embedding": list(emb),
            }
        )
        path = self.path("geo.parquet")

        _, out = _run_quiet(save_dataframe, df, path, precision="float16")

        self.assertEqual(np.load(self.path("geo_cls_embeddings.npy")).dtype, np.float16)
        self.assertIn("(dtype=float16)", out)
        loaded, _ = _run_quiet(load_embeddings, path)
        self.assertEqual(loaded.dtype, np.float32)
        np.testing.assert_array_equal(loaded, emb.astype(np.float16).astype(np.float32))

    def test_save_model_suffix_rules(self):
        df = self._emb_frame()
        _run_quiet(
            save_dataframe, df, self.path("a.parquet"), model_name="org/other-model"
        )
        _run_quiet(
            save_dataframe, df, self.path("b.parquet"), model_name="google/tipsv2-b14"
        )

        self.assertTrue(
            os.path.exists(self.path("a_org_other-model_cls_embeddings.npy"))
        )
        # The default TIPSv2 model gets no model suffix.
        self.assertTrue(os.path.exists(self.path("b_cls_embeddings.npy")))
        self.assertFalse(
            os.path.exists(self.path("b_google_tipsv2-b14_cls_embeddings.npy"))
        )

    def test_save_core_name_strips_cleaned_but_keeps_filtered(self):
        # CHARACTERIZATION: "_cleaned" and "_clustered_k_X" are stripped from the
        # companion name, but "_filtered" is kept.
        df = self._emb_frame()
        for name in [
            "geo_space_cleaned.parquet",
            "geo_space_filtered.parquet",
            "scene_clustered_k_5.parquet",
        ]:
            _run_quiet(save_dataframe, df, self.path(name))

        npy_files = sorted(f for f in os.listdir(self.root) if f.endswith(".npy"))
        self.assertEqual(
            npy_files,
            [
                "geo_space_cls_embeddings.npy",
                "geo_space_filtered_cls_embeddings.npy",
                "scene_cls_embeddings.npy",
            ],
        )

    def test_save_tmp_extension_uses_inner_format_and_final_names(self):
        _run_quiet(save_dataframe, self._emb_frame(), self.path("geo.parquet.tmp"))

        # No ".tmp_" staging files are left behind after the atomic renames.
        self.assertEqual(
            sorted(os.listdir(self.root)),
            [
                "geo.parquet.tmp",
                "geo_cls_embeddings.keys.parquet",
                "geo_cls_embeddings.npy",
            ],
        )

    def test_save_failure_removes_temp_companion_files(self):
        # The parquet write fails after the companion temp files were written.
        with self.assertRaisesRegex(pa.ArrowException, "Unsupported compression"):
            _run_quiet(
                save_dataframe,
                self._emb_frame(),
                self.path("geo.parquet"),
                compression="not_a_codec",
            )

        self.assertEqual([f for f in os.listdir(self.root) if ".tmp_" in f], [])
        self.assertFalse(os.path.exists(self.path("geo_cls_embeddings.npy")))
        self.assertFalse(os.path.exists(self.path("geo_cls_embeddings.keys.parquet")))

    def test_save_unsupported_extension_raises(self):
        with self.assertRaisesRegex(
            ValueError, "Unsupported file format '.txt' for saving dataframe"
        ):
            save_dataframe(pd.DataFrame({"a": [1]}), self.path("out.txt"))


if __name__ == "__main__":
    unittest.main()
