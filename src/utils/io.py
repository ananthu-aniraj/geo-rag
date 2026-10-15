import gc
import glob
import hashlib
import os
import re
import time
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.utils.credentials import get_wildlife_insights_token


def get_core_base_name(base_name):
    """Recursively strips common pipeline suffixes to find the base prefix (e.g. geo_space)."""
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]
    suffixes = [
        "_filtered",
        "_cleaned",
        "_deduplicated",
        "_clustered",
        "_offline",
        "_float16",
        "_float32",
        "_cls",
        "_avg_patch",
        "_cls_avg_patch",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                changed = True
    return base_name


def load_dataframe(file_path, **kwargs):
    """
    Loads a dataframe from CSV, Parquet, or Pickle files dynamically based on extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    elif ext == ".csv":
        # Default low_memory=False for safety on mixed-type data
        if "low_memory" not in kwargs:
            kwargs["low_memory"] = False
        return pd.read_csv(file_path, **kwargs)
    elif ext in (".pkl", ".pickle"):
        return pd.read_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for loading dataframe.")


def save_dataframe(
    df,
    file_path,
    index=False,
    representation_type=None,
    precision=None,
    model_name=None,
    **kwargs,
):
    """
    Saves a dataframe to CSV, Parquet, or Pickle with optimal compression default (Zstd for Parquet).
    Automatically decouples embeddings into a companion .npy file if the column is present.
    """
    # Ensure output parent directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".tmp":
        # Split again to find the actual format extension, e.g. .parquet from .parquet.tmp
        base_without_tmp = os.path.splitext(file_path)[0]
        ext = os.path.splitext(base_without_tmp)[1].lower()
        if not ext:
            ext = ".parquet"  # Default fallback if only .tmp was provided

    if ext == ".parquet":
        # Default to high-performance zstd compression
        if "compression" not in kwargs:
            kwargs["compression"] = "zstd"

        has_embeddings = "embedding" in df.columns

        # Avoid deep copy of multi-gigabyte dataframes.
        # If embeddings are present, drop them immediately to avoid keeping duplicate array copies in RAM.
        if has_embeddings:
            df_to_save = df.drop(columns=["embedding"], errors="ignore")
        else:
            df_to_save = df.copy(deep=False)

        # Generate stable photo_key using fast vectorized PyArrow C++ join if missing
        if (
            "Platform" in df_to_save.columns
            and "Photo_ID" in df_to_save.columns
            and "photo_key" not in df_to_save.columns
        ):
            try:
                plat_arr = pa.Array.from_pandas(df_to_save["Platform"])
                pid_arr = pa.Array.from_pandas(df_to_save["Photo_ID"]).cast(
                    plat_arr.type
                )
                sep = pa.scalar("_", type=plat_arr.type)
                df_to_save["photo_key"] = pc.binary_join_element_wise(
                    plat_arr, pid_arr, sep
                ).to_pandas()
            except Exception:
                df_to_save["photo_key"] = (
                    df_to_save["Platform"].astype(str)
                    + "_"
                    + df_to_save["Photo_ID"].astype(str)
                )

        if has_embeddings:
            effective_path = (
                base_without_tmp
                if os.path.splitext(file_path)[1].lower() == ".tmp"
                else file_path
            )
            db_dir = os.path.dirname(os.path.abspath(effective_path))
            base_name = os.path.splitext(os.path.basename(effective_path))[0]
            if "_clustered_k_" in base_name:
                base_name = base_name.split("_clustered_k_")[0]

            # Determine dimension without loading entire matrix into memory
            first_emb = df["embedding"].iloc[0]
            dim = len(first_emb) if hasattr(first_emb, "__len__") else 768

            # 1. Resolve representation suffix
            if representation_type is not None:
                rep_suffix = representation_type
            else:
                # Auto-detect based on feature dimension
                if dim == 1536:
                    rep_suffix = "cls_avg_patch"
                else:
                    rep_suffix = "cls"

            model_suffix = ""
            if model_name and model_name != "google/tipsv2-b14":
                model_suffix = "_" + model_name.replace("/", "_")

            npy_path = os.path.join(
                db_dir, f"{base_name}{model_suffix}_{rep_suffix}_embeddings.npy"
            )

            # 2. Resolve precision dtype
            dtype = np.float32
            if precision == "float16":
                dtype = np.float16
            elif precision == "float32":
                dtype = np.float32
            else:
                dtype = getattr(first_emb, "dtype", np.float32)

            # Save the companion files to temporary paths first
            tmp_npy_path = file_path + f".tmp_{rep_suffix}_embeddings.npy"
            tmp_keys_path = tmp_npy_path.replace(".npy", ".keys.parquet")

            print(
                f" -> Automatically decoupling embeddings to companion file: {npy_path} (dtype={np.dtype(dtype).name})"
            )

            # Memory-efficient extraction directly into destination dtype
            n_rows = len(df)
            if (
                isinstance(df["embedding"].values, np.ndarray)
                and df["embedding"].values.ndim == 2
            ):
                embs = df["embedding"].values.astype(dtype, copy=False)
            else:
                # Preallocate in target dtype to prevent having simultaneous float32 + float16 full matrices
                embs = np.empty((n_rows, dim), dtype=dtype)
                chunk_size = 50_000
                for i in range(0, n_rows, chunk_size):
                    end_idx = min(i + chunk_size, n_rows)
                    chunk_vals = df["embedding"].iloc[i:end_idx].values
                    if len(chunk_vals) > 0 and isinstance(chunk_vals[0], np.ndarray):
                        embs[i:end_idx] = np.vstack(chunk_vals).astype(
                            dtype, copy=False
                        )
                    else:
                        embs[i:end_idx] = np.array(list(chunk_vals), dtype=dtype)

            np.save(tmp_npy_path, embs)
            del embs
            gc.collect()

            if "photo_key" not in df_to_save.columns:
                df_to_save["photo_key"] = "idx_" + np.arange(len(df_to_save)).astype(
                    str
                )

            # Fast keys parquet writing directly via PyArrow (bypassing Pandas DataFrame allocation)
            keys_path = npy_path.replace(".npy", ".keys.parquet")
            print(f" -> Saving companion keys file to: {keys_path}")
            keys_tbl = pa.Table.from_arrays(
                [pa.array(df_to_save["photo_key"])], names=["photo_key"]
            )
            pq.write_table(keys_tbl, tmp_keys_path, compression="zstd")
            del keys_tbl

            try:
                df_to_save.to_parquet(file_path, index=index, **kwargs)
                # Atomically rename companion files on success
                os.replace(tmp_npy_path, npy_path)
                os.replace(tmp_keys_path, keys_path)
            except Exception as e:
                # Clean up temporary companion files on failure
                if os.path.exists(tmp_npy_path):
                    try:
                        os.remove(tmp_npy_path)
                    except Exception:
                        pass
                if os.path.exists(tmp_keys_path):
                    try:
                        os.remove(tmp_keys_path)
                    except Exception:
                        pass
                raise e
            return

        df_to_save.to_parquet(file_path, index=index, **kwargs)
    elif ext == ".csv":
        df.to_csv(file_path, index=index, **kwargs)
    elif ext in (".pkl", ".pickle"):
        df.to_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for saving dataframe.")


def get_parquet_writer(file_path, schema, **kwargs):
    """
    Returns a PyArrow ParquetWriter configured with Zstandard compression.
    """
    # Ensure parent directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if "compression" not in kwargs:
        kwargs["compression"] = "zstd"

    return pq.ParquetWriter(file_path, schema, **kwargs)


def load_dataset_with_clusters(
    parquet_path,
    k_clusters=50000,
    columns=None,
    representation_type=None,
    model_name=None,
    precision=None,
    **kwargs,
):
    """
    Backward-compatible loader that returns metadata and cluster assignments.
    Merges sidecars automatically if the base parquet does not contain cluster columns.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    # Auto-detect k_clusters if the input filename contains it
    match = re.search(r"_clustered_k_(\d+)", os.path.basename(parquet_path))
    if match:
        k_clusters = int(match.group(1))

    # Auto-extract k_clusters from filename if it has _k_X suffix
    match = re.search(r"_k_(\d+)", os.path.basename(parquet_path))
    if match:
        k_clusters = int(match.group(1))

    # Inspect schema names using PyArrow ParquetFile
    pf = pq.ParquetFile(parquet_path)
    schema_names = pf.schema_arrow.names

    known_sidecar_path = None
    # If it's a decoupled sidecar file (has cluster_id but lacks base columns like Latitude/Longitude),
    # resolve the path to the base metadata file instead.
    if "cluster_id" in schema_names and (
        "Latitude" not in schema_names or "Longitude" not in schema_names
    ):
        known_sidecar_path = parquet_path
        db_dir = os.path.dirname(os.path.abspath(parquet_path))
        parent_dir = os.path.dirname(db_dir)
        candidate_dirs = [db_dir]
        if parent_dir and parent_dir != db_dir:
            candidate_dirs.append(parent_dir)

        sidecar_num_rows = pf.metadata.num_rows
        candidate_base_files = []
        for d in candidate_dirs:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if (
                    f.endswith(".parquet")
                    and not f.endswith(".keys.parquet")
                    and "_clustered_k_" not in f
                    and "_h3_semantic_index" not in f
                ):
                    p = os.path.join(d, f)
                    try:
                        c_pf = pq.ParquetFile(p)
                        if (
                            "Latitude" in c_pf.schema_arrow.names
                            and "Longitude" in c_pf.schema_arrow.names
                        ):
                            candidate_base_files.append(
                                (p, c_pf.metadata.num_rows, "cleaned" in f)
                            )
                    except Exception:
                        pass
            if candidate_base_files:
                break

        if candidate_base_files:
            candidate_base_files.sort(
                key=lambda x: (x[1] == sidecar_num_rows, x[2], -len(x[0])),
                reverse=True,
            )
            parquet_path = candidate_base_files[0][0]
            pf = pq.ParquetFile(parquet_path)
            schema_names = pf.schema_arrow.names

    cluster_cols = [
        "cluster_id",
        "cluster_label",
        "cluster_description",
        "parent_cluster_id",
        "parent_cluster_label",
        "parent_cluster_description",
        "visual_description",
        "parent_visual_description",
        "model_name",
        "representation_type",
        "precision",
    ]

    # Filter which columns belong to cluster variables vs base metadata
    if columns is None:
        requested_cols = list(schema_names) + cluster_cols
    else:
        requested_cols = columns

    req_cluster_cols = [c for c in requested_cols if c in cluster_cols]
    req_meta_cols = [
        c
        for c in requested_cols
        if c not in cluster_cols or c in ["Platform", "Photo_ID"]
    ]

    # Case A: Old format contains cluster columns directly (or we failed to fallback to base)
    if "cluster_id" in schema_names and "Latitude" in schema_names:
        return load_dataframe(parquet_path, columns=columns, **kwargs)

    # Ensure join keys present in schema are loaded so merging never fails with KeyError
    meta_join_cols = [
        c for c in ["photo_key", "Platform", "Photo_ID"] if c in schema_names
    ]
    meta_load_cols = (
        list(dict.fromkeys(req_meta_cols + meta_join_cols))
        if columns is not None
        else req_meta_cols
    )

    # Case B: Decoupled format
    df_meta = load_dataframe(parquet_path, columns=meta_load_cols, **kwargs)

    if known_sidecar_path and os.path.exists(known_sidecar_path):
        sidecar_path = known_sidecar_path
    else:
        db_dir = os.path.dirname(os.path.abspath(parquet_path))
        candidate_dirs = [db_dir]
        parent_dir = os.path.dirname(db_dir)
        if parent_dir and parent_dir != db_dir:
            candidate_dirs.append(parent_dir)

        model_slug = model_name.replace("/", "_") if model_name else None
        rep = representation_type or "cls"

        sidecar_matches = []
        for d in candidate_dirs:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".parquet") and f"_clustered_k_{k_clusters}" in f:
                    sidecar_matches.append(os.path.join(d, f))

        sidecar_path = None
        if sidecar_matches:

            def rank_sidecar(p):
                fn = os.path.basename(p)
                score_model = 1 if (model_slug and model_slug in fn) else 0
                score_rep = 1 if rep in fn else 0
                score_prec = 1 if (precision and precision in fn) else 0
                return (score_model, score_rep, score_prec, -len(fn))

            sidecar_matches.sort(key=rank_sidecar, reverse=True)
            sidecar_path = sidecar_matches[0]

    if sidecar_path and os.path.exists(sidecar_path) and req_cluster_cols:
        # We need join keys in both dataframes for merging
        pf_side = pq.ParquetFile(sidecar_path)
        side_names = pf_side.schema_arrow.names
        join_cols = ["Platform", "Photo_ID"]
        if "photo_key" in side_names and "photo_key" in df_meta.columns:
            join_cols = ["photo_key"]

        sidecar_cols = list(set(join_cols + req_cluster_cols))
        side_avail_cols = [c for c in sidecar_cols if c in side_names]

        df_sidecar = load_dataframe(sidecar_path, columns=side_avail_cols, **kwargs)

        # Optimize memory: encode repeated cluster description text columns as categorical
        desc_cols = {
            "cluster_description",
            "parent_cluster_description",
            "visual_description",
            "parent_visual_description",
        }
        for c in side_avail_cols:
            if c in desc_cols and c in df_sidecar.columns:
                df_sidecar[c] = df_sidecar[c].astype("category")

        # Fast O(1) alignment verification for 1:1 direct column assignment
        is_aligned = False
        if len(df_meta) == len(df_sidecar) and len(df_meta) > 0:
            if "photo_key" in join_cols:
                n_rows = len(df_meta)
                probes = [0, n_rows // 2, n_rows - 1]
                meta_vals = df_meta["photo_key"].iloc[probes].values
                side_vals = df_sidecar["photo_key"].iloc[probes].values
                if (meta_vals == side_vals).all():
                    is_aligned = (
                        df_meta["photo_key"].values == df_sidecar["photo_key"].values
                    ).all()
            elif set(join_cols) == {"Platform", "Photo_ID"}:
                n_rows = len(df_meta)
                probes = [0, n_rows // 2, n_rows - 1]
                if (
                    df_meta["Platform"].iloc[probes].values
                    == df_sidecar["Platform"].iloc[probes].values
                ).all() and (
                    df_meta["Photo_ID"].iloc[probes].values
                    == df_sidecar["Photo_ID"].iloc[probes].values
                ).all():
                    is_aligned = (
                        df_meta["Platform"].values == df_sidecar["Platform"].values
                    ).all() and (
                        df_meta["Photo_ID"].values == df_sidecar["Photo_ID"].values
                    ).all()

        if is_aligned:
            # Direct column assignment (19x faster, 0 extra RAM)
            for col in req_cluster_cols:
                if col in df_sidecar.columns:
                    df_meta[col] = df_sidecar[col]
        else:
            if known_sidecar_path:
                # Caller requested the clustered sidecar directly: preserve sidecar rows and sidecar ordering exactly
                meta_cols_to_merge = [
                    c
                    for c in df_meta.columns
                    if c not in df_sidecar.columns or c in join_cols
                ]
                df_meta = df_sidecar.merge(
                    df_meta[meta_cols_to_merge], on=join_cols, how="left"
                )
            else:
                df_meta = df_meta.merge(df_sidecar, on=join_cols, how="left")

    if columns is not None:
        return df_meta[[c for c in columns if c in df_meta.columns]]

    return df_meta


def load_embeddings(
    parquet_path,
    column="embedding",
    representation_type="cls",
    model_name=None,
    precision=None,
    mmap=False,
):
    """
    Loads embedding matrix matching (model_name, representation_type, precision).
    Supports raw embedding matrices (.npy) and Parquet embedded columns.
    When companion keys (.keys.parquet) are present, verifies and aligns rows.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    # Case 1: Direct .npy file passed
    if parquet_path.endswith(".npy"):
        emb = np.load(parquet_path, mmap_mode="r" if mmap else None)
        target_dtype = np.float16 if precision == "float16" else np.float32
        return emb if mmap else emb.astype(target_dtype, copy=False)

    pf = pq.ParquetFile(parquet_path)
    schema_names = pf.schema_arrow.names

    # Case 2: Embedded in Parquet
    if column in schema_names:
        table = pf.read(columns=[column])
        chunked_arr = table[column]
        num_rows = len(table)
        dim = len(chunked_arr.chunk(0)[0].as_py())
        target_dtype = np.float16 if precision == "float16" else np.float32

        emb_matrix = np.empty((num_rows, dim), dtype=target_dtype)
        current_row = 0
        for chunk in chunked_arr.chunks:
            chunk_len = len(chunk)
            flat_chunk = chunk.flatten().to_numpy()
            emb_matrix[current_row : current_row + chunk_len] = flat_chunk.reshape(
                chunk_len, dim
            )
            current_row += chunk_len
        return emb_matrix

    # Case 3: Decoupled companion .npy file
    # Auto-detect sidecar metadata from schema if not provided
    if model_name is None and "model_name" in schema_names:
        try:
            model_name = pf.read(columns=["model_name"])["model_name"].to_pylist()[0]
        except Exception:
            pass

    if (
        representation_type is None or representation_type == "cls"
    ) and "representation_type" in schema_names:
        try:
            r_val = pf.read(columns=["representation_type"])[
                "representation_type"
            ].to_pylist()[0]
            if r_val:
                representation_type = r_val
        except Exception:
            pass

    if precision is None and "precision" in schema_names:
        try:
            p_val = pf.read(columns=["precision"])["precision"].to_pylist()[0]
            if p_val:
                precision = p_val
        except Exception:
            pass

    db_dir = os.path.dirname(os.path.abspath(parquet_path))
    candidate_dirs = [db_dir]
    parent_dir = os.path.dirname(db_dir)
    if parent_dir and parent_dir != db_dir:
        candidate_dirs.append(parent_dir)

    rep = representation_type or "cls"
    model_slug = model_name.replace("/", "_") if model_name else None
    is_tipsv2 = (model_name is None) or ("tipsv2" in str(model_name).lower())

    patterns = []
    if model_slug and not is_tipsv2:
        if precision:
            patterns.append(f"*{model_slug}*{rep}*{precision}*.npy")
        patterns.append(f"*{model_slug}*{rep}*.npy")
    elif is_tipsv2:
        if precision:
            patterns.append(f"*google_tipsv2-b14*{rep}*{precision}*.npy")
            patterns.append(f"*{rep}*{precision}*.npy")
        patterns.append(f"*google_tipsv2-b14*{rep}*.npy")
        patterns.append(f"*{rep}*embeddings.npy")
        patterns.append(f"*{rep}.npy")
    else:
        patterns.append(f"*{rep}*embeddings.npy")

    def is_rep_match(fn: str, r: str) -> bool:
        fn_lower = fn.lower()
        if r == "cls":
            return "cls" in fn_lower and "avg_patch" not in fn_lower
        elif r == "avg_patch":
            return "avg_patch" in fn_lower and "cls_avg_patch" not in fn_lower
        return r.lower() in fn_lower

    target_rows = None
    try:
        target_rows = pf.metadata.num_rows
    except Exception:
        pass

    base_stem = os.path.splitext(os.path.basename(parquet_path))[0]
    if "_clustered_k_" in base_stem:
        base_stem = base_stem.split("_clustered_k_")[0]

    model_suffix = "_" + model_slug if model_slug and not is_tipsv2 else ""
    direct_npy = os.path.join(db_dir, f"{base_stem}{model_suffix}_{rep}_embeddings.npy")
    direct_keys = direct_npy.replace(".npy", ".keys.parquet")

    npy_path = None
    if os.path.exists(direct_npy) and (
        not os.path.exists(direct_keys)
        or target_rows is None
        or pq.ParquetFile(direct_keys).metadata.num_rows == target_rows
    ):
        npy_path = direct_npy
    else:
        for d in candidate_dirs:
            if not os.path.exists(d):
                continue
            for pat in patterns:
                raw_matches = glob.glob(os.path.join(d, pat))
                matches = [
                    m
                    for m in raw_matches
                    if "checkpoint" not in os.path.basename(m)
                    and not m.endswith(".tmp.npy")
                    and is_rep_match(os.path.basename(m), rep)
                ]
                if matches:
                    if len(matches) > 1:

                        def rank_match(m):
                            fn = os.path.basename(m)
                            k_file = m.replace(".npy", ".keys.parquet")
                            row_score = 1
                            if os.path.exists(k_file) and target_rows is not None:
                                try:
                                    k_rows = pq.ParquetFile(k_file).metadata.num_rows
                                    row_score = 0 if k_rows == target_rows else 2
                                except Exception:
                                    pass
                            stem_match = 0 if fn.startswith(base_stem) else 1
                            return (row_score, stem_match, len(fn))

                        matches.sort(key=rank_match)
                    npy_path = matches[0]
                    break
            if npy_path:
                break

    if not npy_path or not os.path.exists(npy_path):
        if not is_tipsv2:
            raise FileNotFoundError(
                f"Embeddings for model '{model_name}' (representation '{representation_type}') were not found on disk for '{parquet_path}' in '{db_dir}'. "
                f"(Note: Embedding files without a model name are assumed to be TIPSv2). "
                f"Please run 'python -m src.processing.backfill_embeddings --input {parquet_path} --model_name {model_name} --representation_type {representation_type}' "
                f"to compute and backfill embeddings for this model before proceeding."
            )
        else:
            model_display = model_name or "google/tipsv2-b14"
            raise FileNotFoundError(
                f"Embeddings for model '{model_display}' (representation '{representation_type}') were not found on disk for '{parquet_path}' in '{db_dir}'. "
                f"Please run 'python -m src.processing.backfill_embeddings --input {parquet_path} --model_name {model_display} --representation_type {representation_type}' "
                f"to compute and backfill embeddings for this representation before proceeding."
            )

    emb = np.load(npy_path, mmap_mode="r")
    target_dtype = np.float16 if precision == "float16" else np.float32

    # Check companion keys index if present
    keys_path = npy_path.replace(".npy", ".keys.parquet")
    has_photo_key = "photo_key" in schema_names
    has_platform_and_id = "Platform" in schema_names and "Photo_ID" in schema_names

    if (has_photo_key or has_platform_and_id) and os.path.exists(keys_path):
        print(f" -> Resolving embeddings via companion keys index: {keys_path}")
        if has_photo_key:
            meta_keys_table = pf.read(columns=["photo_key"])
            master_keys_table = pq.read_table(keys_path, columns=["photo_key"])

            # Fast O(1) buffer equality check in PyArrow (100% identical order and content)
            if len(meta_keys_table) == len(master_keys_table) and meta_keys_table[
                "photo_key"
            ].equals(master_keys_table["photo_key"]):
                if mmap:
                    return emb
                return emb.astype(target_dtype, copy=False)

            meta_keys = (
                meta_keys_table["photo_key"].to_pandas().astype(str).str.lower().values
            )
            master_keys = pd.Index(
                master_keys_table["photo_key"].to_pandas().astype(str).str.lower()
            )
        else:
            meta_keys_table = pf.read(columns=["Platform", "Photo_ID"])
            df_temp = meta_keys_table.to_pandas()
            meta_keys = (
                df_temp["Platform"].astype(str).str.lower()
                + "_"
                + df_temp["Photo_ID"].astype(str)
            ).values
            master_keys = pd.Index(
                pd.read_parquet(keys_path, columns=["photo_key"])["photo_key"]
                .astype(str)
                .str.lower()
            )

        if master_keys.is_unique:
            indices = master_keys.get_indexer(meta_keys)
        else:
            pos_series = pd.Series(np.arange(len(master_keys)), index=master_keys)
            pos_series = pos_series[~pos_series.index.duplicated(keep="first")]
            indices = pos_series.reindex(meta_keys, fill_value=-1).values

        valid_mask = indices >= 0
        if not valid_mask.all():
            missing_samples = list(meta_keys[~valid_mask][:10])
            print(
                f"Warning: Found {np.sum(~valid_mask):,} missing keys in embeddings keys index. Zero-filling..."
            )
            print(f" -> Sample missing keys: {missing_samples}")
            safe_indices = np.clip(indices, 0, len(emb) - 1)
            sliced_emb = emb[safe_indices].astype(target_dtype, copy=False)
            sliced_emb[~valid_mask] = 0.0
            return sliced_emb

        return emb[indices].astype(target_dtype, copy=False)

    if mmap:
        return emb
    return emb.astype(target_dtype, copy=False)


def iter_embeddings(
    parquet_path,
    batch_size=100_000,
    column="embedding",
    representation_type="cls",
    model_name=None,
    precision="float32",
):
    """
    Yields slices of embeddings in memory-efficient batches without
    materializing the full 7.4M-row matrix in RAM simultaneously.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    # Use load_embeddings with mmap=True to avoid allocating memory for the full matrix
    emb_mmap = load_embeddings(
        parquet_path,
        column=column,
        representation_type=representation_type,
        model_name=model_name,
        precision="float16",
        mmap=True,
    )

    target_dtype = np.float16 if precision == "float16" else np.float32
    n_rows = len(emb_mmap)

    for i in range(0, n_rows, batch_size):
        end_idx = min(i + batch_size, n_rows)
        yield emb_mmap[i:end_idx].astype(target_dtype, copy=False)


# Known censor/placeholder image hashes (e.g. Wildlife Insights "Human in frame" placeholder)
KNOWN_PLACEHOLDER_MD5_HASHES = {
    "6da31bd91c242aa98de0ddae4ad89e3c",  # Wildlife Insights 1600x1200 "Human in frame" PNG placeholder (8,535 bytes)
}


def is_valid_image_file(filepath_or_bytes, reject_placeholders=True):
    """Checks if a file path or raw bytes contains valid image data (JPEG, PNG, GIF, WebP, TIFF)."""
    full_bytes = None
    file_path = None
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        data = filepath_or_bytes[:32]
        full_bytes = filepath_or_bytes
    elif isinstance(filepath_or_bytes, (str, os.PathLike)):
        file_path = filepath_or_bytes
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < 100:
            return False
        try:
            with open(file_path, "rb") as f:
                data = f.read(32)
        except Exception:
            return False
    else:
        return False

    if not data or len(data) < 4:
        return False

    # Check for known censor/placeholder images
    if reject_placeholders:
        try:
            h = None
            if full_bytes is not None:
                h = hashlib.md5(full_bytes).hexdigest()
            elif file_path is not None:
                if os.path.getsize(file_path) == 8535:
                    with open(file_path, "rb") as f:
                        h = hashlib.md5(f.read()).hexdigest()
            if h and h in KNOWN_PLACEHOLDER_MD5_HASHES:
                return False
        except Exception:
            pass

    # Check magic bytes for standard image formats
    if data.startswith(b"\xff\xd8\xff"):  # JPEG
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        return True
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):  # GIF
        return True
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":  # WebP
        return True
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):  # TIFF
        return True

    return False


def build_offline_image_index(image_root_dirs, extensions=None):
    """
    Recursively scans image_root_dirs to build a fast in-memory lookup index
    supporting arbitrary nested directory structures.

    Returns a dict mapping keys (exact filenames, stems, relative subpaths,
    and platform-qualified stems) to absolute file paths on disk.
    """
    if not image_root_dirs:
        return {}

    dirs = [image_root_dirs] if isinstance(image_root_dirs, str) else image_root_dirs
    if extensions is None:
        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tif",
            ".tiff",
            ".gif",
            ".bmp",
        }

    index = {}

    for d in dirs:
        if not d or not os.path.exists(d):
            continue
        abs_d = os.path.abspath(d)
        for root, _, files in os.walk(abs_d):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in extensions:
                    continue
                full_path = os.path.join(root, fn)
                stem = os.path.splitext(fn)[0]
                rel_path = os.path.relpath(full_path, abs_d)

                # Store direct filename and stem
                for k in [fn, fn.lower(), stem, stem.lower()]:
                    if k not in index:
                        index[k] = full_path

                # Store relative subpath
                rel_clean = rel_path.replace("\\", "/")
                for k in [rel_clean, rel_clean.lower()]:
                    if k not in index:
                        index[k] = full_path

                # Subpath components (e.g. "site_A/deployment_1/photo_999.JPG")
                parts = rel_clean.split("/")
                if len(parts) > 1:
                    sub_key = "/".join(parts[-2:])
                    if sub_key not in index:
                        index[sub_key] = full_path
                    if sub_key.lower() not in index:
                        index[sub_key.lower()] = full_path

                    stem_sub_key = f"{parts[-2]}/{stem}"
                    if stem_sub_key not in index:
                        index[stem_sub_key] = full_path
                    if stem_sub_key.lower() not in index:
                        index[stem_sub_key.lower()] = full_path

    return index


def resolve_offline_image_path(
    url, image_root_dirs, photo_id=None, platform=None, image_index=None
):
    """
    Resolves an image URL/ID to a local path on disk by checking an optional
    precomputed recursive image_index, flat files, train/ folders, and nested
    directory structures.
    Returns the absolute path if found, otherwise None.
    """
    # 0. Check precomputed recursive image_index first if provided
    if image_index is not None:
        # A. Try platform + photo_id
        if photo_id:
            photo_str = str(photo_id).strip()
            if photo_str.endswith(".0"):
                photo_str = photo_str[:-2]

            # Fast direct ID check in index
            if photo_str in image_index:
                return image_index[photo_str]
            photo_str_lower = photo_str.lower()
            if photo_str_lower in image_index:
                return image_index[photo_str_lower]

            if platform:
                plat_str = str(platform).strip().lower()
                plat_key = f"{plat_str}/{photo_str_lower}"
                if plat_key in image_index:
                    return image_index[plat_key]
                for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                    candidate = f"{plat_key}{ext.lower()}"
                    if candidate in image_index:
                        return image_index[candidate]

            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                candidate = f"{photo_str}{ext}"
                if candidate in image_index:
                    return image_index[candidate]
                if candidate.lower() in image_index:
                    return image_index[candidate.lower()]

        # B. Try url
        if url and isinstance(url, str):
            clean_url = url.split("://")[1] if "://" in url else url
            clean_url = clean_url.strip()
            if clean_url in image_index:
                return image_index[clean_url]
            clean_url_lower = clean_url.lower()
            if clean_url_lower in image_index:
                return image_index[clean_url_lower]

            url_no_params = clean_url.split("?")[0].split("#")[0]
            if url_no_params != clean_url:
                if url_no_params in image_index:
                    return image_index[url_no_params]
                if url_no_params.lower() in image_index:
                    return image_index[url_no_params.lower()]

            basename = os.path.basename(url_no_params)
            if basename in image_index:
                return image_index[basename]
            basename_lower = basename.lower()
            if basename_lower in image_index:
                return image_index[basename_lower]

            stem = os.path.splitext(basename)[0]
            if stem in image_index:
                return image_index[stem]
            stem_lower = stem.lower()
            if stem_lower in image_index:
                return image_index[stem_lower]

            parts = url_no_params.replace("\\", "/").split("/")
            if len(parts) > 1:
                sub_key = "/".join(parts[-2:])
                if sub_key in image_index:
                    return image_index[sub_key]
                if sub_key.lower() in image_index:
                    return image_index[sub_key.lower()]

        # When precomputed image_index is provided, image_root_dirs have been fully indexed.
        # Avoid expensive fall-through disk checks on non-matching records.
        return None

    if not image_root_dirs:
        env_dirs = os.environ.get("IMAGE_ROOT_DIR") or os.environ.get(
            "OFFLINE_DATASET_DIRS"
        )
        if env_dirs:
            image_root_dirs = env_dirs
        else:
            local_defaults = ["./images", "./data", "images", "data"]
            image_root_dirs = [d for d in local_defaults if os.path.isdir(d)]
            if not image_root_dirs and url and os.path.isfile(url):
                return os.path.abspath(url)

    if not image_root_dirs:
        return None

    if isinstance(image_root_dirs, (str, bytes, os.PathLike)):
        image_root_dirs = [image_root_dirs]

    dirs = []
    for item in image_root_dirs:
        if not item:
            continue
        if isinstance(item, str):
            item = os.path.expanduser(item.strip())
            if " " in item and not os.path.exists(item):
                for part in item.split():
                    p = os.path.expanduser(part.strip())
                    if p and os.path.exists(p):
                        dirs.append(os.path.abspath(p))
                continue
        if os.path.exists(item):
            dirs.append(os.path.abspath(item))

    if not dirs:
        return None

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]
    plat_str = str(platform).strip().lower() if platform else ""
    photo_str = str(photo_id).strip() if photo_id is not None else ""
    if photo_str.endswith(".0"):
        photo_str = photo_str[:-2]

    # Clean and extract relative candidate paths from url
    rel_candidates = []
    if url and isinstance(url, str):
        clean_u = url.split("://")[1] if "://" in url else url
        clean_u = clean_u.split("?")[0].split("#")[0].replace("\\", "/")
        rel_u = clean_u.lstrip("/")
        if rel_u:
            rel_candidates.append(rel_u)
        if plat_str and rel_u.lower().startswith(f"{plat_str}/"):
            rel_candidates.append(rel_u[len(plat_str) + 1 :])
        parts = rel_u.split("/")
        if len(parts) > 1:
            rel_candidates.append("/".join(parts[-2:]))
            if len(parts) > 2:
                rel_candidates.append("/".join(parts[-3:]))
        bname = os.path.basename(rel_u)
        if bname and bname not in rel_candidates:
            rel_candidates.append(bname)

    for d in dirs:
        candidate_subdirs = [d]
        standard_subdirs = ["images", "img", "photos", "train"]
        if plat_str:
            standard_subdirs.extend(
                [
                    plat_str,
                    f"{plat_str}_scraped",
                    f"{plat_str}_scraped/images",
                    f"{plat_str}/images",
                    f"images/{plat_str}",
                    f"{plat_str}_data",
                    f"{plat_str}_subset",
                ]
            )
        for sub in standard_subdirs:
            sub_path = os.path.join(d, sub)
            if os.path.isdir(sub_path):
                candidate_subdirs.append(sub_path)

        for sub_dir in candidate_subdirs:
            # 1. Match against relative URL candidates and case variations
            for rel_c in rel_candidates:
                p = os.path.join(sub_dir, rel_c)
                if os.path.isfile(p):
                    return os.path.abspath(p)
                if "." in rel_c:
                    stem, ext = os.path.splitext(rel_c)
                    for e in [ext.lower(), ext.upper()]:
                        if e != ext:
                            p_alt = os.path.join(sub_dir, f"{stem}{e}")
                            if os.path.isfile(p_alt):
                                return os.path.abspath(p_alt)
                else:
                    for ext in extensions:
                        p_ext = os.path.join(sub_dir, f"{rel_c}{ext}")
                        if os.path.isfile(p_ext):
                            return os.path.abspath(p_ext)

            # 2. Match against photo_id
            if photo_str:
                for ext in extensions:
                    p_pid = os.path.join(sub_dir, f"{photo_str}{ext}")
                    if os.path.isfile(p_pid):
                        return os.path.abspath(p_pid)
                if len(photo_str) == 16:
                    p_nested = os.path.join(
                        sub_dir,
                        photo_str[0],
                        photo_str[1],
                        photo_str[2],
                        f"{photo_str}.jpg",
                    )
                    if os.path.isfile(p_nested):
                        return os.path.abspath(p_nested)

    return None


# Thread-safe global session for download adapters
_http_session = None


def _get_http_session():
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=Retry(
                total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
            ),
        )
        _http_session.mount("https://", adapter)
        _http_session.mount("http://", adapter)
    return _http_session


def resolve_wildlife_insights_url(url, cookie=None, token=None, photo_id=None):
    """Resolves a Wildlife Insights public web viewer URL into a signed Google Cloud Storage image URL."""
    if not url or "wildlifeinsights.org" not in url:
        return None

    m_dl = re.search(r"/download/(\d+)", url)
    m_proj = re.search(r"/project/(\d+)", url)
    m_img = re.search(r"/data-files/([a-zA-Z0-9\-]+)", url)

    dl_id = int(m_dl.group(1)) if m_dl else None
    proj_id = int(m_proj.group(1)) if m_proj else None
    img_uuid = m_img.group(1) if m_img else str(photo_id or "")

    if not dl_id or not img_uuid:
        return None

    auth_token = token or get_wildlife_insights_token(cookie)
    if not auth_token:
        return None

    gql_ep = "https://api.wildlifeinsights.org/graphql"
    query = """query getDataFilePublicDownloadUrl($downloadId: Int!, $projectId: Int, $imageUUID: String!) {
  getDataFilePublicDownloadUrl(downloadId: $downloadId, projectId: $projectId, imageUUID: $imageUUID) {
    url
  }
}"""
    variables = {
        "downloadId": dl_id,
        "projectId": proj_id,
        "imageUUID": img_uuid,
    }
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        r = requests.post(
            gql_ep,
            json={"query": query, "variables": variables},
            headers=headers,
            timeout=10,
        )
        if r.status_code == 401:
            auth_token = get_wildlife_insights_token(cookie, refresh=True)
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
                r = requests.post(
                    gql_ep,
                    json={"query": query, "variables": variables},
                    headers=headers,
                    timeout=10,
                )
        if r.status_code == 200:
            res_data = r.json()
            if (
                "data" in res_data
                and res_data["data"]
                and res_data["data"].get("getDataFilePublicDownloadUrl")
            ):
                return res_data["data"]["getDataFilePublicDownloadUrl"].get("url")
    except Exception:
        pass
    return None


def download_image(
    url,
    mapillary_token,
    photo_id=None,
    platform=None,
    offline_dirs=None,
    image_size=448,
    max_retries=3,
):
    """Loads an image locally if it's part of an offline dataset, or downloads it via connection pool with retries and fallbacks."""
    for attempt in range(max_retries):
        try:
            # 1. Try local direct path
            if url and os.path.exists(url):
                img = Image.open(url).convert("RGB")
                img_resized = img.resize((image_size, image_size))
                return img_resized

            # 2. Try resolving via offline directories
            dirs_to_use = offline_dirs if offline_dirs is not None else []
            if dirs_to_use and url:
                resolved = resolve_offline_image_path(
                    url, dirs_to_use, photo_id, platform
                )
                if resolved and os.path.exists(resolved):
                    img = Image.open(resolved).convert("RGB")
                    img_resized = img.resize((image_size, image_size))
                    return img_resized

            # 3. Fallback to download over HTTP
            if url:
                session = _get_http_session()
                # Mapillary schema resolution
                if url.startswith("mapillary://") or (photo_id and "fbcdn.net" in url):
                    orig_id = str(photo_id) if photo_id else url.split("://")[1]
                    api_url = (
                        f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
                    )
                    headers = {"Authorization": f"OAuth {mapillary_token}"}
                    res = session.get(api_url, headers=headers, timeout=10)
                    if res.status_code == 200:
                        url = res.json().get("thumb_1024_url")
                # KartaView schema resolution
                elif url.startswith("kartaview://"):
                    orig_id = url.split("://")[1]
                    api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
                    res = session.get(api_url, timeout=10)
                    if res.status_code == 200:
                        data = res.json().get("result", {}).get("data", {})
                        url = (
                            data.get("fileurlLTh")
                            or data.get("fileurlTh")
                            or data.get("fileurl")
                        )
                # Wildlife Insights schema resolution
                elif "wildlifeinsights.org" in url or (
                    platform
                    and str(platform).lower()
                    in ["snapshotusa", "snapshot_usa", "wildlife_insights"]
                ):
                    url = resolve_wildlife_insights_url(url, photo_id=photo_id)

                if url:
                    res = session.get(url, timeout=10)
                    if res.status_code == 200:
                        c_type = res.headers.get("content-type", "").lower()
                        if "text/html" not in c_type and is_valid_image_file(
                            res.content
                        ):
                            img = Image.open(BytesIO(res.content)).convert("RGB")
                            img_resized = img.resize((image_size, image_size))
                            return img_resized
        except Exception:
            pass

        # If it fails, wait and try schema fallback if applicable
        if attempt < max_retries - 1:
            time.sleep(1.0 * (attempt + 1))
            if platform:
                plat_lower = platform.lower()
                if (
                    plat_lower == "mapillary"
                    and url
                    and not url.startswith("mapillary://")
                ):
                    url = f"mapillary://{photo_id}"
                elif (
                    plat_lower == "kartaview"
                    and url
                    and not url.startswith("kartaview://")
                ):
                    url = f"kartaview://{photo_id}"

    return None
