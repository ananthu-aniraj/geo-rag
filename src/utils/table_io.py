"""Table read/write helpers, including the embedding companion writer."""

import gc
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.utils.companion_paths import get_core_base_name


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


def _save_dataframe(
    df,
    file_path,
    index=False,
    representation_type=None,
    precision=None,
    model_name=None,
    core_name_resolver=None,
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
            df_to_save = df.drop(
                columns=["embedding", "embedding_idx"], errors="ignore"
            )
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
            core_name = (
                base_name if "_filtered" in base_name else core_name_resolver(base_name)
            )

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
                db_dir, f"{core_name}{model_suffix}_{rep_suffix}_embeddings.npy"
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


def save_dataframe(
    df,
    file_path,
    index=False,
    representation_type=None,
    precision=None,
    model_name=None,
    **kwargs,
):
    """Save a dataframe and any embedding companion files."""
    return _save_dataframe(
        df,
        file_path,
        index=index,
        representation_type=representation_type,
        precision=precision,
        model_name=model_name,
        core_name_resolver=get_core_base_name,
        **kwargs,
    )


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
