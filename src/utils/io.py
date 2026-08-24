import glob
import os
import re
import time
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def get_core_base_name(base_name):
    """Recursively strips common pipeline suffixes to find the base prefix (e.g. geo_space)."""
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]
    suffixes = ["_filtered", "_cleaned", "_deduplicated", "_clustered"]
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

        df_to_save = df.copy()
        # Generate stable photo_key using Platform and Photo_ID for all parquet files
        if "Platform" in df_to_save.columns and "Photo_ID" in df_to_save.columns:
            if "photo_key" not in df_to_save.columns:
                df_to_save["photo_key"] = (
                    df_to_save["Platform"].astype(str)
                    + "_"
                    + df_to_save["Photo_ID"].astype(str)
                )

        if "embedding" in df.columns:
            db_dir = os.path.dirname(os.path.abspath(file_path))
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if "_clustered_k_" in base_name:
                base_name = base_name.split("_clustered_k_")[0]
            core_name = get_core_base_name(base_name)

            embs = np.vstack(df["embedding"].values)
            dim = embs.shape[1]

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
            if model_name:
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
                dtype = embs.dtype

            print(
                f" -> Automatically decoupling embeddings to companion file: {npy_path} (dtype={dtype.__name__})"
            )
            np.save(npy_path, embs.astype(dtype))

            if "photo_key" not in df_to_save.columns:
                df_to_save["photo_key"] = "idx_" + np.arange(len(df_to_save)).astype(
                    str
                )

            # Save the companion keys file
            keys_df = pd.DataFrame({"photo_key": df_to_save["photo_key"]})
            keys_path = os.path.join(
                db_dir, f"{core_name}_{rep_suffix}_embeddings.keys.parquet"
            )
            print(f" -> Saving companion keys file to: {keys_path}")
            keys_df.to_parquet(keys_path, compression="zstd")

            df_to_save = df_to_save.drop(
                columns=["embedding", "embedding_idx"], errors="ignore"
            )
            df_to_save.to_parquet(file_path, index=index, **kwargs)
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
    parquet_path, k_clusters=50000, columns=None, representation_type=None, **kwargs
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

    # If it's a decoupled sidecar file (has cluster_id but lacks base columns like Latitude/Longitude),
    # resolve the path to the base metadata file instead.
    if "cluster_id" in schema_names and (
        "Latitude" not in schema_names or "Longitude" not in schema_names
    ):
        db_dir = os.path.dirname(os.path.abspath(parquet_path))
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        core_name = get_core_base_name(base_name)
        for fallback in [
            f"{core_name}_cleaned.parquet",
            f"{core_name}_deduplicated.parquet",
            f"{core_name}.parquet",
        ]:
            fallback_path = os.path.join(db_dir, fallback)
            if os.path.exists(fallback_path):
                parquet_path = fallback_path
                pf = pq.ParquetFile(parquet_path)
                schema_names = pf.schema_arrow.names
                break

    cluster_cols = [
        "cluster_id",
        "cluster_label",
        "cluster_description",
        "parent_cluster_id",
        "parent_cluster_label",
        "parent_cluster_description",
        "visual_description",
        "parent_visual_description",
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

    # Case B: Decoupled format
    df_meta = load_dataframe(parquet_path, columns=req_meta_cols, **kwargs)

    # Check for and load sidecar file
    db_dir = os.path.dirname(os.path.abspath(parquet_path))
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]

    # Trim '_clustered_k_X' suffix if present to find base name
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]

    core_name = get_core_base_name(base_name)

    # Try finding sidecar with full base_name or core_name
    sidecar_path = os.path.join(db_dir, f"{base_name}_clustered_k_{k_clusters}.parquet")
    if not os.path.exists(sidecar_path):
        sidecar_path = os.path.join(
            db_dir, f"{core_name}_clustered_k_{k_clusters}.parquet"
        )

    if os.path.exists(sidecar_path) and req_cluster_cols:
        # We need Platform and Photo_ID in both dataframes for merging
        sidecar_cols = list(set(["Platform", "Photo_ID"] + req_cluster_cols))
        # Ensure we only load available columns from the sidecar
        pf_side = pq.ParquetFile(sidecar_path)
        side_avail_cols = [c for c in sidecar_cols if c in pf_side.schema_arrow.names]

        df_sidecar = load_dataframe(sidecar_path, columns=side_avail_cols, **kwargs)
        df_meta = df_meta.merge(df_sidecar, on=["Platform", "Photo_ID"], how="left")

    return df_meta


def load_embeddings(
    parquet_path, column="embedding", representation_type="cls", model_name=None
):
    """
    Backward-compatible loader that returns memory-mapped or raw embedding matrices.
    Supports dynamic mapping lookup via 'embedding_idx' to load from a shared base file.
    """

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    if column in pf.schema_arrow.names:
        # Case A: Combined format (read via pyarrow table and stack)
        table = pf.read(columns=[column])
        chunked_arr = table[column]
        num_rows = len(table)
        dim = len(chunked_arr.chunk(0)[0].as_py())

        emb_matrix = np.empty((num_rows, dim), dtype=np.float32)
        current_row = 0
        for chunk in chunked_arr.chunks:
            chunk_len = len(chunk)
            flat_chunk = chunk.flatten().to_numpy()
            emb_matrix[current_row : current_row + chunk_len] = flat_chunk.reshape(
                chunk_len, dim
            )
            current_row += chunk_len
        return emb_matrix

    # Case B: Decoupled format (.npy)
    db_dir = os.path.dirname(os.path.abspath(parquet_path))
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]

    # Trim '_clustered_k_X' suffix if present to find base name
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]

    def get_npy_path(base, model):
        model_suf = f"_{model.replace('/', '_')}" if model else ""
        if column == "embedding":
            name = f"{base}{model_suf}_{representation_type}_embeddings.npy"
        elif column == "patch_embedding":
            name = f"{base}{model_suf}_patch_embeddings.npy"
        else:
            name = f"{base}{model_suf}_{column}_embeddings.npy"
        return os.path.join(db_dir, name)

    # 1. Try resolving with model_name if provided
    npy_path = None
    if model_name:
        for b in [base_name, get_core_base_name(base_name)]:
            path = get_npy_path(b, model_name)
            if os.path.exists(path):
                npy_path = path
                break

    # 2. Fallback to model-agnostic (legacy) path resolution
    if not npy_path:
        for b in [base_name, get_core_base_name(base_name)]:
            path = get_npy_path(b, None)
            if os.path.exists(path) or b == base_name:
                npy_path = path
                break

    def suffix_matches(filename, req_rep):
        has_cls_avg = "cls_avg_patch" in filename
        has_avg = "avg_patch" in filename and not has_cls_avg
        has_cls = "cls" in filename and not has_cls_avg

        if req_rep == "cls_avg_patch":
            return has_cls_avg
        elif req_rep == "avg_patch":
            return has_avg
        elif req_rep == "cls":
            # Allow fallback if the filename doesn't contain any known suffix
            if not has_cls_avg and not has_avg:
                return True
            return has_cls
        return True

    # Fallback: check for shared deduplicated.npy if base file is cleaned.parquet
    if not os.path.exists(npy_path) and "cleaned" in base_name:
        fallback_base = base_name.replace("cleaned", "deduplicated")
        fallback_name = (
            f"{fallback_base}.npy"
            if column == "embedding"
            else f"{fallback_base}_{column}.npy"
        )
        npy_path = os.path.join(db_dir, fallback_name)

    # Wildcard search fallback for different column suffixes (e.g. cls_embeddings)
    if not os.path.exists(npy_path):
        core_name = get_core_base_name(base_name)
        bases = [base_name, core_name]
        if "cleaned" in base_name:
            bases.append(base_name.replace("cleaned", "deduplicated"))

        for b in bases:
            pattern = os.path.join(db_dir, f"{b}*.npy")
            matches = glob.glob(pattern)
            if matches:
                # Filter matches by representation type if default 'embedding' column is requested
                if column == "embedding" and representation_type:
                    matches = [
                        m
                        for m in matches
                        if suffix_matches(os.path.basename(m), representation_type)
                    ]
                if not matches:
                    continue

                # If there's a file matching the specific column name, use it
                col_match = [m for m in matches if column in os.path.basename(m)]
                if col_match:
                    npy_path = col_match[0]
                    break
                # Otherwise, if default 'embedding' was requested, try finding cls_embeddings or similar
                if column == "embedding":
                    preferred = [
                        m
                        for m in matches
                        if "cls_embeddings" in os.path.basename(m)
                        or "embedding" in os.path.basename(m)
                    ]
                    if preferred:
                        npy_path = preferred[0]
                        break
                # Fallback to the first match
                npy_path = matches[0]
                break

    if os.path.exists(npy_path):
        emb = np.load(npy_path, mmap_mode="r")

        # If stable photo_key or platform/ID is available, and companion keys file exists, use it
        keys_path = npy_path.replace(".npy", ".keys.parquet")
        has_photo_key = "photo_key" in pf.schema_arrow.names
        has_platform_and_id = (
            "Platform" in pf.schema_arrow.names and "Photo_ID" in pf.schema_arrow.names
        )

        if (has_photo_key or has_platform_and_id) and os.path.exists(keys_path):
            print(f" -> Resolving embeddings via companion keys index: {keys_path}")
            if has_photo_key:
                meta_keys_table = pf.read(columns=["photo_key"])
                meta_keys = (
                    meta_keys_table["photo_key"]
                    .to_pandas()
                    .astype(str)
                    .str.lower()
                    .values
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
                # If keys contain duplicates, resolve each key to its first occurrence in the matrix
                pos_series = pd.Series(np.arange(len(master_keys)), index=master_keys)
                pos_series = pos_series[~pos_series.index.duplicated(keep="first")]
                indices = pos_series.reindex(meta_keys, fill_value=-1).values

            valid_mask = indices >= 0
            if not valid_mask.all():
                print(
                    f"Warning: Found {np.sum(~valid_mask):,} missing keys in embeddings keys index. Zero-filling..."
                )
                safe_indices = np.clip(indices, 0, len(emb) - 1)
                sliced_emb = emb[safe_indices].astype(np.float32)
                sliced_emb[~valid_mask] = 0.0
                return sliced_emb

            return emb[indices].astype(np.float32)

        # Fallback to older embedding_idx mapping logic
        has_embedding_idx = "embedding_idx" in pf.schema_arrow.names
        pf_for_idx = pf

        if not has_embedding_idx and "Latitude" not in pf.schema_arrow.names:
            # Resolve to base metadata file if this is a sidecar file
            core_name = get_core_base_name(base_name)
            for fallback in [
                f"{base_name}_cleaned.parquet",
                f"{core_name}_cleaned.parquet",
                f"{base_name}_deduplicated.parquet",
                f"{core_name}_deduplicated.parquet",
                f"{base_name}.parquet",
                f"{core_name}.parquet",
            ]:
                fallback_path = os.path.join(db_dir, fallback)
                if os.path.exists(fallback_path):
                    pf_base = pq.ParquetFile(fallback_path)
                    if "embedding_idx" in pf_base.schema_arrow.names:
                        has_embedding_idx = True
                        pf_for_idx = pf_base
                    break

        if has_embedding_idx:
            idx_table = pf_for_idx.read(columns=["embedding_idx"])
            indices = idx_table["embedding_idx"].to_numpy()

            # Check bounds safety against the actual loaded matrix
            valid_mask = (indices >= 0) & (indices < len(emb))
            if not valid_mask.all():
                print(
                    f"Warning: Found {np.sum(~valid_mask):,} out-of-bounds indices in embedding_idx. Clamping and zero-filling..."
                )
                safe_indices = np.clip(indices, 0, len(emb) - 1)
                sliced_emb = emb[safe_indices].astype(np.float32)
                sliced_emb[~valid_mask] = 0.0
                return sliced_emb

            return emb[indices].astype(np.float32)

        return emb.astype(np.float32)

    raise FileNotFoundError(
        f"Could not locate embeddings in parquet schema or matching '{base_name}' in '{db_dir}'"
    )


def resolve_offline_image_path(url, image_root_dirs, photo_id=None, platform=None):
    """
    Resolves an image URL/ID to a local path on disk by checking flat files,
    train/ folders, and nested GLDv2 directory structures.
    Returns the absolute path if found, otherwise None.
    """
    if not image_root_dirs:
        return None

    dirs = [image_root_dirs] if isinstance(image_root_dirs, str) else image_root_dirs
    for d in dirs:
        if not d:
            continue

        # 0. Try direct relative path join first
        p_direct = os.path.join(d, url)
        if os.path.exists(p_direct):
            return os.path.abspath(p_direct)

        # A. Try direct lookup using Photo_ID (flat file, train/ folder, or nested)
        if photo_id:
            photo_str = str(photo_id).strip()
            if photo_str.endswith(".0"):
                photo_str = photo_str[:-2]
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                # 1. Platform-specific subfolder lookup (e.g. d/platform/photo_id.jpg)
                if platform:
                    plat_str = str(platform).strip().lower()
                    p_plat = os.path.join(d, plat_str, f"{photo_str}{ext}")
                    if os.path.exists(p_plat):
                        return os.path.abspath(p_plat)

                # 2. Flat lookup
                p_flat = os.path.join(d, f"{photo_str}{ext}")
                if os.path.exists(p_flat):
                    return os.path.abspath(p_flat)

                # 2. train/ flat lookup
                p_train = os.path.join(d, "train", f"{photo_str}{ext}")
                if os.path.exists(p_train):
                    return os.path.abspath(p_train)

                # 3. Nested GLDv2 lookup (e.g. d/a/b/c/id.jpg)
                if len(photo_str) == 16:
                    p_nested = os.path.join(
                        d, photo_str[0], photo_str[1], photo_str[2], f"{photo_str}{ext}"
                    )
                    if os.path.exists(p_nested):
                        return os.path.abspath(p_nested)
                    p_nested_train = os.path.join(
                        d,
                        "train",
                        photo_str[0],
                        photo_str[1],
                        photo_str[2],
                        f"{photo_str}{ext}",
                    )
                    if os.path.exists(p_nested_train):
                        return os.path.abspath(p_nested_train)

        # B. Fallback to URL basename lookup
        # Strip protocol if present
        clean_url = url
        if "://" in url:
            clean_url = url.split("://")[1]
        basename = os.path.basename(clean_url)

        basenames = [basename]
        if "." not in basename:
            basenames.extend(
                [
                    f"{basename}{ext}"
                    for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]
                ]
            )

        for b in basenames:
            p_base = os.path.join(d, b)
            if os.path.exists(p_base):
                return os.path.abspath(p_base)
            p_base_train = os.path.join(d, "train", b)
            if os.path.exists(p_base_train):
                return os.path.abspath(p_base_train)

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

                if url:
                    res = session.get(url, timeout=10)
                    if res.status_code == 200:
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
