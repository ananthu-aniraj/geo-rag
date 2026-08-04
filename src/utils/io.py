import glob
import os
import re
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import yaml
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
                base_name = base_name[:-len(suffix)]
                changed = True
    return base_name


def load_dataframe(file_path, **kwargs):
    """
    Loads a dataframe from CSV, Parquet, or Pickle files dynamically based on extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    elif ext == '.csv':
        # Default low_memory=False for safety on mixed-type data
        if 'low_memory' not in kwargs:
            kwargs['low_memory'] = False
        return pd.read_csv(file_path, **kwargs)
    elif ext in ('.pkl', '.pickle'):
        return pd.read_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for loading dataframe.")


def save_dataframe(df, file_path, index=False, representation_type=None, precision=None, **kwargs):
    """
    Saves a dataframe to CSV, Parquet, or Pickle with optimal compression default (Zstd for Parquet).
    Automatically decouples embeddings into a companion .npy file if the column is present.
    """
    # Ensure output parent directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.parquet':
        # Default to high-performance zstd compression
        if 'compression' not in kwargs:
            kwargs['compression'] = 'zstd'

        if 'embedding' in df.columns:
            db_dir = os.path.dirname(os.path.abspath(file_path))
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if "_clustered_k_" in base_name:
                base_name = base_name.split("_clustered_k_")[0]
            core_name = get_core_base_name(base_name)

            embs = np.vstack(df['embedding'].values)
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

            npy_path = os.path.join(db_dir, f"{core_name}_{rep_suffix}_embeddings.npy")

            # 2. Resolve precision dtype
            dtype = np.float32
            if precision == 'float16':
                dtype = np.float16
            elif precision == 'float32':
                dtype = np.float32
            else:
                dtype = embs.dtype

            print(f" -> Automatically decoupling embeddings to companion file: {npy_path} (dtype={dtype.__name__})")
            np.save(npy_path, embs.astype(dtype))

            df_to_save = df.copy()
            df_to_save['embedding_idx'] = np.arange(len(df_to_save), dtype=np.int32)
            df_to_save = df_to_save.drop(columns=['embedding'])
            df_to_save.to_parquet(file_path, index=index, **kwargs)
            return

        df.to_parquet(file_path, index=index, **kwargs)
    elif ext == '.csv':
        df.to_csv(file_path, index=index, **kwargs)
    elif ext in ('.pkl', '.pickle'):
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

    if 'compression' not in kwargs:
        kwargs['compression'] = 'zstd'

    return pq.ParquetWriter(file_path, schema, **kwargs)


def load_dataset_with_clusters(parquet_path, k_clusters=50000, columns=None, representation_type=None, **kwargs):
    """
    Backward-compatible loader that returns metadata and cluster assignments.
    Merges sidecars automatically if the base parquet does not contain cluster columns.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    # Auto-detect k_clusters if the input filename contains it
    match = re.search(r'_clustered_k_(\d+)', os.path.basename(parquet_path))
    if match:
        k_clusters = int(match.group(1))

    # Auto-extract k_clusters from filename if it has _k_X suffix
    match = re.search(r'_k_(\d+)', os.path.basename(parquet_path))
    if match:
        k_clusters = int(match.group(1))

    # Inspect schema names using PyArrow ParquetFile
    pf = pq.ParquetFile(parquet_path)
    schema_names = pf.schema_arrow.names

    # If it's a decoupled sidecar file (has cluster_id but lacks base columns like Latitude/Longitude),
    # resolve the path to the base metadata file instead.
    if 'cluster_id' in schema_names and ('Latitude' not in schema_names or 'Longitude' not in schema_names):
        db_dir = os.path.dirname(os.path.abspath(parquet_path))
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        core_name = get_core_base_name(base_name)
        for fallback in [
            f"{core_name}_cleaned.parquet", f"{core_name}_deduplicated.parquet", f"{core_name}.parquet"
        ]:
            fallback_path = os.path.join(db_dir, fallback)
            if os.path.exists(fallback_path):
                parquet_path = fallback_path
                pf = pq.ParquetFile(parquet_path)
                schema_names = pf.schema_arrow.names
                break

    cluster_cols = [
        'cluster_id', 'cluster_label', 'cluster_description',
        'parent_cluster_id', 'parent_cluster_label', 'parent_cluster_description',
        'visual_description', 'parent_visual_description'
    ]

    # Filter which columns belong to cluster variables vs base metadata
    if columns is None:
        requested_cols = list(schema_names) + cluster_cols
    else:
        requested_cols = columns

    req_cluster_cols = [c for c in requested_cols if c in cluster_cols]
    req_meta_cols = [c for c in requested_cols if c not in cluster_cols or c in ['Platform', 'Photo_ID']]

    # Case A: Old format contains cluster columns directly (or we failed to fallback to base)
    if 'cluster_id' in schema_names and 'Latitude' in schema_names:
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
        sidecar_path = os.path.join(db_dir, f"{core_name}_clustered_k_{k_clusters}.parquet")

    if os.path.exists(sidecar_path) and req_cluster_cols:
        # We need Platform and Photo_ID in both dataframes for merging
        sidecar_cols = list(set(['Platform', 'Photo_ID'] + req_cluster_cols))
        # Ensure we only load available columns from the sidecar
        pf_side = pq.ParquetFile(sidecar_path)
        side_avail_cols = [c for c in sidecar_cols if c in pf_side.schema_arrow.names]

        df_sidecar = load_dataframe(sidecar_path, columns=side_avail_cols, **kwargs)
        df_meta = df_meta.merge(df_sidecar, on=['Platform', 'Photo_ID'], how='left')

    return df_meta


def load_embeddings(parquet_path, column='embedding', representation_type='cls'):
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
            emb_matrix[current_row:current_row + chunk_len] = flat_chunk.reshape(chunk_len, dim)
            current_row += chunk_len
        return emb_matrix

    # Case B: Decoupled format (.npy)
    db_dir = os.path.dirname(os.path.abspath(parquet_path))
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]

    # Trim '_clustered_k_X' suffix if present to find base name
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]

    core_name = get_core_base_name(base_name)
    if column == 'embedding':
        suffix = representation_type
        npy_name = f"{core_name}_{suffix}_embeddings.npy"
    elif column == 'patch_embedding':
        npy_name = f"{core_name}_patch_embeddings.npy"
    else:
        npy_name = f"{core_name}_{column}_embeddings.npy"
    npy_path = os.path.join(db_dir, npy_name)

    # Fallback: check for shared deduplicated.npy if base file is cleaned.parquet
    if not os.path.exists(npy_path) and "cleaned" in base_name:
        fallback_base = base_name.replace("cleaned", "deduplicated")
        fallback_name = f"{fallback_base}.npy" if column == 'embedding' else f"{fallback_base}_{column}.npy"
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
                # If there's a file matching the specific column name, use it
                col_match = [m for m in matches if column in os.path.basename(m)]
                if col_match:
                    npy_path = col_match[0]
                    break
                # Otherwise, if default 'embedding' was requested, try finding cls_embeddings or similar
                if column == 'embedding':
                    preferred = [m for m in matches if
                                 'cls_embeddings' in os.path.basename(m) or 'embedding' in os.path.basename(m)]
                    if preferred:
                        npy_path = preferred[0]
                        break
                # Fallback to the first match
                npy_path = matches[0]
                break

    if os.path.exists(npy_path):
        emb = np.load(npy_path, mmap_mode="r")
        # If 'embedding_idx' is in the parquet columns, map indices dynamically
        if 'embedding_idx' in pf.schema_arrow.names:
            idx_table = pf.read(columns=['embedding_idx'])
            indices = idx_table['embedding_idx'].to_numpy()
            return emb[indices].astype(np.float32)
        return emb.astype(np.float32)

    raise FileNotFoundError(f"Could not locate embeddings in parquet schema or matching '{base_name}' in '{db_dir}'")


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
            if photo_str.endswith('.0'):
                photo_str = photo_str[:-2]
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG']:
                # 1. Flat lookup
                p_flat = os.path.join(d, f"{photo_str}{ext}")
                if os.path.exists(p_flat):
                    return os.path.abspath(p_flat)

                # 2. train/ flat lookup
                p_train = os.path.join(d, "train", f"{photo_str}{ext}")
                if os.path.exists(p_train):
                    return os.path.abspath(p_train)

                # 3. Nested GLDv2 lookup (e.g. d/a/b/c/id.jpg)
                if len(photo_str) == 16:
                    p_nested = os.path.join(d, photo_str[0], photo_str[1], photo_str[2], f"{photo_str}{ext}")
                    if os.path.exists(p_nested):
                        return os.path.abspath(p_nested)
                    p_nested_train = os.path.join(d, "train", photo_str[0], photo_str[1], photo_str[2],
                                                  f"{photo_str}{ext}")
                    if os.path.exists(p_nested_train):
                        return os.path.abspath(p_nested_train)

        # B. Fallback to URL basename lookup
        # Strip protocol if present
        clean_url = url
        if "://" in url:
            clean_url = url.split("://")[1]
        basename = os.path.basename(clean_url)

        basenames = [basename]
        if '.' not in basename:
            basenames.extend([f"{basename}{ext}" for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG']])

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
_MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


def _get_http_session():
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        )
        _http_session.mount("https://", adapter)
        _http_session.mount("http://", adapter)
    return _http_session


def download_image(url, photo_id=None, platform=None, offline_dirs=None, image_size=448):
    """Loads an image locally if it's part of an offline dataset, or downloads it via connection pool."""
    # 1. Try local direct path
    if url and os.path.exists(url):
        try:
            img = Image.open(url).convert("RGB")
            img_resized = img.resize((image_size, image_size))
            return img_resized
        except Exception:
            pass

    # 2. Try resolving via offline directories
    dirs_to_use = offline_dirs
    if dirs_to_use is None:
        dirs_to_use = []
        if os.path.exists("params.yaml"):
            try:
                with open("params.yaml", "r") as f:
                    params = yaml.safe_load(f)
                pipeline_params = params.get("pipeline", {}) if isinstance(params, dict) else {}
                offline_dirs_str = pipeline_params.get("offline_dataset_dirs", "") if isinstance(pipeline_params,
                                                                                                 dict) else ""
                if not offline_dirs_str and isinstance(params, dict):
                    offline_dirs_str = params.get("offline_dataset_dirs", "")
                if offline_dirs_str:
                    dirs_to_use = [d.strip() for d in offline_dirs_str.split() if d.strip()]
            except Exception:
                pass

    if dirs_to_use and url:
        try:
            resolved = resolve_offline_image_path(url, dirs_to_use, photo_id, platform)
            if resolved and os.path.exists(resolved):
                img = Image.open(resolved).convert("RGB")
                img_resized = img.resize((image_size, image_size))
                return img_resized
        except Exception:
            pass

    # 3. Fallback to download over HTTP
    try:
        session = _get_http_session()
        if url.startswith("mapillary://") or (photo_id and "fbcdn.net" in url):
            orig_id = str(photo_id) if photo_id else url.split("://")[1]
            api_url = f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {_MAPILLARY_TOKEN}"}
            res = session.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
        elif url.startswith("kartaview://"):
            orig_id = url.split("://")[1]
            api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
            res = session.get(api_url, timeout=10)
            if res.status_code == 200:
                data = res.json().get("result", {}).get("data", {})
                url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")

        if not url:
            return None

        res = session.get(url, timeout=10)
        if res.status_code == 200:
            img = Image.open(BytesIO(res.content)).convert("RGB")
            img_resized = img.resize((image_size, image_size))
            img.close()
            return img_resized
    except Exception:
        pass
    return None
