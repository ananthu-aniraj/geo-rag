import os

import pandas as pd
import pyarrow.parquet as pq


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


def save_dataframe(df, file_path, index=False, **kwargs):
    """
    Saves a dataframe to CSV, Parquet, or Pickle with optimal compression default (Zstd for Parquet).
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


def load_dataset_with_clusters(parquet_path, k_clusters=50000, columns=None, **kwargs):
    """
    Backward-compatible loader that returns metadata and cluster assignments.
    Merges sidecars automatically if the base parquet does not contain cluster columns.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    # Inspect schema names using PyArrow ParquetFile
    pf = pq.ParquetFile(parquet_path)
    schema_names = pf.schema_arrow.names

    cluster_cols = [
        'cluster_id', 'cluster_label', 'cluster_description', 
        'parent_cluster_id', 'parent_cluster_label', 'parent_cluster_description',
        'visual_description', 'parent_visual_description'
    ]

    # Filter which columns belong to cluster variables vs base metadata
    requested_cols = columns if columns is not None else schema_names
    req_cluster_cols = [c for c in requested_cols if c in cluster_cols]
    req_meta_cols = [c for c in requested_cols if c not in cluster_cols or c in ['Platform', 'Photo_ID']]

    # Case A: Old format contains cluster columns directly
    if 'cluster_id' in schema_names:
        return load_dataframe(parquet_path, columns=columns, **kwargs)

    # Case B: Decoupled format
    df_meta = load_dataframe(parquet_path, columns=req_meta_cols, **kwargs)

    # Check for and load sidecar file
    db_dir = os.path.dirname(os.path.abspath(parquet_path))
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]
    
    # Trim '_clustered_k_X' suffix if present to find base name
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]
        
    sidecar_path = os.path.join(db_dir, f"{base_name}_clustered_k_{k_clusters}.parquet")

    if os.path.exists(sidecar_path) and req_cluster_cols:
        # We need Platform and Photo_ID in both dataframes for merging
        sidecar_cols = list(set(['Platform', 'Photo_ID'] + req_cluster_cols))
        # Ensure we only load available columns from the sidecar
        pf_side = pq.ParquetFile(sidecar_path)
        side_avail_cols = [c for c in sidecar_cols if c in pf_side.schema_arrow.names]
        
        df_sidecar = load_dataframe(sidecar_path, columns=side_avail_cols, **kwargs)
        df_meta = df_meta.merge(df_sidecar, on=['Platform', 'Photo_ID'], how='left')

    return df_meta


def load_embeddings(parquet_path, column='embedding'):
    """
    Backward-compatible loader that returns memory-mapped or raw embedding matrices.
    Supports dynamic mapping lookup via 'embedding_idx' to load from a shared base file.
    """
    import numpy as np
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    if column in pf.schema_arrow.names:
        # Case A: Combined format (read via pyarrow table and stack)
        table = pf.read_table(columns=[column])
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
        
    npy_name = f"{base_name}.npy" if column == 'embedding' else f"{base_name}_{column}.npy"
    npy_path = os.path.join(db_dir, npy_name)

    # Fallback: check for shared deduplicated.npy if base file is cleaned.parquet
    if not os.path.exists(npy_path) and "cleaned" in base_name:
        fallback_base = base_name.replace("cleaned", "deduplicated")
        fallback_name = f"{fallback_base}.npy" if column == 'embedding' else f"{fallback_base}_{column}.npy"
        npy_path = os.path.join(db_dir, fallback_name)

    if os.path.exists(npy_path):
        emb = np.load(npy_path, mmap_mode="r")
        # If 'embedding_idx' is in the parquet columns, map indices dynamically
        if 'embedding_idx' in pf.schema_arrow.names:
            idx_table = pf.read_table(columns=['embedding_idx'])
            indices = idx_table['embedding_idx'].to_numpy()
            return emb[indices]
        return emb

    raise FileNotFoundError(f"Could not locate embeddings in parquet schema or at '{npy_path}'")
