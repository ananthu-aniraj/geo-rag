import argparse
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import load_dataframe, save_dataframe


def main():
    parser = argparse.ArgumentParser(description="Decouple existing clustered Parquet database into memory-mapped NumPy arrays and lightweight sidecars.")
    parser.add_argument("--input", type=str, required=True, help="Path to the existing clustered/cleaned Parquet database.")
    parser.add_argument("--k_clusters", type=int, default=None, help="Number of clusters (k) for naming the output sidecar file. Auto-detected if not specified.")
    parser.add_argument("--delete_csv", action="store_true", help="Delete the matching duplicate CSV file if it exists.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    db_dir = os.path.dirname(os.path.abspath(args.input))
    base_file_name = os.path.basename(args.input)
    
    # Auto-detect k_clusters if not provided
    k_clusters = args.k_clusters
    if k_clusters is None:
        match = re.search(r'_k_(\d+)', base_file_name)
        if match:
            k_clusters = int(match.group(1))
            print(f"Auto-detected k_clusters = {k_clusters} from input filename.")
        else:
            k_clusters = 50000  # Default fallback
    
    print(f"Reading input database: {base_file_name}...")
    t0 = pd.Timestamp.now()
    
    # 1. Load schema to inspect columns without loading heavy embeddings first
    parquet_file = pq.ParquetFile(args.input)
    available_cols = parquet_file.schema_arrow.names
    
    embedding_cols = [c for c in ['embedding', 'patch_embedding'] if c in available_cols]
    metadata_cols = [c for c in available_cols if c not in embedding_cols]
    
    # 2. Load the metadata columns into pandas
    print("Loading metadata...")
    df_meta = load_dataframe(args.input, columns=metadata_cols)
    print(f" -> Loaded metadata for {len(df_meta):,} rows.")

    # 3. Load and decouple heavy embeddings into NumPy binary files (.npy)
    # Define filenames based on clean naming matching run_full_pipeline.sh
    # E.g., if input is ${BASE_NAME}_cleaned_clustered_k_X.parquet -> base metadata is ${BASE_NAME}_cleaned
    if "_clustered_k_" in base_file_name:
        meta_name = base_file_name.split("_clustered_k_")[0]
    else:
        meta_name = os.path.splitext(base_file_name)[0]

    # Map cleaned rows to deduplicated row indices if we are processing a cleaned file
    skip_npy = False
    if "cleaned" in meta_name:
        # Check if deduplicated metadata parquet exists in same directory
        dedup_base = meta_name.replace("cleaned", "deduplicated")
        dedup_path = os.path.join(db_dir, f"{dedup_base}.parquet")
        if os.path.exists(dedup_path):
            print(f"\nFound deduplicated metadata at: {dedup_path}. Mapping cleaned row indices...")
            df_dedup = load_dataframe(dedup_path, columns=['Platform', 'Photo_ID'])
            df_dedup['embedding_idx'] = np.arange(len(df_dedup), dtype=np.int32)
            df_meta = df_meta.merge(df_dedup[['Platform', 'Photo_ID', 'embedding_idx']], on=['Platform', 'Photo_ID'], how='left')
            df_meta['embedding_idx'] = df_meta['embedding_idx'].fillna(-1).astype(np.int32)
            print(f" -> Mapped {len(df_meta):,} rows. Skipping cleaned .npy generation to share {dedup_base}.npy.")
            skip_npy = True
    elif "deduplicated" in meta_name:
        # Assign direct sequential embedding indices to deduplicated base metadata
        df_meta['embedding_idx'] = np.arange(len(df_meta), dtype=np.int32)
        print(" -> Assigned sequential embedding_idx to deduplicated dataset.")

    for col in embedding_cols:
        if skip_npy:
            continue
        npy_path = os.path.join(db_dir, f"{meta_name}.npy" if col == "embedding" else f"{meta_name}_{col}.npy")
        print(f"Extracting and saving '{col}' to binary: {npy_path}...")
        
        # Load raw embedding column table
        table = pq.read_table(args.input, columns=[col])
        chunked_arr = table[col]
        num_rows = len(table)
        dim = len(chunked_arr.chunk(0)[0].as_py())
        
        # Allocate contiguous float32 numpy array
        emb_matrix = np.empty((num_rows, dim), dtype=np.float32)
        current_row = 0
        for chunk in chunked_arr.chunks:
            chunk_len = len(chunk)
            flat_chunk = chunk.flatten().to_numpy()
            emb_matrix[current_row:current_row + chunk_len] = flat_chunk.reshape(chunk_len, dim)
            current_row += chunk_len
            
        np.save(npy_path, emb_matrix)
        print(f" -> Saved {emb_matrix.shape} matrix ({os.path.getsize(npy_path)/1024**2:.1f} MB).")
        del emb_matrix
        del table

    # 4. Split and write the cluster sidecar file if cluster_id is present
    cluster_cols = [
        'cluster_id', 'cluster_label', 'cluster_description', 
        'parent_cluster_id', 'parent_cluster_label', 'parent_cluster_description',
        'visual_description', 'parent_visual_description'
    ]
    active_cluster_cols = [c for c in cluster_cols if c in df_meta.columns]

    if active_cluster_cols:
        sidecar_name = f"{meta_name}_clustered_k_{k_clusters}.parquet"
        sidecar_path = os.path.join(db_dir, sidecar_name)
        print(f"\nDecoupling cluster columns into sidecar: {sidecar_name}...")
        
        # Sidecar file only holds keys (Platform, Photo_ID) + cluster parameters
        sidecar_df = df_meta[['Platform', 'Photo_ID'] + active_cluster_cols].copy()
        save_dataframe(sidecar_df, sidecar_path)
        print(f" -> Saved sidecar database ({os.path.getsize(sidecar_path)/1024**2:.1f} MB).")
        
        # Drop the cluster columns from the base metadata df
        df_meta = df_meta.drop(columns=active_cluster_cols)

    # 5. Save the clean base metadata parquet
    base_parquet_path = os.path.join(db_dir, f"{meta_name}.parquet")
    print(f"\nSaving clean base metadata to: {meta_name}.parquet...")
    save_dataframe(df_meta, base_parquet_path)
    print(f" -> Saved base metadata ({os.path.getsize(base_parquet_path)/1024**2:.1f} MB).")

    # 6. Delete duplicate CSV file if requested
    if args.delete_csv:
        csv_file_name = os.path.splitext(base_file_name)[0] + ".csv"
        csv_path = os.path.join(db_dir, csv_file_name)
        if os.path.exists(csv_path):
            print(f"\nDeleting duplicate CSV file: {csv_file_name}...")
            os.remove(csv_path)
            print(" -> Deleted.")
        else:
            print(f"\nNo duplicate CSV file found matching: {csv_file_name}")

    t_diff = pd.Timestamp.now() - t0
    print(f"\n✅ Decoupling completed successfully in {t_diff.total_seconds():.2f}s!")
    print(f"Outputs located in directory: {db_dir}")


if __name__ == "__main__":
    main()
