import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Migrate existing decoupled embeddings to stable photo_key maps.")
    parser.add_argument("--parquet", type=str, required=True, help="Path to the Parquet metadata file.")
    parser.add_argument("--npy", type=str, required=True, help="Path to the .npy embeddings file.")
    parser.add_argument("--update_parquet", action="store_true",
                        help="Add the 'photo_key' column directly to the input Parquet file.")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        raise FileNotFoundError(f"Parquet file not found: {args.parquet}")
    if not os.path.exists(args.npy):
        raise FileNotFoundError(f"Npy file not found: {args.npy}")

    print(f"Loading metadata from: {args.parquet}...")
    df = pd.read_parquet(args.parquet)

    print(f"Loading embeddings shape from: {args.npy}...")
    # Use mmap to avoid loading the massive array into RAM
    emb = np.load(args.npy, mmap_mode="r")
    npy_len = len(emb)
    print(f" -> Found {len(df):,} metadata rows.")
    print(f" -> Found {npy_len:,} embeddings rows in npy.")

    # 1. Ensure Platform and Photo_ID are present to generate stable keys
    if 'Platform' not in df.columns or 'Photo_ID' not in df.columns:
        raise ValueError("Metadata Parquet must contain 'Platform' and 'Photo_ID' columns to generate stable keys.")

    df['photo_key'] = df['Platform'].astype(str) + "_" + df['Photo_ID'].astype(str)

    # 2. Build the master keys list aligned with the npy rows
    print("Building master key index...")
    master_keys = [None] * npy_len

    # Check if df matches the npy length exactly and is aligned
    is_aligned_1to1 = False
    if len(df) == npy_len:
        if 'embedding_idx' not in df.columns:
            is_aligned_1to1 = True
        else:
            is_aligned_1to1 = (df['embedding_idx'] == np.arange(npy_len)).all()

    if is_aligned_1to1:
        print(" -> Parquet and npy are 1-to-1 aligned. Mapping keys directly.")
        master_keys = df['photo_key'].tolist()
    else:
        print(" -> Aligning keys using 'embedding_idx' pointers...")
        if 'embedding_idx' not in df.columns:
            raise ValueError(
                "Parquet row count does not match npy row count, and no 'embedding_idx' column is present to align them.")

        indices = df['embedding_idx'].values
        keys = df['photo_key'].values

        # Populate master keys list
        for idx, key in zip(indices, keys):
            if 0 <= idx < npy_len:
                master_keys[idx] = key
            else:
                print(f"Warning: Index {idx} is out of bounds for npy length {npy_len}. Skipping.")

        # Fill missing values with placeholder/fallback key
        for i in range(npy_len):
            if master_keys[i] is None:
                master_keys[i] = f"unknown_idx_{i}"

    # 3. Save the companion keys file
    keys_df = pd.DataFrame({'photo_key': master_keys})
    keys_path = args.npy.replace(".npy", ".keys.parquet")
    print(f"Saving companion keys index to: {keys_path}...")
    keys_df.to_parquet(keys_path, compression='zstd')

    # 4. Optionally update the original parquet file to include the photo_key column
    if args.update_parquet:
        print(f"Appending 'photo_key' column to original Parquet file: {args.parquet}...")
        df.to_parquet(args.parquet, compression='zstd')
        print(" -> Parquet file successfully updated.")

    print("Migration complete! Stable hash-map association is now active.")


if __name__ == "__main__":
    main()
