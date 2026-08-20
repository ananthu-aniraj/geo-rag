import argparse
import os
import random
import sys

import h3
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.io import load_dataframe, save_dataframe


def main():
    parser = argparse.ArgumentParser(
        description="Stratified Spatial Sampling of Geolocated Datasets using H3 Index."
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to input geolocated CSV file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the sampled CSV. If not specified, appends resolution/max suffixes.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=11,
        help="H3 spatial resolution for grid partitioning (0-15).",
    )
    parser.add_argument(
        "--max_per_cell",
        type=int,
        default=1,
        help="Maximum number of images to sample per H3 cell.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling reproducibility."
    )
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.csv_path):
        print(f"Error: Input CSV file '{args.csv_path}' not found.")
        sys.exit(1)

    print(f"Loading geolocated metadata from: {args.csv_path}...")
    df = load_dataframe(args.csv_path)
    original_count = len(df)
    print(f"Total rows in original dataset: {original_count:,}")

    # 1. Identify coordinate columns
    lat_col = next(
        (col for col in df.columns if col.lower() in ["latitude", "lat"]), None
    )
    lon_col = next(
        (col for col in df.columns if col.lower() in ["longitude", "lon", "lng"]), None
    )

    if not lat_col or not lon_col:
        print("Error: Could not locate Latitude/Longitude columns in the CSV.")
        sys.exit(1)

    print(f"Found coordinate columns: Lat='{lat_col}', Lon='{lon_col}'")

    # Drop rows with invalid coordinates
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    valid_coord_count = len(df)

    if valid_coord_count < original_count:
        print(
            f"Dropped {original_count - valid_coord_count:,} rows with missing or invalid coordinates."
        )

    if len(df) == 0:
        print("Error: No valid geolocated rows to sample.")
        sys.exit(0)

    # 2. Compute H3 cells
    print(f"Indexing coordinates into H3 grid at resolution {args.resolution}...")
    h3_cells = []
    for lat, lon in tqdm(
        zip(df[lat_col], df[lon_col]), total=len(df), desc="H3 Hashing"
    ):
        try:
            cell = h3.latlng_to_cell(lat, lon, args.resolution)
            h3_cells.append(cell)
        except Exception:
            h3_cells.append(None)

    df["h3_cell"] = h3_cells
    df = df.dropna(subset=["h3_cell"]).reset_index(drop=True)
    unique_cells = df["h3_cell"].nunique()
    print(
        f"Mapped rows to {unique_cells:,} unique H3 cells at resolution {args.resolution}."
    )

    # 3. Perform stratified sampling per cell
    print(f"Applying spatial stratified sampling (max_per_cell={args.max_per_cell})...")

    # Custom sampling function per group
    def sample_group(group):
        if len(group) <= args.max_per_cell:
            return group
        return group.sample(n=args.max_per_cell, random_state=args.seed)

    sampled_df = (
        df.groupby("h3_cell", group_keys=False)
        .apply(sample_group)
        .reset_index(drop=True)
    )

    # Drop temporary column if desired, or keep for transparency
    # Let's keep it so user knows which cell it belonged to

    final_count = len(sampled_df)
    reduction_pct = (1.0 - (final_count / original_count)) * 100.0
    print("\nSampling Summary:")
    print(f"- Original dataset: {original_count:,} rows")
    print(f"- Unique spatial cells: {unique_cells:,} cells")
    print(f"- Sampled dataset: {final_count:,} rows (reduced by {reduction_pct:.2f}%)")
    print(f"- Average density: {final_count / unique_cells:.2f} rows per active cell")

    # 4. Resolve output path
    if not args.output_path:
        base, ext = os.path.splitext(args.csv_path)
        args.output_path = (
            f"{base}_sampled_h3_res{args.resolution}_max{args.max_per_cell}{ext}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    save_dataframe(sampled_df, args.output_path)
    print(
        f"\nSampled dataset saved successfully to: {os.path.abspath(args.output_path)}"
    )


if __name__ == "__main__":
    main()
