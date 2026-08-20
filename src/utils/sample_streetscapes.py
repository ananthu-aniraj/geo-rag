import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


def build_city_h3_groups(simplemaps_path, h3_path):
    """
    Loads simplemaps.csv and h3.csv columns, merges them on uuid,
    and returns a DataFrame of ['uuid', 'city_id', 'h3_8'].
    """
    print("Loading and merging city mappings and H3 indexes...")
    start_time = time.time()

    # Load only necessary columns with memory-optimized dtypes
    m = pd.read_csv(
        simplemaps_path,
        usecols=["uuid", "city_id"],
        dtype={"uuid": "string", "city_id": "category"},
    )
    h = pd.read_csv(
        h3_path, usecols=["uuid", "h3_8"], dtype={"uuid": "string", "h3_8": "category"}
    )

    # Merge on uuid
    merged = pd.merge(m, h, on="uuid")
    print(
        f"Loaded and merged {len(merged):,} rows in {time.time() - start_time:.2f} seconds."
    )
    return merged


def sample_geographically(df_merged, max_per_city=4000):
    """
    Samples at most max_per_city images for each city, distributing the sample
    evenly across the H3 cells (round-robin) to maximize geographic spread.
    """
    print(
        f"Performing geographically stratified sampling (max {max_per_city} per city)..."
    )
    start_time = time.time()

    selected_uuids = []

    # Group by city_id
    city_groups = df_merged.groupby("city_id", observed=True)

    for city_id, grp in city_groups:
        if len(grp) <= max_per_city:
            selected_uuids.extend(grp["uuid"].tolist())
            continue

        # Group this city's rows by H3 cell
        h3_groups = grp.groupby("h3_8", observed=True)

        # Build lists of UUIDs per cell
        group_lists = []
        for cell, cell_grp in h3_groups:
            uuids = cell_grp["uuid"].tolist()
            # Shuffle locally to avoid sequence ordering bias
            np.random.seed(42)
            np.random.shuffle(uuids)
            group_lists.append(uuids)

        # Round-robin sampling
        city_sampled = []
        while len(city_sampled) < max_per_city and group_lists:
            # Keep only non-empty lists
            group_lists = [g for g in group_lists if len(g) > 0]
            if not group_lists:
                break

            for g in group_lists:
                if len(city_sampled) >= max_per_city:
                    break
                city_sampled.append(g.pop(0))

        selected_uuids.extend(city_sampled)

    selected_uuids_set = set(selected_uuids)
    print(f"Sampling complete in {time.time() - start_time:.2f} seconds.")
    print(f"Total selected UUIDs: {len(selected_uuids_set):,}")
    return selected_uuids_set


def filter_metadata(metadata_path, selected_uuids, output_path, chunk_size=1000000):
    """
    Filters metadata_common_attributes.csv in chunks to keep only selected UUIDs
    and writes the final balanced CSV.
    """
    print(f"Filtering metadata sheet and writing output to {output_path}...")
    start_time = time.time()

    if os.path.exists(output_path):
        os.remove(output_path)

    chunks = pd.read_csv(metadata_path, chunksize=chunk_size, dtype=str)

    cols_to_keep = ["uuid", "source", "orig_id", "lat", "lon", "datetime_local"]
    total_written = 0

    chunk_idx = 0
    for chunk in chunks:
        # Filter rows belonging to selected UUIDs
        filtered_chunk = chunk[chunk["uuid"].isin(selected_uuids)]

        if not filtered_chunk.empty:
            filtered_chunk = filtered_chunk[cols_to_keep]

            # Write to CSV
            mode = "w" if total_written == 0 else "a"
            header = total_written == 0
            filtered_chunk.to_csv(output_path, mode=mode, header=header, index=False)
            total_written += len(filtered_chunk)

        chunk_idx += 1
        print(
            f"  Processed {chunk_idx * chunk_size:,} metadata rows... (Written: {total_written:,})"
        )

    print(
        f"Finished writing in {time.time() - start_time:.2f} seconds. Output contains {total_written:,} records."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced, geographically diverse subset of Global Streetscapes."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/user/aaniraj/home/Documents/Projects/data/global-streetscapes",
        help="Path to the global-streetscapes directory containing metadata files.",
    )
    parser.add_argument(
        "--max_per_city",
        type=int,
        default=4000,
        help="Maximum images to select per city (default: 4000).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="streetscapes_balanced_subset.csv",
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    simplemaps_path = os.path.join(args.base_dir, "simplemaps.csv")
    h3_path = os.path.join(args.base_dir, "h3.csv")
    metadata_path = os.path.join(args.base_dir, "metadata_common_attributes.csv")

    for p in [simplemaps_path, h3_path, metadata_path]:
        if not os.path.exists(p):
            print(f"Error: Required file '{p}' not found.")
            sys.exit(1)

    df_merged = build_city_h3_groups(simplemaps_path, h3_path)
    selected_uuids = sample_geographically(df_merged, args.max_per_city)
    filter_metadata(metadata_path, selected_uuids, args.out)


if __name__ == "__main__":
    main()
