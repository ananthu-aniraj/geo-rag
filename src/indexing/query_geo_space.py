import argparse
import os
import sys

import h3
import pandas as pd


def format_cell_output(df_cell, target_cell, res, total_images):
    print("\n" + "=" * 80)
    print(f"📍 QUERY RESULT FOR H3 CELL: {target_cell} (Resolution {res})")
    print(f"📊 Total street-level images observed in this area: {total_images}")
    print("=" * 80)

    # Sort by count descending
    df_sorted = df_cell.sort_values(by='image_count', ascending=False)

    print(f"{'CLUSTER LABEL':<35} | {'PERCENTAGE':<10} | {'IMAGE COUNT':<12}")
    print("-" * 80)
    for _, row in df_sorted.iterrows():
        pct = (row['image_count'] / total_images) * 100
        print(f"{row['cluster_label'][:35]:<35} | {pct:>9.2f}% | {row['image_count']:>11}")

    print("\n" + "-" * 80)
    print("🗂️  DETAILED CLUSTER DESCRIPTIONS:")
    print("-" * 80)
    for idx, (index_val, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"\n{idx}. {row['cluster_label'].upper()} ({row['image_count']} images)")
        print(f"   Description: {row['cluster_description']}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Query H3 spatial-semantic index for local land-cover and activities.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the query location.")
    parser.add_argument("--lng", type=float, required=True, help="Longitude of the query location.")
    parser.add_argument("--res", type=int, default=8, choices=list(range(5, 12)),
                        help="H3 resolution for aggregation (default: 8, ~0.7 sq km). Range: 5-11.")
    parser.add_argument("--index", type=str, default="full_pipeline_output/geo_space_h3_semantic_index.parquet",
                        help="Path to the pre-aggregated index parquet file.")
    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"Error: Pre-aggregated index file not found at '{args.index}'.")
        print("Please build the index first using build_spatial_semantic_index.py.")
        sys.exit(1)

    # Convert query coordinates to H3 Cell at target resolution
    target_cell = h3.latlng_to_cell(args.lat, args.lng, args.res)

    print(f"Loading H3 spatial-semantic index from {args.index}...")
    df = pd.read_parquet(args.index)

    # Filter by resolution and cell
    df_cell = df[(df['resolution'] == args.res) & (df['query_cell'] == target_cell)]

    if not df_cell.empty:
        total_images = df_cell['image_count'].sum()
        format_cell_output(df_cell, target_cell, args.res, total_images)
    else:
        print(f"\n⚠️  No direct observations found in H3 cell {target_cell} at resolution {args.res}.")
        print("Searching neighboring cells for nearby observations...")

        # fallback: do a ring search (k-ring = 1, i.e., immediate neighbors)
        neighbors = h3.grid_disk(target_cell, 1)
        df_neighbors = df[(df['resolution'] == args.res) & (df['query_cell'].isin(neighbors))]

        if not df_neighbors.empty:
            neighbor_cells_found = df_neighbors['query_cell'].unique()
            print(
                f"Found observations in {len(neighbor_cells_found)} neighboring cell(s): {', '.join(neighbor_cells_found)}")

            # Aggregate neighbor results
            df_agg = df_neighbors.groupby(
                ['cluster_id', 'cluster_label', 'cluster_description']
            )['image_count'].sum().reset_index()

            total_images = df_agg['image_count'].sum()
            format_cell_output(df_agg, f"{target_cell} (Aggregated neighbors)", args.res, total_images)
        else:
            print("❌ No observations found in the immediate neighborhood at this resolution.")


if __name__ == "__main__":
    main()
