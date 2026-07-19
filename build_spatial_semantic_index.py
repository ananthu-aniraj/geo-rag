import pyarrow.parquet as pq
import pandas as pd
import h3
import time
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Build a lightweight pre-aggregated H3 spatial-semantic index.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input clustered parquet file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the aggregated index parquet file.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    print(f"Reading metadata columns from {args.input}...")
    start = time.time()

    # Read necessary columns, dynamically checking what's available
    parquet_file = pq.ParquetFile(args.input)
    available_cols = parquet_file.schema.names
    
    read_cols = ['cluster_id']
    
    # Handle H3 Cell fallback
    has_h3 = 'H3_Cell' in available_cols
    if has_h3:
        read_cols.append('H3_Cell')
    else:
        if 'Latitude' in available_cols and 'Longitude' in available_cols:
            print("[WARNING] H3_Cell column is missing in input dataset. Reading coordinates to reconstruct H3_Cell...")
            read_cols.extend(['Latitude', 'Longitude'])
        else:
            print("Error: Input dataset is missing required columns H3_Cell (or Latitude & Longitude).")
            sys.exit(1)
            
    # Check other columns
    has_label = 'cluster_label' in available_cols
    if has_label:
        read_cols.append('cluster_label')
        
    has_desc = 'cluster_description' in available_cols
    if has_desc:
        read_cols.append('cluster_description')

    has_season = 'Season' in available_cols
    if has_season:
        read_cols.append('Season')
    
    has_tod = 'Time_Of_Day' in available_cols
    if has_tod:
        read_cols.append('Time_Of_Day')
        
    has_parent_id = 'parent_cluster_id' in available_cols
    has_parent_label = 'parent_cluster_label' in available_cols
    if has_parent_id:
        read_cols.append('parent_cluster_id')
    if has_parent_label:
        read_cols.append('parent_cluster_label')
        
    table = pq.read_table(args.input, columns=read_cols)
    df = table.to_pandas()
    print(f"Loaded {len(df)} rows in {time.time() - start:.2f} seconds.")

    # Reconstruct H3_Cell if missing
    if 'H3_Cell' not in df.columns:
        print("Reconstructing H3 cells at resolution 11 from coordinates...")
        df['H3_Cell'] = [
            h3.latlng_to_cell(float(lat), float(lon), 11) if pd.notna(lat) and pd.notna(lon) else None
            for lat, lon in zip(df['Latitude'], df['Longitude'])
        ]

    # Drop rows with null H3 cells or cluster ids
    df = df.dropna(subset=['H3_Cell', 'cluster_id'])
    
    # Default populate other missing columns to ensure schema consistency
    if 'cluster_label' not in df.columns:
        df['cluster_label'] = df['cluster_id'].apply(lambda x: f"Cluster {x}")
    if 'cluster_description' not in df.columns:
        df['cluster_description'] = "No description available"
        
    if 'parent_cluster_id' not in df.columns:
        df['parent_cluster_id'] = df['cluster_id'] // 80
    if 'parent_cluster_label' not in df.columns:
        df['parent_cluster_label'] = df['parent_cluster_id'].apply(lambda x: f"Parent Cluster {x}")

    if 'Season' not in df.columns:
        df['Season'] = 'Unknown'
    else:
        df['Season'] = df['Season'].fillna('Unknown')

    if 'Time_Of_Day' not in df.columns:
        df['Time_Of_Day'] = 'Unknown'
    else:
        df['Time_Of_Day'] = df['Time_Of_Day'].fillna('Unknown')

    # Build aggregated tables for resolutions 1 through 11
    resolutions = list(range(1, 12))
    aggregated_dfs = []

    for res in resolutions:
        print(f"Aggregating at H3 Resolution {res}...")
        res_start = time.time()

        # If resolution is 11, use raw cell directly
        if res == 11:
            df_res = df.copy()
            df_res['query_cell'] = df_res['H3_Cell']
        else:
            df_res = df.copy()
            df_res['query_cell'] = df_res['H3_Cell'].apply(lambda x: h3.cell_to_parent(x, res))

        # Group by query cell, season, time of day, and cluster/parent metadata
        group_cols = ['query_cell', 'Season']
        if has_tod:
            group_cols.append('Time_Of_Day')
        group_cols.extend(['cluster_id', 'cluster_label', 'cluster_description'])
        if has_parent_id:
            group_cols.append('parent_cluster_id')
        if has_parent_label:
            group_cols.append('parent_cluster_label')

        grouped = df_res.groupby(group_cols, observed=True).size().reset_index(name='image_count')

        grouped['resolution'] = res
        aggregated_dfs.append(grouped)
        print(f"  -> Generated {len(grouped)} records in {time.time() - res_start:.2f} seconds.")

    print("Combining all resolutions...")
    final_df = pd.concat(aggregated_dfs, ignore_index=True)

    # Reorder columns, dynamically preserving parent and time of day info
    final_cols = ['resolution', 'query_cell', 'Season']
    if has_tod:
        final_cols.append('Time_Of_Day')
    final_cols.extend(['cluster_id', 'cluster_label', 'cluster_description'])
    if has_parent_id:
        final_cols.append('parent_cluster_id')
    if has_parent_label:
        final_cols.append('parent_cluster_label')
    final_cols.append('image_count')
    final_df = final_df[final_cols]

    print(f"Saving aggregated index of {len(final_df)} rows to {args.output}...")

    # Ensure output parent directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final_df.to_parquet(args.output, index=False)
    print(f"Index built successfully in {time.time() - start:.2f} seconds.")


if __name__ == '__main__':
    main()
