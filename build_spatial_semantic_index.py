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
    
    # Read only the necessary metadata columns to prevent high memory usage
    table = pq.read_table(
        args.input,
        columns=['H3_Cell', 'cluster_id', 'cluster_label', 'cluster_description']
    )
    df = table.to_pandas()
    print(f"Loaded {len(df)} rows in {time.time() - start:.2f} seconds.")
    
    # Drop rows with null H3 cells or cluster ids
    df = df.dropna(subset=['H3_Cell', 'cluster_id'])
    
    # Build aggregated tables for resolutions 5, 6, 7, 8, 9, 10, 11
    resolutions = [5, 6, 7, 8, 9, 10, 11]
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
            
        # Group by query cell and cluster metadata
        grouped = df_res.groupby(
            ['query_cell', 'cluster_id', 'cluster_label', 'cluster_description'],
            observed=True
        ).size().reset_index(name='image_count')
        
        grouped['resolution'] = res
        aggregated_dfs.append(grouped)
        print(f"  -> Generated {len(grouped)} records in {time.time() - res_start:.2f} seconds.")
        
    print("Combining all resolutions...")
    final_df = pd.concat(aggregated_dfs, ignore_index=True)
    
    # Reorder columns
    final_df = final_df[['resolution', 'query_cell', 'cluster_id', 'cluster_label', 'cluster_description', 'image_count']]
    
    print(f"Saving aggregated index of {len(final_df)} rows to {args.output}...")
    
    # Ensure output parent directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    final_df.to_parquet(args.output, index=False)
    print(f"Index built successfully in {time.time() - start:.2f} seconds.")

if __name__ == '__main__':
    main()
