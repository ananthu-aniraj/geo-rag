import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Purge locked-latitude coordinate anomalies from the database.")
    parser.add_argument("--input", type=str, default="full_pipeline_output/geo_space_deduplicated.parquet", help="Path to the Parquet dataset.")
    parser.add_argument("--csv", type=str, default="full_pipeline_output/geo_space_deduplicated.csv", help="Path to the CSV metadata file.")
    parser.add_argument("--output", type=str, default=None, help="Path to write the cleaned Parquet dataset. Defaults to input path.")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to write the cleaned CSV metadata file. Defaults to input CSV path.")
    args = parser.parse_args()

    parquet_path = args.input
    csv_path = args.csv
    output_parquet_path = args.output if args.output else parquet_path
    output_csv_path = args.output_csv if args.output_csv else csv_path

    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}")
        return

    print("================================================================================")
    print("🧹 GEOSPATIAL COORDINATE ANOMALY CLEANUP (STREAMING VERSION)")
    print("================================================================================")

    t0 = time.time()
    print("Opening Parquet file in streaming mode...")
    pf = pq.ParquetFile(parquet_path)
    num_row_groups = pf.num_row_groups
    print(f" -> Found {num_row_groups} row groups.")

    # 1. Read coordinates only (extremely fast and memory efficient)
    lats = []
    lons = []
    for rg in range(num_row_groups):
        tbl_rg = pf.read_row_group(rg, columns=["Latitude", "Longitude"])
        lats.append(tbl_rg["Latitude"].to_numpy())
        lons.append(tbl_rg["Longitude"].to_numpy())

    df_meta = pd.DataFrame({
        "Latitude": np.concatenate(lats),
        "Longitude": np.concatenate(lons)
    })
    df_meta["Latitude"] = pd.to_numeric(df_meta["Latitude"], errors='coerce')
    df_meta["Longitude"] = pd.to_numeric(df_meta["Longitude"], errors='coerce')
    print(f" -> Loaded coordinates metadata for {len(df_meta):,} records in {time.time() - t0:.2f}s.")

    # 2. Group by rounded latitude to detect locked lines
    df_meta["lat_round"] = df_meta["Latitude"].round(5)
    stats = df_meta.groupby("lat_round").agg(
        count=("Longitude", "count"),
        min_lon=("Longitude", "min"),
        max_lon=("Longitude", "max")
    )
    stats["span"] = stats["max_lon"] - stats["min_lon"]

    # Flag: high frequency (>10 images) across a global span (>1.0 degree longitude difference)
    # This prevents dropping dense urban centers (which have a tiny span, <0.05 degree)
    anomalies = stats[(stats["count"] > 10) & (stats["span"] > 1.0)]

    if anomalies.empty:
        print("✅ No locked-latitude coordinate anomalies found in the dataset.")
        # Ensure output files exist (copy input to output if they differ)
        if parquet_path != output_parquet_path:
            import shutil
            print(f"Copying clean Parquet database to output: {output_parquet_path}")
            shutil.copy2(parquet_path, output_parquet_path)
        if csv_path != output_csv_path and os.path.exists(csv_path):
            import shutil
            print(f"Copying clean CSV metadata to output: {output_csv_path}")
            shutil.copy2(csv_path, output_csv_path)
        return

    print(f"\n🚨 Found {len(anomalies)} locked-latitude coordinate lines:")
    print("================================================================================")
    print(anomalies[["count", "span"]].to_string())
    print("================================================================================")

    flagged_lats = set(anomalies.index)

    # 3. Stream write filtered row groups to a temporary Parquet file
    temp_parquet_path = output_parquet_path + ".tmp"
    schema = pf.schema_arrow

    print("\nStreaming row groups and filtering in C++ memory...")
    t_stream = time.time()
    removed_count = 0

    with pq.ParquetWriter(temp_parquet_path, schema) as writer:
        for rg in range(num_row_groups):
            tbl_rg = pf.read_row_group(rg)
            rg_lat = tbl_rg["Latitude"].to_numpy()
            rg_lat_round = np.round(rg_lat, 5)

            # Keep indices not matching flagged lats
            keep_mask = ~np.isin(rg_lat_round, list(flagged_lats))

            filtered_tbl = tbl_rg.filter(pa.array(keep_mask))
            writer.write_table(filtered_tbl)

            removed_count += (len(tbl_rg) - len(filtered_tbl))

    os.replace(temp_parquet_path, output_parquet_path)
    print(f" -> Saved cleaned Parquet in {time.time() - t_stream:.2f}s.")

    # 4. Clean matching CSV if it exists
    if os.path.exists(csv_path):
        print("Cleaning CSV file...")
        t_csv = time.time()
        temp_csv_path = output_csv_path + ".tmp"
        first = True
        for chunk in pd.read_csv(csv_path, chunksize=100000, dtype={'Platform': str, 'Photo_ID': str}):
            chunk["Latitude"] = pd.to_numeric(chunk["Latitude"], errors='coerce')
            chunk["Longitude"] = pd.to_numeric(chunk["Longitude"], errors='coerce')
            chunk["lat_round"] = chunk["Latitude"].round(5)
            cleaned_chunk = chunk[~chunk["lat_round"].isin(flagged_lats)].drop(columns=["lat_round"])

            cleaned_chunk.to_csv(
                temp_csv_path,
                mode="a" if not first else "w",
                index=False,
                header=first
            )
            first = False

        os.replace(temp_csv_path, output_csv_path)
        print(f" -> Saved CSV in {time.time() - t_csv:.2f}s.")

    print(f"\n🎉 Streaming cleanup completed successfully! Purged {removed_count:,} records.")


if __name__ == "__main__":
    main()
