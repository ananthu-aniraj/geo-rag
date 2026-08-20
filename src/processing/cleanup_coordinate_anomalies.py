import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Purge locked-latitude coordinate anomalies from the database."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="full_pipeline_output/geo_space_deduplicated.parquet",
        help="Path to the Parquet dataset.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="full_pipeline_output/geo_space_deduplicated.csv",
        help="Path to the CSV metadata file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the cleaned Parquet dataset. Defaults to input path.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to write the cleaned CSV metadata file. Defaults to input CSV path.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Restrict anomaly detection and cleanup to this specific platform (e.g. 'mapillary').",
    )
    parser.add_argument(
        "--continent",
        type=str,
        default=None,
        help="Restrict anomaly detection and cleanup to this specific continent (e.g. 'africa').",
    )
    args = parser.parse_args()

    parquet_path = args.input
    csv_path = args.csv
    output_parquet_path = args.output if args.output else parquet_path
    output_csv_path = args.output_csv if args.output_csv else csv_path

    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}")
        return

    print(
        "================================================================================"
    )
    print("🧹 GEOSPATIAL COORDINATE ANOMALY CLEANUP (STREAMING VERSION)")
    print(
        "================================================================================"
    )

    t0 = time.time()
    print("Opening Parquet file in streaming mode...")
    pf = pq.ParquetFile(parquet_path)
    num_row_groups = pf.num_row_groups
    print(f" -> Found {num_row_groups} row groups.")

    # Detect if continent column is present in the schema
    schema = pf.schema_arrow
    has_continent = "continent" in schema.names or "Continent" in schema.names
    continent_col = (
        "continent"
        if "continent" in schema.names
        else ("Continent" if "Continent" in schema.names else None)
    )

    # 1. Read coordinates and group columns only (extremely fast and memory efficient)
    load_cols = ["Latitude", "Longitude", "Platform"]
    if has_continent and continent_col:
        load_cols.append(continent_col)

    lats = []
    lons = []
    plats = []
    conts = []

    for rg in range(num_row_groups):
        tbl_rg = pf.read_row_group(rg, columns=load_cols)
        lats.append(tbl_rg["Latitude"].to_numpy())
        lons.append(tbl_rg["Longitude"].to_numpy())
        plats.append(tbl_rg["Platform"].to_numpy().astype(str))
        if has_continent and continent_col:
            conts.append(tbl_rg[continent_col].to_numpy().astype(str))

    meta_dict = {
        "Latitude": np.concatenate(lats),
        "Longitude": np.concatenate(lons),
        "Platform": np.concatenate(plats),
    }
    if has_continent and continent_col:
        meta_dict[continent_col] = np.concatenate(conts)

    df_meta = pd.DataFrame(meta_dict)
    df_meta["Latitude"] = pd.to_numeric(df_meta["Latitude"], errors="coerce")
    df_meta["Longitude"] = pd.to_numeric(df_meta["Longitude"], errors="coerce")
    df_meta["Platform"] = df_meta["Platform"].fillna("").astype(str)
    if has_continent and continent_col:
        df_meta[continent_col] = df_meta[continent_col].fillna("").astype(str)

    print(
        f" -> Loaded coordinates metadata for {len(df_meta):,} records in {time.time() - t0:.2f}s."
    )

    # Filter target platform/continent before analyzing anomalies if requested
    df_meta_for_stats = df_meta.copy()
    if args.platform:
        print(f"Filtering detection scope to platform: {args.platform}")
        df_meta_for_stats = df_meta_for_stats[
            df_meta_for_stats["Platform"].str.lower() == args.platform.lower()
        ]
    if args.continent:
        if has_continent and continent_col:
            print(f"Filtering detection scope to continent: {args.continent}")
            df_meta_for_stats = df_meta_for_stats[
                df_meta_for_stats[continent_col].str.lower() == args.continent.lower()
            ]
        else:
            print(
                f"Warning: --continent '{args.continent}' specified but no continent column was found in schema. Skipping."
            )

    # 2. Group by Platform (+ Continent) and rounded latitude to detect locked lines per platform
    df_meta_for_stats["lat_round"] = df_meta_for_stats["Latitude"].round(5)

    groupby_cols = ["Platform", "lat_round"]
    if has_continent and continent_col:
        groupby_cols.insert(1, continent_col)

    stats = (
        df_meta_for_stats.groupby(groupby_cols)
        .agg(
            count=("Longitude", "count"),
            min_lon=("Longitude", "min"),
            max_lon=("Longitude", "max"),
        )
        .reset_index()
    )

    stats["span"] = stats["max_lon"] - stats["min_lon"]

    # Flag: high frequency (>10 images) across a global span (>1.0 degree longitude difference)
    anomalies = stats[(stats["count"] > 10) & (stats["span"] > 1.0)]

    if anomalies.empty:
        print("✅ No locked-latitude coordinate anomalies found in the dataset scope.")
        if parquet_path != output_parquet_path:
            import shutil

            print(f"Copying clean Parquet database to output: {output_parquet_path}")
            shutil.copy2(parquet_path, output_parquet_path)
        if csv_path != output_csv_path and os.path.exists(csv_path):
            import shutil

            print(f"Copying clean CSV metadata to output: {output_csv_path}")
            shutil.copy2(csv_path, output_csv_path)
        return

    print(f"\n🚨 Found {len(anomalies)} locked-latitude coordinate anomaly lines:")
    print(
        "================================================================================"
    )
    print(anomalies.to_string(index=False))
    print(
        "================================================================================"
    )

    # Build lookup set of flagged keys
    if has_continent and continent_col:
        flagged_keys = set(
            zip(anomalies["Platform"], anomalies[continent_col], anomalies["lat_round"])
        )
    else:
        flagged_keys = set(zip(anomalies["Platform"], anomalies["lat_round"]))

    # 3. Stream write filtered row groups to a temporary Parquet file
    temp_parquet_path = output_parquet_path + ".tmp"
    schema = pf.schema_arrow

    print("\nStreaming row groups and filtering in C++ memory...")
    t_stream = time.time()
    removed_count = 0

    from src.utils.io import get_parquet_writer

    with get_parquet_writer(temp_parquet_path, schema) as writer:
        for rg in range(num_row_groups):
            tbl_rg = pf.read_row_group(rg)
            rg_lat = tbl_rg["Latitude"].to_numpy()
            rg_lat_round = np.round(rg_lat, 5)
            rg_plat = tbl_rg["Platform"].to_numpy().astype(str)

            if has_continent and continent_col:
                rg_cont = tbl_rg[continent_col].to_numpy().astype(str)
                keys = list(zip(rg_plat, rg_cont, rg_lat_round))
            else:
                keys = list(zip(rg_plat, rg_lat_round))

            # Keep indices not matching flagged keys
            keep_mask = np.array([k not in flagged_keys for k in keys], dtype=bool)

            filtered_tbl = tbl_rg.filter(pa.array(keep_mask))
            writer.write_table(filtered_tbl)

            removed_count += len(tbl_rg) - len(filtered_tbl)

    os.replace(temp_parquet_path, output_parquet_path)
    print(f" -> Saved cleaned Parquet in {time.time() - t_stream:.2f}s.")

    # 5. Clean matching CSV if it exists
    if os.path.exists(csv_path):
        print("Cleaning CSV file...")
        t_csv = time.time()
        temp_csv_path = output_csv_path + ".tmp"
        first = True
        for chunk in pd.read_csv(
            csv_path, chunksize=100000, dtype={"Platform": str, "Photo_ID": str}
        ):
            chunk["Latitude"] = pd.to_numeric(chunk["Latitude"], errors="coerce")
            chunk["Longitude"] = pd.to_numeric(chunk["Longitude"], errors="coerce")
            chunk["lat_round"] = chunk["Latitude"].round(5)
            chunk["Platform"] = chunk["Platform"].fillna("").astype(str)

            if has_continent and continent_col:
                chunk[continent_col] = chunk[continent_col].fillna("").astype(str)
                keys = list(
                    zip(chunk["Platform"], chunk[continent_col], chunk["lat_round"])
                )
            else:
                keys = list(zip(chunk["Platform"], chunk["lat_round"]))

            keep_mask = [k not in flagged_keys for k in keys]
            cleaned_chunk = chunk[keep_mask].drop(columns=["lat_round"])

            cleaned_chunk.to_csv(
                temp_csv_path, mode="a" if not first else "w", index=False, header=first
            )
            first = False

        os.replace(temp_csv_path, output_csv_path)
        print(f" -> Saved CSV in {time.time() - t_csv:.2f}s.")

    print(
        f"\n🎉 Streaming cleanup completed successfully! Purged {removed_count:,} records."
    )


if __name__ == "__main__":
    main()
