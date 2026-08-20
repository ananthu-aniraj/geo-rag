import argparse
import os
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Restore backfilled License column from a license-rich metadata file into a pre-decoupled database."
    )
    parser.add_argument(
        "--pre_decoupled",
        type=str,
        required=True,
        help="Path to the uncorrupted pre-decoupled Parquet file containing the 'embedding' column.",
    )
    parser.add_argument(
        "--latest",
        type=str,
        required=True,
        help="Path to the latest Parquet metadata file containing the backfilled 'License' column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output merged Parquet file (will still contain the 'embedding' column).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pre_decoupled):
        print(f"Error: Pre-decoupled file not found: {args.pre_decoupled}")
        sys.exit(1)
    if not os.path.exists(args.latest):
        print(f"Error: Latest license-rich file not found: {args.latest}")
        sys.exit(1)

    print(f"Loading pre-decoupled database from: {args.pre_decoupled}...")
    df_pre = pd.read_parquet(args.pre_decoupled)
    print(f" -> Loaded {len(df_pre):,} rows.")

    if "embedding" not in df_pre.columns:
        print(
            "Error: The pre-decoupled file must contain the raw 'embedding' column to preserve alignment."
        )
        sys.exit(1)

    print(f"Loading license metadata from: {args.latest}...")
    # Load only necessary columns to save memory
    df_latest = pd.read_parquet(
        args.latest, columns=["Platform", "Photo_ID", "License"]
    )
    print(f" -> Loaded {len(df_latest):,} rows containing license mappings.")

    # Drop old License column if present in pre-decoupled database to avoid suffix collisions
    if "License" in df_pre.columns:
        print("Dropping old License column from pre-decoupled metadata...")
        df_pre = df_pre.drop(columns=["License"])

    print("Merging license entries based on stable ['Platform', 'Photo_ID'] keys...")
    df_merged = df_pre.merge(df_latest, on=["Platform", "Photo_ID"], how="left")

    # Verify if licenses were mapped successfully
    license_count = df_merged["License"].notna().sum()
    print(
        f" -> Merged successfully. Found {license_count:,} non-null license entries in the merged output."
    )

    # Ensure parent directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving merged database to: {args.output}...")
    df_merged.to_parquet(args.output, compression="zstd")
    print("Restore complete! You can now run decouple_dataset.py on this output file.")


if __name__ == "__main__":
    main()
