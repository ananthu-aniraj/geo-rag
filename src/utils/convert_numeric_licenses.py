import argparse
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

from src.utils.licensing import FLICKR_LICENSE_MAP


def convert_file(file_path, dry_run=False):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    try:
        if file_path.endswith(".parquet"):
            pf = pq.ParquetFile(file_path)
            if "License" not in pf.schema_arrow.names:
                return False
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path, nrows=5)
            if "License" not in df.columns:
                return False
            df = pd.read_csv(file_path)
        else:
            return False

        # Find unique licenses in the dataset
        unique_licenses = df["License"].dropna().unique()
        numeric_keys = ["11", "12", "13", "14", "15", "16"]
        numeric_found = [
            l for l in unique_licenses
            if str(l).split('.')[0].strip() in numeric_keys
        ]

        if not numeric_found:
            return False

        print(f"Found numeric licenses {numeric_found} in {file_path}")
        if dry_run:
            print("Dry run: no changes will be saved.")
            return True

        # Perform conversion
        def map_license(l):
            if pd.isna(l):
                return l
            l_str = str(l).split('.')[0].strip()
            if l_str in FLICKR_LICENSE_MAP:
                return FLICKR_LICENSE_MAP[l_str]
            return l

        df["License"] = df["License"].map(map_license)
        
        # Save back
        if file_path.endswith(".parquet"):
            df.to_parquet(file_path, index=False, compression="zstd")
        else:
            df.to_csv(file_path, index=False)
            
        print(f"Successfully converted and saved: {file_path}")
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Flickr numeric license codes (11-16) to human-readable strings in Parquet/CSV datasets."
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to a file or directory to scan for Parquet/CSV datasets."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and show numeric licenses without modifying any files."
    )
    args = parser.parse_args()

    if os.path.isfile(args.path):
        convert_file(args.path, dry_run=args.dry_run)
    elif os.path.isdir(args.path):
        print(f"Scanning directory: {args.path} for data files...")
        
        # Gather all parquet and csv files recursively
        parquet_files = glob.glob(os.path.join(args.path, "*.parquet"))
        csv_files = glob.glob(os.path.join(args.path, "*.csv"))
        
        all_files = sorted(list(set(parquet_files + csv_files)))
        
        processed_count = 0
        converted_count = 0
        for f in all_files:
            # Exclude version control, IDE, env folders
            if any(x in f for x in [".git/", ".idea/", ".venv/", ".ruff_cache/"]):
                continue
            
            processed_count += 1
            if convert_file(f, dry_run=args.dry_run):
                converted_count += 1
                
        print(f"\nScan complete. Scanned {processed_count} data files. Converted {converted_count} files.")
    else:
        print(f"Error: Invalid path: {args.path}")


if __name__ == "__main__":
    main()
