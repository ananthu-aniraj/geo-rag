import argparse
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def rename_in_csv(file_path, old_name, new_name):
    print(f"Loading CSV file: {file_path}...")
    temp_path = file_path + ".tmp"
    chunksize = 100000

    first_chunk = True
    print(f"Modifying '{old_name}' -> '{new_name}' in column 'Platform'...")

    try:
        reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            if 'Platform' in chunk.columns:
                chunk['Platform'] = chunk['Platform'].replace(old_name, new_name)

            if first_chunk:
                chunk.to_csv(temp_path, index=False, mode='w')
                first_chunk = False
            else:
                chunk.to_csv(temp_path, index=False, mode='a', header=False)

        os.replace(temp_path, file_path)
        print(f"Successfully updated CSV: {file_path}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Failed to update CSV {file_path}: {e}")


def rename_in_parquet(file_path, old_name, new_name):
    if pq is None or pa is None:
        print("Error: pyarrow is required to edit Parquet files. Please install it first.")
        sys.exit(1)

    print(f"Processing Parquet file: {file_path}...")
    temp_path = file_path + ".tmp"

    try:
        reader = pq.ParquetFile(file_path)
        schema = reader.schema_arrow

        # Check if 'Platform' exists in schema before processing
        if 'Platform' not in schema.names:
            print(f"Skipping Parquet {file_path}: 'Platform' column not found in schema.")
            return

        writer = None
        # Stream row groups to avoid loading entire 15GB into memory
        for i in range(reader.num_row_groups):
            print(f" -> Processing row group {i + 1}/{reader.num_row_groups}...")
            table = reader.read_row_group(i)
            df = table.to_pandas()

            if 'Platform' in df.columns:
                df['Platform'] = df['Platform'].replace(old_name, new_name)

            # Convert back to PyArrow table
            new_table = pa.Table.from_pandas(df, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(temp_path, schema)

            writer.write_table(new_table)

        if writer is not None:
            writer.close()

        os.replace(temp_path, file_path)
        print(f"Successfully updated Parquet: {file_path}")
    except Exception as e:
        print(f"Failed to update Parquet {file_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(description="Rename Platform names in dataset files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file_path", type=str, help="Path to a single CSV or Parquet file.")
    group.add_argument("--dir_path", type=str, help="Path to a directory containing CSV and Parquet files.")

    parser.add_argument("--old_name", type=str, default="GoogleLandmarks", help="Platform name to replace.")
    parser.add_argument("--new_name", type=str, default="Wikimedia", help="New platform name.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively scan subdirectories when --dir_path is used.")
    args = parser.parse_args()

    files_to_process = []

    if args.file_path:
        if not os.path.exists(args.file_path):
            print(f"Error: File '{args.file_path}' not found.")
            sys.exit(1)
        files_to_process.append(args.file_path)
    else:
        if not os.path.exists(args.dir_path):
            print(f"Error: Directory '{args.dir_path}' not found.")
            sys.exit(1)

        print(f"Scanning directory '{args.dir_path}'...")
        if args.recursive:
            for root, _, files in os.walk(args.dir_path):
                for filename in files:
                    if filename.lower().endswith(('.csv', '.parquet')):
                        files_to_process.append(os.path.join(root, filename))
        else:
            for entry in os.scandir(args.dir_path):
                if entry.is_file() and entry.name.lower().endswith(('.csv', '.parquet')):
                    files_to_process.append(entry.path)

    if not files_to_process:
        print("No CSV or Parquet files found to process.")
        sys.exit(0)

    print(f"Found {len(files_to_process)} file(s) to process.")
    for file_path in files_to_process:
        file_ext = os.path.splitext(file_path)[1].lower()
        print("\n" + "=" * 50)
        if file_ext == '.csv':
            rename_in_csv(file_path, args.old_name, args.new_name)
        elif file_ext == '.parquet':
            rename_in_parquet(file_path, args.old_name, args.new_name)
        print("=" * 50)


if __name__ == "__main__":
    main()
