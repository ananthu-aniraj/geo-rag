import argparse
import os
import sys
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

def rename_in_csv(file_path, old_name, new_name):
    print(f"Loading CSV file: {file_path}...")
    # Read in chunks to prevent memory issues on large files
    temp_path = file_path + ".tmp"
    chunksize = 100000
    
    first_chunk = True
    # Count rows for progress
    try:
        total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore')) - 1
    except Exception:
        total_rows = None

    print(f"Modifying '{old_name}' -> '{new_name}' in column 'Platform'...")
    
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


def rename_in_parquet(file_path, old_name, new_name):
    if pq is None or pa is None:
        print("Error: pyarrow is required to edit Parquet files. Please install it first.")
        sys.exit(1)

    print(f"Processing Parquet file: {file_path}...")
    temp_path = file_path + ".tmp"
    
    reader = pq.ParquetFile(file_path)
    schema = reader.schema_arrow
    
    writer = None
    try:
        # Stream row groups to avoid loading entire 15GB into memory
        for i in range(reader.num_row_groups):
            print(f" -> Processing row group {i+1}/{reader.num_row_groups}...")
            table = reader.read_row_group(i)
            df = table.to_pandas()
            
            if 'Platform' in df.columns:
                df['Platform'] = df['Platform'].replace(old_name, new_name)
            
            # Convert back to PyArrow table
            new_table = pa.Table.from_pandas(df, schema=schema)
            
            if writer is None:
                writer = pq.ParquetWriter(temp_path, schema)
            
            writer.write_table(new_table)
            
    finally:
        if writer is not None:
            writer.close()
            
    os.replace(temp_path, file_path)
    print(f"Successfully updated Parquet: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Rename Platform names in dataset files.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV or Parquet file.")
    parser.add_argument("--old_name", type=str, default="GoogleLandmarks", help="Platform name to replace.")
    parser.add_argument("--new_name", type=str, default="Wikimedia", help="New platform name.")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    file_ext = os.path.splitext(args.file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            rename_in_csv(args.file_path, args.old_name, args.new_name)
        elif file_ext == '.parquet':
            rename_in_parquet(args.file_path, args.old_name, args.new_name)
        else:
            print(f"Error: Unsupported file format '{file_ext}'. Only CSV and Parquet are supported.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during modification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
