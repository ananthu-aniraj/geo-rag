import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Convert npy embedding matrix from float32 to float16 memory-efficiently.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy file (float32).")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to output .npy file. Defaults to [input]_fp16.npy.")
    parser.add_argument("--overwrite", action="store_true", 
                        help="Overwrite the input file directly (deletes original float32 file).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not args.input.endswith(".npy"):
        print("Error: Input file must be a .npy binary file.")
        sys.exit(1)

    # Determine output path
    if args.overwrite:
        output_path = args.input + ".tmp_fp16.npy"
    else:
        if args.output:
            output_path = args.output
        else:
            output_path = args.input.replace(".npy", "_fp16.npy")

    print(f"Opening source file in memory-mapped mode: {args.input}")
    src_mmap = np.load(args.input, mmap_mode="r")
    shape = src_mmap.shape
    src_dtype = src_mmap.dtype

    print(f" -> Matrix shape: {shape[0]:,} rows x {shape[1]:,} dimensions")
    print(f" -> Source precision: {src_dtype}")

    if src_dtype == np.float16:
        print("Warning: Input matrix is already float16. No conversion needed.")
        sys.exit(0)

    # Get initial file size
    orig_size_bytes = os.path.getsize(args.input)
    orig_size_gb = orig_size_bytes / (1024 ** 3)
    print(f" -> Original file size: {orig_size_gb:.2f} GB")

    print(f"Creating output file: {output_path} (dtype=float16)...")
    dst_mmap = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.float16, shape=shape)

    # Copy in chunks of 250,000 rows (approx. 760MB of float32 values per chunk)
    chunk_size = 250000
    total_rows = shape[0]

    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        print(f" -> Converting rows {start_idx:,} to {end_idx:,} / {total_rows:,}...")
        dst_mmap[start_idx:end_idx] = src_mmap[start_idx:end_idx].astype(np.float16)
        dst_mmap.flush()

    # Close memmaps
    del src_mmap
    del dst_mmap

    new_size_bytes = os.path.getsize(output_path)
    new_size_gb = new_size_bytes / (1024 ** 3)
    print(f" -> New file size: {new_size_gb:.2f} GB")

    # If overwrite requested, perform rename
    if args.overwrite:
        print("Overwriting original file...")
        try:
            os.replace(output_path, args.input)
            print(f"Successfully replaced original file with float16 version: {args.input}")
        except Exception as e:
            print(f"Error replacing original file: {e}. The converted file is saved at: {output_path}")
            sys.exit(1)
    else:
        print(f"Conversion complete! Converted file saved to: {output_path}")

if __name__ == "__main__":
    main()
