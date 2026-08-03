import argparse
import os

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Prune extracted Google Landmarks v2 images to match your sampled CSV.")
    parser.add_argument("--csv", type=str, required=True, 
                        help="Path to the sampled Google Landmarks CSV.")
    parser.add_argument("--img_dir", type=str, required=True, 
                        help="Path to the extracted GLDv2 image directory (where train/ resides).")
    parser.add_argument("--dry_run", action="store_true", 
                        help="Perform a dry run to see how many files would be deleted without deleting them.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at: {args.csv}")
        return

    if not os.path.exists(args.img_dir):
        print(f"Error: Image directory not found at: {args.img_dir}")
        return

    print(f"Loading active image IDs from {args.csv}...")
    # Load only the Photo_ID column to keep memory usage minimal
    df = pd.read_csv(args.csv, usecols=['Photo_ID'])
    active_ids = set(df['Photo_ID'].astype(str).str.strip().values)
    print(f" -> Found {len(active_ids):,} active target images in CSV.")

    print("\nScanning image directory and pruning files...")
    total_scanned = 0
    deleted_count = 0
    space_freed = 0
    
    # We walk the directory tree from the bottom up (topdown=False) 
    # to safely delete empty directories after their contents are pruned.
    for root, dirs, files in tqdm(os.walk(args.img_dir, topdown=False), desc="Pruning directories"):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_scanned += 1
                img_id = os.path.splitext(file_name)[0]
                
                if img_id not in active_ids:
                    file_path = os.path.join(root, file_name)
                    try:
                        file_size = os.path.getsize(file_path)
                        if not args.dry_run:
                            os.remove(file_path)
                        deleted_count += 1
                        space_freed += file_size
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                        
        # Delete empty subdirectories (only if not a dry run)
        if not args.dry_run:
            # Check if directory contains no files and no subdirectories
            try:
                if not os.listdir(root) and root != args.img_dir:
                    os.rmdir(root)
            except Exception:
                pass

    print("\n" + "="*50)
    if args.dry_run:
        print("🎒 [DRY RUN RESULTS]")
        print(f"Scan complete. Would have deleted {deleted_count:,} / {total_scanned:,} scanned images.")
        print(f"Estimated space to be freed: {space_freed / 1024**3:.2f} GB")
    else:
        print("✅ [PRUNING COMPLETED SUCCESSFULLY]")
        print(f"Pruned {deleted_count:,} / {total_scanned:,} scanned images.")
        print(f"Actual disk space freed: {space_freed / 1024**3:.2f} GB")
    print("="*50)


if __name__ == "__main__":
    main()
