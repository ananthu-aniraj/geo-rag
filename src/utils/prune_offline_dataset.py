import argparse
import glob
import math
import os

import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from src.utils.io import resolve_offline_image_path


def clean_id(x):
    if pd.isna(x):
        return ""
    # If float, convert to int first to discard decimal digits, then string
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        if x.is_integer():
            return str(int(x))
        return str(x)
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Prune offline datasets in-place to align with deduplicated database."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="full_pipeline_output/geo_space_cleaned.parquet",
        help="Path to the deduplicated/cleaned parquet database.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print stats without making any changes."
    )
    parser.add_argument(
        "--no_delete_images",
        action="store_true",
        help="Only prune metadata CSVs; do not delete image files.",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default="config/pipeline/params.yaml",
        help="Path to the parameters YAML file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        print(f"Error: Parquet file '{args.parquet}' does not exist.")
        return

    # Load active IDs and Platforms from the Parquet database
    print(f"Loading deduplicated database from {args.parquet}...")
    try:
        # Read only the necessary columns to save memory
        table = pq.read_table(args.parquet, columns=["Photo_ID", "Platform"])
        pids = table["Photo_ID"].to_pylist()
        plats = table["Platform"].to_pylist()
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Create a set of active (platform, photo_id) tuples via fast zip iteration
    active_keys = set()
    for plat, pid in tqdm(
        zip(plats, pids), total=len(pids), desc="Indexing active database keys"
    ):
        plat_str = str(plat).strip().lower()
        pid_str = clean_id(pid)
        if pid_str:
            active_keys.add((plat_str, pid_str))

    print(
        f"Found {len(active_keys):,} active unique (Platform, Photo_ID) entries in the deduplicated database."
    )

    # Read offline directories from params.yaml
    params_path = args.params_path
    offline_dirs = []
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        pipeline_params = params.get("pipeline", {}) if isinstance(params, dict) else {}
        offline_dirs_str = ""
        if isinstance(pipeline_params, dict):
            offline_dirs_str = pipeline_params.get("offline_dataset_dirs", "")
        if not offline_dirs_str and isinstance(params, dict):
            offline_dirs_str = params.get("offline_dataset_dirs", "")
        offline_dirs = [d.strip() for d in offline_dirs_str.split() if d.strip()]

    if not offline_dirs:
        print("Warning: No offline dataset directories found in params.yaml.")
        return

    print(f"Configured offline directories to scan: {offline_dirs}")

    # Gather all CSVs in those directories
    csv_files = []
    for d in offline_dirs:
        if os.path.exists(d):
            found_csvs = glob.glob(os.path.join(d, "*.csv"))
            csv_files.extend(found_csvs)
        else:
            print(f"Warning: Directory '{d}' does not exist.")

    if not csv_files:
        print("No metadata CSV files found in the configured offline directories.")
        return

    print(f"Found {len(csv_files)} metadata CSV files to process: {csv_files}")

    # Step 1: Scan and identify all files we MUST keep
    # (files that are in active_keys)
    keep_image_paths = set()
    delete_image_candidates = []

    print("\nScanning offline metadata CSV files...")
    for f_path in csv_files:
        print("-" * 60)
        print(f"Processing: {f_path}")
        try:
            df = pd.read_csv(f_path)
            total_rows = len(df)
            if total_rows == 0:
                print("CSV is empty. Skipping.")
                continue

            # Identify columns case-insensitively
            photo_col = next(
                (c for c in df.columns if c.lower() in ["photo_id", "id"]), None
            )
            platform_col = next(
                (c for c in df.columns if c.lower() == "platform"), None
            )
            url_col = next(
                (
                    c
                    for c in df.columns
                    if c.lower() in ["image_location", "image_url", "url"]
                ),
                None,
            )

            if not photo_col or not url_col:
                print(
                    "Error: Could not find ID/Photo_ID or Image URL column. Skipping."
                )
                continue

            rows_to_keep = []
            removed_count = 0

            for _, row in tqdm(
                df.iterrows(), total=total_rows, desc="Scanning metadata rows"
            ):
                pid = clean_id(row[photo_col])
                plat = (
                    str(row[platform_col]).strip().lower()
                    if platform_col
                    else "unknown"
                )
                url = row[url_col]

                # Key check (allow loose match for platform strings like "inat" vs "inaturalist")
                is_active = False
                if pid:
                    # Match exact (plat, pid)
                    if (plat, pid) in active_keys:
                        is_active = True
                    # Match relaxed platform
                    elif "inaturalist" in plat or plat == "inat":
                        is_active = ("inaturalist", pid) in active_keys or (
                            "inat",
                            pid,
                        ) in active_keys

                resolved_path = resolve_offline_image_path(
                    url,
                    offline_dirs,
                    row[photo_col],
                    row[platform_col] if platform_col else None,
                )

                if is_active:
                    rows_to_keep.append(row)
                    if resolved_path:
                        keep_image_paths.add(resolved_path)
                else:
                    removed_count += 1
                    if resolved_path and os.path.exists(resolved_path):
                        delete_image_candidates.append(resolved_path)

            kept_df = pd.DataFrame(rows_to_keep)
            print("CSV Pruning Stats:")
            print(f"  - Original rows: {total_rows:,}")
            print(f"  - Rows to keep:  {len(kept_df):,}")
            print(f"  - Rows to prune: {removed_count:,}")

            if not args.dry_run:
                if len(kept_df) > 0:
                    kept_df.to_csv(f_path, index=False)
                    print(f"  -> Successfully updated CSV in place: {f_path}")
                else:
                    # If all rows are pruned, keep an empty CSV with matching headers
                    empty_df = df.iloc[0:0]
                    empty_df.to_csv(f_path, index=False)
                    print(
                        f"  -> Successfully emptied CSV (kept headers) in place: {f_path}"
                    )

        except Exception as e:
            print(f"Error processing CSV {f_path}: {e}")

    # Step 2: Delete images that are candidates and NOT in keep_image_paths
    if not args.no_delete_images and delete_image_candidates:
        print("\n" + "=" * 60)
        print("Starting physical image file pruning...")

        # Unique list of paths to delete (ensuring we don't try to delete something we decided to keep)
        paths_to_delete = list(
            set(p for p in delete_image_candidates if p not in keep_image_paths)
        )
        print(f"Found {len(paths_to_delete):,} candidate image files to delete.")

        bytes_saved = 0
        deleted_count = 0

        for p in tqdm(paths_to_delete, desc="Pruning image files"):
            if os.path.exists(p):
                try:
                    f_size = os.path.getsize(p)
                    if not args.dry_run:
                        os.remove(p)
                    bytes_saved += f_size
                    deleted_count += 1
                except Exception as e:
                    print(f"Warning: Failed to delete '{p}': {e}")

        mb_saved = bytes_saved / (1024 * 1024)
        print("\nPhysical Pruning Summary:")
        if args.dry_run:
            print(
                f"  [DRY RUN] Would have deleted {deleted_count:,} files, saving {mb_saved:.2f} MB of disk space."
            )
        else:
            print(
                f"  Successfully deleted {deleted_count:,} files, saving {mb_saved:.2f} MB of disk space."
            )

        # Clean up empty subdirectories
        if not args.dry_run:
            print("Cleaning up empty subdirectories...")
            for d in offline_dirs:
                if os.path.exists(d):
                    for root, dirs_list, files in os.walk(d, topdown=False):
                        for name in dirs_list:
                            dir_path = os.path.join(root, name)
                            try:
                                if not os.listdir(dir_path):
                                    os.rmdir(dir_path)
                            except Exception:
                                pass
    else:
        print(
            "\nPhysical image pruning was skipped (either no candidates or --no_delete_images flag set)."
        )


if __name__ == "__main__":
    main()
