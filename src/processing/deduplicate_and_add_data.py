import argparse
import glob
import os

import pandas as pd


def clean_photo_id(val):
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.endswith(".0"):
        val_str = val_str[:-2]
    return val_str


def main():
    parser = argparse.ArgumentParser(
        description="Clean new metadata CSVs by removing duplicates (within the files and against the existing dataset)."
    )
    parser.add_argument(
        "--existing_csv",
        type=str,
        required=True,
        help="Path to the current consolidated CSV file containing already processed data.",
    )
    parser.add_argument(
        "--new_data_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing the new metadata CSV files to clean.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix to append to cleaned files (e.g. '_cleaned'). If empty, overwrites the files.",
    )
    args = parser.parse_args()

    # 1. Load existing dataset keys to exclude
    seen_keys = set()
    if os.path.exists(args.existing_csv):
        print(f"Loading existing dataset keys from: {args.existing_csv}")
        # Load only key columns to optimize memory and enforce string format to prevent warnings
        df_existing = pd.read_csv(
            args.existing_csv,
            usecols=["Platform", "Photo_ID"],
            dtype={"Platform": str, "Photo_ID": str},
        )
        df_existing["Photo_ID"] = df_existing["Photo_ID"].apply(clean_photo_id)
        seen_keys = set(zip(df_existing["Platform"], df_existing["Photo_ID"]))
        print(f" -> Loaded {len(seen_keys)} existing keys.")
        del df_existing
    else:
        print(
            f"Existing CSV not found at '{args.existing_csv}'. Proceeding with empty duplicate database."
        )

    # 2. Gather new metadata CSVs from the specified directories
    new_csv_files = []
    for d in args.new_data_dirs:
        if os.path.isdir(d):
            found_csvs = glob.glob(
                os.path.join(d, "**", "*.csv"), recursive=True
            ) + glob.glob(os.path.join(d, "*.csv"))
            found_csvs = list(set(found_csvs))
            print(f"Found {len(found_csvs)} CSV files in directory: {d}")
            new_csv_files.extend(found_csvs)
        elif os.path.isfile(d) and d.endswith(".csv"):
            new_csv_files.append(d)

    # Make paths absolute and remove the existing CSV itself
    existing_abs = os.path.abspath(args.existing_csv)
    new_csv_files = [
        os.path.abspath(f) for f in new_csv_files if os.path.abspath(f) != existing_abs
    ]

    print(f"Starting file-by-file cleaning of {len(new_csv_files)} files...")

    # 3. Process and clean each CSV file individually
    for f in new_csv_files:
        try:
            df = pd.read_csv(f, dtype=str)
            original_len = len(df)
            if original_len == 0:
                print(f"Skipping empty file: {f}")
                continue

            # Standardize Column Names (case-insensitive check)
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ["photo_id", "image_id", "id"]:
                    col_mapping[col] = "Photo_ID"
                elif col_lower == "platform":
                    col_mapping[col] = "Platform"
                elif col_lower == "latitude":
                    col_mapping[col] = "Latitude"
                elif col_lower == "longitude":
                    col_mapping[col] = "Longitude"
                elif col_lower in ["image_url", "url"]:
                    col_mapping[col] = "Image_URL"
                elif col_lower in ["captured_at", "timestamp", "time"]:
                    col_mapping[col] = "Captured_At"
                elif col_lower == "h3_cell":
                    col_mapping[col] = "H3_Cell"

            df = df.rename(columns=col_mapping)
            if "Photo_ID" not in df.columns or "Platform" not in df.columns:
                print(f"Skipping '{f}' (missing Photo_ID or Platform column)")
                continue

            # Clean photo IDs
            df["Photo_ID"] = df["Photo_ID"].apply(clean_photo_id)

            # Drop duplicates within the file itself
            df = df.drop_duplicates(subset=["Platform", "Photo_ID"])
            file_dedup_len = len(df)

            # Filter out records already in seen_keys (existing dataset + previously cleaned files in this run)
            keys = list(zip(df["Platform"], df["Photo_ID"]))
            is_new = [k not in seen_keys for k in keys]
            df_cleaned = df[is_new]
            final_len = len(df_cleaned)

            # Update our globally seen keys with the kept records from this file
            seen_keys.update(zip(df_cleaned["Platform"], df_cleaned["Photo_ID"]))

            # Save the cleaned file
            if args.suffix:
                base, ext = os.path.splitext(f)
                out_path = f"{base}{args.suffix}{ext}"
            else:
                out_path = f

            df_cleaned.to_csv(out_path, index=False)

            duplicates_removed = original_len - final_len
            print(
                f"Cleaned '{os.path.basename(f)}': {original_len} -> {final_len} rows (Removed {duplicates_removed} duplicates). Saved to '{os.path.basename(out_path)}'"
            )

        except Exception as e:
            print(f"Error processing file '{f}': {e}")

    print("\nProcessing complete. All new CSV files have been cleaned in-place.")


if __name__ == "__main__":
    main()
