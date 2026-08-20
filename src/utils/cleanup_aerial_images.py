import argparse
import glob
import os

from tqdm import tqdm

from src.utils.io import load_dataframe, save_dataframe


def main():
    parser = argparse.ArgumentParser(
        description="Clean up OSM scraped database files by purging aerial images with 'View_of_' in their name."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing the scraped OSM CSV/Parquet files.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist.")
        return

    # Find all CSV and Parquet files in the specified directory
    files = glob.glob(os.path.join(args.dir, "*.csv")) + glob.glob(
        os.path.join(args.dir, "*.parquet")
    )

    if not files:
        print(f"No CSV or Parquet files found in '{args.dir}'.")
        return

    print(f"Scanning {len(files)} files in '{args.dir}'...")
    total_removed = 0
    cleaned_files = 0

    for file_path in tqdm(files, desc="Cleaning files"):
        try:
            df = load_dataframe(file_path)
        except Exception as e:
            print(f"Warning: Could not read {os.path.basename(file_path)}: {e}")
            continue

        if df.empty:
            continue

        # Look for the image URL column case-insensitively
        url_col = next(
            (c for c in df.columns if c.lower() in ["image_url", "url"]), None
        )

        if not url_col:
            continue

        # Process URL string check
        urls = df[url_col].astype(str).str.lower()
        clean_urls = urls.str.split("?").str[0]

        # 1. Check for "view_of_" in URL
        view_of_mask = urls.str.contains("view_of_", na=False)

        # 2. Ensure only standard image extensions are allowed (.jpg, .jpeg, .png, .webp)
        valid_ext_mask = clean_urls.str.endswith(
            (".jpg", ".jpeg", ".png", ".webp"), na=False
        )

        # 3. Reject non-image original formats (even if converted to .jpg thumbnails by MediaWiki)
        bad_orig_mask = clean_urls.str.contains(
            r"\.(tif|tiff|svg|pdf|djvu|gif|ogg|ogv|webm|mp3|mp4|wav|zip|gz|tar)\b",
            case=False,
            regex=True,
        )

        # Combined filter mask for removal
        remove_mask = view_of_mask | (~valid_ext_mask) | bad_orig_mask
        removed_count = remove_mask.sum()

        if removed_count > 0:
            df_cleaned = df[~remove_mask].copy()
            try:
                save_dataframe(df_cleaned, file_path)
                total_removed += removed_count
                cleaned_files += 1
                # Use tqdm.write to print without messing up progress bar
                tqdm.write(
                    f" -> Removed {removed_count:,} non-image/aerial records from {os.path.basename(file_path)}"
                )
            except Exception as e:
                tqdm.write(f"Error saving {os.path.basename(file_path)}: {e}")

    print(
        "\n================================================================================"
    )
    print("Cleanup Summary:")
    print(f"- Total files scanned: {len(files)}")
    print(f"- Files updated: {cleaned_files}")
    print(f"- Total non-image/aerial records removed: {total_removed:,}")
    print(
        "================================================================================"
    )


if __name__ == "__main__":
    main()
