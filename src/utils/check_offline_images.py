import glob
import os

import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.io import resolve_offline_image_path


def main():
    params_path = "params.yaml"
    if not os.path.exists(params_path):
        print(f"Error: Could not find {params_path} in the current project directory.")
        return

    print(f"Reading configuration from {params_path}...")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Support both nested 'pipeline' block and root level config
    pipeline_params = params.get("pipeline", {}) if isinstance(params, dict) else {}
    offline_dirs_str = ""
    if isinstance(pipeline_params, dict):
        offline_dirs_str = pipeline_params.get("offline_dataset_dirs", "")
    if not offline_dirs_str and isinstance(params, dict):
        offline_dirs_str = params.get("offline_dataset_dirs", "")

    if not offline_dirs_str:
        print(
            "No offline dataset directories configured in params.yaml (pipeline.offline_dataset_dirs)."
        )
        return

    offline_dirs = [d.strip() for d in offline_dirs_str.split() if d.strip()]
    print(f"Configured offline directories: {offline_dirs}")

    # Gather all CSV files in those directories
    csv_files = []
    for d in offline_dirs:
        if os.path.exists(d):
            found_csvs = glob.glob(os.path.join(d, "*.csv"))
            print(
                f"Found {len(found_csvs)} metadata CSV files in directory '{d}': {found_csvs}"
            )
            csv_files.extend(found_csvs)
        else:
            print(f"Warning: Directory '{d}' does not exist on this machine.")

    if not csv_files:
        print("No metadata CSV files found in the configured offline directories.")
        return

    print("\nStarting offline images accessibility check...")
    for f in csv_files:
        print("=" * 60)
        print(f"Checking metadata file: {f}")
        try:
            # Load metadata
            df = pd.read_csv(f)
            total_rows = len(df)
            print(f"Total entries in metadata: {total_rows:,}")

            if total_rows == 0:
                print("Metadata file is empty. Skipping.")
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
                    "Error: Could not find Photo_ID or Image_Location/Image_URL columns in this CSV."
                )
                continue

            found_count = 0
            missing_samples = []

            # Check each image using our unified resolve function
            for _, row in tqdm(
                df.iterrows(), total=total_rows, desc="Verifying local files"
            ):
                photo_id = row[photo_col]
                platform = row[platform_col] if platform_col else None
                url = row[url_col]

                # Resolve path using the exact pipeline logic
                resolved = resolve_offline_image_path(
                    url, offline_dirs, photo_id, platform
                )
                if resolved and os.path.exists(resolved):
                    found_count += 1
                else:
                    if len(missing_samples) < 5:
                        missing_samples.append((photo_id, url))

            ratio = found_count / total_rows
            print(f"\nResults for {os.path.basename(f)}:")
            print(f" - Found: {found_count:,} / {total_rows:,} images on disk")
            print(f" - Accessibility Ratio: {ratio * 100:.2f}%")

            if missing_samples:
                print(" - Sample missing entries (first 5):")
                for pid, url in missing_samples:
                    print(f"   * Photo_ID: {pid} | Configured Path: {url}")
            else:
                print(" - All images are successfully accessible on disk!")

        except Exception as e:
            print(f"Error processing {f}: {e}")


if __name__ == "__main__":
    main()
