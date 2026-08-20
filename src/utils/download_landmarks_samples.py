import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm


def download_image(url, output_path, session, delay=0.2):
    """Downloads a single image from url and writes it to output_path with rate-limit retries."""
    if delay > 0:
        time.sleep(delay)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(r.content)
                return True
            elif r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_time = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else (5 * (attempt + 1))
                )
                time.sleep(sleep_time)
            else:
                break
        except Exception:
            pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download sample images from Google Landmarks CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/user/aaniraj/home/Documents/Projects/data/google_landmarks_v2/google_landmarks_metadata_sampled_h3_res11_max100.csv",
        help="Path to Google Landmarks CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/user/aaniraj/home/Documents/Projects/data/google_landmarks_v2/images",
        help="Output directory to save images.",
    )
    parser.add_argument(
        "--num_images", type=int, default=100, help="Number of images to download."
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Parallel download threads."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay in seconds between requests per thread.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at: {args.csv}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)

    # Drop rows without Image_URL or file_name
    df = df.dropna(subset=["Image_URL", "file_name"])

    if len(df) == 0:
        print("Error: No valid URLs or file names found in CSV.")
        return

    sample_df = df.head(args.num_images)
    print(f"Planning to download {len(sample_df)} images to: {args.out_dir}...")

    session = requests.Session()
    # Configure custom User-Agent to prevent Wikimedia HTTP 403 Forbidden blocks
    session.headers.update(
        {
            "User-Agent": "Geo-RAG-Landmark-Downloader/1.0 (aaniraj@home; contact: aaniraj@home.com)"
        }
    )

    success_count = 0
    from functools import partial

    download_fn = partial(download_image, session=session, delay=args.delay)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for _, row in sample_df.iterrows():
            url = row["Image_URL"]
            file_name = row["file_name"]
            out_path = os.path.join(args.out_dir, file_name)
            futures[executor.submit(download_fn, url, out_path)] = file_name

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            file_name = futures[fut]
            if fut.result():
                success_count += 1

    print(
        f"\n✅ Completed! Successfully downloaded {success_count}/{len(sample_df)} images."
    )


if __name__ == "__main__":
    main()
