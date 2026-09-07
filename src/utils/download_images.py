import argparse
import concurrent.futures
import hashlib
import os
import sys
import threading

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

from src.utils.io import (
    KNOWN_PLACEHOLDER_MD5_HASHES,
    is_valid_image_file,
    load_dataframe,
    load_embeddings,
    resolve_offline_image_path,
    resolve_wildlife_insights_url,
    save_dataframe,
)

# Try to load .env variables if not already set
if os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    val = v.strip().strip('"').strip("'")
                    if k.strip() == "MAPILLARY_TOKEN" and not os.environ.get(
                        "MAPILLARY_TOKEN"
                    ):
                        os.environ["MAPILLARY_TOKEN"] = val
                    elif k.strip() == "WILDLIFE_INSIGHTS_COOKIE" and not os.environ.get(
                        "WILDLIFE_INSIGHTS_COOKIE"
                    ):
                        os.environ["WILDLIFE_INSIGHTS_COOKIE"] = val
                    elif k.strip() == "WILDLIFE_INSIGHTS_TOKEN" and not os.environ.get(
                        "WILDLIFE_INSIGHTS_TOKEN"
                    ):
                        os.environ["WILDLIFE_INSIGHTS_TOKEN"] = val
    except Exception:
        pass

MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "")


def download_image(url, output_path, photo_id, platform, timeout=10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Configure session with retries and backoff
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    platform_lower = str(platform).strip().lower() if platform else ""
    photo_str = str(photo_id).strip() if photo_id else ""
    if photo_str.endswith(".0"):
        photo_str = photo_str[:-2]

    is_mapillary = (
        platform_lower == "mapillary" or "mapillary" in url or "fbcdn.net" in url
    )
    is_kartaview = (
        platform_lower == "kartaview" or "kartaview" in url or "openstreetcam" in url
    )
    is_wildlife = (
        platform_lower in ["snapshotusa", "snapshot_usa", "wildlife_insights"]
        or "wildlifeinsights.org" in url
    )

    def try_fetch(target_url, headers=None):
        if not target_url:
            return False
        try:
            res = session.get(target_url, headers=headers, timeout=timeout, stream=True)
            if res.status_code == 200:
                c_type = res.headers.get("content-type", "").lower()
                if "text/html" in c_type:
                    return False

                chunk_iter = res.iter_content(chunk_size=8192)
                first_chunk = next(chunk_iter, None)
                if not first_chunk or not is_valid_image_file(first_chunk):
                    return False

                temp_path = f"{output_path}.tmp_{os.getpid()}_{threading.get_ident()}"
                hasher = hashlib.md5()
                hasher.update(first_chunk)
                with open(temp_path, "wb") as f:
                    f.write(first_chunk)
                    for chunk in chunk_iter:
                        hasher.update(chunk)
                        f.write(chunk)

                # Reject known censor/placeholder images (e.g. Wildlife Insights "Human in frame")
                if hasher.hexdigest() in KNOWN_PLACEHOLDER_MD5_HASHES:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                    return False

                os.replace(temp_path, output_path)
                return True
        except Exception:
            pass
        return False

    # 1. For Wildlife Insights: NEVER direct GET the web viewer URL
    if is_wildlife:
        if not ("storage.googleapis.com" in url or "googleusercontent.com" in url):
            fresh_url = resolve_wildlife_insights_url(url, photo_id=photo_str)
            if fresh_url and try_fetch(fresh_url):
                return True
        else:
            if try_fetch(url):
                return True
        return False

    # 2. Standard direct image URLs (Flickr, direct CDN, etc.)
    if not (url.startswith("mapillary://") or url.startswith("kartaview://")):
        if try_fetch(url):
            return True

    # 3. Dynamic recovery for Mapillary or KartaView expired signatures
    if is_mapillary and photo_str:
        api_url = f"https://graph.mapillary.com/{photo_str}?fields=thumb_1024_url"
        headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
        try:
            api_res = session.get(api_url, headers=headers, timeout=timeout)
            if api_res.status_code == 200:
                fresh_url = api_res.json().get("thumb_1024_url")
                if fresh_url and try_fetch(fresh_url):
                    return True
        except Exception:
            pass

    elif is_kartaview and photo_str:
        api_url = f"https://api.openstreetcam.org/2.0/photo/{photo_str}"
        try:
            api_res = session.get(api_url, timeout=timeout)
            if api_res.status_code == 200:
                data = api_res.json().get("result", {}).get("data", {})
                fresh_url = (
                    data.get("fileurlLTh")
                    or data.get("fileurlTh")
                    or data.get("fileurl")
                )
                if fresh_url and try_fetch(fresh_url):
                    return True
        except Exception:
            pass

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and archive online dataset images for robust offline evaluations."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input Parquet dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save downloaded images.",
    )
    parser.add_argument(
        "--image_root_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of existing local image directories to check before downloading.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the updated offline metadata file (.csv or .parquet). Defaults to [input_base]_offline.csv.",
    )
    parser.add_argument(
        "--threads", type=int, default=32, help="Number of download threads."
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        default="cls",
        choices=["cls", "avg_patch", "cls_avg_patch"],
        help="Type of representation embedding to update.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Floating point precision format for stored embeddings (float32 or float16).",
    )
    parser.add_argument(
        "--wildlife_cookie",
        type=str,
        default=None,
        help="Wildlife Insights session cookie (connect.sid=...) for camera trap downloads.",
    )
    parser.add_argument(
        "--wildlife_token",
        type=str,
        default=None,
        help="Wildlife Insights JWT authorization token.",
    )
    args = parser.parse_args()

    if args.wildlife_cookie:
        os.environ["WILDLIFE_INSIGHTS_COOKIE"] = args.wildlife_cookie
    if args.wildlife_token:
        os.environ["WILDLIFE_INSIGHTS_TOKEN"] = args.wildlife_token

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # 1. Load DataFrame
    print(f"Loading dataset metadata from {args.input}...")
    df = load_dataframe(args.input)
    print(f" -> Loaded {len(df):,} metadata records.")

    # 2. Load Embeddings
    print("Checking companion embeddings matrix...")
    embeddings = None
    try:
        embeddings = load_embeddings(
            args.input, representation_type=args.representation_type
        )
        print(f" -> Loaded embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(
            f" -> No companion embeddings found ({e}). Proceeding with metadata-only download."
        )

    # Validate row alignment if embeddings exist
    if embeddings is not None:
        if "embedding_idx" in df.columns:
            valid_mask = (df["embedding_idx"] >= 0) & (
                df["embedding_idx"] < len(embeddings)
            )
            if not valid_mask.all():
                print(
                    f"Warning: Found {np.sum(~valid_mask):,} rows with out-of-bounds embedding_idx. Slicing embeddings..."
                )
                df = df.iloc[valid_mask.values].reset_index(drop=True)
                embeddings = embeddings[df["embedding_idx"].values]
        else:
            if len(df) != len(embeddings):
                print(
                    f"Error: Shape mismatch. Metadata has {len(df)} rows, but embeddings has {len(embeddings)} rows."
                )
                sys.exit(1)

    # 3. Identify images to download
    print("Checking local image cache...")

    check_dirs = []
    if args.image_root_dirs:
        check_dirs.extend(args.image_root_dirs)
    if args.output_dir not in check_dirs:
        check_dirs.append(args.output_dir)

    to_download = []
    successful_indices = []

    for i, row in enumerate(
        tqdm(df.itertuples(), total=len(df), desc="Scanning image status")
    ):
        url = getattr(row, "Image_URL", "")
        photo_id = getattr(row, "Photo_ID", "")
        platform = getattr(row, "Platform", "")

        # Check if already exists in check_dirs and is a valid image
        existing_path = resolve_offline_image_path(
            url, check_dirs, photo_id=photo_id, platform=platform
        )
        if existing_path and is_valid_image_file(existing_path):
            successful_indices.append(i)
        else:
            if existing_path and os.path.isfile(existing_path):
                try:
                    os.remove(existing_path)
                except Exception:
                    pass
            # Build target output path
            platform_str = str(platform).strip().lower() or "unknown"
            photo_str = str(photo_id).strip()
            if photo_str.endswith(".0"):
                photo_str = photo_str[:-2]
            output_name = f"{photo_str}.jpg"
            target_path = os.path.join(args.output_dir, platform_str, output_name)

            to_download.append((i, url, target_path, photo_id, platform))

    print(f" -> Found {len(df) - len(to_download):,} images already offline.")
    print(f" -> Need to download {len(to_download):,} online images.")

    # 4. Multi-threaded download
    download_success_count = 0
    if to_download:
        print(f"Starting downloads using {args.threads} threads...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.threads
        ) as executor:
            futures = {
                executor.submit(
                    download_image, item[1], item[2], item[3], item[4]
                ): item
                for item in to_download
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Downloading images",
            ):
                item = futures[future]
                idx = item[0]
                success = future.result()
                if success:
                    successful_indices.append(idx)
                    download_success_count += 1

    print(
        f"\nDownload run finished. Successfully downloaded {download_success_count:,} / {len(to_download):,} images."
    )

    # 5. Filter and save output
    successful_indices = sorted(successful_indices)
    print(
        f"\nFiltering dataset to include only successfully resolved images ({len(successful_indices):,} / {len(df):,})..."
    )

    df_clean = df.iloc[successful_indices].copy()
    embeddings_clean = (
        embeddings[successful_indices] if embeddings is not None else None
    )

    # Resolve output paths
    if args.output:
        out_metadata = args.output
    else:
        in_dir = os.path.dirname(os.path.abspath(args.input))
        in_base = os.path.splitext(os.path.basename(args.input))[0]
        out_metadata = os.path.join(in_dir, f"{in_base}_offline.csv")

    out_dir = os.path.dirname(os.path.abspath(out_metadata))
    out_base = os.path.splitext(os.path.basename(out_metadata))[0]

    # Compute relative Image_Location and file_name columns to make it drop-in compatible with offline datasets like iwildcam_subset
    file_names = []
    local_locations = []

    for row in df_clean.itertuples():
        photo_id = getattr(row, "Photo_ID", "")
        platform = getattr(row, "Platform", "")
        platform_str = str(platform).strip().lower() or "unknown"
        photo_str = str(photo_id).strip()
        if photo_str.endswith(".0"):
            photo_str = photo_str[:-2]
        name = f"{photo_str}.jpg"

        # Determine output absolute path
        abs_img_path = os.path.abspath(
            os.path.join(args.output_dir, platform_str, name)
        )

        # Calculate path relative to the metadata output directory
        try:
            rel_path = "./" + os.path.relpath(abs_img_path, out_dir)
        except Exception:
            rel_path = os.path.join(args.output_dir, platform_str, name)

        file_names.append(name)
        local_locations.append(rel_path)

    df_clean["file_name"] = file_names
    df_clean["Image_Location"] = local_locations
    if "Image_URL" in df_clean.columns:
        df_clean["Image_URL"] = local_locations
    if "url" in df_clean.columns:
        df_clean["url"] = local_locations

    temp_parquet_path = os.path.join(out_dir, f"{out_base}.parquet")
    if embeddings_clean is not None:
        # Drop existing embedding_idx so save_dataframe will rebuild it for the new 1-to-1 matrix
        if "embedding_idx" in df_clean.columns:
            df_clean = df_clean.drop(columns=["embedding_idx"])

        # Re-insert embedding to let save_dataframe decouple it dynamically
        df_clean["embedding"] = list(embeddings_clean)

        # Leverage the tested save_dataframe logic by writing to a temporary parquet file (which handles the .npy decoupling)
        print("Utilizing save_dataframe() to decouple companion embeddings matrix...")
        save_dataframe(
            df_clean,
            temp_parquet_path,
            representation_type=args.representation_type,
            precision=args.precision,
        )
    else:
        print(f"Saving updated offline metadata to {temp_parquet_path}...")
        save_dataframe(df_clean, temp_parquet_path)

    # Load back the decoupled dataframe containing the generated 'embedding_idx' column
    df_decoupled = load_dataframe(temp_parquet_path)

    # Save to CSV or Parquet based on requested format
    ext = os.path.splitext(out_metadata)[1].lower()
    if ext == ".parquet":
        # Re-save the clean parquet to the requested location
        if out_metadata != temp_parquet_path:
            os.replace(temp_parquet_path, out_metadata)
    else:
        # Save CSV metadata
        print(f"Saving offline dataset metadata to CSV: {out_metadata}...")
        df_decoupled.to_csv(out_metadata, index=False)
        # Clean up temporary parquet file
        if os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)

    out_npy = os.path.join(
        out_dir, f"{out_base}_{args.representation_type}_embeddings.npy"
    )
    print("\n🎉 Offline dataset created successfully!")
    print(f" -> Metadata: {out_metadata}")
    print(f" -> Embeddings: {out_npy}")


if __name__ == "__main__":
    main()
