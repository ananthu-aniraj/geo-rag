import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
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


def bulk_resolve_mapillary_urls(
    photo_ids,
    access_token,
    cache_path=None,
    batch_size=250,
    max_workers=4,
    max_retries=5,
):
    """
    Bulk resolves Mapillary photo IDs to fresh 30-day signed CDN URLs (thumb_1024_url)
    using the multi-ID endpoint (GET /?ids=id1,id2,...&fields=thumb_1024_url).
    Supports persistent disk caching, multi-threading, and automatic backoff on 429 rate limits.
    """
    if not photo_ids or not access_token:
        return {}

    unique_ids = list(
        dict.fromkeys(
            str(pid).strip()[:-2]
            if str(pid).strip().endswith(".0")
            else str(pid).strip()
            for pid in photo_ids
            if pid
        )
    )

    resolved = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            resolved = dict(
                zip(
                    cache_df["photo_id"].astype(str),
                    cache_df["url"].astype(str),
                )
            )
            print(
                f" -> Loaded {len(resolved):,} previously resolved Mapillary URLs from cache: {cache_path}"
            )
        except Exception as e:
            print(f" -> Warning: Could not read Mapillary URL cache ({e}).")

    missing_ids = [pid for pid in unique_ids if pid not in resolved]
    if not missing_ids:
        return resolved

    print(
        f" -> Querying Mapillary Graph API for {len(missing_ids):,} unresolved photo IDs (batch size: {batch_size})..."
    )

    batches = [
        missing_ids[i : i + batch_size] for i in range(0, len(missing_ids), batch_size)
    ]
    session = requests.Session()

    def fetch_batch(batch):
        ids_str = ",".join(batch)
        url = f"https://graph.mapillary.com/?ids={ids_str}&fields=thumb_1024_url"
        headers = {"Authorization": f"OAuth {access_token}"}
        for attempt in range(max_retries):
            try:
                res = session.get(url, headers=headers, timeout=25)
                if res.status_code == 200:
                    data = res.json()
                    # Check usage header and pace if nearing limit
                    usage_header = res.headers.get("x-app-usage", "")
                    if usage_header and '"call_volume":' in usage_header:
                        try:
                            usage_dict = json.loads(usage_header)
                            if usage_dict.get("call_volume", 0) > 85:
                                time.sleep(5)
                        except Exception:
                            pass

                    out = {}
                    for pid, val in data.items():
                        if isinstance(val, dict) and val.get("thumb_1024_url"):
                            out[str(pid)] = val["thumb_1024_url"]
                    return out
                elif res.status_code == 429:
                    wait_time = min(120, 15 * (2**attempt))
                    time.sleep(wait_time)
                elif res.status_code >= 500:
                    time.sleep(2 * (attempt + 1))
                else:
                    return {}
            except Exception:
                time.sleep(2 * (attempt + 1))
        return {}

    new_resolutions = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_res in tqdm(
            executor.map(fetch_batch, batches),
            total=len(batches),
            desc="Resolving Mapillary CDN URLs",
        ):
            resolved.update(batch_res)
            new_resolutions += len(batch_res)

    if cache_path and new_resolutions > 0:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            pd.DataFrame(
                {
                    "photo_id": list(resolved.keys()),
                    "url": list(resolved.values()),
                }
            ).to_parquet(cache_path, compression="zstd")
            print(f" -> Saved updated Mapillary URL cache to {cache_path}")
        except Exception as e:
            print(f" -> Warning: Could not write Mapillary URL cache ({e}).")

    return resolved


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
            elif api_res.status_code == 429:
                time.sleep(10)
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
        help="Path to write the updated offline metadata file (.parquet or .csv). Defaults to [input_base]_offline.parquet.",
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
    parser.add_argument(
        "--mapillary_token",
        type=str,
        default=None,
        help="Mapillary access token (overrides MAPILLARY_TOKEN env/secret).",
    )
    parser.add_argument(
        "--mapillary_batch_size",
        type=int,
        default=250,
        help="Batch size of photo IDs per Mapillary Graph API request (default: 250).",
    )
    parser.add_argument(
        "--mapillary_resolver_threads",
        type=int,
        default=4,
        help="Number of threads for bulk Mapillary URL resolution (default: 4).",
    )
    args = parser.parse_args()

    if args.wildlife_cookie:
        os.environ["WILDLIFE_INSIGHTS_COOKIE"] = args.wildlife_cookie
    if args.wildlife_token:
        os.environ["WILDLIFE_INSIGHTS_TOKEN"] = args.wildlife_token
    if args.mapillary_token:
        os.environ["MAPILLARY_TOKEN"] = args.mapillary_token
        global MAPILLARY_TOKEN
        MAPILLARY_TOKEN = args.mapillary_token

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

    # Pre-resolve Mapillary virtual URIs in bulk using Graph API multi-ID lookups
    mapillary_missing_ids = []
    for item in to_download:
        item_url, item_photo_id, item_platform = item[1], item[3], item[4]
        plat_lower = str(item_platform).strip().lower() if item_platform else ""
        if plat_lower == "mapillary" and (
            item_url.startswith("mapillary://") or "thumb_1024_url" not in item_url
        ):
            photo_str = str(item_photo_id).strip()
            if photo_str.endswith(".0"):
                photo_str = photo_str[:-2]
            if photo_str:
                mapillary_missing_ids.append(photo_str)

    if mapillary_missing_ids:
        token = (
            args.mapillary_token
            or MAPILLARY_TOKEN
            or os.environ.get("MAPILLARY_TOKEN", "")
        )
        if not token:
            print(
                "\nWarning: MAPILLARY_TOKEN not found in env, .env, or CLI args. "
                "Mapillary virtual URIs cannot be resolved."
            )
        else:
            cache_file = os.path.join(args.output_dir, ".mapillary_url_cache.parquet")
            print(
                f"\nPre-resolving {len(mapillary_missing_ids):,} Mapillary photo IDs via bulk Graph API..."
            )
            resolved_mapillary = bulk_resolve_mapillary_urls(
                mapillary_missing_ids,
                access_token=token,
                cache_path=cache_file,
                batch_size=args.mapillary_batch_size,
                max_workers=args.mapillary_resolver_threads,
            )
            print(
                f" -> Successfully resolved {len(resolved_mapillary):,} Mapillary CDN URLs."
            )

            # Update to_download tuples with fresh CDN URLs
            updated_to_download = []
            for item in to_download:
                idx, url, target_path, photo_id, platform = item
                photo_str = str(photo_id).strip()
                if photo_str.endswith(".0"):
                    photo_str = photo_str[:-2]
                if photo_str in resolved_mapillary:
                    url = resolved_mapillary[photo_str]
                updated_to_download.append((idx, url, target_path, photo_id, platform))
            to_download = updated_to_download

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

    # Resolve output paths (preserving dataset format, defaulting to .parquet)
    if args.output:
        out_metadata = args.output
    else:
        in_dir = os.path.dirname(os.path.abspath(args.input))
        in_base = os.path.splitext(os.path.basename(args.input))[0]
        in_ext = os.path.splitext(args.input)[1].lower()
        if in_ext not in [".parquet", ".csv"]:
            in_ext = ".parquet"
        out_metadata = os.path.join(in_dir, f"{in_base}_offline{in_ext}")

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

    if embeddings_clean is not None:
        # Drop existing embedding_idx so save_dataframe will rebuild companion files for the new 1-to-1 matrix
        if "embedding_idx" in df_clean.columns:
            df_clean = df_clean.drop(columns=["embedding_idx"])

        # Re-insert embedding to let save_dataframe decouple it dynamically
        df_clean["embedding"] = list(embeddings_clean)

    print(
        f"Saving offline dataset metadata to {out_metadata} using save_dataframe()..."
    )
    save_dataframe(
        df_clean,
        out_metadata,
        representation_type=args.representation_type,
        precision=args.precision,
    )

    out_npy = os.path.join(
        out_dir, f"{out_base}_{args.representation_type}_embeddings.npy"
    )
    print("\n🎉 Offline dataset created successfully!")
    print(f" -> Metadata: {out_metadata}")
    if embeddings_clean is not None:
        print(f" -> Embeddings: {out_npy}")


if __name__ == "__main__":
    main()
