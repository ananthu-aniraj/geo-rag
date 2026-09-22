import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
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


def save_stream_checkpoint(
    out_metadata,
    df_combined,
    embeddings_combined=None,
    representation_type="cls",
    precision="float32",
):
    """
    Atomically writes the combined metadata DataFrame and companion embeddings
    (.npy and .keys.parquet) to disk using temporary files and atomic replaces.
    Ensures that metadata, embeddings matrix, and key index remain in lockstep.
    """
    out_dir = os.path.dirname(os.path.abspath(out_metadata))
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.splitext(os.path.basename(out_metadata))[0]
    out_ext = os.path.splitext(out_metadata)[1].lower()

    rep_suffix = representation_type or "cls"
    dtype = np.float16 if precision == "float16" else np.float32

    npy_path = os.path.join(out_dir, f"{out_base}_{rep_suffix}_embeddings.npy")
    keys_path = os.path.join(
        out_dir, f"{out_base}_{rep_suffix}_embeddings.keys.parquet"
    )

    tmp_out = f"{out_metadata}.tmp_stream"
    tmp_npy = f"{out_metadata}.tmp_{rep_suffix}_embeddings.npy"
    tmp_keys = f"{out_metadata}.tmp_{rep_suffix}_embeddings.keys.parquet"

    # 1. Embeddings & Keys
    if embeddings_combined is not None:
        embs_arr = np.asarray(embeddings_combined, dtype=dtype)
        np.save(tmp_npy, embs_arr)
        keys_df = pd.DataFrame({"photo_key": df_combined["photo_key"].astype(str)})
        keys_df.to_parquet(tmp_keys, compression="zstd")

    # 2. DataFrame Metadata
    df_to_save = df_combined.drop(
        columns=["embedding", "embedding_idx"], errors="ignore"
    )
    if out_ext == ".csv":
        df_to_save.to_csv(tmp_out, index=False)
    else:
        df_to_save.to_parquet(tmp_out, index=False, compression="zstd")

    # 3. Atomic replacement
    if os.path.exists(tmp_out):
        os.replace(tmp_out, out_metadata)
    if embeddings_combined is not None:
        if os.path.exists(tmp_npy):
            os.replace(tmp_npy, npy_path)
        if os.path.exists(tmp_keys):
            os.replace(tmp_keys, keys_path)


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
        "--resume",
        action="store_true",
        help="Resume downloading by skipping images already present in the output Parquet file and streaming updates as new images are downloaded.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to an existing output Parquet file to resume from (defaults to --output or [input_base]_offline.parquet if --resume is set).",
    )
    parser.add_argument(
        "--verify_existing",
        action="store_true",
        help="Re-verify on disk that all existing records in the resume file are present and valid image files (by default, resume trusts records already written to the resume file for fast resumption).",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=60,
        help="Interval in seconds for periodic streaming updates / checkpoints to the output Parquet file during downloads (default: 60, set to 0 to disable periodic streaming).",
    )
    parser.add_argument(
        "--copy_offline_images",
        action="store_true",
        help="Copy existing offline images found in --image_root_dirs into --output_dir and update their paths in the output dataset.",
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

    # 1. Resolve output metadata destination
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
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load DataFrame
    print(f"Loading dataset metadata from {args.input}...")
    df = load_dataframe(args.input)
    print(f" -> Loaded {len(df):,} metadata records.")

    # Ensure deterministic, lowercase photo_key
    if "photo_key" not in df.columns:
        plat_col = (
            df["Platform"].astype(str).str.strip().str.lower()
            if "Platform" in df.columns
            else ["unknown"] * len(df)
        )
        pid_col = [
            str(pid).strip()[:-2]
            if str(pid).strip().endswith(".0")
            else str(pid).strip()
            for pid in df["Photo_ID"]
        ]
        df["photo_key"] = [f"{plat}_{pid}" for plat, pid in zip(plat_col, pid_col)]
    else:
        df["photo_key"] = df["photo_key"].astype(str)

    # 3. Load Embeddings
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

    # 4. Check for existing output / resume
    is_resuming = args.resume or (args.resume_from is not None)
    resume_file = args.resume_from or (out_metadata if is_resuming else None)

    df_existing = None
    existing_embeddings = None
    existing_keys = set()

    if is_resuming and resume_file:
        if os.path.exists(resume_file):
            print(f"\nResuming from existing output file: {resume_file}")
            df_existing = load_dataframe(resume_file)
            if "photo_key" not in df_existing.columns:
                plat_col = (
                    df_existing["Platform"].astype(str).str.strip().str.lower()
                    if "Platform" in df_existing.columns
                    else ["unknown"] * len(df_existing)
                )
                pid_col = [
                    str(pid).strip()[:-2]
                    if str(pid).strip().endswith(".0")
                    else str(pid).strip()
                    for pid in df_existing["Photo_ID"]
                ]
                df_existing["photo_key"] = [
                    f"{plat}_{pid}" for plat, pid in zip(plat_col, pid_col)
                ]
            else:
                df_existing["photo_key"] = df_existing["photo_key"].astype(str)

            try:
                existing_embeddings = load_embeddings(
                    resume_file, representation_type=args.representation_type
                )
                print(
                    f" -> Loaded {len(existing_embeddings):,} companion embeddings from resume file."
                )
            except Exception as e:
                existing_embeddings = None
                print(f" -> No companion embeddings found for resume file ({e}).")

            # Verify existing records on disk
            if args.verify_existing:
                print(
                    f" -> Verifying {len(df_existing):,} existing records on disk (--verify_existing enabled)..."
                )

                def check_existing_row(row):
                    loc = getattr(row, "Image_Location", None) or getattr(
                        row, "Image_URL", ""
                    )
                    photo_id = getattr(row, "Photo_ID", "")
                    platform = getattr(row, "Platform", "")
                    platform_str = str(platform).strip().lower() or "unknown"
                    photo_str = str(photo_id).strip()
                    if photo_str.endswith(".0"):
                        photo_str = photo_str[:-2]
                    target_name = f"{photo_str}.jpg"
                    target_path = os.path.join(
                        args.output_dir, platform_str, target_name
                    )

                    # Resolve path on disk
                    abs_p = (
                        os.path.abspath(os.path.join(out_dir, loc))
                        if loc and not os.path.isabs(loc)
                        else loc
                    )
                    valid = bool(abs_p and is_valid_image_file(abs_p))

                    if valid and args.copy_offline_images:
                        if os.path.abspath(abs_p) != os.path.abspath(target_path):
                            if not (
                                os.path.exists(target_path)
                                and is_valid_image_file(target_path)
                            ):
                                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                shutil.copy2(abs_p, target_path)
                            abs_p = target_path

                    if valid and os.path.exists(abs_p):
                        try:
                            rel_p = "./" + os.path.relpath(abs_p, out_dir)
                        except Exception:
                            rel_p = abs_p
                        return True, rel_p, os.path.basename(abs_p)
                    return False, None, None

                verify_threads = min(args.threads, 32)
                valid_existing_mask = []
                verified_locations = []
                verified_filenames = []

                if len(df_existing) > 500 and verify_threads > 1:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=verify_threads
                    ) as executor:
                        results = list(
                            tqdm(
                                executor.map(
                                    check_existing_row,
                                    df_existing.itertuples(index=False),
                                    chunksize=256,
                                ),
                                total=len(df_existing),
                                desc="Verifying existing records on disk",
                            )
                        )
                    for is_valid, rel_p, fn in results:
                        valid_existing_mask.append(is_valid)
                        verified_locations.append(rel_p)
                        verified_filenames.append(fn)
                else:
                    for row in tqdm(
                        df_existing.itertuples(index=False),
                        total=len(df_existing),
                        desc="Verifying existing records on disk",
                    ):
                        is_valid, rel_p, fn = check_existing_row(row)
                        valid_existing_mask.append(is_valid)
                        verified_locations.append(rel_p)
                        verified_filenames.append(fn)

                valid_mask_arr = np.array(valid_existing_mask, dtype=bool)
                if not valid_mask_arr.all():
                    num_inv = int(np.sum(~valid_mask_arr))
                    print(
                        f" -> Found {num_inv:,} records in resume file with missing/corrupt image files on disk. These will be re-downloaded."
                    )
                    df_existing = df_existing.iloc[valid_mask_arr].reset_index(
                        drop=True
                    )
                    if existing_embeddings is not None:
                        existing_embeddings = existing_embeddings[valid_mask_arr]
                    verified_locations = [
                        loc for loc, v in zip(verified_locations, valid_mask_arr) if v
                    ]
                    verified_filenames = [
                        fn for fn, v in zip(verified_filenames, valid_mask_arr) if v
                    ]

                df_existing["Image_Location"] = verified_locations
                df_existing["file_name"] = verified_filenames
                if "Image_URL" in df_existing.columns:
                    df_existing["Image_URL"] = verified_locations
                if "url" in df_existing.columns:
                    df_existing["url"] = verified_locations

                existing_keys = set(df_existing["photo_key"].dropna().astype(str))
                print(
                    f" -> Resuming: {len(existing_keys):,} valid images verified in output Parquet ({resume_file}). Skipping these."
                )
            else:
                existing_keys = set(df_existing["photo_key"].dropna().astype(str))
                print(
                    f" -> Fast resume: {len(existing_keys):,} existing records loaded from output Parquet ({resume_file}). Skipping disk verification."
                )
                print(
                    "    (Tip: Use --verify_existing if you wish to re-verify all image files on disk)."
                )
        else:
            print(
                f" -> Resume requested, but output file '{resume_file}' does not exist yet. Starting fresh download."
            )

    # 5. Filter input DataFrame to remaining candidate images
    print(
        f"\nFiltering input dataset ({len(df):,} records) against existing resume keys..."
    )
    remaining_indices = [
        i for i, pk in enumerate(df["photo_key"]) if pk not in existing_keys
    ]
    print(
        f" -> Candidate images remaining to process: {len(remaining_indices):,} / {len(df):,}."
    )

    if not remaining_indices:
        print(
            "\nAll candidate images from the input dataset are already downloaded and verified in the output file! Nothing to download."
        )
        if df_existing is not None and not df_existing.empty:
            save_stream_checkpoint(
                out_metadata,
                df_existing,
                existing_embeddings,
                representation_type=args.representation_type,
                precision=args.precision,
            )
            print(f" -> Output verified and up-to-date at: {out_metadata}")
        return

    # 6. Scan local cache for offline images vs online downloads
    check_dirs = []
    if args.image_root_dirs:
        check_dirs.extend(args.image_root_dirs)
    if args.output_dir not in check_dirs:
        check_dirs.append(args.output_dir)
    in_dir = os.path.dirname(os.path.abspath(args.input))
    if in_dir not in check_dirs:
        check_dirs.append(in_dir)

    offline_to_copy = []
    offline_ready = []
    to_download = []

    print("\nScanning local image cache for remaining images...")
    for i in tqdm(remaining_indices, desc="Scanning image status"):
        row = df.iloc[i]
        url = (
            getattr(row, "Image_Location", None)
            or getattr(row, "Image_URL", None)
            or getattr(row, "file_name", None)
            or ""
        )
        photo_id = getattr(row, "Photo_ID", "")
        platform = getattr(row, "Platform", "")
        platform_str = str(platform).strip().lower() or "unknown"
        photo_str = str(photo_id).strip()
        if photo_str.endswith(".0"):
            photo_str = photo_str[:-2]
        output_name = f"{photo_str}.jpg"
        target_path = os.path.join(args.output_dir, platform_str, output_name)
        try:
            rel_target_path = "./" + os.path.relpath(target_path, out_dir)
        except Exception:
            rel_target_path = target_path

        existing_path = resolve_offline_image_path(
            url, check_dirs, photo_id=photo_id, platform=platform
        )

        if existing_path and is_valid_image_file(existing_path):
            if args.copy_offline_images:
                if os.path.abspath(existing_path) != os.path.abspath(target_path):
                    offline_to_copy.append(
                        (i, existing_path, target_path, rel_target_path, output_name)
                    )
                else:
                    offline_ready.append(
                        (i, existing_path, rel_target_path, output_name)
                    )
            else:
                try:
                    rel_p = "./" + os.path.relpath(existing_path, out_dir)
                except Exception:
                    rel_p = existing_path
                offline_ready.append(
                    (i, existing_path, rel_p, os.path.basename(existing_path))
                )
        else:
            if (
                existing_path
                and os.path.isfile(existing_path)
                and os.path.abspath(existing_path).startswith(
                    os.path.abspath(args.output_dir)
                )
            ):
                try:
                    os.remove(existing_path)
                except Exception:
                    pass
            to_download.append(
                (
                    i,
                    url,
                    target_path,
                    photo_id,
                    platform,
                    photo_str,
                    platform_str,
                    rel_target_path,
                    output_name,
                )
            )

    print(
        f" -> Found {len(offline_to_copy) + len(offline_ready):,} offline images already on disk."
    )
    print(f" -> Need to download {len(to_download):,} online images.")

    # 7. Copy offline images to output_dir if requested
    if offline_to_copy:
        print(
            f"\nCopying {len(offline_to_copy):,} existing offline images to {args.output_dir}..."
        )

        def copy_worker(item):
            idx, src, dst, rel_p, fn = item
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not (os.path.exists(dst) and is_valid_image_file(dst)):
                shutil.copy2(src, dst)
            return (idx, dst, rel_p, fn)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.threads, 32)
        ) as executor:
            copied_items = list(
                tqdm(
                    executor.map(copy_worker, offline_to_copy),
                    total=len(offline_to_copy),
                    desc="Copying offline images",
                )
            )
        offline_ready.extend(copied_items)

    # 8. Integrate offline ready images into the output stream immediately
    if offline_ready:
        print(
            f"\nIntegrating {len(offline_ready):,} offline images into output dataset..."
        )
        offline_indices = [item[0] for item in offline_ready]
        df_offline = df.iloc[offline_indices].copy()
        df_offline["Image_Location"] = [item[2] for item in offline_ready]
        df_offline["file_name"] = [item[3] for item in offline_ready]
        df_offline["Image_URL"] = df_offline["Image_Location"]
        if "url" in df_offline.columns:
            df_offline["url"] = df_offline["Image_Location"]

        offline_embs = embeddings[offline_indices] if embeddings is not None else None

        if df_existing is not None and not df_existing.empty:
            df_existing = pd.concat([df_existing, df_offline], ignore_index=True)
            if existing_embeddings is not None and offline_embs is not None:
                existing_embeddings = np.concatenate(
                    [existing_embeddings, offline_embs], axis=0
                )
            elif offline_embs is not None:
                existing_embeddings = offline_embs
        else:
            df_existing = df_offline.reset_index(drop=True)
            existing_embeddings = offline_embs

        save_stream_checkpoint(
            out_metadata,
            df_existing,
            existing_embeddings,
            representation_type=args.representation_type,
            precision=args.precision,
        )
        print(
            f" -> Stream updated {out_metadata} ({len(df_existing):,} total images saved)."
        )

    # 9. Pre-resolve Mapillary virtual URIs in bulk using Graph API multi-ID lookups
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
                (
                    idx,
                    url,
                    target_path,
                    photo_id,
                    platform,
                    photo_str,
                    platform_str,
                    rel_target_path,
                    output_name,
                ) = item
                if photo_str in resolved_mapillary:
                    url = resolved_mapillary[photo_str]
                updated_to_download.append(
                    (
                        idx,
                        url,
                        target_path,
                        photo_id,
                        platform,
                        photo_str,
                        platform_str,
                        rel_target_path,
                        output_name,
                    )
                )
            to_download = updated_to_download

    # 10. Multi-threaded download with periodic streaming updates
    download_success_count = 0
    if to_download:
        print(
            f"\nStarting downloads for {len(to_download):,} images using {args.threads} threads..."
        )
        last_checkpoint_time = time.time()
        pending_downloaded = []

        def flush_pending():
            nonlocal \
                df_existing, \
                existing_embeddings, \
                pending_downloaded, \
                last_checkpoint_time
            if not pending_downloaded:
                return
            pending_downloaded.sort(key=lambda x: x[0])
            batch_indices = [item[0] for item in pending_downloaded]
            df_batch = df.iloc[batch_indices].copy()
            df_batch["Image_Location"] = [item[7] for item in pending_downloaded]
            df_batch["file_name"] = [item[8] for item in pending_downloaded]
            df_batch["Image_URL"] = df_batch["Image_Location"]
            if "url" in df_batch.columns:
                df_batch["url"] = df_batch["Image_Location"]

            batch_embs = embeddings[batch_indices] if embeddings is not None else None

            if df_existing is not None and not df_existing.empty:
                df_existing = pd.concat([df_existing, df_batch], ignore_index=True)
                if existing_embeddings is not None and batch_embs is not None:
                    existing_embeddings = np.concatenate(
                        [existing_embeddings, batch_embs], axis=0
                    )
                elif batch_embs is not None:
                    existing_embeddings = batch_embs
            else:
                df_existing = df_batch.reset_index(drop=True)
                existing_embeddings = batch_embs

            save_stream_checkpoint(
                out_metadata,
                df_existing,
                existing_embeddings,
                representation_type=args.representation_type,
                precision=args.precision,
            )
            print(
                f"\n -> Stream checkpoint saved: {len(df_existing):,} total images recorded in {out_metadata}."
            )
            pending_downloaded = []
            last_checkpoint_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.threads
        ) as executor:
            futures = {
                executor.submit(
                    download_image, item[1], item[2], item[3], item[4]
                ): item
                for item in to_download
            }

            try:
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Downloading images",
                ):
                    item = futures[future]
                    success = future.result()
                    if success:
                        pending_downloaded.append(item)
                        download_success_count += 1

                    if (
                        args.checkpoint_interval > 0
                        and (
                            time.time() - last_checkpoint_time
                            > args.checkpoint_interval
                        )
                        and pending_downloaded
                    ):
                        flush_pending()
            except KeyboardInterrupt:
                print(
                    "\n\nExecution interrupted by user! Flushing pending downloads to disk..."
                )
                flush_pending()
                print(
                    f"Safe checkpoint saved to {out_metadata}. You can resume later with --resume."
                )
                sys.exit(0)

        # Final flush for any remaining downloaded items
        flush_pending()
        print(
            f"\nDownload run finished. Successfully downloaded {download_success_count:,} / {len(to_download):,} images."
        )

    out_npy = os.path.join(
        out_dir, f"{out_base}_{args.representation_type}_embeddings.npy"
    )
    if df_existing is not None and not df_existing.empty:
        print("\n🎉 Offline dataset created successfully!")
        print(f" -> Total images recorded: {len(df_existing):,}")
        print(f" -> Metadata: {out_metadata}")
        if existing_embeddings is not None:
            print(f" -> Embeddings: {out_npy}")
    else:
        print("\n[!] No images were successfully downloaded or resolved.")


if __name__ == "__main__":
    main()
