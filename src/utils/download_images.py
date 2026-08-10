import argparse
import concurrent.futures
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm

from src.utils.io import load_dataframe, load_embeddings

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


def download_image(url, output_path, photo_id, platform, timeout=10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Configure session with retries and backoff
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        # 1. Try direct GET download
        res = session.get(url, timeout=timeout, stream=True)
        if res.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception:
        pass

    # 2. Dynamic recovery for Mapillary or KartaView expired signatures
    if not photo_id or not platform:
        return False

    platform_lower = str(platform).strip().lower()
    photo_str = str(photo_id).strip()
    if photo_str.endswith('.0'):
        photo_str = photo_str[:-2]

    is_mapillary = platform_lower == 'mapillary' or 'mapillary' in url or 'fbcdn.net' in url
    is_kartaview = platform_lower == 'kartaview' or 'kartaview' in url or 'openstreetcam' in url

    if not (is_mapillary or is_kartaview):
        return False

    try:
        fresh_url = None
        if is_mapillary:
            api_url = f"https://graph.mapillary.com/{photo_str}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
            api_res = session.get(api_url, headers=headers, timeout=timeout)
            if api_res.status_code == 200:
                fresh_url = api_res.json().get("thumb_1024_url")
        elif is_kartaview:
            api_url = f"https://api.openstreetcam.org/2.0/photo/{photo_str}"
            api_res = session.get(api_url, timeout=timeout)
            if api_res.status_code == 200:
                data = api_res.json().get("result", {}).get("data", {})
                fresh_url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")

        if fresh_url:
            res = session.get(fresh_url, timeout=timeout, stream=True)
            if res.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
    except Exception:
        pass

    return False


def main():
    parser = argparse.ArgumentParser(description="Download and archive online dataset images for robust offline evaluations.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input Parquet dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save downloaded images.")
    parser.add_argument("--image_root_dirs", type=str, nargs="*", default=None, 
                        help="Optional list of existing local image directories to check before downloading.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write the updated offline Parquet file. Defaults to [input_base]_offline.parquet.")
    parser.add_argument("--threads", type=int, default=32, help="Number of download threads.")
    parser.add_argument("--representation_type", type=str, default="cls", choices=["cls", "avg_patch", "cls_avg_patch"],
                        help="Type of representation embedding to update.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # 1. Load DataFrame
    print(f"Loading dataset metadata from {args.input}...")
    df = load_dataframe(args.input)
    print(f" -> Loaded {len(df):,} metadata records.")

    # 2. Load Embeddings
    print("Loading companion embeddings matrix...")
    try:
        embeddings = load_embeddings(args.input, representation_type=args.representation_type)
        print(f" -> Loaded embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error loading companion embeddings: {e}")
        sys.exit(1)

    # Validate row alignment
    if 'embedding_idx' in df.columns:
        valid_mask = (df['embedding_idx'] >= 0) & (df['embedding_idx'] < len(embeddings))
        if not valid_mask.all():
            print(f"Warning: Found {np.sum(~valid_mask):,} rows with out-of-bounds embedding_idx. Slicing embeddings...")
            df = df.iloc[valid_mask.values].reset_index(drop=True)
            embeddings = embeddings[df['embedding_idx'].values]
    else:
        if len(df) != len(embeddings):
            print(f"Error: Shape mismatch. Metadata has {len(df)} rows, but embeddings has {len(embeddings)} rows.")
            sys.exit(1)

    # 3. Identify images to download
    print("Checking local image cache...")
    from src.utils.io import resolve_offline_image_path
    
    check_dirs = []
    if args.image_root_dirs:
        check_dirs.extend(args.image_root_dirs)
    if args.output_dir not in check_dirs:
        check_dirs.append(args.output_dir)

    to_download = []
    successful_indices = []

    for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Scanning image status")):
        url = getattr(row, "Image_URL", "")
        photo_id = getattr(row, "Photo_ID", "")
        platform = getattr(row, "Platform", "")
        
        # Check if already exists in check_dirs
        existing_path = resolve_offline_image_path(url, check_dirs, photo_id=photo_id, platform=platform)
        if existing_path:
            successful_indices.append(i)
        else:
            # Build target output path
            platform_str = str(platform).strip().lower() or "unknown"
            photo_str = str(photo_id).strip()
            if photo_str.endswith('.0'):
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(download_image, item[1], item[2], item[3], item[4]): item
                for item in to_download
            }
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading images"):
                item = futures[future]
                idx = item[0]
                success = future.result()
                if success:
                    successful_indices.append(idx)
                    download_success_count += 1

    print(f"\nDownload run finished. Successfully downloaded {download_success_count:,} / {len(to_download):,} images.")

    # 5. Filter and save output
    successful_indices = sorted(successful_indices)
    print(f"\nFiltering dataset to include only successfully resolved images ({len(successful_indices):,} / {len(df):,})...")
    
    df_clean = df.iloc[successful_indices].copy()
    embeddings_clean = embeddings[successful_indices]

    # Resolve output paths
    if args.output:
        out_parquet = args.output
    else:
        in_dir = os.path.dirname(os.path.abspath(args.input))
        in_base = os.path.splitext(os.path.basename(args.input))[0]
        out_parquet = os.path.join(in_dir, f"{in_base}_offline.parquet")

    out_dir = os.path.dirname(os.path.abspath(out_parquet))
    out_base = os.path.splitext(os.path.basename(out_parquet))[0]

    # Update indices to be a dense, gap-free sequence in the new offline file
    df_clean['embedding_idx'] = np.arange(len(df_clean), dtype=np.int32)
    
    print(f"Saving offline dataset metadata to {out_parquet}...")
    df_clean.to_parquet(out_parquet, index=False, compression='zstd')
    
    out_npy = os.path.join(out_dir, f"{out_base}_{args.representation_type}_embeddings.npy")
    print(f"Saving aligned embeddings to {out_npy}...")
    np.save(out_npy, embeddings_clean)

    print(f"\n🎉 Offline dataset created successfully!")
    print(f" -> Parquet: {out_parquet}")
    print(f" -> Embeddings: {out_npy}")


if __name__ == "__main__":
    main()
