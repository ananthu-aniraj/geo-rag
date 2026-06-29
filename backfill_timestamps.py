import os
import time
import argparse
import datetime
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'
FLICKR_API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'

# Polite delay for Flickr API requests (seconds)
FLICKR_DELAY = 1.1


def fetch_mapillary_timestamps(photo_ids):
    """Fetches captured_at timestamps for multiple Mapillary photo IDs in batches."""
    if not photo_ids:
        return {}

    url = f"https://graph.mapillary.com/images?ids={','.join(photo_ids)}&fields=id,captured_at"
    headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
    results = {}

    try:
        # Mapillary rate limits: fetch up to 50 nodes per call
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            for item in res.json().get('data', []):
                pid = str(item.get('id'))
                cap_ms = item.get('captured_at')
                if cap_ms:
                    # Convert ms epoch to UTC ISO 8601
                    results[pid] = datetime.datetime.fromtimestamp(cap_ms / 1000.0, datetime.timezone.utc).strftime(
                        '%Y-%m-%dT%H:%M:%SZ')
        else:
            print(f"Mapillary API error: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"Error fetching Mapillary batch: {e}")
    return results


def fetch_flickr_timestamp(photo_id):
    """Fetches the date taken for a single Flickr photo ID."""
    url = (
        f"https://www.flickr.com/services/rest/"
        f"?method=flickr.photos.getInfo"
        f"&api_key={FLICKR_API_KEY}"
        f"&photo_id={photo_id}"
        f"&format=json"
        f"&nojsoncallback=1"
    )
    try:
        # Enforce rate limits by sleeping
        time.sleep(FLICKR_DELAY)
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get('stat') == 'ok':
                taken = data.get('photo', {}).get('dates', {}).get('taken', '')
                if taken:
                    # Convert to ISO format
                    return photo_id, taken.replace(" ", "T")
            else:
                print(f"Flickr API warning for ID {photo_id}: {data.get('message')}")
    except Exception as e:
        print(f"Error fetching Flickr ID {photo_id}: {e}")
    return photo_id, None


def fetch_kartaview_timestamp(photo_id):
    """Fetches the shotDate for a single KartaView photo ID."""
    url = f"https://api.openstreetcam.org/2.0/photo/{photo_id}"
    try:
        # Enforce polite delay
        time.sleep(0.5)
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json().get("result", {}).get("data", {})
            shot_date = data.get("shotDate")
            if shot_date:
                # Convert space to T and drop trailing milliseconds for ISO format
                # e.g., '2016-05-03 11:53:13.000' -> '2016-05-03T11:53:13'
                if '.' in shot_date:
                    shot_date = shot_date.split('.')[0]
                return photo_id, shot_date.replace(" ", "T")
    except Exception as e:
        print(f"Error fetching KartaView ID {photo_id}: {e}")
    return photo_id, None


def main():
    parser = argparse.ArgumentParser(description="Backfill Captured_At timestamps for existing dataset.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to geo_embedding_space.parquet or .csv")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save the enriched output (defaults to overwriting file_path)")
    args = parser.parse_args()

    save_path = args.save_path or args.file_path

    # 1. Load Data
    print(f"Loading dataset from {args.file_path}...")
    if args.file_path.endswith('.parquet'):
        df = pd.read_parquet(args.file_path)
    else:
        df = pd.read_csv(args.file_path)

    if 'Captured_At' not in df.columns:
        df['Captured_At'] = None

    # Convert Photo_ID to string for reliable matching
    df['Photo_ID'] = df['Photo_ID'].astype(str)

    # Identify missing entries
    missing_mask = df['Captured_At'].isna() | (df['Captured_At'] == "")
    df_missing = df[missing_mask]

    print(f"Total rows: {len(df)}. Rows missing Captured_At: {len(df_missing)}.")
    if len(df_missing) == 0:
        print("No missing timestamps to backfill.")
        return

    # Mapillary Batch Processing
    mapillary_ids = df_missing[df_missing['Platform'].str.lower() == 'mapillary']['Photo_ID'].tolist()
    print(f"Found {len(mapillary_ids)} Mapillary images to backfill.")

    mapillary_timestamps = {}
    batch_size = 50
    for i in tqdm(range(0, len(mapillary_ids), batch_size), desc="Fetching Mapillary Timestamps"):
        batch = mapillary_ids[i:i + batch_size]
        results = fetch_mapillary_timestamps(batch)
        mapillary_timestamps.update(results)
        # Avoid hitting Mapillary rate limits
        time.sleep(0.5)

    # Flickr Sequential/Throttled Processing
    flickr_ids = df_missing[df_missing['Platform'].str.lower() == 'flickr']['Photo_ID'].tolist()
    print(f"Found {len(flickr_ids)} Flickr images to backfill.")

    flickr_timestamps = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_flickr_timestamp, pid): pid for pid in flickr_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Flickr Timestamps"):
            pid, timestamp = future.result()
            if timestamp:
                flickr_timestamps[pid] = timestamp

    # KartaView (OpenStreetCam) Processing
    kartaview_ids = df_missing[df_missing['Platform'].str.lower().isin(['kartaview', 'openstreetcam'])][
        'Photo_ID'].tolist()
    print(f"Found {len(kartaview_ids)} KartaView images to backfill.")

    kartaview_timestamps = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_kartaview_timestamp, pid): pid for pid in kartaview_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching KartaView Timestamps"):
            pid, timestamp = future.result()
            if timestamp:
                kartaview_timestamps[pid] = timestamp

    # Merge back into DataFrame
    all_fetched = {**mapillary_timestamps, **flickr_timestamps, **kartaview_timestamps}

    def get_timestamp(row):
        pid = row['Photo_ID']
        if pid in all_fetched:
            return all_fetched[pid]
        return row['Captured_At']

    df['Captured_At'] = df.apply(get_timestamp, axis=1)

    # 2. Save Output
    print(f"Saving enriched dataset to {save_path}...")
    if save_path.endswith('.parquet'):
        df.to_parquet(save_path, index=False)
    else:
        df.to_csv(save_path, index=False)
    print("Backfill complete!")


if __name__ == "__main__":
    main()
