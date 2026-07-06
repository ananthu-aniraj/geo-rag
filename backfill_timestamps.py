import os
import time
import argparse
import datetime
import glob
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'
FLICKR_API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'
FLICKR_DELAY = 1.1


def fetch_mapillary_timestamps(photo_ids):
    """Fetches captured_at timestamps for multiple Mapillary photo IDs in batches of 100."""
    if not photo_ids:
        return {}

    url = f"https://graph.mapillary.com/images?ids={','.join(photo_ids)}&fields=id,captured_at"
    headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
    results = {}

    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            for item in res.json().get('data', []):
                pid = str(item.get('id'))
                cap_ms = item.get('captured_at')
                if cap_ms:
                    results[pid] = datetime.datetime.fromtimestamp(cap_ms / 1000.0, datetime.timezone.utc).strftime(
                        '%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        pass
    return results


def fetch_flickr_bbox_timestamps(bbox_str):
    """Fetches photo IDs and date_taken timestamps for a bounding box in bulk (up to 500 per call)."""
    results = {}

    # Priority 1: Outdoors (2). Priority 2: Unlabelled (0).
    for geo_context in [2, 0]:
        page = 1
        total_pages = 1
        while page <= total_pages:
            url = (
                f"https://www.flickr.com/services/rest/"
                f"?method=flickr.photos.search"
                f"&api_key={FLICKR_API_KEY}"
                f"&bbox={bbox_str}"
                f"&has_geo=1"
                f"&geo_context={geo_context}"
                f"&extras=date_taken"
                f"&per_page=250"
                f"&page={page}"
                f"&format=json"
                f"&nojsoncallback=1"
            )
            try:
                time.sleep(FLICKR_DELAY)
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    if data.get('stat') == 'ok':
                        # Since the scraper capped photos at 100, and per_page is 250,
                        # all scraped photos are guaranteed to be on Page 1.
                        total_pages = 1

                        photos = data.get('photos', {}).get('photo', [])
                        if not photos:
                            break

                        for p in photos:
                            pid = str(p.get('id'))
                            taken = p.get('datetaken', '')
                            if taken:
                                results[pid] = taken.replace(" ", "T")

                        page += 1
                    else:
                        break
                else:
                    break
            except Exception:
                break
    return results


def fetch_flickr_individual_timestamp(photo_id):
    """Fetches the date taken for a single Flickr photo ID (fallback approach)."""
    url = (
        f"https://www.flickr.com/services/rest/"
        f"?method=flickr.photos.getInfo"
        f"&api_key={FLICKR_API_KEY}"
        f"&photo_id={photo_id}"
        f"&format=json"
        f"&nojsoncallback=1"
    )
    try:
        time.sleep(FLICKR_DELAY)
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get('stat') == 'ok':
                taken = data.get('photo', {}).get('dates', {}).get('taken', '')
                if taken:
                    return photo_id, taken.replace(" ", "T")
    except Exception:
        pass
    return photo_id, None


def fetch_kartaview_timestamp(photo_id):
    """Fetches the shotDate for a single KartaView photo ID."""
    url = f"https://api.openstreetcam.org/2.0/photo/{photo_id}"
    try:
        time.sleep(0.3)
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json().get("result", {}).get("data", {})
            shot_date = data.get("shotDate")
            if shot_date:
                if '.' in shot_date:
                    shot_date = shot_date.split('.')[0]
                return photo_id, shot_date.replace(" ", "T")
    except Exception:
        pass
    return photo_id, None


def main():
    parser = argparse.ArgumentParser(description="Consolidated Backfill of Captured_At timestamps.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to geo_embedding_space.parquet or .csv")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save the enriched output (defaults to overwriting file_path)")
    parser.add_argument("--log_dirs", nargs="+", default=None,
                        help="Optional list of folders containing the completed_boxes log files (enables 250x faster bulk search backfill for Flickr).")
    args = parser.parse_args()

    save_path = args.save_path or args.file_path

    # 1. Load Dataset
    print(f"Loading dataset from {args.file_path}...")
    if args.file_path.endswith('.parquet'):
        df = pd.read_parquet(args.file_path)
    else:
        df = pd.read_csv(args.file_path)

    if 'Captured_At' not in df.columns:
        df['Captured_At'] = None
    df['Photo_ID'] = df['Photo_ID'].astype(str)

    missing_mask = df['Captured_At'].isna() | (df['Captured_At'] == "")
    df_missing = df[missing_mask]

    print(f"Total rows: {len(df)}. Missing timestamps: {len(df_missing)}.")
    if len(df_missing) == 0:
        print("No missing timestamps to backfill.")
        return

    # --- 1. Flickr (Bulk Box Search or Fallback Individual Queries) ---
    flickr_ids = set(df_missing[df_missing['Platform'].str.lower() == 'flickr']['Photo_ID'].tolist())
    flickr_timestamps = {}
    
    if flickr_ids:
        if args.log_dirs:
            # Optimized Bulk BBox Search
            log_files = []
            for folder in args.log_dirs:
                log_files.extend(glob.glob(os.path.join(folder, "flickr_completed_boxes_chunk_*.txt")))
            
            print(f"Found {len(log_files)} Flickr completed boxes logs. Reading search coordinates...")
            bboxes = set()
            for f in log_files:
                try:
                    with open(f, 'r') as file:
                        for line in file:
                            box_id = line.strip()
                            if box_id:
                                bboxes.add(box_id)
                except Exception:
                    pass
            
            print(f"Discovered {len(bboxes)} unique bounding boxes.")
            
            # Filter bboxes using spatial join to only query those containing missing points
            active_bboxes = []
            try:
                import geopandas as gpd
                from shapely.geometry import box
                
                print("Filtering boxes using spatial indexing to find boxes that contain our images...")
                # Create GeoDataFrame for bboxes
                box_geoms = []
                box_ids = []
                for bbox_str in bboxes:
                    try:
                        coords = [float(x) for x in bbox_str.split(',')]
                        box_geoms.append(box(coords[0], coords[1], coords[2], coords[3]))
                        box_ids.append(bbox_str)
                    except Exception:
                        pass
                
                gdf_boxes = gpd.GeoDataFrame({'bbox_str': box_ids}, geometry=box_geoms, crs="EPSG:4326")
                
                # Create GeoDataFrame for missing Flickr points
                df_flickr_missing = df_missing[df_missing['Platform'].str.lower() == 'flickr']
                gdf_points = gpd.GeoDataFrame(
                    df_flickr_missing,
                    geometry=gpd.points_from_xy(df_flickr_missing['Longitude'], df_flickr_missing['Latitude']),
                    crs="EPSG:4326"
                )
                
                # Spatial join
                joined = gpd.sjoin(gdf_boxes, gdf_points, how="inner", predicate="intersects")
                
                # Map each bbox to the set of Photo_IDs it contains
                box_to_photos = joined.groupby('bbox_str')['Photo_ID'].apply(set).to_dict()
                
                # Sort bboxes by count of missing photos descending
                active_bboxes = sorted(box_to_photos.keys(), key=lambda b: len(box_to_photos[b]), reverse=True)
                print(f"Filtered and sorted to {len(active_bboxes)} active bounding boxes containing missing Flickr images.")
            except Exception as se:
                print(f"Spatial join optimization failed or geopandas not available: {se}")
                print("Falling back to scanning all discovered bounding boxes...")
                active_bboxes = list(bboxes)
                box_to_photos = {}
            
            if active_bboxes:
                print(f"Running optimized bulk search scan on active boxes...")
                for bbox in tqdm(active_bboxes, desc="Bulk Scan Flickr BBoxes"):
                    # Dynamic Pruning: Skip if all photos in this box have already been resolved
                    if box_to_photos and bbox in box_to_photos:
                        box_photos = box_to_photos[bbox]
                        needed_photos = box_photos - set(flickr_timestamps.keys())
                        if not needed_photos:
                            # Skip this box completely as all its images were resolved by overlapping queries
                            continue
                            
                    res_box = fetch_flickr_bbox_timestamps(bbox)
                    for pid, timestamp in res_box.items():
                        if pid in flickr_ids:
                            flickr_timestamps[pid] = timestamp
                            
                    # Early Termination: Stop if all missing Flickr IDs are resolved
                    if len(flickr_timestamps) >= len(flickr_ids):
                        print("\nAll missing Flickr timestamps successfully backfilled! Terminating early...")
                        break
                        
            print(f"Retrieved {len(flickr_timestamps)} Flickr timestamps using bulk search.")
        else:
            # Fallback Individual Queries (Warning user first)
            print(f"\n[WARNING] No --log_dirs provided. Querying Flickr API individually for {len(flickr_ids)} photos.")
            print(f"This will take approximately {len(flickr_ids) / 3600:.1f} hours due to Flickr's rate limit.")
            print("Provide --log_dirs with your completed box text files to speed this up by 250x.")
 
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(fetch_flickr_individual_timestamp, pid): pid for pid in flickr_ids}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Fetch Flickr (1-by-1)"):
                    pid, timestamp = future.result()
                    if timestamp:
                        flickr_timestamps[pid] = timestamp

    # --- 2. Mapillary (Batch ID Queries) ---
    mapillary_ids = df_missing[df_missing['Platform'].str.lower() == 'mapillary']['Photo_ID'].tolist()
    mapillary_timestamps = {}
    if mapillary_ids:
        print(f"Found {len(mapillary_ids)} Mapillary images. Running batch queries...")
        batch_size = 100
        for i in tqdm(range(0, len(mapillary_ids), batch_size), desc="Bulk Fetch Mapillary"):
            batch = mapillary_ids[i:i + batch_size]
            res_batch = fetch_mapillary_timestamps(batch)
            mapillary_timestamps.update(res_batch)
            time.sleep(0.1)

    # --- 3. KartaView (Throttled Parallel Queries) ---
    kartaview_ids = df_missing[df_missing['Platform'].str.lower().isin(['kartaview', 'openstreetcam'])][
        'Photo_ID'].tolist()
    kartaview_timestamps = {}
    if kartaview_ids:
        print(f"Found {len(kartaview_ids)} KartaView images. Querying individual timestamps...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_kartaview_timestamp, pid): pid for pid in kartaview_ids}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetch KartaView"):
                pid, timestamp = future.result()
                if timestamp:
                    kartaview_timestamps[pid] = timestamp

    # --- Merge and Save ---
    all_fetched = {**mapillary_timestamps, **flickr_timestamps, **kartaview_timestamps}

    def merge_timestamp(row):
        pid = row['Photo_ID']
        if pid in all_fetched:
            return all_fetched[pid]
        return row['Captured_At']

    df['Captured_At'] = df.apply(merge_timestamp, axis=1)

    print(f"Saving enriched dataset to {save_path}...")
    output_base, _ = os.path.splitext(save_path)
    
    if save_path.endswith('.parquet'):
        df.to_parquet(save_path, index=False)
        
        # Auto-discover and enrich corresponding CSV file
        csv_pair = output_base + ".csv"
        if os.path.exists(csv_pair):
            print(f"Auto-discovered corresponding CSV file: {csv_pair}. Enriching it...")
            try:
                df_csv = pd.read_csv(csv_pair)
                df_csv['Photo_ID'] = df_csv['Photo_ID'].astype(str)
                if 'Captured_At' not in df_csv.columns:
                    df_csv['Captured_At'] = None
                df_csv['Captured_At'] = df_csv.apply(merge_timestamp, axis=1)
                df_csv.to_csv(csv_pair, index=False)
                print("CSV file enriched successfully!")
            except Exception as e:
                print(f"Error enriching corresponding CSV file: {e}")
    else:
        df.to_csv(save_path, index=False)
        
        # Auto-discover and enrich corresponding Parquet file
        parquet_pair = output_base + ".parquet"
        if os.path.exists(parquet_pair):
            print(f"Auto-discovered corresponding Parquet file: {parquet_pair}. Enriching it...")
            try:
                df_pq = pd.read_parquet(parquet_pair)
                df_pq['Photo_ID'] = df_pq['Photo_ID'].astype(str)
                if 'Captured_At' not in df_pq.columns:
                    df_pq['Captured_At'] = None
                df_pq['Captured_At'] = df_pq.apply(merge_timestamp, axis=1)
                df_pq.to_parquet(parquet_pair, index=False)
                print("Parquet file enriched successfully!")
            except Exception as e:
                print(f"Error enriching corresponding Parquet file: {e}")
                
    print("Backfill complete!")


if __name__ == "__main__":
    main()
