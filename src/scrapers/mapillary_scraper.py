import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box
from tqdm import tqdm

from src.utils.credentials import get_mapillary_token
from src.utils.mapillary_trajectory_validator import MapillaryTrajectoryValidator

# --- 1. Configuration ---
# Global Region for a representative scan
REGION = (-180, -90, 180, 90)


def parse_args():
    argparser = argparse.ArgumentParser(description="Mapillary 5km Grid Search")
    argparser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Which chunk of the grid to process (0-based index)",
    )
    argparser.add_argument(
        "--total_chunks",
        type=int,
        default=10000,
        help="Total number of chunks to split the grid into",
    )
    argparser.add_argument(
        "--base_dir", type=str, default=".", help="Base directory for output files"
    )
    argparser.add_argument(
        "--step_km", type=float, default=5, help="Grid step size in kilometers"
    )
    argparser.add_argument(
        "--max_photos_per_box",
        type=int,
        default=100,
        help="Maximum number of photos to fetch per grid box",
    )
    argparser.add_argument(
        "--delay_between_calls",
        type=float,
        default=1.8,
        help="Delay between API calls in seconds",
    )
    argparser.add_argument(
        "--uncovered_shapefile",
        type=str,
        default="shapefiles/uncovered_land_areas_test.shp",
        help="Path to the uncovered land areas shapefile",
    )
    argparser.add_argument(
        "--access_token", type=str, default=None, help="Mapillary API access token"
    )
    argparser.add_argument(
        "--validate_sequences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate sequence trajectories and purge invalid/stationary sequences at end of chunk",
    )
    return argparser.parse_args()


args = parse_args()
# --- Splitting Variables ---
TOTAL_CHUNKS = args.total_chunks  # How many pieces to split the region into
CURRENT_CHUNK = args.chunk  # Which piece THIS script will process (0 through 9)
DELAY_BETWEEN_CALLS = args.delay_between_calls
UNCOVERED_SHAPEFILE = args.uncovered_shapefile
STEP_KM = args.step_km
MAX_PHOTOS_PER_BOX = args.max_photos_per_box
ACCESS_TOKEN = args.access_token or get_mapillary_token() or ""

# File Setup
Path(args.base_dir).mkdir(parents=True, exist_ok=True)  # Ensure base directory exists
OUTPUT_FILE = os.path.join(args.base_dir, f"mapillary_data_chunk_{CURRENT_CHUNK}.csv")
STAGING_FILE = os.path.join(
    args.base_dir, f"mapillary_data_chunk_{CURRENT_CHUNK}.staging.csv"
)
LOG_FILE = os.path.join(
    args.base_dir, f"mapillary_completed_boxes_chunk_{CURRENT_CHUNK}.txt"
)


# --- 2. Helper Functions ---
def is_in_uncovered_area(bbox_coords, uncovered_gdf):
    """Checks if a bounding box intersects with any uncovered land area."""
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)

    # Use spatial indexing for fast lookups
    possible_matches_index = list(
        uncovered_gdf.sindex.intersection(bbox_polygon.bounds)
    )

    if len(possible_matches_index) == 0:
        return False

    possible_matches = uncovered_gdf.iloc[possible_matches_index]
    return any(possible_matches.intersects(bbox_polygon))


def fetch_mapillary_photos(bbox_coords=None, next_url=None):
    """Fetches photos using a bounding box OR a pagination URL."""
    headers = {"Authorization": f"OAuth {ACCESS_TOKEN}"}

    # If we have a next_url from a previous request, use that.
    # Otherwise, build the initial URL from the bounding box.
    if next_url:
        url = next_url
    else:
        bbox_str = (
            f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
        )
        # Requesting ID, coordinates, sequence ID, and the 1024px thumbnail URL
        url = (
            f"https://graph.mapillary.com/images"
            f"?bbox={bbox_str}"
            f"&fields=id,geometry,thumb_1024_url,captured_at,sequence"
            f"&limit=50"  # Max allowed per request is usually higher, but 50 aligns with our goal
        )

    try:
        time.sleep(DELAY_BETWEEN_CALLS)
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json().get("data", [])
            # Mapillary passes the pagination link in the response headers!
            # Python's requests library automatically parses this into `response.links`
            new_next_url = response.links.get("next", {}).get("url")
            return {"stat": "ok", "data": data, "next_url": new_next_url}
        else:
            return {
                "stat": "fail",
                "message": f"HTTP {response.status_code}: {response.text}",
            }
    except Exception as e:
        return {"stat": "fail", "message": str(e)}


# --- 3. Initialization ---
completed_boxes = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        completed_boxes = set(line.strip() for line in f)

print(f"Loading Uncovered Land Areas map: {UNCOVERED_SHAPEFILE}...")
uncovered_gdf = gpd.read_file(UNCOVERED_SHAPEFILE, engine="pyogrio")

print("Generating global virtual grid...")
my_boxes = []
total_boxes_count = 0
current_lat = REGION[1]
lat_step = STEP_KM / 111.32

while current_lat < REGION[3]:
    # Adjust longitude step based on current latitude
    cos_lat = math.cos(math.radians(max(-89.9, min(89.9, current_lat))))
    lon_step = STEP_KM / (111.32 * cos_lat)

    current_lon = REGION[0]
    while current_lon < REGION[2]:
        if total_boxes_count % TOTAL_CHUNKS == CURRENT_CHUNK:
            my_boxes.append(
                (
                    current_lon,
                    current_lat,
                    current_lon + lon_step,
                    current_lat + lat_step,
                )
            )

        current_lon += lon_step
        total_boxes_count += 1
    current_lat += lat_step

# Shuffle the specific boxes assigned to THIS chunk to avoid sequential processing
random.Random(42 + CURRENT_CHUNK).shuffle(my_boxes)

print(f"Total virtual boxes in global grid: {total_boxes_count}")
print(f"Boxes assigned to Chunk {CURRENT_CHUNK}: {len(my_boxes)}")
print(f"Already completed: {len(completed_boxes)}")

# --- 4. Main Execution ---
with open(STAGING_FILE, mode="a", newline="", encoding="utf-8") as staging_file:
    staging_writer = csv.writer(staging_file)

    for grid_box in tqdm(my_boxes, desc=f"Processing Chunk {CURRENT_CHUNK}"):
        box_id = (
            f"{grid_box[0]:.4f},{grid_box[1]:.4f},{grid_box[2]:.4f},{grid_box[3]:.4f}"
        )

        # 1. Save-State Check
        if box_id in completed_boxes:
            continue

        # 2. Check Uncovered Mask: Skip if it does NOT intersect an uncovered land area
        if not is_in_uncovered_area(grid_box, uncovered_gdf):
            with open(LOG_FILE, "a") as log:
                log.write(box_id + "\n")
            completed_boxes.add(box_id)
            continue

        # 4. Fetch Mapillary Photos
        current_url = None
        photos_saved_this_box = 0

        while True:
            # Pass the grid_box (for the first request) or the current_url (for pagination)
            result = fetch_mapillary_photos(bbox_coords=grid_box, next_url=current_url)

            if result["stat"] == "ok":
                images = result.get("data", [])

                # If no images are returned, break out of the pagination loop
                if not images:
                    break

                for img in images:
                    img_id = img.get("id")
                    # Mapillary returns GeoJSON [Longitude, Latitude]
                    lon, lat = img.get("geometry", {}).get("coordinates", [None, None])
                    image_url = img.get("thumb_1024_url")
                    captured_at_ms = img.get("captured_at")
                    seq_id = str(img.get("sequence") or "")
                    captured_at = ""
                    if captured_at_ms:
                        import datetime

                        captured_at = datetime.datetime.fromtimestamp(
                            captured_at_ms / 1000.0, datetime.timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ")

                    if image_url and lat and lon:
                        staging_writer.writerow(
                            [
                                img_id,
                                "Mapillary",
                                lat,
                                lon,
                                image_url,
                                captured_at,
                                seq_id,
                            ]
                        )
                        photos_saved_this_box += 1

                    if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                        break

                # If we hit our limit, break the pagination loop
                if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                    break

                # Update the URL for the next page. If it's None, we've hit the end.
                current_url = result.get("next_url")
                if not current_url:
                    break

            else:
                # Silently break on error to keep the loop moving
                break

        # 5. Update Save-State
        with open(LOG_FILE, "a") as log:
            log.write(box_id + "\n")
        completed_boxes.add(box_id)

print(f"\nChunk {CURRENT_CHUNK} boxes finished. Auditing and finalizing...")

# --- 5. End-of-Chunk Sequence Validation & Final Output ---
if os.path.exists(STAGING_FILE) and os.path.getsize(STAGING_FILE) > 0:
    import pandas as pd

    df_staging = pd.read_csv(
        STAGING_FILE,
        names=[
            "Photo_ID",
            "Platform",
            "Latitude",
            "Longitude",
            "Image_URL",
            "Captured_At",
            "Sequence_ID",
        ],
        dtype=str,
        header=None,
    )

    if not df_staging.empty:
        total_photos = len(df_staging)
        print(f" -> Collected {total_photos:,} staged photos in chunk {CURRENT_CHUNK}.")

        if args.validate_sequences and ACCESS_TOKEN:
            validator = MapillaryTrajectoryValidator(token=ACCESS_TOKEN)
            unique_seqs = [
                s for s in df_staging["Sequence_ID"].dropna().unique() if str(s).strip()
            ]
            print(
                f" -> Auditing {len(unique_seqs):,} unique sequence track(s) for kinematic validity..."
            )

            invalid_seqs = set()
            for s in tqdm(unique_seqs, desc="Validating sequence kinematics"):
                res = validator.validate_sequence(str(s).strip())
                if not res.is_valid:
                    invalid_seqs.add(str(s).strip())

            if invalid_seqs:
                valid_mask = ~df_staging["Sequence_ID"].isin(invalid_seqs)
                purged_count = total_photos - int(valid_mask.sum())
                df_staging = df_staging[valid_mask]
                print(
                    f" -> Discarded {purged_count:,} photo(s) from {len(invalid_seqs):,} invalid/stationary sequence(s)."
                )
            else:
                print(" -> All sequences passed kinematic validation.")

        # Project to standard 6 columns for downstream compatibility
        df_clean = df_staging[
            [
                "Photo_ID",
                "Platform",
                "Latitude",
                "Longitude",
                "Image_URL",
                "Captured_At",
            ]
        ]
        write_header = not os.path.exists(OUTPUT_FILE)
        df_clean.to_csv(OUTPUT_FILE, mode="a", index=False, header=write_header)
        print(f" -> Saved {len(df_clean):,} clean records to {OUTPUT_FILE}.")

    try:
        os.remove(STAGING_FILE)
    except OSError:
        pass

print(f"\nChunk {CURRENT_CHUNK} complete!")
