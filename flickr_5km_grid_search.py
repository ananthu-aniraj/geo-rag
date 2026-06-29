import requests
import math
import csv
import os
import time
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box

# --- 1. Configuration ---
API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'  # <-- REPLACE THIS WITH YOUR ACTUAL FLICKR API KEY
STEP_KM = 5
# Global Region for a representative scan
REGION = (-180, -90, 180, 90)
MAX_PHOTOS_PER_BOX = 100
DELAY_BETWEEN_CALLS = (3600 / 3600) * 1.1

# Uncovered Land Configuration
UNCOVERED_SHAPEFILE = "uncovered_land_areas_test.shp"

argparser = argparse.ArgumentParser(description="Flickr 5km Grid Search")
argparser.add_argument('--chunk', type=int, default=0, help='Which chunk of the grid to process (0-based index)')
argparser.add_argument('--total_chunks', type=int, default=10000, help='Total number of chunks to split the grid into')
argparser.add_argument('--base_dir', type=str, default='.', help='Base directory for output files')
args = argparser.parse_args()

# --- Splitting Variables ---
TOTAL_CHUNKS = args.total_chunks  # How many pieces to split the region into
CURRENT_CHUNK = args.chunk  # Which piece THIS script will process (0 through 9)

# File Setup
Path(args.base_dir).mkdir(parents=True, exist_ok=True)  # Ensure base directory exists
OUTPUT_FILE = os.path.join(args.base_dir, f'flickr_data_chunk_{CURRENT_CHUNK}.csv')
LOG_FILE = os.path.join(args.base_dir, f'flickr_completed_boxes_chunk_{CURRENT_CHUNK}.txt')


# --- 2. Helper Functions ---
def is_in_uncovered_area(bbox_coords, uncovered_gdf):
    """Checks if a bounding box intersects with any uncovered land area."""
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)

    # Use spatial indexing for lightning-fast lookups
    possible_matches_index = list(uncovered_gdf.sindex.intersection(bbox_polygon.bounds))

    if len(possible_matches_index) == 0:
        return False

    possible_matches = uncovered_gdf.iloc[possible_matches_index]
    return any(possible_matches.intersects(bbox_polygon))


# Pass geo_context as a parameter (defaulting to 2)
def fetch_photos(bbox_coords, page=1, geo_context=2):
    """Fetches geo-tagged photos for a specific context."""
    bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
    url = (
        f"https://www.flickr.com/services/rest/"
        f"?method=flickr.photos.search"
        f"&api_key={API_KEY}"
        f"&bbox={bbox_str}"
        f"&has_geo=1"
        f"&geo_context={geo_context}"  # <--- Now dynamically injected
        f"&extras=url_m,geo,date_taken"
        f"&per_page=250"
        f"&page={page}"
        f"&format=json"
        f"&nojsoncallback=1"
    )

    try:
        time.sleep(DELAY_BETWEEN_CALLS)
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception as e:
        return {'stat': 'fail', 'message': str(e)}


# --- 3. Initialization ---
completed_boxes = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        completed_boxes = set(line.strip() for line in f)

print(f"Loading Uncovered Land Areas map: {UNCOVERED_SHAPEFILE}...")
try:
    uncovered_gdf = gpd.read_file(UNCOVERED_SHAPEFILE, engine='pyogrio')
except Exception as e:
    print(f"ERROR: Could not load {UNCOVERED_SHAPEFILE}: {e}")
    exit()

print(f"Initializing global grid sampling for Chunk {CURRENT_CHUNK}...")
# Memory-efficient approach: Instead of generating and shuffling 26M+ boxes in RAM,
# we use a virtual grid. Each chunk picks every Nth box from the virtual grid.
# This ensures global diversity (representative scan) for every chunk.

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
            my_boxes.append((current_lon, current_lat, current_lon + lon_step, current_lat + lat_step))

        current_lon += lon_step
        total_boxes_count += 1
    current_lat += lat_step

# Shuffle the specific boxes assigned to THIS chunk to avoid sequential processing
random.Random(42 + CURRENT_CHUNK).shuffle(my_boxes)

print(f"Total virtual boxes in global grid: {total_boxes_count}")
print(f"Boxes assigned to Chunk {CURRENT_CHUNK}: {len(my_boxes)}")
print(f"Already completed: {len(completed_boxes)}")

# --- 4. Main Execution ---
file_exists = os.path.exists(OUTPUT_FILE)
with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(['Photo_ID', 'Title', 'Latitude', 'Longitude', 'Image_URL', 'Captured_At'])

    # Use tqdm to create a progress bar
    # CHANGE: Renamed 'box' to 'grid_box' to avoid overwriting Shapely's 'box' function
    for grid_box in tqdm(my_boxes, desc=f"Processing Chunk {CURRENT_CHUNK}"):

        # Create a unique ID for this box to log it
        box_id = f"{grid_box[0]:.4f},{grid_box[1]:.4f},{grid_box[2]:.4f},{grid_box[3]:.4f}"

        # 1. Check Save-State
        if box_id in completed_boxes:
            continue

        # 2. Check Uncovered Mask: Skip if it does NOT intersect an uncovered land area
        # This replaces both the land mask and the urban mask
        if not is_in_uncovered_area(grid_box, uncovered_gdf):
            with open(LOG_FILE, 'a') as log:
                log.write(box_id + '\n')
            completed_boxes.add(box_id)
            continue

        # 4. Fetch Photos
        current_page = 1
        total_pages = 1
        photos_saved_this_box = 0

        # Priority 1: Outdoors (2). Priority 2: Unlabelled (0).
        for current_context in [2, 0]:

            # If the bucket is already full from the previous priority, skip!
            if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                break

            current_page = 1
            total_pages = 1

            while current_page <= total_pages:
                # Call our updated function with the current context
                data = fetch_photos(grid_box, page=current_page, geo_context=current_context)

                if data.get('stat') == 'ok':
                    if current_page == 1:
                        total_pages = data['photos']['pages']

                    photos = data.get('photos', {}).get('photo', [])
                    if not photos:
                        break  # No more photos for this context, move on

                    for photo in photos:
                        photo_id = photo.get('id')
                        title = photo.get('title', 'Untitled')
                        lat = photo.get('latitude')
                        lon = photo.get('longitude')
                        image_url = photo.get('url_m')
                        captured_at = photo.get('datetaken', '')
                        if captured_at:
                            captured_at = captured_at.replace(" ", "T")

                        if image_url and lat and lon:
                            writer.writerow([photo_id, title, lat, lon, image_url, captured_at])
                            photos_saved_this_box += 1

                        # Break out if we hit the limit while looping through photos
                        if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                            break

                    # Break out of the pagination loop if limit is reached
                    if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                        break

                    current_page += 1

                else:
                    break  # API Error, break pagination loop

        # 5. Update Save-State
        with open(LOG_FILE, 'a') as log:
            log.write(box_id + '\n')
        completed_boxes.add(box_id)
print(f"\nChunk {CURRENT_CHUNK} finished!")
