import requests
import math
import csv
import os
import time
import random
from tqdm import tqdm
import geopandas as gpd
import argparse
from pathlib import Path
from shapely.geometry import box

# --- 1. Configuration ---
ACCESS_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'  # <-- REPLACE WITH YOUR MAPILLARY TOKEN
STEP_KM = 5
# Global Region for a representative scan
REGION = (-180, -90, 180, 90) 
MAX_PHOTOS_PER_BOX = 100
DELAY_BETWEEN_CALLS = 1.8  # Be polite to the Mapillary servers

# Uncovered Land Configuration
UNCOVERED_SHAPEFILE = "uncovered_land_areas_test.shp"

argparser = argparse.ArgumentParser(description="Mapillary 5km Grid Search")
argparser.add_argument('--chunk', type=int, default=0, help='Which chunk of the grid to process (0-based index)')
argparser.add_argument('--total_chunks', type=int, default=10000, help='Total number of chunks to split the grid into')
argparser.add_argument('--base_dir', type=str, default='.', help='Base directory for output files')
args = argparser.parse_args()


# --- Splitting Variables ---
TOTAL_CHUNKS = args.total_chunks     # How many pieces to split the region into
CURRENT_CHUNK = args.chunk     # Which piece THIS script will process (0 through 9)

# File Setup
Path(args.base_dir).mkdir(parents=True, exist_ok=True)  # Ensure base directory exists
OUTPUT_FILE = os.path.join(args.base_dir, f'mapillary_data_chunk_{CURRENT_CHUNK}.csv')
LOG_FILE = os.path.join(args.base_dir, f'mapillary_completed_boxes_chunk_{CURRENT_CHUNK}.txt')

# --- 2. Helper Functions ---
def is_in_uncovered_area(bbox_coords, uncovered_gdf):
    """Checks if a bounding box intersects with any uncovered land area."""
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
    
    # Use spatial indexing for fast lookups
    possible_matches_index = list(uncovered_gdf.sindex.intersection(bbox_polygon.bounds))
    
    if len(possible_matches_index) == 0:
        return False
        
    possible_matches = uncovered_gdf.iloc[possible_matches_index]
    return any(possible_matches.intersects(bbox_polygon))

def fetch_mapillary_photos(bbox_coords=None, next_url=None):
    """Fetches photos using a bounding box OR a pagination URL."""
    headers = {
        "Authorization": f"OAuth {ACCESS_TOKEN}"
    }
    
    # If we have a next_url from a previous request, use that. 
    # Otherwise, build the initial URL from the bounding box.
    if next_url:
        url = next_url
    else:
        bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
        # Requesting ID, coordinates, and the 1024px thumbnail URL
        url = (
            f"https://graph.mapillary.com/images"
            f"?bbox={bbox_str}"
            f"&fields=id,geometry,thumb_1024_url"
            f"&limit=50" # Max allowed per request is usually higher, but 50 aligns with our goal
        )
    
    try:
        time.sleep(DELAY_BETWEEN_CALLS) 
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json().get('data', [])
            # Mapillary passes the pagination link in the response headers!
            # Python's requests library automatically parses this into `response.links`
            new_next_url = response.links.get('next', {}).get('url')
            return {'stat': 'ok', 'data': data, 'next_url': new_next_url}
        else:
            return {'stat': 'fail', 'message': f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {'stat': 'fail', 'message': str(e)}

# --- 3. Initialization ---
completed_boxes = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        completed_boxes = set(line.strip() for line in f)

print(f"Loading Uncovered Land Areas map: {UNCOVERED_SHAPEFILE}...")
uncovered_gdf = gpd.read_file(UNCOVERED_SHAPEFILE, engine='pyogrio')

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
        writer.writerow(['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL'])
    
    for grid_box in tqdm(my_boxes, desc=f"Processing Chunk {CURRENT_CHUNK}"):
        box_id = f"{grid_box[0]:.4f},{grid_box[1]:.4f},{grid_box[2]:.4f},{grid_box[3]:.4f}"
        
        # 1. Save-State Check
        if box_id in completed_boxes:
            continue
            
        # 2. Check Uncovered Mask: Skip if it does NOT intersect an uncovered land area
        if not is_in_uncovered_area(grid_box, uncovered_gdf):
            with open(LOG_FILE, 'a') as log: log.write(box_id + '\n')
            completed_boxes.add(box_id)
            continue
            
        # 4. Fetch Mapillary Photos
        current_url = None
        photos_saved_this_box = 0
        
        while True:
            # Pass the grid_box (for the first request) or the current_url (for pagination)
            result = fetch_mapillary_photos(bbox_coords=grid_box, next_url=current_url)
            
            if result['stat'] == 'ok':
                images = result.get('data', [])
                
                # If no images are returned, break out of the pagination loop
                if not images:
                    break
                    
                for img in images:
                    img_id = img.get('id')
                    # Mapillary returns GeoJSON [Longitude, Latitude]
                    lon, lat = img.get('geometry', {}).get('coordinates', [None, None])
                    image_url = img.get('thumb_1024_url')
                    
                    if image_url and lat and lon:
                        writer.writerow([img_id, "Mapillary", lat, lon, image_url])
                        photos_saved_this_box += 1
                        
                    if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                        break
                
                # If we hit our limit, break the pagination loop
                if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                    break 
                
                # Update the URL for the next page. If it's None, we've hit the end.
                current_url = result.get('next_url')
                if not current_url:
                    break
                
            else:
                # Silently break on error to keep the loop moving
                break

        # 5. Update Save-State
        with open(LOG_FILE, 'a') as log:
            log.write(box_id + '\n')
        completed_boxes.add(box_id)

print(f"\nChunk {CURRENT_CHUNK} finished!")