import requests
import math
import csv
import os
import time
from tqdm import tqdm
from global_land_mask import globe
import geopandas as gpd
import argparse
from shapely.geometry import box

# --- 1. Configuration ---
ACCESS_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'  # <-- REPLACE WITH YOUR MAPILLARY TOKEN
STEP_KM = 5
REGION = (-180, -90, -25, 90) 
MAX_PHOTOS_PER_BOX = 50
DELAY_BETWEEN_CALLS = 1.8  # Be polite to the Mapillary servers

# Urban Mask Configuration
URBAN_SHAPEFILE = "ne_10m_urban_areas.shp"

argparser = argparse.ArgumentParser(description="Flickr 5km Grid Search")
argparser.add_argument('--chunk', type=int, default=0, help='Which chunk of the grid to process (0-based index)')
argparser.add_argument('--total_chunks', type=int, default=10000, help='Total number of chunks to split the grid into')
argparser.add_argument('--base_dir', type=str, default='.', help='Base directory for output files')
args = argparser.parse_args()


# --- Splitting Variables ---
TOTAL_CHUNKS = args.total_chunks     # How many pieces to split the region into
CURRENT_CHUNK = args.chunk     # Which piece THIS script will process (0 through 9)

# File Setup
OUTPUT_FILE = os.path.join(args.base_dir, f'mapillary_data_chunk_{CURRENT_CHUNK}.csv')
LOG_FILE = os.path.join(args.base_dir, f'mapillary_completed_boxes_chunk_{CURRENT_CHUNK}.txt')

# --- 2. Helper Functions ---
def generate_5km_grid(region, step_km):
    min_lon, min_lat, max_lon, max_lat = region
    lat_step = step_km / 111.32
    avg_lat = math.radians((min_lat + max_lat) / 2)
    lon_step = step_km / (111.32 * math.cos(avg_lat))
    
    boxes = []
    current_lat = min_lat
    while current_lat < max_lat:
        current_lon = min_lon
        while current_lon < max_lon:
            boxes.append((current_lon, current_lat, current_lon + lon_step, current_lat + lat_step))
            current_lon += lon_step
        current_lat += lat_step
    return boxes

def is_urban(bbox_coords, urban_dataframe):
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
    
    possible_matches_index = list(urban_dataframe.sindex.intersection(bbox_polygon.bounds))
    if len(possible_matches_index) == 0:
        return False
        
    possible_matches = urban_dataframe.iloc[possible_matches_index]
    exact_matches = possible_matches[possible_matches.intersects(bbox_polygon)]
    return not exact_matches.empty

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

print("Loading Urban Areas map...")
urban_gdf = gpd.read_file(URBAN_SHAPEFILE)

print("Generating global grid...")
all_boxes = generate_5km_grid(REGION, STEP_KM)

chunk_size = math.ceil(len(all_boxes) / TOTAL_CHUNKS)
start_idx = CURRENT_CHUNK * chunk_size
end_idx = min(start_idx + chunk_size, len(all_boxes))
my_boxes = all_boxes[start_idx:end_idx]

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
            
        center_lon = (grid_box[0] + grid_box[2]) / 2
        center_lat = (grid_box[1] + grid_box[3]) / 2
        
        # 2. Land Mask Check
        if not globe.is_land(center_lat, center_lon):
            with open(LOG_FILE, 'a') as log: log.write(box_id + '\n')
            completed_boxes.add(box_id)
            continue
            
        # 3. Urban Mask Check
        if is_urban(grid_box, urban_gdf):
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