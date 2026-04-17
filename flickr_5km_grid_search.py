import requests
import math
import csv
import os
import time
from tqdm import tqdm
from global_land_mask import globe
import geopandas as gpd
from shapely.geometry import box

# --- 1. Configuration ---
API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'  # <-- REPLACE THIS WITH YOUR ACTUAL FLICKR API KEY
STEP_KM = 5
# Example: The "Western" Hemisphere (The Americas)
REGION = (-180, -90, -25, 90) 
MAX_PHOTOS_PER_BOX = 50
DELAY_BETWEEN_CALLS = (3600 / 3600) * 1.1

# Urban Mask Configuration
URBAN_SHAPEFILE = "ne_10m_urban_areas.shp"

# --- Splitting Variables ---
TOTAL_CHUNKS = 100000     # How many pieces to split the region into
CURRENT_CHUNK = 0     # Which piece THIS script will process (0 through 9)

# File Setup
OUTPUT_FILE = f'flickr_data_chunk_{CURRENT_CHUNK}.csv'
LOG_FILE = f'completed_boxes_chunk_{CURRENT_CHUNK}.txt'

# --- 2. Helper Functions ---
def generate_5km_grid(region, step_km):
    """Slices a large bounding box into smaller boxes."""
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
    """Checks if a bounding box intersects with any urban area using a spatial index."""
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
    
    # Use spatial indexing for lightning-fast lookups
    possible_matches_index = list(urban_dataframe.sindex.intersection(bbox_polygon.bounds))
    
    if len(possible_matches_index) == 0:
        return False
        
    possible_matches = urban_dataframe.iloc[possible_matches_index]
    exact_matches = possible_matches[possible_matches.intersects(bbox_polygon)]
    
    return not exact_matches.empty

def fetch_outdoor_photos(bbox_coords, page=1):
    """Fetches outdoor photos for a specific bounding box and page."""
    bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
    url = (
        f"https://www.flickr.com/services/rest/"
        f"?method=flickr.photos.search"
        f"&api_key={API_KEY}"
        f"&bbox={bbox_str}"
        f"&has_geo=1"
        f"&geo_context=2"
        f"&extras=url_m,geo"
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

print("Loading Urban Areas map... (This takes a few seconds)")
try:
    urban_gdf = gpd.read_file(URBAN_SHAPEFILE)
except FileNotFoundError:
    print(f"ERROR: Could not find {URBAN_SHAPEFILE}. Please download it from Natural Earth and put it in this folder.")
    exit()

print("Generating global grid... (this might take a moment)")
all_boxes = generate_5km_grid(REGION, STEP_KM)

# Calculate chunk sizes
chunk_size = math.ceil(len(all_boxes) / TOTAL_CHUNKS)
start_idx = CURRENT_CHUNK * chunk_size
end_idx = min(start_idx + chunk_size, len(all_boxes))
my_boxes = all_boxes[start_idx:end_idx]

print(f"Total boxes in region: {len(all_boxes)}")
print(f"Boxes in Chunk {CURRENT_CHUNK}: {len(my_boxes)}")
print(f"Already completed: {len(completed_boxes)}")

# --- 4. Main Execution ---
file_exists = os.path.exists(OUTPUT_FILE)
with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    
    if not file_exists:
        writer.writerow(['Photo_ID', 'Title', 'Latitude', 'Longitude', 'Image_URL'])
    
    # Use tqdm to create a progress bar
    # CHANGE: Renamed 'box' to 'grid_box' to avoid overwriting Shapely's 'box' function
    for grid_box in tqdm(my_boxes, desc=f"Processing Chunk {CURRENT_CHUNK}"):
        
        # Create a unique ID for this box to log it
        box_id = f"{grid_box[0]:.4f},{grid_box[1]:.4f},{grid_box[2]:.4f},{grid_box[3]:.4f}"
        
        # 1. Check Save-State
        if box_id in completed_boxes:
            continue
            
        center_lon = (grid_box[0] + grid_box[2]) / 2
        center_lat = (grid_box[1] + grid_box[3]) / 2
        
        # 2. Check Land Mask: Log and skip if in the ocean
        if not globe.is_land(center_lat, center_lon):
            with open(LOG_FILE, 'a') as log:
                log.write(box_id + '\n')
            completed_boxes.add(box_id)
            continue
            
        # 3. Check Urban Mask: Log and skip if it intersects an urban area
        if is_urban(grid_box, urban_gdf):
            with open(LOG_FILE, 'a') as log:
                log.write(box_id + '\n')
            completed_boxes.add(box_id)
            continue
            
        # 4. Fetch Photos
        current_page = 1
        total_pages = 1
        photos_saved_this_box = 0
        
        while current_page <= total_pages:
            data = fetch_outdoor_photos(grid_box, current_page)
            
            if data.get('stat') == 'ok':
                if current_page == 1:
                    total_pages = data['photos']['pages']
                
                photos = data.get('photos', {}).get('photo', [])
                if not photos:
                    break
                    
                for photo in photos:
                    photo_id = photo.get('id')
                    title = photo.get('title', 'Untitled')
                    lat = photo.get('latitude')
                    lon = photo.get('longitude')
                    image_url = photo.get('url_m')
                    
                    if image_url and lat and lon:
                        writer.writerow([photo_id, title, lat, lon, image_url])
                        photos_saved_this_box += 1
                        
                    if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                        break
                
                if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                    break 
                
                current_page += 1
                
            else:
                break

        # 5. Update Save-State
        with open(LOG_FILE, 'a') as log:
            log.write(box_id + '\n')
        completed_boxes.add(box_id)

print(f"\nChunk {CURRENT_CHUNK} finished!")