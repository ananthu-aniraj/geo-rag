import requests
import math
import csv
import os
from global_land_mask import globe
from tqdm import tqdm
import time

# --- 1. Configuration ---
API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'
STEP_KM = 5
# Example: The "Western" Hemisphere (The Americas)
REGION = (-180, -90, -25, 90) 

# Add a 10% safety buffer (1.1 seconds instead of exactly 1.0)
DELAY_BETWEEN_CALLS = (3600 / 3600) * 1.1

# --- Splitting Variables ---
TOTAL_CHUNKS = 10     # How many pieces to split the region into
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
        f"&per_page=250"       # Request the maximum per page
        f"&page={page}"        # Specify which page we want
        f"&format=json"
        f"&nojsoncallback=1"
    )
    
    try:
        # Rate limiter applies to EVERY page we fetch
        time.sleep(DELAY_BETWEEN_CALLS) 
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception as e:
        return {'stat': 'fail', 'message': str(e)}

# --- 3. Save-State Initialization ---
completed_boxes = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        # Load all previously completed boxes into memory
        completed_boxes = set(line.strip() for line in f)

# --- 4. Main Execution ---
print("Generating global grid... (this might take a moment)")
all_boxes = generate_5km_grid(REGION, STEP_KM)

# Calculate chunk sizes
chunk_size = math.ceil(len(all_boxes) / TOTAL_CHUNKS)
start_idx = CURRENT_CHUNK * chunk_size
end_idx = min(start_idx + chunk_size, len(all_boxes))

# Extract only the boxes assigned to this specific chunk
my_boxes = all_boxes[start_idx:end_idx]
print(f"Total boxes in region: {len(all_boxes)}")
print(f"Boxes in Chunk {CURRENT_CHUNK}: {len(my_boxes)}")
print(f"Already completed: {len(completed_boxes)}")

# Open CSV in 'append' mode ('a') so we don't overwrite existing data
file_exists = os.path.exists(OUTPUT_FILE)
with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    
    # Only write headers if the file was just created
    if not file_exists:
        writer.writerow(['Photo_ID', 'Title', 'Latitude', 'Longitude', 'Image_URL'])
    
    # Use tqdm to create a progress bar
    for box in tqdm(my_boxes, desc=f"Processing Chunk {CURRENT_CHUNK}"):
        
        # Create a unique ID for this box to log it
        box_id = f"{box[0]:.4f},{box[1]:.4f},{box[2]:.4f},{box[3]:.4f}"
        
        # 1. Check Save-State: Skip if already processed
        if box_id in completed_boxes:
            continue
            
        center_lon = (box[0] + box[2]) / 2
        center_lat = (box[1] + box[3]) / 2
        
        # 2. Check Land Mask: Log it as complete immediately if in the ocean
        if not globe.is_land(center_lat, center_lon):
            with open(LOG_FILE, 'a') as log:
                log.write(box_id + '\n')
            continue
            
        # 3. Fetch Photos with Pagination
        current_page = 1
        total_pages = 1  # We start at 1, but update this after the first API call
        
        while current_page <= total_pages:
            data = fetch_outdoor_photos(box, current_page)
            
            if data.get('stat') == 'ok':
                # On the first page, find out how many pages there are in total
                if current_page == 1:
                    total_pages = data['photos']['pages']
                    # Optional: Print out how many total photos we expect for this box
                    total_photos = data['photos']['total']
                    if int(total_photos) > 0:
                        print(f"\n  -> Found {total_photos} photos in this box across {total_pages} pages.")
                
                photos = data.get('photos', {}).get('photo', [])
                
                # Safety break if a page comes back empty
                if not photos:
                    break
                    
                # Write the photos for this page
                for photo in photos:
                    photo_id = photo.get('id')
                    title = photo.get('title', 'Untitled')
                    lat = photo.get('latitude')
                    lon = photo.get('longitude')
                    image_url = photo.get('url_m')
                    
                    if image_url and lat and lon:
                        writer.writerow([photo_id, title, lat, lon, image_url])
                
                current_page += 1  # Move to the next page
                
            else:
                print(f"  -> Error on page {current_page}: {data.get('message')}")
                break  # Exit the pagination loop if an error occurs

        # 4. Update Save-State (Only log the box as complete after ALL pages are done)
        with open(LOG_FILE, 'a') as log:
            log.write(box_id + '\n')
        completed_boxes.add(box_id)

print(f"\nChunk {CURRENT_CHUNK} finished!")