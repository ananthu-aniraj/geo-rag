import csv
import math
import time

import requests

# --- 1. Configuration ---
ACCESS_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'  # <-- REPLACE WITH YOUR MAPILLARY TOKEN
MONTPELLIER_REGION = (3.85, 43.59, 3.91, 43.63) # min_lon, min_lat, max_lon, max_lat
STEP_KM = 1  # 1x1 km squares for a fine-grained local test
MAX_PHOTOS_PER_BOX = 10  # Keep it low so the test runs instantly
OUTPUT_FILE = 'montpellier_mapillary_test.csv'

# --- 2. Helper Functions ---
def generate_grid(region, step_km):
    """Slices a bounding box into smaller boxes."""
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

def fetch_mapillary_photos(bbox_coords=None, next_url=None):
    """Fetches photos using a bounding box OR a pagination URL."""
    headers = {"Authorization": f"OAuth {ACCESS_TOKEN}"}
    
    if next_url:
        url = next_url
    else:
        bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
        url = (
            f"https://graph.mapillary.com/images"
            f"?bbox={bbox_str}"
            f"&fields=id,geometry,thumb_1024_url"
            f"&limit=50" 
        )
    
    try:
        time.sleep(0.5) # Slight pause to be polite to the API
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json().get('data', [])
            new_next_url = response.links.get('next', {}).get('url')
            return {'stat': 'ok', 'data': data, 'next_url': new_next_url}
        else:
            return {'stat': 'fail', 'message': f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {'stat': 'fail', 'message': str(e)}

# --- 3. Main Execution ---
print(f"Generating {STEP_KM}x{STEP_KM} km grid for Montpellier...")
grid_boxes = generate_grid(MONTPELLIER_REGION, STEP_KM)
print(f"Created {len(grid_boxes)} tiny boxes to test.\n")

with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL'])
    
    for i, grid_box in enumerate(grid_boxes):
        print(f"Box {i+1}/{len(grid_boxes)}: Fetching Mapillary images...")
        
        current_url = None
        photos_saved_this_box = 0
        
        while True:
            result = fetch_mapillary_photos(bbox_coords=grid_box, next_url=current_url)
            
            if result['stat'] == 'ok':
                images = result.get('data', [])
                
                if not images:
                    print("  -> No images found in this specific box.")
                    break
                    
                for img in images:
                    img_id = img.get('id')
                    lon, lat = img.get('geometry', {}).get('coordinates', [None, None])
                    image_url = img.get('thumb_1024_url')
                    
                    if image_url and lat and lon:
                        writer.writerow([img_id, "Mapillary", lat, lon, image_url])
                        photos_saved_this_box += 1
                        
                    if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                        break
                
                if photos_saved_this_box >= MAX_PHOTOS_PER_BOX:
                    print(f"  -> Saved {MAX_PHOTOS_PER_BOX} images.")
                    break 
                
                current_url = result.get('next_url')
                if not current_url:
                    print(f"  -> Reached end of available images ({photos_saved_this_box} saved).")
                    break
                
            else:
                print(f"  -> API Error: {result.get('message')}")
                break

print(f"\nTest finished! Open '{OUTPUT_FILE}' to see your data.")