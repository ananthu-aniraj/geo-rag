import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Target Bounding Boxes (min_lon, min_lat, max_lon, max_lat)
# Centered around the landmarks (~1.5km to 3km boxes)
preset_locs = {
    # New Seven Wonders of the World + Giza + Dubai
    "Taj Mahal": (78.037, 27.170, 78.047, 27.180),
    "Colosseum": (12.487, 41.885, 12.497, 41.895),
    "Chichen Itza": (-88.573, 20.679, -88.563, 20.689),
    "Machu Picchu": (-72.550, -13.168, -72.540, -13.158),
    "Christ the Redeemer": (-43.216, -22.957, -43.205, -22.947),
    "Petra": (35.479, 30.315, 35.489, 30.326),
    "Great Wall of China (Mutianyu)": (116.559, 40.426, 116.569, 40.436),
    "Giza Pyramids": (31.129, 29.974, 31.139, 29.984),
    "Dubai (Downtown)": (55.263, 25.179, 55.293, 25.209),

    # Famous Waterfalls (Continental Representation)
    "Victoria Falls (Africa)": (25.841, -17.939, 25.871, -17.909),
    "Iguazu Falls (South America)": (-54.465, -25.681, -54.435, -25.651),
    "Niagara Falls (North America)": (-79.086, 43.065, -79.056, 43.095),
    "Angel Falls (South America)": (-62.550, 5.952, -62.520, 5.982),

    # Special Natural Landmarks
    "Mount Everest": (86.910, 27.973, 86.940, 28.003),
    "Mount Fuji": (138.712, 35.345, 138.742, 35.375),
    "Salar de Uyuni": (-67.504, -20.148, -67.474, -20.118),
    "Grand Canyon (Mather Point)": (-112.152, 36.039, -112.122, 36.069),

    # Newly Added Visually Striking & Remote Landmarks
    "Easter Island (Moai)": (-109.365, -27.131, -109.335, -27.101),
    "Bora Bora Lagoon": (-151.748, -16.498, -151.718, -16.468),
    "Svalbard (Longyearbyen)": (15.603, 78.201, 15.663, 78.231),
    "Deception Island (Antarctica)": (-60.655, -62.965, -60.611, -62.935),
    "Angkor Wat": (103.851, 13.385, 103.881, 13.415),
    "Stonehenge": (-1.832, 51.151, -1.801, 51.182),
    "Acropolis of Athens": (23.701, 37.951, 23.732, 37.982),
    "Arches National Park (Moab)": (-109.265, 38.668, -109.235, 38.698),
    "Halong Bay": (107.185, 20.885, 107.215, 20.915),
    "Pamukkale Travertines": (29.101, 37.901, 29.132, 37.932),
    "Lake Louise (Banff)": (-116.232, 51.385, -116.201, 51.415),
    "Santorini Caldera": (25.401, 36.385, 25.432, 36.415),

    # Remote Sahara Desert & Polar/Subarctic Landmarks
    "Richat Structure (Eye of the Sahara)": (-11.409, 21.100, -11.379, 21.130),
    "Tassili n'Ajjer National Park (Sahara)": (8.985, 25.485, 9.015, 25.515),
    "Ilulissat Icefjord (Greenland)": (-49.575, 69.135, -49.525, 69.165),
    "Nuuk (Greenland)": (-51.745, 64.168, -51.698, 64.199),
    "Yellowknife (Northern Canada)": (-114.392, 62.439, -114.348, 62.469),
    "Virginia Falls (Nahanni, Canada)": (-125.760, 61.592, -125.713, 61.622),
    "Olkhon Island (Lake Baikal, Russia)": (107.378, 53.135, 107.422, 53.165),
    "Lena Pillars (Siberia, Russia)": (127.561, 61.131, 127.608, 61.161),

    # Famous Man-made Landmarks (Continental Representation)
    "Eiffel Tower": (2.285, 48.852, 2.304, 48.864),
    "Leaning Tower of Pisa": (10.389, 43.717, 10.404, 43.729),
    "Statue of Liberty": (-74.052, 40.683, -74.037, 40.695),
    "Sydney Opera House": (151.202, -33.860, 151.216, -33.848),
    "Forbidden City": (116.381, 39.907, 116.401, 39.925),
    "Djinguereber Mosque (Timbuktu)": (-3.020, 16.762, -3.000, 16.781),
    "Fushimi Inari Shrine (Kyoto)": (135.762, 34.959, 135.783, 34.975),
    "Palace of Westminster (Big Ben)": (-0.132, 51.493, -0.117, 51.505),
    "Sagrada Família": (2.164, 41.397, 2.184, 41.410),
    "Golden Gate Bridge": (-122.488, 37.813, -122.468, 37.826),
    "Petronas Twin Towers": (101.701, 3.151, 101.721, 3.164),
    "Neuschwanstein Castle": (10.739, 47.551, 10.760, 47.564),
    "Mont-Saint-Michel": (-1.521, 48.630, -1.501, 48.642),
    "Mount Rushmore": (-103.469, 43.872, -103.449, 43.885),
    "Panama Canal (Miraflores Locks)": (-79.598, 8.985, -79.578, 8.998),
    "Great Mosque of Djenné (Mali)": (-4.565, 13.899, -4.545, 13.911),

    # Microstates / City-States (Fully covered in a single API query)
    "Singapore": (103.6000, 1.1500, 104.0500, 1.4800),
    "Monaco": (7.4000, 43.7300, 7.4400, 43.7500),
    "Liechtenstein": (9.4700, 47.0500, 9.6400, 47.2800),
    "San Marino": (12.4000, 43.8800, 12.5200, 44.0000),
    "Vatican City": (12.4450, 41.8990, 12.4600, 41.9080),
    "Gibraltar": (-5.3750, 36.1000, -5.3400, 36.1800),
    "Rotterdam": (4.3793, 51.8617, 4.6018, 51.9943),
    "Delft": (4.3202, 51.9663, 4.4079, 52.0326),
    "Maastricht": (5.6389, 50.8038, 5.7629, 50.9120),
    "Wageningen": (5.6058, 51.9364, 5.7244, 52.0007),
    "Arnhem": (5.8030, 51.9335, 5.9903, 52.0779)
}


def fetch_flickr_photos(bbox_coords, api_key, page=1, geo_context=2, delay=2.0):
    """Fetches geo-tagged photos from Flickr REST API for a bounding box."""
    bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
    url = (
        f"https://www.flickr.com/services/rest/"
        f"?method=flickr.photos.search"
        f"&api_key={api_key}"
        f"&bbox={bbox_str}"
        f"&has_geo=1"
        f"&geo_context={geo_context}"
        f"&extras=url_m,geo,date_taken,license"
        f"&per_page=250"
        f"&page={page}"
        f"&format=json"
        f"&nojsoncallback=1"
    )
    try:
        time.sleep(delay)
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error connecting to Flickr: {e}")
    return {'stat': 'fail'}


def geocode_location(location_name):
    """Geocodes a location name to a bounding box (min_lon, min_lat, max_lon, max_lat) using Nominatim.
    Automatically pads landmark-sized bounding boxes to at least 2km x 2km to capture camera standpoints."""
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(location_name)}&format=json&limit=1"
    headers = {"User-Agent": "geo-rag-density-profiler"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                bbox = data[0].get('boundingbox')
                if bbox and len(bbox) == 4:
                    # Nominatim returns [min_lat, max_lat, min_lon, max_lon]
                    min_lat = float(bbox[0])
                    max_lat = float(bbox[1])
                    min_lon = float(bbox[2])
                    max_lon = float(bbox[3])
                    
                    # Pad out small landmark bounding boxes (less than ~2.2km wide)
                    width = max_lon - min_lon
                    height = max_lat - min_lat
                    if width < 0.02 or height < 0.02:
                        center_lon = (min_lon + max_lon) / 2
                        center_lat = (min_lat + max_lat) / 2
                        min_lon = center_lon - 0.01
                        max_lon = center_lon + 0.01
                        min_lat = center_lat - 0.01
                        max_lat = center_lat + 0.01
                        print(f" -> Small bounding box detected. Padded '{location_name}' to 2km x 2km buffer.")
                        
                    return (min_lon, min_lat, max_lon, max_lat)
    except Exception as e:
        print(f"Geocoding error for '{location_name}': {e}")
    return None


def generate_grid_boxes(bbox, step_km=5.0):
    """Divides a bounding box into a grid of step_km * step_km sub-boxes (accounting for latitude cosine)."""
    import math
    min_lon, min_lat, max_lon, max_lat = bbox
    
    lat_step = step_km / 111.32
    
    sub_boxes = []
    current_lat = min_lat
    while current_lat < max_lat:
        next_lat = min(current_lat + lat_step, max_lat)
        
        # Calculate cos_lat for the center of this band
        mid_lat = (current_lat + next_lat) / 2.0
        cos_lat = math.cos(math.radians(max(-89.9, min(89.9, mid_lat))))
        lon_step = step_km / (111.32 * cos_lat)
        
        current_lon = min_lon
        while current_lon < max_lon:
            next_lon = min(current_lon + lon_step, max_lon)
            sub_boxes.append((current_lon, current_lat, next_lon, next_lat))
            current_lon += lon_step
            
        current_lat += lat_step
        
    return sub_boxes


def main():
    parser = argparse.ArgumentParser(description="Scrape Flickr outdoor images of locations.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum photos to collect per landmark (only used if grid_size=0).")
    parser.add_argument("--limit_per_box", type=int, default=100, help="Maximum photos to collect per grid sub-box (default: 100).")
    parser.add_argument("--grid_size", type=float, default=5.0, help="Grid size in km (default: 5.0). Set to 0 to disable grid splitting.")
    parser.add_argument("--out", type=str, default="seven_wonders_flickr.csv", help="Output CSV path.")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls in seconds (default: 2.0).")
    parser.add_argument("--api_key", type=str, required=True, help="Flickr API key")
    parser.add_argument("--location", type=str, default=None, help="Dynamic location name to geocode and scrape.")
    parser.add_argument("--bbox", type=str, default=None, help="Manual bounding box coords (min_lon,min_lat,max_lon,max_lat).")
    args = parser.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Determine target bounding boxes
    targets = {}
    if args.location:
        print(f"Geocoding location: '{args.location}'...")
        coords = geocode_location(args.location)
        if coords:
            print(f" -> Found coordinates: {coords}")
            targets[args.location] = coords
        else:
            print(f"Error: Could not geocode location '{args.location}'.")
            sys.exit(1)
    elif args.bbox:
        try:
            coords = tuple(map(float, args.bbox.split(',')))
            if len(coords) != 4:
                raise ValueError("Bounding box must contain exactly 4 coordinates.")
            targets["Custom_BBox"] = coords
            print(f"Using manual bounding box: {coords}")
        except Exception as e:
            print(f"Error parsing bounding box '{args.bbox}': {e}")
            sys.exit(1)
    else:
        targets = preset_locs

    # Ingest data
    output_exists = os.path.exists(args.out)
    processed_landmarks = set()

    # Establish log path for searched boxes
    log_path = args.out.replace('.csv', '_completed_boxes.txt')
    completed_boxes = set()
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            completed_boxes = set(line.strip() for line in f)
        print(f"Found {len(completed_boxes)} completed boxes in log: {log_path}")

    if output_exists:
        try:
            df_temp = pd.read_csv(args.out)
            if not df_temp.empty and 'Landmark' in df_temp.columns:
                processed_landmarks = set(df_temp['Landmark'].unique())
                print(f"Found existing output file. Landmarks already processed: {processed_landmarks}")
        except Exception:
            pass

    csv_file = open(args.out, mode='a', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)

    if not output_exists:
        writer.writerow(['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'Captured_At', 'Landmark'])

    print("\n--- Starting Flickr Density Profiler ---")
    for name, coords in targets.items():
        # Generate sub-boxes
        if args.grid_size > 0:
            sub_boxes = generate_grid_boxes(coords, step_km=args.grid_size)
            print(f"\nProcessing '{name}' | Divided into {len(sub_boxes)} grid cells of {args.grid_size}km x {args.grid_size}km.")
        else:
            sub_boxes = [coords]

        for idx, box in enumerate(sub_boxes):
            box_id = f"{box[0]:.4f},{box[1]:.4f},{box[2]:.4f},{box[3]:.4f}"
            if box_id in completed_boxes:
                print(f" -> Sub-box #{idx+1}/{len(sub_boxes)} ({box_id}) already scraped. Skipping...")
                continue

            print(f"\n -> Scraping Sub-box #{idx+1}/{len(sub_boxes)} ({box_id})...")
            photos_saved = 0
            limit_val = args.limit_per_box if args.grid_size > 0 else args.limit

            # Priority 1: Outdoors (2). Priority 2: Unlabelled (0).
            for context in [2, 0]:
                if photos_saved >= limit_val:
                    break
                    
                page = 1
                total_pages = 1
                
                while page <= total_pages:
                    print(f"    - Querying Flickr [Context: {context}, Page: {page}/{total_pages}]...")
                    data = fetch_flickr_photos(box, api_key=args.api_key, page=page, geo_context=context, delay=args.delay)
                    if data.get('stat') == 'ok':
                        if page == 1:
                            total_pages = min(data.get('photos', {}).get('pages', 1), 16)  # limit paging to avoid rate limits
                        
                        photos = data.get('photos', {}).get('photo', [])
                        if not photos:
                            break
                            
                        for photo in photos:
                            p_id = photo.get('id')
                            lat = photo.get('latitude')
                            lon = photo.get('longitude')
                            url = photo.get('url_m')
                            captured = photo.get('datetaken', '')
                            if captured:
                                captured = captured.replace(" ", "T")
                                
                            if url and lat and lon:
                                writer.writerow([p_id, "Flickr", lat, lon, url, captured, name])
                                photos_saved += 1
                                
                            if photos_saved >= limit_val:
                                break
                                
                        if photos_saved >= limit_val:
                            break
                            
                        page += 1
                    else:
                        print("    - API call failed. Breaking pagination loop.")
                        break

            print(f"Finished scraping sub-box #{idx+1}/{len(sub_boxes)}! Saved: {photos_saved}")

            # Log this box to file for future backfills / resume tracking
            with open(log_path, 'a') as log:
                log.write(box_id + '\n')
            completed_boxes.add(box_id)

            csv_file.flush()  # Force write to disk

    csv_file.close()
    print(f"Completed boxes log saved to: {os.path.abspath(log_path)}")
    print(f"\nAll operations complete! Data saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
