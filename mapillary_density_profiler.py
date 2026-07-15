import os
import time
import csv
import requests
import datetime
import pandas as pd
from tqdm import tqdm
from pathlib import Path

ACCESS_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'
DELAY_BETWEEN_CALLS = 1.8  # Be polite to Mapillary servers

# Target Bounding Boxes (min_lon, min_lat, max_lon, max_lat)
# Centered around the landmarks (~1.5km to 3km boxes)
WONDERS = {
    # New Seven Wonders of the World + Giza + Dubai
    # "Taj Mahal": (78.037, 27.170, 78.047, 27.180),
    # "Colosseum": (12.487, 41.885, 12.497, 41.895),
    # "Chichen Itza": (-88.573, 20.679, -88.563, 20.689),
    # "Machu Picchu": (-72.550, -13.168, -72.540, -13.158),
    # "Christ the Redeemer": (-43.216, -22.957, -43.205, -22.947),
    # "Petra": (35.479, 30.315, 35.489, 30.326),
    # "Great Wall of China (Mutianyu)": (116.559, 40.426, 116.569, 40.436),
    # "Giza Pyramids": (31.129, 29.974, 31.139, 29.984),
    # "Dubai (Downtown)": (55.263, 25.179, 55.293, 25.209),
    #
    # # Famous Waterfalls (Continental Representation)
    # "Victoria Falls (Africa)": (25.841, -17.939, 25.871, -17.909),
    # "Iguazu Falls (South America)": (-54.465, -25.681, -54.435, -25.651),
    # "Niagara Falls (North America)": (-79.086, 43.065, -79.056, 43.095),
    # "Angel Falls (South America)": (-62.550, 5.952, -62.520, 5.982),
    #
    # # Special Natural Landmarks
    # "Mount Everest": (86.910, 27.973, 86.940, 28.003),
    # "Mount Fuji": (138.712, 35.345, 138.742, 35.375),
    # "Salar de Uyuni": (-67.504, -20.148, -67.474, -20.118),
    # "Grand Canyon (Mather Point)": (-112.152, 36.039, -112.122, 36.069),
    #
    # # Newly Added Visually Striking & Remote Landmarks
    # "Easter Island (Moai)": (-109.365, -27.131, -109.335, -27.101),
    # "Bora Bora Lagoon": (-151.748, -16.498, -151.718, -16.468),
    # "Svalbard (Longyearbyen)": (15.603, 78.201, 15.663, 78.231),
    # "Deception Island (Antarctica)": (-60.655, -62.965, -60.611, -62.935),
    # "Angkor Wat": (103.851, 13.385, 103.881, 13.415),
    # "Stonehenge": (-1.832, 51.151, -1.801, 51.182),
    # "Acropolis of Athens": (23.701, 37.951, 23.732, 37.982),
    # "Arches National Park (Moab)": (-109.265, 38.668, -109.235, 38.698),
    # "Halong Bay": (107.185, 20.885, 107.215, 20.915),
    # "Pamukkale Travertines": (29.101, 37.901, 29.132, 37.932),
    # "Lake Louise (Banff)": (-116.232, 51.385, -116.201, 51.415),
    # "Santorini Caldera": (25.401, 36.385, 25.432, 36.415),
    #
    # # Remote Sahara Desert & Polar/Subarctic Landmarks
    # "Richat Structure (Eye of the Sahara)": (-11.409, 21.100, -11.379, 21.130),
    # "Tassili n'Ajjer National Park (Sahara)": (8.985, 25.485, 9.015, 25.515),
    # "Ilulissat Icefjord (Greenland)": (-49.575, 69.135, -49.525, 69.165),
    # "Nuuk (Greenland)": (-51.745, 64.168, -51.698, 64.199),
    # "Yellowknife (Northern Canada)": (-114.392, 62.439, -114.348, 62.469),
    # "Virginia Falls (Nahanni, Canada)": (-125.760, 61.592, -125.713, 61.622),
    # "Olkhon Island (Lake Baikal, Russia)": (107.378, 53.135, 107.422, 53.165),
    # "Lena Pillars (Siberia, Russia)": (127.561, 61.131, 127.608, 61.161),
    #
    # # Famous Man-made Landmarks (Continental Representation)
    # "Eiffel Tower": (2.285, 48.852, 2.304, 48.864),
    # "Leaning Tower of Pisa": (10.389, 43.717, 10.404, 43.729),
    # "Statue of Liberty": (-74.052, 40.683, -74.037, 40.695),
    # "Sydney Opera House": (151.202, -33.860, 151.216, -33.848),
    # "Forbidden City": (116.381, 39.907, 116.401, 39.925),
    # "Djinguereber Mosque (Timbuktu)": (-3.020, 16.762, -3.000, 16.781),
    # "Fushimi Inari Shrine (Kyoto)": (135.762, 34.959, 135.783, 34.975),
    # "Palace of Westminster (Big Ben)": (-0.132, 51.493, -0.117, 51.505),
    # "Sagrada Família": (2.164, 41.397, 2.184, 41.410),
    # "Golden Gate Bridge": (-122.488, 37.813, -122.468, 37.826),
    # "Petronas Twin Towers": (101.701, 3.151, 101.721, 3.164),
    # "Neuschwanstein Castle": (10.739, 47.551, 10.760, 47.564),
    # "Mont-Saint-Michel": (-1.521, 48.630, -1.501, 48.642),
    # "Mount Rushmore": (-103.469, 43.872, -103.449, 43.885),
    # "Panama Canal (Miraflores Locks)": (-79.598, 8.985, -79.578, 8.998),
    # "Great Mosque of Djenné (Mali)": (-4.565, 13.899, -4.545, 13.911),
    #
    # # Microstates / City-States (Fully covered in a single API query)
    # "Singapore": (103.6000, 1.1500, 104.0500, 1.4800),
    # "Monaco": (7.4000, 43.7300, 7.4400, 43.7500),
    # "Liechtenstein": (9.4700, 47.0500, 9.6400, 47.2800),
    # "San Marino": (12.4000, 43.8800, 12.5200, 44.0000),
    # "Vatican City": (12.4450, 41.8990, 12.4600, 41.9080),
    # "Gibraltar": (-5.3750, 36.1000, -5.3400, 36.1800),
    "Rotterdam": (4.3793, 51.8617, 4.6018, 51.9943),
    "Delft": (4.3202, 51.9663, 4.4079, 52.0326),
    "Maastricht": (5.6389, 50.8038, 5.7629, 50.9120),
    "Wageningen": (5.6058, 51.9364, 5.7244, 52.0007),
    "Arnhem": (5.8030, 51.9335, 5.9903, 52.0779)
}


def fetch_mapillary_photos(bbox_coords=None, next_url=None, delay=3.0):
    """Fetches Mapillary photos using a bounding box OR a pagination URL."""
    headers = {
        "Authorization": f"OAuth {ACCESS_TOKEN}"
    }

    if next_url:
        url = next_url
    else:
        bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
        url = (
            f"https://graph.mapillary.com/images"
            f"?bbox={bbox_str}"
            f"&fields=id,geometry,thumb_1024_url,captured_at"
            f"&limit=50"
        )

    try:
        time.sleep(delay)
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json().get('data', [])
            new_next_url = response.links.get('next', {}).get('url')
            return {'stat': 'ok', 'data': data, 'next_url': new_next_url}
        else:
            return {'stat': 'fail', 'message': f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {'stat': 'fail', 'message': f"Timeout or connection error: {str(e)}"}


def collect_images_for_bbox(bbox, limit, delay, depth=0):
    """
    Recursively fetches images for a bounding box.
    If Mapillary throws an HTTP 500, a data density error, or a read timeout, we subdivide the box.
    """
    images_collected = []
    
    # Try fetching the first page
    res = fetch_mapillary_photos(bbox_coords=bbox, next_url=None, delay=delay)
    
    if res['stat'] == 'fail':
        msg = res['message'].lower()
        # If we hit Mapillary's data density or timeout thresholds, split into 4 quadrants
        if depth < 5 and ("reduce" in msg or "500" in msg or "limit" in msg or "code: 1" in msg or "timeout" in msg or "timed out" in msg):
            print(f"    - [Density Alert] Bbox too dense at depth {depth}. Splitting into 4 sub-quadrants...")
            min_lon, min_lat, max_lon, max_lat = bbox
            mid_lon = (min_lon + max_lon) / 2
            mid_lat = (min_lat + max_lat) / 2
            
            quadrants = [
                (min_lon, min_lat, mid_lon, mid_lat),  # SW
                (mid_lon, min_lat, max_lon, mid_lat),  # SE
                (min_lon, mid_lat, mid_lon, max_lat),  # NW
                (mid_lon, mid_lat, max_lon, max_lat)   # NE
            ]
            
            for quad in quadrants:
                needed = limit - len(images_collected)
                if needed <= 0:
                    break
                sub_images = collect_images_for_bbox(quad, needed, delay, depth + 1)
                images_collected.extend(sub_images)
            return images_collected
        else:
            print(f"    - Mapillary request failed: {res['message']}")
            return []
            
    # Page 1 succeeded, append results
    images_collected.extend(res.get('data', []))
    current_url = res.get('next_url')
    
    # Paginate through remaining records
    while current_url and len(images_collected) < limit:
        res = fetch_mapillary_photos(bbox_coords=None, next_url=current_url, delay=delay)
        if res['stat'] == 'ok':
            images = res.get('data', [])
            if not images:
                break
            images_collected.extend(images)
            current_url = res.get('next_url')
        else:
            print(f"    - Pagination failed: {res['message']}")
            break
            
    return images_collected[:limit]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape Mapillary street-level images of landmarks and microstates.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum photos to collect per landmark.")
    parser.add_argument("--out", type=str, default="seven_wonders_mapillary.csv", help="Output CSV path.")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between API calls in seconds (default: 3.0).")
    args = parser.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Ingest data
    output_exists = os.path.exists(args.out)
    processed_landmarks = set()

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

    print("\n--- Starting Seven Wonders Mapillary Scraper ---")
    for name, coords in WONDERS.items():
        box_id = f"{coords[0]:.4f},{coords[1]:.4f},{coords[2]:.4f},{coords[3]:.4f}"
        if name in processed_landmarks or box_id in completed_boxes:
            print(f"[{name}] (Box: {box_id}) already scraped or logged. Skipping...")
            completed_boxes.add(box_id)  # Sync state
            continue

        print(f"\nProcessing '{name}' | Coords: {coords}")

        photos = collect_images_for_bbox(coords, args.limit, args.delay)
        
        photos_saved = 0
        for img in photos:
            img_id = img.get('id')
            lon, lat = img.get('geometry', {}).get('coordinates', [None, None])
            image_url = img.get('thumb_1024_url')
            captured_at_ms = img.get('captured_at')
            captured_at = ""
            if captured_at_ms:
                captured_at = datetime.datetime.fromtimestamp(captured_at_ms / 1000.0,
                                                              datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

            if image_url and lat and lon:
                writer.writerow([img_id, "Mapillary", lat, lon, image_url, captured_at, name])
                photos_saved += 1

        print(f"Finished scraping [{name}]! Total images saved: {photos_saved}")

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
