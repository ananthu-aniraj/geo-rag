import os
import time
import csv
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path

API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'
DELAY_BETWEEN_CALLS = 1.1

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

    # Microstates / City-States (Fully covered in a single API query)
    "Singapore": (103.6000, 1.1500, 104.0500, 1.4800),
    "Monaco": (7.4000, 43.7300, 7.4400, 43.7500),
    "Liechtenstein": (9.4700, 47.0500, 9.6400, 47.2800),
    "San Marino": (12.4000, 43.8800, 12.5200, 44.0000),
    "Vatican City": (12.4450, 41.8990, 12.4600, 41.9080),
    "Gibraltar": (-5.3750, 36.1000, -5.3400, 36.1800)
}


def fetch_flickr_photos(bbox_coords, page=1, geo_context=2):
    """Fetches geo-tagged photos from Flickr REST API for a bounding box."""
    bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
    url = (
        f"https://www.flickr.com/services/rest/"
        f"?method=flickr.photos.search"
        f"&api_key={API_KEY}"
        f"&bbox={bbox_str}"
        f"&has_geo=1"
        f"&geo_context={geo_context}"
        f"&extras=url_m,geo,date_taken"
        f"&per_page=250"
        f"&page={page}"
        f"&format=json"
        f"&nojsoncallback=1"
    )
    try:
        time.sleep(DELAY_BETWEEN_CALLS)
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error connecting to Flickr: {e}")
    return {'stat': 'fail'}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape Flickr outdoor images of the Seven Wonders of the World.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum photos to collect per landmark.")
    parser.add_argument("--out", type=str, default="seven_wonders_flickr.csv", help="Output CSV path.")
    args = parser.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

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

    print("\n--- Starting Seven Wonders Flickr Scraper ---")
    for name, coords in WONDERS.items():
        box_id = f"{coords[0]:.4f},{coords[1]:.4f},{coords[2]:.4f},{coords[3]:.4f}"
        if name in processed_landmarks or box_id in completed_boxes:
            print(f"[{name}] (Box: {box_id}) already scraped or logged. Skipping...")
            completed_boxes.add(box_id)  # Sync state
            continue

        print(f"\nProcessing '{name}' | Coords: {coords}")

        photos_saved = 0

        # Priority 1: Outdoors (2). Priority 2: Unlabelled (0).
        for context in [2, 0]:
            if photos_saved >= args.limit:
                break
                
            page = 1
            total_pages = 1
            
            while page <= total_pages:
                print(f" -> Querying Flickr API [Context: {context}, Page: {page}/{total_pages}]...")
                data = fetch_flickr_photos(coords, page=page, geo_context=context)

                if data.get('stat') == 'ok':
                    if page == 1:
                        total_pages = data.get('photos', {}).get('pages', 1)

                    photos = data.get('photos', {}).get('photo', [])
                    if not photos:
                        break

                    for p in photos:
                        if photos_saved >= args.limit:
                            break

                        p_id = p.get('id')
                        lat = p.get('latitude')
                        lon = p.get('longitude')
                        url = p.get('url_m')
                        captured = p.get('datetaken', '')
                        if captured:
                            captured = captured.replace(" ", "T")

                        if url and lat and lon:
                            writer.writerow([p_id, 'Flickr', lat, lon, url, captured, name])
                            photos_saved += 1

                    print(f"    - Collected {photos_saved}/{args.limit} photos so far...")
                    if photos_saved >= args.limit:
                        break
                    page += 1
                else:
                    print("    - API call failed. Breaking pagination loop.")
                    break

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
