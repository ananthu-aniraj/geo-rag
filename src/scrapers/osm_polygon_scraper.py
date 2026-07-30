import argparse
import csv
import datetime
import math
import os
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
from tqdm import tqdm

# --- API Rate Limit Calculations ---
# Wikimedia Commons & KartaView both recommend keeping anonymous/public requests 
# below 1 request per second to prevent temporary IP bans. 
# Mathematically: 3600 seconds / 3600 max requests/hour = 1.0 second minimum delay.
# We pad this to 1.2s for safety.
DEFAULT_DELAY = 1.2

USER_AGENT = "Geo-RAG-OSM-Scraper/1.0 (contact: aaniraj@example.com)"


def make_request_with_backoff(url, params=None, headers=None, max_retries=5, initial_delay=1.5):
    """Executes a request with exponential backoff for HTTP 429 or connection issues."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                print(f"\n[HTTP 429] Rate limit hit. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
            elif response.status_code == 403:
                print(f"\n[HTTP 403] Forbidden: Access blocked. Your IP or User-Agent might be blocked by the server.")
                return None
            else:
                print(f"\n[HTTP {response.status_code}] Warning. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
        except Exception as e:
            print(f"\nConnection error: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2.0
    return None


def fetch_boundary_from_local_countries(query_str, shapefile_path="shapefiles/ne_10m_admin_0_countries.shp"):
    """Attempts to find the boundary of a country or continent in the local shapefile."""
    if not os.path.exists(shapefile_path):
        return None
    
    try:
        print(f"Checking local country/continent shapefile for: '{query_str}'...")
        gdf = gpd.read_file(shapefile_path)
        
        # Try matching country columns case-insensitively
        for col in ['NAME', 'SOVEREIGNT', 'ADMIN', 'NAME_LONG']:
            if col in gdf.columns:
                matches = gdf[gdf[col].astype(str).str.lower() == query_str.lower()]
                if not matches.empty:
                    print(f"Matched country '{query_str}' under column '{col}' in local shapefile.")
                    return unary_union(matches.geometry)
                    
        # Check continent match
        if 'CONTINENT' in gdf.columns:
            matches = gdf[gdf['CONTINENT'].astype(str).str.lower() == query_str.lower()]
            if not matches.empty:
                print(f"Matched continent '{query_str}' in local shapefile.")
                return unary_union(matches.geometry)
                
    except Exception as e:
        print(f"Warning: Failed reading local countries shapefile: {e}")
        
    return None


def fetch_boundary_by_query(query_str, shapefile_path="shapefiles/ne_10m_admin_0_countries.shp"):
    """Fetches a boundary polygon using local country shapefile query or falls back to Nominatim."""
    local_poly = fetch_boundary_from_local_countries(query_str, shapefile_path)
    if local_poly is not None:
        return local_poly

    print(f"Searching Nominatim for query: '{query_str}'...")
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query_str,
        "format": "json",
        "polygon_geojson": 1,
        "limit": 1
    }
    headers = {"User-Agent": USER_AGENT}
    
    response = make_request_with_backoff(url, params=params, headers=headers)
    if response is None:
        return None
        
    try:
        data = response.json()
        if not data:
            print(f"No boundary found for query: '{query_str}'")
            return None
            
        geojson = data[0].get("geojson")
        if not geojson:
            print(f"No geometry found in Nominatim result for: '{query_str}'")
            return None
            
        return shape(geojson)
    except Exception as e:
        print(f"Failed to parse Nominatim search response: {e}")
        return None


def fetch_boundary_by_relation(relation_id):
    """Fetches a boundary polygon from Nominatim using an OSM Relation ID."""
    print(f"Fetching boundary for OSM Relation: R{relation_id}...")
    url = "https://nominatim.openstreetmap.org/lookup"
    params = {
        "osm_ids": f"R{relation_id}",
        "format": "json",
        "polygon_geojson": 1
    }
    headers = {"User-Agent": USER_AGENT}
    
    response = make_request_with_backoff(url, params=params, headers=headers)
    if response is None:
        return None
        
    try:
        data = response.json()
        if not data:
            print(f"No boundary found for relation ID: {relation_id}")
            return None
            
        geojson = data[0].get("geojson")
        if not geojson:
            print(f"No geometry found in Nominatim result for relation: R{relation_id}")
            return None
            
        return shape(geojson)
    except Exception as e:
        print(f"Failed to parse Nominatim lookup response: {e}")
        return None


def fetch_wikimedia_batch(grid_box, polygon, max_images, delay, continue_params=None):
    """Fetches a single batch of images from Wikimedia Commons in a grid box."""
    min_lon, min_lat, max_lon, max_lat = grid_box
    url = "https://commons.wikimedia.org/w/api.php"
    
    if continue_params is None:
        continue_params = {}
        
    params = {
        "action": "query",
        "generator": "geosearch",
        "ggsnamespace": 6,  # File namespace
        "ggsbbox": f"{max_lat}|{min_lon}|{min_lat}|{max_lon}",
        "ggslimit": 500,
        "prop": "coordinates|imageinfo",
        "iiprop": "url|timestamp|extmetadata",
        "coprimary": "all",
        "format": "json",
        **continue_params
    }
    headers = {"User-Agent": USER_AGENT}
    
    time.sleep(delay)
    response = make_request_with_backoff(url, params=params, headers=headers)
    if response is None:
        return {'stat': 'fail', 'data': [], 'continue': None}
        
    try:
        data = response.json()
        query_data = data.get("query", {})
        pages = query_data.get("pages", {})
        
        results = []
        for page_id, page in pages.items():
            coords = page.get("coordinates", [{}])[0]
            lat = coords.get("lat")
            lon = coords.get("lon")
            
            imageinfo = page.get("imageinfo", [{}])[0]
            img_url = imageinfo.get("url")
            timestamp = imageinfo.get("timestamp", "")
            
            # Extract license short name
            extmetadata = imageinfo.get("extmetadata", {})
            license_val = extmetadata.get("LicenseShortName", {}).get("value", "unknown")
            
            if lat is not None and lon is not None and img_url:
                pt = Point(lon, lat)
                if polygon.contains(pt):
                    results.append({
                        "Photo_ID": page_id,
                        "Platform": "Wikimedia",
                        "Latitude": lat,
                        "Longitude": lon,
                        "Image_URL": img_url,
                        "Captured_At": timestamp,
                        "License": license_val
                    })
                    
        new_continue = data.get("continue")
        return {'stat': 'ok', 'data': results, 'continue': new_continue}
    except Exception as e:
        return {'stat': 'fail', 'data': [], 'continue': None, 'message': str(e)}


def fetch_kartaview_batch(grid_box, polygon, max_images, delay, page=1):
    """Fetches a single batch of images from KartaView in a grid box."""
    min_lon, min_lat, max_lon, max_lat = grid_box
    
    # KartaView API limits bounding box size to 0.04 degrees per side.
    # If the box exceeds this limit, we split it into smaller sub-boxes recursively.
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    
    if lon_span > 0.04 or lat_span > 0.04:
        mid_lon = (min_lon + max_lon) / 2.0
        mid_lat = (min_lat + max_lat) / 2.0
        
        sub_boxes = [
            (min_lon, min_lat, mid_lon, mid_lat),  # Bottom-left
            (mid_lon, min_lat, max_lon, mid_lat),  # Bottom-right
            (min_lon, mid_lat, mid_lon, max_lat),  # Top-left
            (mid_lon, mid_lat, max_lon, max_lat)   # Top-right
        ]
        
        combined_results = []
        for s_box in sub_boxes:
            res = fetch_kartaview_batch(s_box, polygon, max_images, delay, page)
            if res['stat'] == 'ok':
                combined_results.extend(res['data'])
            if len(combined_results) >= max_images:
                break
        return {'stat': 'ok', 'data': combined_results[:max_images]}

    url = "https://api.openstreetcam.org/2.0/photo/"
    
    params = {
        "nwLat": max_lat,
        "nwLng": min_lon,
        "seLat": min_lat,
        "seLng": max_lon,
        "itemsPerPage": 150,
        "page": page
    }
    
    time.sleep(delay)
    response = make_request_with_backoff(url, params=params)
    if response is None:
        return {'stat': 'fail', 'data': []}
        
    try:
        data = response.json()
        photos = data.get("result", {}).get("data", [])
        
        results = []
        for photo in photos:
            photo_id = photo.get("id")
            lat = photo.get("latitude") or photo.get("lat")
            lon = photo.get("longitude") or photo.get("lng")
            img_url = photo.get("fileurl") or photo.get("fileurlLd") or photo.get("thumbnail")
            captured_at = photo.get("shotDate") or photo.get("dateAdded") or ""
            
            if lat is not None and lon is not None and img_url:
                lat_f = float(lat)
                lon_f = float(lon)
                pt = Point(lon_f, lat_f)
                if polygon.contains(pt):
                    results.append({
                        "Photo_ID": photo_id,
                        "Platform": "KartaView",
                        "Latitude": lat_f,
                        "Longitude": lon_f,
                        "Image_URL": img_url,
                        "Captured_At": captured_at,
                        "License": "CC BY-SA 4.0"  # Platform standard
                    })
        return {'stat': 'ok', 'data': results}
    except Exception as e:
        return {'stat': 'fail', 'data': [], 'message': str(e)}


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


def main():
    parser = argparse.ArgumentParser(description="Grid-based OpenStreetMap Polygon Scraper.")
    parser.add_argument("--osm_relation", type=int, help="OSM Relation ID (e.g. 74263 for Paris).")
    parser.add_argument("--osm_query", type=str, help="Free-text search query (e.g. 'Central Park, New York').")
    parser.add_argument("--geojson", type=str, help="Path to local GeoJSON/Shapefile containing target boundary.")
    
    # Grid sampling & Chunking properties (matches flickr_5km_grid_search)
    parser.add_argument("--chunk", type=int, default=0, help="Which chunk of the grid to process.")
    parser.add_argument("--total_chunks", type=int, default=1, help="Total chunks to split the grid into.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory for output files.")
    
    parser.add_argument("--platforms", type=str, default="all", choices=["wikimedia", "kartaview", "all"],
                        help="Which platforms to scrape.")
    parser.add_argument("--max_images_per_box", type=int, default=100, help="Max images to retrieve per grid box.")
    parser.add_argument("--countries_shp", type=str, default="shapefiles/ne_10m_admin_0_countries.shp",
                        help="Path to local countries shapefile.")
    parser.add_argument("--uncovered_shp", type=str, default="shapefiles/uncovered_land_areas_test.shp",
                        help="Path to local uncovered land areas shapefile.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay in seconds between API requests.")
    args = parser.parse_args()

    # Create target directory
    Path(args.base_dir).mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = os.path.join(args.base_dir, f"osm_data_chunk_{args.chunk}.csv")
    LOG_FILE = os.path.join(args.base_dir, f"osm_completed_boxes_chunk_{args.chunk}.txt")

    # Load polygon boundary (checking local base_dir cache first)
    polygon = None
    local_geojson_path = os.path.join(args.base_dir, "boundary.geojson")
    
    if os.path.exists(local_geojson_path):
        print(f"Loading cached boundary from local file: {local_geojson_path}...")
        try:
            gdf = gpd.read_file(local_geojson_path)
            polygon = unary_union(gdf.geometry)
        except Exception as e:
            print(f"Warning: Failed to load cached boundary from {local_geojson_path}: {e}. Re-fetching...")
            polygon = None

    if polygon is None or polygon.is_empty:
        if args.osm_relation:
            polygon = fetch_boundary_by_relation(args.osm_relation)
        elif args.osm_query:
            polygon = fetch_boundary_by_query(args.osm_query, args.countries_shp)
        elif args.geojson:
            if not os.path.exists(args.geojson):
                print(f"Error: GeoJSON file '{args.geojson}' not found.")
                sys.exit(1)
            print(f"Loading boundary from local file: {args.geojson}...")
            gdf = gpd.read_file(args.geojson)
            polygon = unary_union(gdf.geometry)
        else:
            print("Error: You must provide one of: --osm_relation, --osm_query, or --geojson.")
            sys.exit(1)

        if polygon is not None and not polygon.is_empty:
            try:
                print(f"Caching resolved boundary polygon to: {local_geojson_path}...")
                gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326").to_file(local_geojson_path, driver="GeoJSON")
            except Exception as e:
                print(f"Warning: Failed to cache boundary polygon: {e}")

    if polygon is None or polygon.is_empty:
        print("Error: Failed to obtain a valid boundary geometry.")
        sys.exit(1)

    # 1. Load Completed Box Logs
    completed_boxes = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            completed_boxes = set(line.strip() for line in f)

    # Load Uncovered Land Areas shapefile if it exists
    uncovered_gdf = None
    if args.uncovered_shp and os.path.exists(args.uncovered_shp):
        print(f"Loading Uncovered Land Areas map: {args.uncovered_shp}...")
        try:
            uncovered_gdf = gpd.read_file(args.uncovered_shp, engine='pyogrio')
        except Exception as e:
            try:
                uncovered_gdf = gpd.read_file(args.uncovered_shp)
            except Exception as e2:
                print(f"Warning: Could not load uncovered shapefile: {e2}")

    # 2. Partition Bounding Box into 5km x 5km grid boxes
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    STEP_KM = 5.0
    lat_step = STEP_KM / 111.32
    current_lat = min_lat

    virtual_grid = []
    while current_lat < max_lat:
        cos_lat = math.cos(math.radians(max(-89.9, min(89.9, current_lat))))
        lon_step = STEP_KM / (111.32 * cos_lat)
        
        current_lon = min_lon
        while current_lon < max_lon:
            grid_box = box(current_lon, current_lat, current_lon + lon_step, current_lat + lat_step)
            # Only process if grid box intersects polygon boundary to optimize speed
            if grid_box.intersects(polygon):
                virtual_grid.append((current_lon, current_lat, current_lon + lon_step, current_lat + lat_step))
            current_lon += lon_step
        current_lat += lat_step

    # Sort boxes deterministically
    virtual_grid.sort()
    
    # 3. Pick boxes assigned to this chunk
    my_boxes = []
    for idx, g_box in enumerate(virtual_grid):
        if idx % args.total_chunks == args.chunk:
            my_boxes.append(g_box)

    # Shuffle the specific boxes assigned to THIS chunk to avoid sequential processing
    random.Random(42 + args.chunk).shuffle(my_boxes)

    print(f"Total virtual boxes intersecting polygon: {len(virtual_grid)}")
    print(f"Boxes assigned to Chunk {args.chunk}: {len(my_boxes)}")
    print(f"Already completed: {len(completed_boxes)}")

    if not my_boxes:
        print("No boxes assigned to this chunk. Exiting.")
        return

    # 4. Main Scraping Loop
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'Captured_At', 'License'])

        for grid_box in tqdm(my_boxes, desc=f"Processing Chunk {args.chunk}"):
            box_id = f"{grid_box[0]:.4f},{grid_box[1]:.4f},{grid_box[2]:.4f},{grid_box[3]:.4f}"
            
            if box_id in completed_boxes:
                continue

            # Check Uncovered Mask: Skip if it does NOT intersect an uncovered land area
            if uncovered_gdf is not None:
                if not is_in_uncovered_area(grid_box, uncovered_gdf):
                    with open(LOG_FILE, 'a') as log:
                        log.write(box_id + '\n')
                    completed_boxes.add(box_id)
                    continue

            saved_count = 0

            # A. Wikimedia Commons geosearch
            if args.platforms in ["all", "wikimedia"]:
                continue_params = {}
                while saved_count < args.max_images_per_box:
                    res = fetch_wikimedia_batch(grid_box, polygon, args.max_images_per_box, args.delay, continue_params)
                    if res['stat'] == 'ok':
                        for item in res['data']:
                            if saved_count >= args.max_images_per_box:
                                break
                            writer.writerow([
                                item["Photo_ID"],
                                item["Platform"],
                                item["Latitude"],
                                item["Longitude"],
                                item["Image_URL"],
                                item["Captured_At"],
                                item["License"]
                            ])
                            saved_count += 1
                        
                        continue_params = res['continue']
                        if not continue_params:
                            break
                    else:
                        break

            # B. KartaView geosearch
            if args.platforms in ["all", "kartaview"] and saved_count < args.max_images_per_box:
                page = 1
                while saved_count < args.max_images_per_box:
                    res = fetch_kartaview_batch(grid_box, polygon, args.max_images_per_box, args.delay, page)
                    if res['stat'] == 'ok':
                        photos = res['data']
                        if not photos:
                            break
                        for item in photos:
                            if saved_count >= args.max_images_per_box:
                                break
                            writer.writerow([
                                item["Photo_ID"],
                                item["Platform"],
                                item["Latitude"],
                                item["Longitude"],
                                item["Image_URL"],
                                item["Captured_At"],
                                item["License"]
                            ])
                            saved_count += 1
                        page += 1
                    else:
                        break

            # Log completion of this grid box
            with open(LOG_FILE, 'a') as log:
                log.write(box_id + '\n')
            completed_boxes.add(box_id)

    print(f"\nChunk {args.chunk} finished successfully!")


if __name__ == "__main__":
    main()
