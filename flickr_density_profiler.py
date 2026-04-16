import requests
import geopandas as gpd
from shapely.geometry import box

API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'

# 1. Load the Natural Earth Urban Areas shapefile
# (Make sure the path matches where you extracted the downloaded files)
print("Loading Urban Areas map... (This takes a few seconds)")
path_urban_areas = "/user/aaniraj/home/Documents/Projects/data/ne_10m_urban_areas/ne_10m_urban_areas.shp"
urban_gdf = gpd.read_file(path_urban_areas)

# A dictionary of our test boxes (min_lon, min_lat, max_lon, max_lat)
test_boxes = {
    "New York (Midtown)": (-74.015, 40.735, -73.955, 40.780),
    "Central Paris": (2.30, 48.83, 2.37, 48.88),
    "Montpellier": (3.85, 43.59, 3.91, 43.63),
    "Cévennes National Park": (3.55, 44.20, 3.61, 44.24)
}

def is_urban(bbox_coords, urban_dataframe):
    """
    Checks if a bounding box intersects with any urban area.
    bbox_coords: tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    
    # Create a Shapely polygon out of your bounding box coordinates
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
    
    # --- OPTIMIZATION: Spatial Indexing ---
    # Instead of checking every city in the world, the spatial index (sindex) 
    # instantly filters down to only the cities right next to your box.
    possible_matches_index = list(urban_dataframe.sindex.intersection(bbox_polygon.bounds))
    
    # If there are no cities even close by, it's definitely rural
    if len(possible_matches_index) == 0:
        return False
        
    # If there ARE cities nearby, do an exact check to see if the borders touch/overlap
    possible_matches = urban_dataframe.iloc[possible_matches_index]
    exact_matches = possible_matches[possible_matches.intersects(bbox_polygon)]
    
    # If exact_matches is not empty, it means our box overlaps an urban area!
    return not exact_matches.empty

def check_total_photos(box_name, bbox_str):
    url = (
        f"https://www.flickr.com/services/rest/"
        f"?method=flickr.photos.search"
        f"&api_key={API_KEY}"
        f"&bbox={bbox_str}"
        f"&has_geo=1"
        f"&geo_context=2" # Outdoors only
        f"&per_page=1"    # We only need 1 photo to get the metadata
        f"&format=json"
        f"&nojsoncallback=1"
    )
    
    response = requests.get(url)
    data = response.json()
    
    if data.get('stat') == 'ok':
        total = data['photos']['total']
        print(f"{box_name}:")
        print(f"  -> Total outdoor photos available: {total}\n")
    else:
        print(f"Error checking {box_name}: {data.get('message')}")

# Run the test
print("\n--- Urban vs Rural Profiler ---")
for name, coords in test_boxes.items():
    if is_urban(coords, urban_gdf):
        print(f"[{name}] is URBAN -> Skip API, use public dataset.")
    else:
        print(f"[{name}] is RURAL -> Fetch photos from API!")