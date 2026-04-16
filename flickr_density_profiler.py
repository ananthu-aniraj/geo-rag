import requests

API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'

# A dictionary of our test boxes (min_lon, min_lat, max_lon, max_lat)
test_boxes = {
    "New York (Midtown Manhattan)": "-74.015,40.735,-73.955,40.780",
    "Central Paris (Hyper-Dense)": "2.30,48.83,2.37,48.88",
    "Montpellier (Medium City)": "3.85,43.59,3.91,43.63",
    "Cévennes National Park (Rural)": "3.55,44.20,3.61,44.24"
}

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
print("--- Flickr Density Profiler ---\n")
for name, coords in test_boxes.items():
    check_total_photos(name, coords)