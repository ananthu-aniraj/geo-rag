import requests

# 1. Insert your actual API key here
API_KEY = 'FLICKR_API_KEY_PLACEHOLDER'
SECRET_KEY = '435314db576530e9'
# 2. Define your geographical square (min_lon, min_lat, max_lon, max_lat)
# Example: A bounding box around central Paris
BBOX = '2.31,48.83,2.37,48.89'

# 3. Construct the API request URL
url = (
    f"https://www.flickr.com/services/rest/"
    f"?method=flickr.photos.search"
    f"&api_key={API_KEY}"
    f"&bbox={BBOX}"
    f"&has_geo=1"          # Ensures photos have location data
    f"&geo_context=2"      # 2 = OUTDOORS ONLY
    f"&extras=url_m,geo"   # Requests the medium image URL and geo-coordinates
    f"&format=json"
    f"&nojsoncallback=1"   # Returns clean JSON
)

# 4. Make the request and parse the data
response = requests.get(url)
data = response.json()

# 5. Extract the image links
if data['stat'] == 'ok':
    photos = data['photos']['photo']
    print(f"Found {data['photos']['total']} photos. Here are the first few:")
    
    for photo in photos[:5]: # Let's just look at the first 5
        title = photo.get('title', 'Untitled')
        image_url = photo.get('url_m')
        lat = photo.get('latitude')
        lon = photo.get('longitude')
        
        if image_url:
            print(f"- {title}")
            print(f"  Location: {lat}, {lon}")
            print(f"  URL: {image_url}\n")
else:
    print("Error fetching data:", data.get('message'))