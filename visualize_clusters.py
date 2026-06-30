import pickle
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import argparse
import os
import h3
import math

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


def get_clean_boundary(cell):
    """Get hexagon boundary and handle antimeridian crossing."""
    coords = h3.cell_to_boundary(cell)
    lngs = [c[1] for c in coords]
    if max(lngs) - min(lngs) > 180:
        # Shift negative longitudes by 360 to keep the polygon contiguous
        coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
    return coords


def create_map(pkl_path, output_html, max_markers=1000):
    print(f"Loading clustered data from {pkl_path}...")
    if pkl_path.endswith('.pkl'):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # Assume Parquet
        df = pd.read_parquet(pkl_path)
        data = df.to_dict('records')

    print(f"Total unique images: {len(data)}")

    # Use a subset for the map if there are too many, to keep the HTML responsive
    if len(data) > max_markers:
        print(f"Limiting visualization to first {max_markers} images for performance.")
        plot_data = data[:max_markers]
    else:
        plot_data = data

    # Batch resolve expired Mapillary or Kartaview URLs dynamically
    import concurrent.futures
    import requests

    print(f"Resolving {len(plot_data)} image URLs in parallel...")

    def resolve_item_url(item, timeout=10):
        url = item.get('Image_URL')
        photo_id = item.get('Photo_ID')
        platform = item.get('Platform')
        
        if not url or not photo_id or not platform:
            return
            
        platform_lower = str(platform).strip().lower()
        photo_str = str(photo_id).strip()
        if photo_str.endswith('.0'):
            photo_str = photo_str[:-2]
            
        is_mapillary = platform_lower == 'mapillary' or 'mapillary' in url or 'fbcdn.net' in url
        is_kartaview = platform_lower == 'kartaview' or 'kartaview' in url or 'openstreetcam' in url
        
        if not (is_mapillary or is_kartaview):
            return

        try:
            if is_mapillary:
                api_url = f"https://graph.mapillary.com/{photo_str}?fields=thumb_1024_url"
                headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
                res = requests.get(api_url, headers=headers, timeout=timeout)
                if res.status_code == 200:
                    fresh_url = res.json().get("thumb_1024_url")
                    if fresh_url:
                        item["Image_URL"] = fresh_url
            elif is_kartaview:
                api_url = f"https://api.openstreetcam.org/2.0/photo/{photo_str}"
                res = requests.get(api_url, timeout=timeout)
                if res.status_code == 200:
                    data = res.json().get("result", {}).get("data", {})
                    fresh_url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")
                    if fresh_url:
                        item["Image_URL"] = fresh_url
        except Exception:
            pass

    max_workers = min(32, (len(plot_data) + 4) // 5 or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(resolve_item_url, plot_data))
    print("URL resolution complete.")

    # Calculate center of the map (robustly handling the antimeridian)
    avg_lat = sum(item['Latitude'] for item in plot_data) / len(plot_data)
    
    # Use vector averaging for longitude to handle wrap-around
    x = sum(math.cos(math.radians(item['Longitude'])) for item in plot_data) / len(plot_data)
    y = sum(math.sin(math.radians(item['Longitude'])) for item in plot_data) / len(plot_data)
    avg_lon = math.degrees(math.atan2(y, x))

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12, tiles='CartoDB Positron')
    
    # Prepare H3 Cell Polygons
    unique_cells = set(item['H3_Cell'] for item in plot_data if 'H3_Cell' in item)
    features = []
    for cell in unique_cells:
        try:
            boundary = get_clean_boundary(cell)
            # GeoJSON expects [lng, lat] and a closed loop
            geojson_coords = [[lng, lat] for lat, lng in boundary]
            geojson_coords.append(geojson_coords[0])
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geojson_coords]
                },
                "properties": {"cell": cell}
            })
        except Exception:
            continue
            
    if features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            style_function=lambda x: {
                "fillColor": "gray",
                "color": "gray",
                "weight": 1,
                "fillOpacity": 0.1
            },
            tooltip=folium.GeoJsonTooltip(fields=["cell"], aliases=["H3 Cell:"])
        ).add_to(m)

    marker_cluster = MarkerCluster().add_to(m)

    # Define available folium colors
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black'
    ]

    for item in plot_data:
        # Determine color based on cluster_id or platform
        if 'cluster_id' in item:
            color = colors[item['cluster_id'] % len(colors)]
            label = item.get('cluster_label', 'Unlabeled')
            cluster_text = f"<b>Cluster ID:</b> {item['cluster_id']}<br><b>Labels:</b> {label}<br>"
            if 'cluster_description' in item and item['cluster_description']:
                cluster_text += f"<b>Description:</b> <span style='font-style: italic; font-size: 0.9em; color: #555;'>{item['cluster_description']}</span><br>"
        else:
            color = 'blue' if item['Platform'] == 'Flickr' else 'green'
            cluster_text = ""

        # Create a popup with the image and metadata
        taken_text = f"<b>Captured At:</b> {item['Captured_At']}<br>" if 'Captured_At' in item and item['Captured_At'] else ""
        html = f"""
            <div style="width:220px">
                <img src="{item['Image_URL']}" width="100%" style="border-radius: 4px;">
                <p style="font-size: 11px; margin-top: 5px; line-height: 1.4; font-family: sans-serif;">
                <b>ID:</b> {item['Photo_ID']}<br>
                <b>Platform:</b> {item['Platform']}<br>
                {taken_text}
                {cluster_text}
                <b>H3 Cell:</b> {item['H3_Cell']}<br>
                <a href="{item['Image_URL']}" target="_blank" style="color: #1a73e8; text-decoration: none; font-weight: bold;">Full Image</a></p>
            </div>
        """
        iframe_height = 350 if ('cluster_description' in item and item['cluster_description']) else 280
        iframe = folium.IFrame(html=html, width=240, height=iframe_height)
        popup = folium.Popup(iframe, max_width=285)

        folium.Marker(
            location=[item['Latitude'], item['Longitude']],
            popup=popup,
            icon=folium.Icon(color=color, icon='camera')
        ).add_to(marker_cluster)

    m.save(output_html)
    print(f"Interactive map saved to: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize kept images and cluster centroids on a map.")
    parser.add_argument("--pkl_file", type=str, required=True, help="Path to the .pkl file.")
    parser.add_argument("--output", type=str, default="cluster_map.html", help="Output HTML file name.")
    parser.add_argument("--max_markers", type=int, default=2000, help="Max markers to show.")
    args = parser.parse_args()

    create_map(args.pkl_file, args.output, args.max_markers)
