import pickle
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import argparse
import os
import h3
import math


def get_clean_boundary(cell):
    """Get hexagon boundary and handle antimeridian crossing."""
    coords = h3.cell_to_boundary(cell)
    lngs = [c[1] for c in coords]
    if max(lngs) - min(lngs) > 180:
        # Shift negative longitudes by 360 to keep the polygon contiguous
        coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
    return coords


def create_map(pkl_path, output_html, max_markers=1000):
    print(f"Loading embedding space from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Total unique images: {len(data)}")

    # Use a subset for the map if there are too many, to keep the HTML responsive
    if len(data) > max_markers:
        print(f"Limiting visualization to first {max_markers} images for performance.")
        plot_data = data[:max_markers]
    else:
        plot_data = data

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
            cluster_text = f"<b>Cluster ID:</b> {item['cluster_id']}<br>"
        else:
            color = 'blue' if item['Platform'] == 'Flickr' else 'green'
            cluster_text = ""

        # Create a popup with the image and metadata
        html = f"""
            <div style="width:200px">
                <img src="{item['Image_URL']}" width="100%">
                <p><b>ID:</b> {item['Photo_ID']}<br>
                <b>Platform:</b> {item['Platform']}<br>
                {cluster_text}
                <b>H3 Cell:</b> {item['H3_Cell']}<br>
                <a href="{item['Image_URL']}" target="_blank">Full Image</a></p>
            </div>
        """
        iframe = folium.IFrame(html=html, width=220, height=280)
        popup = folium.Popup(iframe, max_width=265)

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
