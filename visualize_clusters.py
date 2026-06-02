import pickle
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import argparse
import os

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

    # Calculate center of the map
    avg_lat = sum(item['Latitude'] for item in plot_data) / len(plot_data)
    avg_lon = sum(item['Longitude'] for item in plot_data) / len(plot_data)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12, tiles='OpenStreetMap')
    marker_cluster = MarkerCluster().add_to(m)

    for item in plot_data:
        # Create a popup with the image and metadata
        html = f"""
            <div style="width:200px">
                <img src="{item['Image_URL']}" width="100%">
                <p><b>ID:</b> {item['Photo_ID']}<br>
                <b>Platform:</b> {item['Platform']}<br>
                <b>H3 Cell:</b> {item['H3_Cell']}<br>
                <a href="{item['Image_URL']}" target="_blank">Full Image</a></p>
            </div>
        """
        iframe = folium.IFrame(html=html, width=220, height=280)
        popup = folium.Popup(iframe, max_width=265)
        
        color = 'blue' if item['Platform'] == 'Flickr' else 'green'
        
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
