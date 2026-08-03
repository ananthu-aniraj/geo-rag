import argparse
import math
import os
import pickle
import re

import folium
import h3
import pyarrow.parquet as pq
from folium.plugins import MarkerCluster

from src.utils.io import load_dataset_with_clusters

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


def get_clean_boundary(cell):
    """Get hexagon boundary and handle antimeridian crossing."""
    coords = h3.cell_to_boundary(cell)
    lngs = [c[1] for c in coords]
    if max(lngs) - min(lngs) > 180:
        # Shift negative longitudes by 360 to keep the polygon contiguous
        coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
    return coords


def create_map(pkl_path, output_html, max_markers=2000, image_root_dir=None):
    print(f"Loading data from {pkl_path}...")
    if pkl_path.endswith('.pkl'):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
       # Auto-extract k_clusters from filename
        k_clusters = 50000
        match = re.search(r'_k_(\d+)', pkl_path)
        if match:
            k_clusters = int(match.group(1))

        try:
            parquet_file = pq.ParquetFile(pkl_path)
            available_cols = parquet_file.schema_arrow.names

            # Since cluster columns could be in sidecar, we list them all as targets
            target_cols = [
                'Latitude', 'Longitude', 'H3_Cell', 'cluster_id', 'cluster_label',
                'cluster_description', 'parent_cluster_id', 'parent_cluster_label',
                'parent_cluster_description', 'Platform', 'Captured_At', 'Image_URL', 'Photo_ID',
                'Koppen_Code', 'Koppen_Desc', 'Season'
            ]

            # Load sidecar columns if available in index or sidecar
            load_cols = [c for c in target_cols]
            df = load_dataset_with_clusters(pkl_path, k_clusters=k_clusters, columns=load_cols)
        except Exception:
            df = load_dataset_with_clusters(pkl_path, k_clusters=k_clusters)
        data = df.to_dict('records')

    # Load marker popup template
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    popup_template_path = os.path.join(root_dir, "templates", "marker_popup.html")
    if not os.path.exists(popup_template_path):
        popup_template_path = "templates/marker_popup.html"

    with open(popup_template_path, 'r', encoding='utf-8') as f:
        popup_template = f.read()

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
            # Color by parent_cluster_id for cohesive visual grouping if available
            group_id = item.get('parent_cluster_id', item['cluster_id'])
            color = colors[group_id % len(colors)]
            label = item.get('cluster_label', 'Unlabeled')

            parent_lbl = item.get('parent_cluster_label', '')
            parent_text = f"<b>Parent Cluster:</b> {parent_lbl} (ID: {item['parent_cluster_id']})<br>" if parent_lbl else ""

            cluster_text = f"<b>Cluster ID:</b> {item['cluster_id']}<br><b>Labels:</b> {label}<br>{parent_text}"
            if 'cluster_description' in item and item['cluster_description']:
                cluster_text += f"<b>Description:</b> <span style='font-style: italic; font-size: 0.9em; color: #555;'>{item['cluster_description']}</span><br>"
        else:
            color = 'blue' if item['Platform'] == 'Flickr' else 'green'
            cluster_text = ""

        # Create a popup with the image and metadata
        taken_text = f"<b>Captured At:</b> {item['Captured_At']}<br>" if 'Captured_At' in item and item[
            'Captured_At'] else ""
        koppen_text = f"<b>Climate:</b> {item['Koppen_Code']} - {item['Koppen_Desc']}<br>" if 'Koppen_Code' in item and \
                                                                                              item[
                                                                                                  'Koppen_Code'] else ""
        season_text = f"<b>Season:</b> {item['Season']}<br>" if 'Season' in item and item['Season'] else ""

        # Resolve offline paths if root dir is provided
        image_url = item['Image_URL']
        if image_root_dir:
            from src.utils.io import resolve_offline_image_path
            resolved_path = resolve_offline_image_path(
                image_url, image_root_dir, 
                photo_id=item.get('Photo_ID'), 
                platform=item.get('Platform')
            )
            if resolved_path:
                image_url = "file://" + os.path.abspath(resolved_path)

        html = popup_template.format(
            image_url=image_url,
            photo_id=item['Photo_ID'],
            platform=item['Platform'],
            taken_text=taken_text,
            season_text=season_text,
            koppen_text=koppen_text,
            cluster_text=cluster_text,
            h3_cell=item['H3_Cell']
        )
        iframe_height = 360 if ('cluster_description' in item and item['cluster_description']) else 300
        iframe = folium.IFrame(html=html, width=240, height=iframe_height)
        popup = folium.Popup(iframe, max_width=285)

        folium.Marker(
            location=[item['Latitude'], item['Longitude']],
            popup=popup,
            icon=folium.Icon(color=color, icon='camera')
        ).add_to(marker_cluster)

    # Inject global JS helper template from templates/image_error_handler.js
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    js_template_path = os.path.join(root_dir, "templates", "image_error_handler.js")
    if not os.path.exists(js_template_path):
        js_template_path = "templates/image_error_handler.js"

    with open(js_template_path, 'r', encoding='utf-8') as f:
        js_code = f.read().replace("{{MAPILLARY_TOKEN}}", MAPILLARY_TOKEN)

    js_header = f"<script>\n{js_code}\n</script>"
    m.get_root().html.add_child(folium.Element(js_header))

    m.save(output_html)
    print(f"Interactive map saved to: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize kept images and cluster centroids on a map.")
    parser.add_argument("--pkl_file", type=str, required=True, help="Path to the .pkl file.")
    parser.add_argument("--output", type=str, default="cluster_map.html", help="Output HTML file name.")
    parser.add_argument("--max_markers", type=int, default=2000, help="Max markers to show.")
    parser.add_argument("--image_root_dir", type=str, nargs="+", default=None, 
                        help="Optional local offline image directories.")
    args = parser.parse_args()

    create_map(args.pkl_file, args.output, args.max_markers, args.image_root_dir)
