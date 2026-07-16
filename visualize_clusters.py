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
        import pyarrow.parquet as pq
        try:
            meta = pq.read_metadata(pkl_path)
            available_cols = meta.schema.names
            target_cols = [
                'Latitude', 'Longitude', 'H3_Cell', 'cluster_id', 'cluster_label', 
                'cluster_description', 'parent_cluster_id', 'parent_cluster_label', 
                'parent_cluster_description', 'Platform', 'Captured_At', 'Image_URL', 'Photo_ID'
            ]
            load_cols = [c for c in target_cols if c in available_cols]
            df = pd.read_parquet(pkl_path, columns=load_cols)
        except Exception:
            df = pd.read_parquet(pkl_path)
        data = df.to_dict('records')

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
        taken_text = f"<b>Captured At:</b> {item['Captured_At']}<br>" if 'Captured_At' in item and item['Captured_At'] else ""
        html = f"""
            <div style="width:220px">
                <img src="{item['Image_URL']}" onerror="handleImageError(this, '{item['Photo_ID']}', '{item['Platform']}')" width="100%" style="border-radius: 4px;">
                <p style="font-size: 11px; margin-top: 5px; line-height: 1.4; font-family: sans-serif;">
                <b>ID:</b> {item['Photo_ID']}<br>
                <b>Platform:</b> {item['Platform']}<br>
                {taken_text}
                {cluster_text}
                <b>H3 Cell:</b> {item['H3_Cell']}<br>
                <a href="{item['Image_URL']}" target="_blank" style="color: #1a73e8; text-decoration: none; font-weight: bold;">Full Image</a></p>
            </div>
            <script>
                const MAPILLARY_TOKEN = '{MAPILLARY_TOKEN}';
                function handleImageError(img, photoId, platform) {{
                    if (img.dataset.retryAttempt) return;
                    img.dataset.retryAttempt = '1';
                    
                    const platformLower = String(platform).toLowerCase().trim();
                    const isMapillary = platformLower === 'mapillary' || img.src.includes('mapillary') || img.src.includes('fbcdn.net');
                    const isKartaview = platformLower === 'kartaview' || img.src.includes('kartaview') || img.src.includes('openstreetcam');
                    
                    let cleanPhotoId = String(photoId).trim();
                    if (cleanPhotoId.endsWith('.0')) {{
                        cleanPhotoId = cleanPhotoId.slice(0, -2);
                    }}
                    
                    if (isMapillary && cleanPhotoId && cleanPhotoId !== 'null' && cleanPhotoId !== 'undefined' && cleanPhotoId !== 'NaN') {{
                        const apiUrl = 'https://graph.mapillary.com/' + cleanPhotoId + '?fields=thumb_1024_url';
                        fetch(apiUrl, {{
                            headers: {{ 'Authorization': 'OAuth ' + MAPILLARY_TOKEN }}
                        }})
                        .then(res => res.json())
                        .then(resData => {{
                            if (resData.thumb_1024_url) {{
                                img.src = resData.thumb_1024_url;
                                const link = img.closest('div').querySelector('a');
                                if (link) link.href = resData.thumb_1024_url;
                            }}
                        }})
                        .catch(err => console.error('Error fetching Mapillary fresh URL:', err));
                    }} else if (isKartaview && cleanPhotoId && cleanPhotoId !== 'null' && cleanPhotoId !== 'undefined' && cleanPhotoId !== 'NaN') {{
                        const apiUrl = 'https://api.openstreetcam.org/2.0/photo/' + cleanPhotoId;
                        fetch(apiUrl)
                        .then(res => res.json())
                        .then(resData => {{
                            const data = resData.result && resData.result.data;
                            if (data) {{
                                const freshUrl = data.fileurlLTh || data.fileurlTh || data.fileurl;
                                if (freshUrl) {{
                                    img.src = freshUrl;
                                    const link = img.closest('div').querySelector('a');
                                    if (link) link.href = freshUrl;
                                }}
                            }}
                        }})
                        .catch(err => console.error('Error fetching Kartaview fresh URL:', err));
                    }}
                }}
            </script>
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
