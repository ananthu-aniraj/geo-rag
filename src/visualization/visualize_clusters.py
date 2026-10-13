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

try:
    from src.visualization.map_utils import (
        create_carto_tile_layer,
        create_folium_map,
    )
except ImportError:
    from map_utils import (
        create_carto_tile_layer,
        create_folium_map,
    )

try:
    from src.utils.credentials import get_mapillary_token
except ImportError:
    from utils.credentials import get_mapillary_token

MAPILLARY_TOKEN = get_mapillary_token()


def get_clean_boundary(cell):
    """Get hexagon boundary and handle antimeridian crossing."""
    coords = h3.cell_to_boundary(cell)
    lngs = [c[1] for c in coords]
    if max(lngs) - min(lngs) > 180:
        # Shift negative longitudes by 360 to keep the polygon contiguous
        coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
    return coords


def create_map(
    pkl_path,
    output_html,
    max_markers=2000,
    image_root_dir=None,
    find_prefix=None,
    replace_prefix=None,
    tiles="CartoDB Dark_Matter",
):
    print(f"Loading data from {pkl_path}...")
    if pkl_path.endswith(".pkl"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        total_unique = len(data)
        if max_markers and len(data) > max_markers:
            plot_data = data[:max_markers]
        else:
            plot_data = data
    else:
        # Auto-extract k_clusters from filename
        k_clusters = 50000
        match = re.search(r"_k_(\d+)", pkl_path)
        if match:
            k_clusters = int(match.group(1))

        try:
            parquet_file = pq.ParquetFile(pkl_path)
            available_cols = parquet_file.schema_arrow.names

            # Since cluster columns could be in sidecar, we list them all as targets
            target_cols = [
                "Latitude",
                "Longitude",
                "H3_Cell",
                "cluster_id",
                "cluster_label",
                "cluster_description",
                "parent_cluster_id",
                "parent_cluster_label",
                "parent_cluster_description",
                "Platform",
                "Captured_At",
                "Image_URL",
                "Photo_ID",
                "Koppen_Code",
                "Koppen_Desc",
                "Season",
            ]

            # Load sidecar columns if available in index or sidecar
            load_cols = [c for c in target_cols]
            df = load_dataset_with_clusters(
                pkl_path, k_clusters=k_clusters, columns=load_cols
            )
        except Exception:
            df = load_dataset_with_clusters(pkl_path, k_clusters=k_clusters)

        total_unique = len(df)
        if max_markers and len(df) > max_markers:
            plot_data = df.iloc[:max_markers].to_dict("records")
        else:
            plot_data = df.to_dict("records")

    # Load marker popup template
    popup_template_path = os.path.join(
        os.path.dirname(__file__), "templates", "marker_popup.html"
    )
    with open(popup_template_path, "r", encoding="utf-8") as f:
        popup_template = f.read()

    print(f"Total unique images: {total_unique}")

    # Use a subset for the map if there are too many, to keep the HTML responsive
    if total_unique > max_markers:
        print(f"Limiting visualization to first {max_markers} images for performance.")

    # Calculate center of the map (robustly handling the antimeridian)
    avg_lat = sum(item["Latitude"] for item in plot_data) / len(plot_data)

    # Use vector averaging for longitude to handle wrap-around
    x = sum(math.cos(math.radians(item["Longitude"])) for item in plot_data) / len(
        plot_data
    )
    y = sum(math.sin(math.radians(item["Longitude"])) for item in plot_data) / len(
        plot_data
    )
    avg_lon = math.degrees(math.atan2(y, x))

    m = create_folium_map(location=[avg_lat, avg_lon], zoom_start=12, tiles=tiles)

    # Add alternative basemaps for user flexibility and contrast
    if tiles == "CartoDB Dark_Matter":
        create_carto_tile_layer("light_all", name="CartoDB Positron (Light)").add_to(m)
    else:
        create_carto_tile_layer("dark_all", name="CartoDB Dark_Matter (Dark)").add_to(m)

    create_carto_tile_layer(
        "rastertiles/voyager", name="CartoDB Voyager (Color)"
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # Prepare H3 Cell Polygons
    unique_cells = set(item["H3_Cell"] for item in plot_data if "H3_Cell" in item)
    features = []
    for cell in unique_cells:
        try:
            boundary = get_clean_boundary(cell)
            # GeoJSON expects [lng, lat] and a closed loop
            geojson_coords = [[lng, lat] for lat, lng in boundary]
            geojson_coords.append(geojson_coords[0])

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [geojson_coords]},
                    "properties": {"cell": cell},
                }
            )
        except Exception:
            continue

    if features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            name="H3 Clusters",
            style_function=lambda x: {
                "fillColor": "#38bdf8",
                "color": "#38bdf8",
                "weight": 1.5,
                "fillOpacity": 0.08,
            },
            tooltip=folium.GeoJsonTooltip(fields=["cell"], aliases=["H3 Cell:"]),
        ).add_to(m)

    marker_cluster = MarkerCluster(name="Cluster Markers").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    # Define available folium colors
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightred",
        "beige",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
        "pink",
        "lightblue",
        "lightgreen",
        "gray",
        "black",
    ]

    auto_detected_base = find_prefix or ""

    for item in plot_data:
        # Determine color based on cluster_id or platform
        if "cluster_id" in item:
            # Color by parent_cluster_id for cohesive visual grouping if available
            group_id = item.get("parent_cluster_id", item["cluster_id"])
            color = colors[group_id % len(colors)]
            label = item.get("cluster_label", "Unlabeled")

            parent_lbl = item.get("parent_cluster_label", "")
            parent_text = (
                f"<span style='color: #94a3b8;'>Parent Cluster:</span> <span style='color: #f8fafc;'>{parent_lbl}</span> (ID: <span style='color: #a78bfa;'>{item['parent_cluster_id']}</span>)<br>"
                if parent_lbl
                else ""
            )

            cluster_text = f"<span style='color: #94a3b8;'>Cluster ID:</span> <span style='color: #a78bfa; font-weight: 600;'>{item['cluster_id']}</span><br><span style='color: #94a3b8;'>Labels:</span> <span style='color: #f8fafc; font-weight: 600;'>{label}</span><br>{parent_text}"
            if "cluster_description" in item and item["cluster_description"]:
                cluster_text += f"<span style='color: #94a3b8;'>Description:</span> <span style='font-style: italic; font-size: 0.9em; color: #cbd5e1;'>{item['cluster_description']}</span><br>"
        else:
            color = "blue" if item.get("Platform") == "Flickr" else "green"
            cluster_text = ""

        # Create a popup with the image and metadata
        taken_text = (
            f"<span style='color: #94a3b8;'>Captured At:</span> <span style='color: #f8fafc;'>{item['Captured_At']}</span><br>"
            if "Captured_At" in item and item["Captured_At"]
            else ""
        )
        koppen_text = (
            f"<span style='color: #94a3b8;'>Climate:</span> <span style='color: #f8fafc;'>{item['Koppen_Code']} - {item['Koppen_Desc']}</span><br>"
            if "Koppen_Code" in item and item["Koppen_Code"]
            else ""
        )
        season_text = (
            f"<span style='color: #94a3b8;'>Season:</span> <span style='color: #f8fafc;'>{item['Season']}</span><br>"
            if "Season" in item and item["Season"]
            else ""
        )

        # Resolve offline paths if root dir is provided
        image_url = item["Image_URL"]
        if image_root_dir:
            from src.utils.io import resolve_offline_image_path

            resolved_path = resolve_offline_image_path(
                image_url,
                image_root_dir,
                photo_id=item.get("Photo_ID"),
                platform=item.get("Platform"),
            )
            if resolved_path:
                image_url = "file://" + os.path.abspath(resolved_path)

        if not auto_detected_base and isinstance(image_url, str):
            if image_url.startswith("file://"):
                last_slash = image_url.rfind("/")
                if last_slash != -1:
                    auto_detected_base = image_url[: last_slash + 1]

        html = popup_template.format(
            image_url=image_url,
            photo_id=item.get("Photo_ID", ""),
            platform=item.get("Platform", ""),
            taken_text=taken_text,
            season_text=season_text,
            koppen_text=koppen_text,
            cluster_text=cluster_text,
            h3_cell=item.get("H3_Cell", ""),
        )
        safe_html = html.replace("`", "&#96;").replace("${", "&#36;{")
        popup = folium.Popup(folium.Html(safe_html, script=True), max_width=285)

        marker_tooltip = (
            f"<b>Cluster {item.get('cluster_id', '')}</b>: {item.get('cluster_label', 'Photo')}"
            if "cluster_id" in item
            else f"<b>{item.get('Platform', 'Photo')}</b>: {item.get('Photo_ID', '')}"
        )
        folium.Marker(
            location=[item["Latitude"], item["Longitude"]],
            popup=popup,
            tooltip=folium.Tooltip(marker_tooltip, sticky=True),
            icon=folium.Icon(color=color, icon="camera"),
        ).add_to(marker_cluster)

    if not auto_detected_base and image_root_dir:
        first_dir = (
            image_root_dir[0] if isinstance(image_root_dir, list) else image_root_dir
        )
        auto_detected_base = "file://" + os.path.abspath(first_dir).rstrip("/") + "/"
    if not auto_detected_base:
        for item in plot_data:
            u = item.get("Image_URL", "")
            if isinstance(u, str) and u.startswith("/"):
                last_slash = u.rfind("/")
                if last_slash != -1:
                    auto_detected_base = u[: last_slash + 1]
                    break

    # Inject global JS helper template from templates/image_error_handler.js
    js_template_path = os.path.join(
        os.path.dirname(__file__), "templates", "image_error_handler.js"
    )
    with open(js_template_path, "r", encoding="utf-8") as f:
        js_code = (
            f.read()
            .replace("__MAPILLARY_TOKEN__", MAPILLARY_TOKEN)
            .replace("{{MAPILLARY_TOKEN}}", MAPILLARY_TOKEN)
            .replace("__MAP_NAME__", m.get_name())
            .replace("{{MAP_NAME}}", m.get_name())
            .replace("__AUTO_DETECTED_BASE__", auto_detected_base)
            .replace("{{AUTO_DETECTED_BASE}}", auto_detected_base)
            .replace("__DEFAULT_REPLACE_PREFIX__", replace_prefix or "")
            .replace("{{DEFAULT_REPLACE_PREFIX}}", replace_prefix or "")
        )

    js_header = f"<script>\n{js_code}\n</script>"
    m.get_root().html.add_child(folium.Element(js_header))

    m.save(output_html)
    print(f"Interactive map saved to: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize kept images and cluster centroids on a map."
    )
    parser.add_argument(
        "--pkl_file", type=str, required=True, help="Path to the .pkl file."
    )
    parser.add_argument(
        "--output", type=str, default="cluster_map.html", help="Output HTML file name."
    )
    parser.add_argument(
        "--max_markers", type=int, default=2000, help="Max markers to show."
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        nargs="+",
        default=None,
        help="Optional local offline image directories.",
    )
    parser.add_argument(
        "--find_prefix",
        type=str,
        default=None,
        help="Optional local path prefix to replace (auto-detected if omitted).",
    )
    parser.add_argument(
        "--replace_prefix",
        type=str,
        default=None,
        help="Optional replacement URL prefix (e.g. http://localhost:8000/).",
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default="CartoDB Dark_Matter",
        help="Default basemap tile style (default: 'CartoDB Dark_Matter').",
    )
    args = parser.parse_args()

    create_map(
        args.pkl_file,
        args.output,
        args.max_markers,
        args.image_root_dir,
        find_prefix=args.find_prefix,
        replace_prefix=args.replace_prefix,
        tiles=args.tiles,
    )
