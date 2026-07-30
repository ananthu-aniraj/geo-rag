import argparse
import json
import os

import branca.colormap as cm
import folium
import h3
import numpy as np
from tqdm import tqdm

from src.utils.io import load_dataframe


def main():
    parser = argparse.ArgumentParser(description="Generate an interactive H3 spatial-semantic world map.")
    parser.add_argument("--index", type=str, default="full_pipeline_output/geo_space_h3_semantic_index.parquet",
                        help="Path to the pre-aggregated H3 semantic index Parquet file.")
    parser.add_argument("--output", type=str, default="full_pipeline_output/global_h3_semantic_map.html",
                        help="Path to save the output HTML map.")
    parser.add_argument("--res", type=int, default=6, choices=list(range(5, 12)),
                        help="H3 resolution to display on the map (default: 6, ~36 sq km).")
    parser.add_argument("--min_count", type=int, default=10,
                        help="Minimum number of images in a cell to display on the map (default: 10).")
    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"Error: Aggregated index file not found at '{args.index}'.")
        print("Please build the index first using build_spatial_semantic_index.py.")
        return

    print(f"Loading spatial-semantic index from {args.index}...")
    df = load_dataframe(args.index)

    # Filter for the target resolution
    df_res = df[df['resolution'] == args.res]

    if df_res.empty:
        print(f"Error: No records found for resolution {args.res} in the index.")
        return

    # Calculate global cell totals to set a stable density colormap scale
    global_cell_totals = df_res.groupby('query_cell')['image_count'].sum().reset_index()
    global_cell_totals = global_cell_totals[global_cell_totals['image_count'] >= args.min_count]

    if global_cell_totals.empty:
        print(f"No H3 cells found with at least {args.min_count} images at resolution {args.res}.")
        return

    # Set up global colormap based on log density of images
    counts = global_cell_totals['image_count'].values
    min_log = np.log10(min(counts))
    max_log = np.log10(max(counts))
    if min_log == max_log:
        max_log += 0.1

    colormap = cm.linear.YlOrRd_09.scale(min_log, max_log)
    colormap.caption = f"Image Density (Log10 scale, H3 Res {args.res})"

    # Initialize Folium Map centered globally
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB Positron")

    def get_clean_boundary(cell):
        """Get cell boundary coordinates, adjusting for antimeridian crossing."""
        coords = h3.cell_to_boundary(cell)
        lngs = [c[1] for c in coords]
        if max(lngs) - min(lngs) > 180:
            coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
        return coords

    # Style function mapping density to color
    def style_fn(feature):
        log_val = feature["properties"]["log_count"]
        fill_color = colormap(log_val)
        return {
            "fillColor": fill_color,
            "color": fill_color,
            "weight": 1,
            "fillOpacity": 0.6,
        }

    # Discover unique seasons in the index (backward compatible if Season is not present)
    unique_seasons = sorted(df_res['Season'].unique()) if 'Season' in df_res.columns else ['Unknown']

    # Load JS tooltip builder template from templates/h3_semantic_tooltip.js
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    js_template_path = os.path.join(root_dir, "templates", "h3_semantic_tooltip.js")
    if not os.path.exists(js_template_path):
        js_template_path = "templates/h3_semantic_tooltip.js"

    with open(js_template_path, 'r', encoding='utf-8') as f:
        js_code_str = f.read().replace("{{RES}}", str(args.res))
    js_tooltip_builder = folium.utilities.JsCode(js_code_str)

    for season in unique_seasons:
        df_season = df_res[df_res['Season'] == season] if 'Season' in df_res.columns else df_res
        
        # Calculate cell totals for this season
        cell_totals = df_season.groupby('query_cell')['image_count'].sum().reset_index()
        cell_totals = cell_totals[cell_totals['image_count'] >= args.min_count]
        
        if cell_totals.empty:
            continue

        print(f"Preparing map layer for Season: {season} ({len(cell_totals)} cells)...")

        # Pre-aggregate top 5 clusters per cell for this season in one vectorized operation
        agg_cols = ['query_cell', 'cluster_label', 'cluster_description']
        has_parent_col = 'parent_cluster_label' in df_season.columns
        if has_parent_col:
            agg_cols.append('parent_cluster_label')
        
        df_cell_agg = df_season.groupby(agg_cols, observed=True)['image_count'].sum().reset_index()
        df_cell_agg = df_cell_agg.sort_values(by=['query_cell', 'image_count'], ascending=[True, False])
        df_top5 = df_cell_agg.groupby('query_cell').head(5)

        # Build fast O(1) lookup dictionary for cell clusters
        cell_clusters_dict = {}
        for cell, group in df_top5.groupby('query_cell'):
            cell_clusters_dict[cell] = group.to_dict('records')

        features = []

        for _, item in tqdm(cell_totals.iterrows(), total=len(cell_totals), desc=f"Hexagons ({season})"):
            cell = item['query_cell']
            total_images = item['image_count']

            try:
                boundary = get_clean_boundary(cell)
                geojson_coords = [[lng, lat] for lat, lng in boundary]
                geojson_coords.append(geojson_coords[0])

                # O(1) dictionary lookup instead of linear filtering
                top_clusters = cell_clusters_dict.get(cell, [])
                cluster_list = []
                for row in top_clusters:
                    pct = float((row['image_count'] / total_images) * 100)
                    desc = row['cluster_description'] if row['cluster_description'] else ""
                    parent_label = row.get('parent_cluster_label', '') if has_parent_col else ''
                    cluster_list.append([row['cluster_label'], pct, int(row['image_count']), desc[:200], parent_label])

                clusters_json = json.dumps(cluster_list)
                log_val = np.log10(total_images)

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [geojson_coords]
                    },
                    "properties": {
                        "cell": cell,
                        "count": int(total_images),
                        "log_count": float(log_val),
                        "clusters": clusters_json
                    }
                }
                features.append(feature)

            except Exception:
                continue

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        # Show first season (or Summer) by default, others hidden to avoid collision
        is_default = (season == 'Summer') or (len(unique_seasons) == 1) or (season == unique_seasons[0] and 'Summer' not in unique_seasons)
        layer_name = f"🍂 Season: {season}" if season != 'Unknown' else "📍 All Observations"
        
        fg = folium.FeatureGroup(name=layer_name, show=is_default)
        
        folium.GeoJson(
            geojson_data,
            style_function=style_fn,
            on_each_feature=js_tooltip_builder
        ).add_to(fg)
        
        fg.add_to(m)

    m.add_child(colormap)
    folium.LayerControl(collapsed=False).add_to(m)

    # Save the output HTML file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"\nSuccessfully generated interactive spatial-semantic map at: {args.output}")


if __name__ == "__main__":
    main()
