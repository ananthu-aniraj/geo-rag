import os
import pandas as pd
import h3
import folium
import argparse
import numpy as np
from tqdm import tqdm
import branca.colormap as cm
import json

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
    df = pd.read_parquet(args.index)
    
    # Filter for the target resolution
    df_res = df[df['resolution'] == args.res]
    
    if df_res.empty:
        print(f"Error: No records found for resolution {args.res} in the index.")
        return

    # Calculate total image counts per H3 cell at this resolution
    cell_totals = df_res.groupby('query_cell')['image_count'].sum().reset_index()
    
    # Filter by minimum image count
    cell_totals = cell_totals[cell_totals['image_count'] >= args.min_count]
    
    if cell_totals.empty:
        print(f"No H3 cells found with at least {args.min_count} images at resolution {args.res}.")
        return

    print(f"Preparing map features for {len(cell_totals)} cells...")
    
    # Initialize Folium Map centered globally
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB Positron")

    # Set up colormap based on log density of images
    counts = cell_totals['image_count'].values
    min_log = np.log10(min(counts))
    max_log = np.log10(max(counts))
    if min_log == max_log:
        max_log += 0.1
        
    colormap = cm.linear.YlOrRd_09.scale(min_log, max_log)
    colormap.caption = f"Image Density (Log10 scale, H3 Res {args.res})"

    def get_clean_boundary(cell):
        """Get cell boundary coordinates, adjusting for antimeridian crossing."""
        coords = h3.cell_to_boundary(cell)
        lngs = [c[1] for c in coords]
        if max(lngs) - min(lngs) > 180:
            coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
        return coords

    # Create GeoJSON Features List
    features = []
    
    for _, item in tqdm(cell_totals.iterrows(), total=len(cell_totals), desc="Processing Hexagons"):
        cell = item['query_cell']
        total_images = item['image_count']
        
        try:
            boundary = get_clean_boundary(cell)
            # GeoJSON polygon expects [lng, lat] and a closed loop
            geojson_coords = [[lng, lat] for lat, lng in boundary]
            geojson_coords.append(geojson_coords[0])
            
            # Retrieve cluster breakdown for this cell, sorted by image count
            df_cell = df_res[df_res['query_cell'] == cell].sort_values(by='image_count', ascending=False)
            
            # Construct a lightweight JSON string of cluster data to prevent HTML duplication bloat
            cluster_list = []
            for _, row in df_cell.head(5).iterrows():
                pct = float((row['image_count'] / total_images) * 100)
                # Keep only first 200 chars of description to reduce JSON size
                desc = row['cluster_description'] if row['cluster_description'] else ""
                cluster_list.append([row['cluster_label'], pct, int(row['image_count']), desc[:200]])
            
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
                    "clusters": clusters_json  # Compact serialized data
                }
            }
            features.append(feature)
            
        except Exception as e:
            continue

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

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

    # Javascript dynamic HTML tooltip builder (corrected quoting issues)
    js_tooltip_builder = f"""
    function(feature, layer) {{
        var props = feature.properties;
        var countFormatted = Number(props.count).toLocaleString();
        
        var html = '<div style="font-family: Arial, sans-serif; min-width: 320px; max-width: 400px; font-size: 12px; padding: 6px;">' +
                   '<b style="font-size: 14px; color: #2c3e50; display: block; margin-bottom: 5px;">📍 H3 Cell: ' + props.cell + ' (Res {args.res})</b>' +
                   '<b>📊 Total Images:</b> ' + countFormatted + '<br/>' +
                   '<hr style="margin: 6px 0; border: 0; border-top: 1px solid #ddd;"/>' +
                   '<b style="color: #16a085; text-transform: uppercase; font-size: 10px; display: block; margin-bottom: 4px;">Dominant Land Use / Cover:</b>' +
                   '<ul style="margin: 0; padding-left: 14px; list-style-type: square; color: #34495e;">';
        
        try {{
            var clusters = JSON.parse(props.clusters);
            for (var i = 0; i < clusters.length; i++) {{
                var pct = Number(clusters[i][1]).toFixed(1);
                var cCount = Number(clusters[i][2]).toLocaleString();
                var label = clusters[i][0];
                var desc = clusters[i][3];
                
                // Truncate description if too long
                if (desc.length > 150) {{
                    desc = desc.substring(0, 147) + '...';
                }}
                
                html += '<li style="margin-bottom: 6px;">' +
                        '<b>' + label + ':</b> ' + pct + '% (' + cCount + ' images)' +
                        '<br/><span style="color: #7f8c8d; font-size: 10.5px; font-style: italic;">' + desc + '</span>' +
                        '</li>';
            }}
        }} catch(e) {{
            html += '<li>Error parsing cluster details</li>';
        }}
        
        html += '</ul></div>';
        layer.bindTooltip(html, {{
            sticky: true,
            direction: "auto",
            opacity: 0.95
        }});
    }}
    """

    print("Adding GeoJSON layer to map with dynamic JavaScript tooltips...")
    folium.GeoJson(
        geojson_data,
        style_function=style_fn,
        on_each_feature=js_tooltip_builder
    ).add_to(m)

    m.add_child(colormap)
    
    # Save the output HTML file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"\nSuccessfully generated interactive spatial-semantic map at: {args.output}")

if __name__ == "__main__":
    main()
