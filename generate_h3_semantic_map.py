import os
import pandas as pd
import h3
import folium
import argparse
import numpy as np
from tqdm import tqdm
import branca.colormap as cm

def main():
    parser = argparse.ArgumentParser(description="Generate an interactive H3 spatial-semantic world map.")
    parser.add_argument("--index", type=str, default="full_pipeline_output/geo_space_h3_semantic_index.parquet",
                        help="Path to the pre-aggregated H3 semantic index Parquet file.")
    parser.add_argument("--output", type=str, default="full_pipeline_output/global_h3_semantic_map.html",
                        help="Path to save the output HTML map.")
    parser.add_argument("--res", type=int, default=6, choices=list(range(5, 12)),
                        help="H3 resolution to display on the map (default: 6, ~36 sq km).")
    parser.add_argument("--min_count", type=int, default=1,
                        help="Minimum number of images in a cell to display on the map.")
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
            
            # Construct rich HTML tooltip
            tooltip_html = f"""
            <div style="font-family: 'Helvetica Neue', Arial, sans-serif; min-width: 280px; font-size: 12px; padding: 4px;">
                <b style="font-size: 13px; color: #2c3e50; display: block; margin-bottom: 5px;">📍 H3 Cell: {cell} (Res {args.res})</b>
                <b>📊 Total Images:</b> {total_images:,}<br/>
                <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ddd;"/>
                <b style="color: #16a085; text-transform: uppercase; font-size: 10px; display: block; margin-bottom: 4px;">Dominant Land Use / Cover:</b>
                <ul style="margin: 0; padding-left: 14px; list-style-type: square; color: #34495e;">
            """
            
            # Show top 5 clusters in the cell tooltip
            for _, row in df_cell.head(5).iterrows():
                pct = (row['image_count'] / total_images) * 100
                tooltip_html += f"""
                    <li style="margin-bottom: 4px;">
                        <b>{row['cluster_label']}:</b> {pct:.1f}% ({row['image_count']:,} images)
                    </li>
                """
            tooltip_html += "</ul></div>"
            
            log_val = np.log10(total_images)
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geojson_coords]
                },
                "properties": {
                    "cell": cell,
                    "count": total_images,
                    "log_count": log_val,
                    "tooltip_content": tooltip_html
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

    print("Adding GeoJSON layer to map...")
    # Add GeoJSON layer in one batch
    folium.GeoJson(
        geojson_data,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip_content"],
            aliases=[""],
            labels=False,
            style="background-color: white; border: 1px solid #ccc; border-radius: 4px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);"
        )
    ).add_to(m)

    m.add_child(colormap)
    
    # Save the output HTML file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"\nSuccessfully generated interactive spatial-semantic map at: {args.output}")

if __name__ == "__main__":
    main()
