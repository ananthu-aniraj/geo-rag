import os
import glob
import pandas as pd
import h3
import folium
import argparse
import numpy as np
from tqdm import tqdm
import branca.colormap as cm
import json


def main():
    parser = argparse.ArgumentParser(description="Generate a global H3-based occupancy map from image metadata.")
    parser.add_argument("--dirs", nargs="+", required=True, help="List of directories containing CSV files.")
    parser.add_argument("--output", type=str, default="global_h3_occupancy_map.html", help="Path to save the HTML map.")
    parser.add_argument("--res", type=int, default=4, help="H3 resolution for aggregation (default 4, ~177km edge).")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum number of images in a cell to display.")
    args = parser.parse_args()

    # 1. Gather all CSVs
    csv_files = []
    for d in args.dirs:
        if os.path.isdir(d):
            csv_files.extend(glob.glob(os.path.join(d, "*.csv")))
        elif os.path.isfile(d) and d.endswith(".csv"):
            csv_files.append(d)

    if not csv_files:
        print("No CSV files found in the specified directories.")
        return

    print(f"Found {len(csv_files)} CSV files.")

    # 2. Aggregate Data
    h3_stats = {} # cell -> {'total': count, 'platforms': {name: count}, 'photos': [ids]}
    seen_photo_ids = set()
    total_images = 0

    for f in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue

            # Identify standard columns (consistent with process_scraped_data.py)
            lat_col = next((c for c in df.columns if c.lower() in ['latitude', 'lat']), None)
            lon_col = next((c for c in df.columns if c.lower() in ['longitude', 'lon']), None)
            id_col = next((c for c in df.columns if c.lower() in ['photo_id', 'id', 'orig_id', 'uuid']), None)
            platform_col = next((c for c in df.columns if c.lower() in ['platform', 'source']), None)
            h3_col = 'H3_Cell' if 'H3_Cell' in df.columns else None

            if not (lat_col and lon_col) and not h3_col:
                print(f"Skipping {f}: Missing location columns.")
                continue
            
            # Infer platform if not explicitly in columns
            inferred_platform = 'Flickr' if 'flickr' in f.lower() else 'Mapillary'

            for _, row in df.iterrows():
                try:
                    # Deduplication
                    photo_id = str(row[id_col]) if id_col else None
                    if photo_id and photo_id in seen_photo_ids:
                        continue

                    if h3_col:
                        # Use existing H3 cell and coarsen if needed
                        cell = row[h3_col]
                        if h3.get_resolution(cell) != args.res:
                            cell = h3.cell_to_parent(cell, args.res)
                    else:
                        # Compute H3 cell from lat/lon
                        lat, lon = float(row[lat_col]), float(row[lon_col])
                        cell = h3.latlng_to_cell(lat, lon, args.res)

                    platform = str(row[platform_col]) if platform_col else inferred_platform
                    
                    if cell not in h3_stats:
                        h3_stats[cell] = {'total': 0, 'platforms': {}, 'photos': []}
                    
                    h3_stats[cell]['total'] += 1
                    h3_stats[cell]['platforms'][platform] = h3_stats[cell]['platforms'].get(platform, 0) + 1
                    if photo_id:
                        h3_stats[cell]['photos'].append(photo_id)
                        seen_photo_ids.add(photo_id)
                    
                    total_images += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not h3_stats:
        print("No valid data processed.")
        return

    print(f"Total unique images: {total_images}")
    print(f"Unique H3 cells at resolution {args.res}: {len(h3_stats)}")

    # 3. Create Interactive Map
    print("Generating Folium map...")
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB Positron")

    # Filter by min_count
    display_cells = {k: v for k, v in h3_stats.items() if v['total'] >= args.min_count}
    if not display_cells:
        print(f"No cells with at least {args.min_count} images.")
        return

    counts = [v['total'] for v in display_cells.values()]
    min_val = np.log10(min(counts))
    max_val = np.log10(max(counts))

    # Handle case where all counts are the same
    if min_val == max_val:
        max_val += 0.1

    colormap = cm.linear.YlOrRd_09.scale(min_val, max_val)
    colormap.caption = f"Image Density (Log10 scale, H3 Res {args.res})"

    def get_clean_boundary(cell):
        """Get hexagon boundary and handle antimeridian crossing."""
        coords = h3.cell_to_boundary(cell)
        lngs = [c[1] for c in coords]
        if max(lngs) - min(lngs) > 180:
            # Shift negative longitudes by 360 to keep the polygon contiguous
            # Leaflet handles longitudes > 180 gracefully.
            coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
        return coords

    # Prepare GeoJSON features for better performance
    features = []
    for cell, stats in tqdm(display_cells.items(), desc="Preparing GeoJSON"):
        try:
            boundary = get_clean_boundary(cell)
            # GeoJSON expects [lng, lat] and a closed loop
            geojson_coords = [[lng, lat] for lat, lng in boundary]
            geojson_coords.append(geojson_coords[0])

            # Format platform breakdown for the tooltip
            p_str = ", ".join([f"{k}: {v}" for k, v in stats['platforms'].items()])

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geojson_coords]
                },
                "properties": {
                    "cell": cell,
                    "count": stats['total'],
                    "platforms": p_str,
                    "log_count": np.log10(stats['total'])
                }
            }
            features.append(feature)
        except Exception:
            continue

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    def style_fn(feature):
        return {
            "fillColor": colormap(feature["properties"]["log_count"]),
            "color": colormap(feature["properties"]["log_count"]),
            "weight": 1,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        geojson_data,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["cell", "count", "platforms"],
            aliases=["H3 Cell:", "Total Images:", "Platform Breakdown:"],
            localize=True
        )
    ).add_to(m)

    m.add_child(colormap)
    m.save(args.output)
    print(f"Successfully saved global occupancy map to: {args.output}")


if __name__ == "__main__":
    main()
