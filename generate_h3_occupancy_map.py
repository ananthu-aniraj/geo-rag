import os
import glob
import pandas as pd
import h3
import folium
import argparse
import numpy as np
from tqdm import tqdm
import branca.colormap as cm

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
    h3_counts = {}
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
            h3_col = 'H3_Cell' if 'H3_Cell' in df.columns else None

            if not (lat_col and lon_col) and not h3_col:
                print(f"Skipping {f}: Missing location columns.")
                continue

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

                    h3_counts[cell] = h3_counts.get(cell, 0) + 1
                    if photo_id:
                        seen_photo_ids.add(photo_id)
                    total_images += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not h3_counts:
        print("No valid data processed.")
        return

    print(f"Total unique images: {total_images}")
    print(f"Unique H3 cells at resolution {args.res}: {len(h3_counts)}")

    # 3. Create Interactive Map
    print("Generating Folium map...")
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB Positron")

    # Filter by min_count
    display_cells = {k: v for k, v in h3_counts.items() if v >= args.min_count}
    if not display_cells:
        print(f"No cells with at least {args.min_count} images.")
        return

    counts = list(display_cells.values())
    min_val = np.log10(min(counts))
    max_val = np.log10(max(counts))

    # Handle case where all counts are the same
    if min_val == max_val:
        max_val += 0.1

    colormap = cm.linear.YlOrRd_09.scale(min_val, max_val)
    colormap.caption = f"Image Density (Log10 scale, H3 Res {args.res})"

    for cell, count in tqdm(display_cells.items(), desc="Adding hexagons to map"):
        try:
            # Get hexagon boundary
            boundary = h3.cell_to_boundary(cell)
            
            # Map count to color (log scale)
            color = colormap(np.log10(count))
            
            folium.Polygon(
                locations=boundary,
                fill=True,
                fill_color=color,
                color=color,
                weight=1,
                fill_opacity=0.7,
                tooltip=f"H3 Cell: {cell}<br>Image Count: {count}"
            ).add_to(m)
        except Exception as e:
            continue

    m.add_child(colormap)
    m.save(args.output)
    print(f"Successfully saved global occupancy map to: {args.output}")

if __name__ == "__main__":
    main()
