import argparse
import glob
import os

import branca.colormap as cm
import folium
import h3
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Generate a global H3-based occupancy map from image metadata."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="List of directories containing CSV files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="global_h3_occupancy_map.html",
        help="Path to save the HTML map.",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=4,
        help="H3 resolution for aggregation (default 4, ~177km edge).",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum number of images in a cell to display.",
    )
    args = parser.parse_args()

    # 1. Gather all CSVs and Parquets
    csv_files = []
    for d in args.dirs:
        if os.path.isdir(d):
            csv_files.extend(glob.glob(os.path.join(d, "*.csv")))
            csv_files.extend(glob.glob(os.path.join(d, "*.parquet")))
        elif os.path.isfile(d) and (d.endswith(".csv") or d.endswith(".parquet")):
            csv_files.append(d)

    if not csv_files:
        print("No CSV or Parquet files found in the specified paths.")
        return

    print(f"Found {len(csv_files)} files (CSV/Parquet).")

    # 2. Aggregate Data
    h3_stats = {}  # cell -> {'total': count, 'platforms': {name: count}}
    total_images = 0

    import pyarrow.parquet as pq

    for f in tqdm(csv_files, desc="Reading files"):
        try:
            if f.endswith(".parquet"):
                avail_cols = pq.ParquetFile(f).schema_arrow.names
                read_cols = [
                    c
                    for c in [
                        "Latitude",
                        "Longitude",
                        "Photo_ID",
                        "Platform",
                        "H3_Cell",
                    ]
                    if c in avail_cols
                ]
                df = pd.read_parquet(f, columns=read_cols)
            else:
                # We only read the necessary columns to save memory
                df = pd.read_csv(
                    f,
                    usecols=lambda col: col.lower()
                    in [
                        "latitude",
                        "lat",
                        "longitude",
                        "lon",
                        "photo_id",
                        "id",
                        "platform",
                        "source",
                        "h3_cell",
                    ],
                    dtype={
                        "photo_id": str,
                        "Photo_ID": str,
                        "id": str,
                        "ID": str,
                        "platform": str,
                        "Platform": str,
                        "source": str,
                        "Source": str,
                    },
                )
            if df.empty:
                continue

            # Standardize column names
            lat_col = next(
                (c for c in df.columns if c.lower() in ["latitude", "lat"]), None
            )
            lon_col = next(
                (c for c in df.columns if c.lower() in ["longitude", "lon"]), None
            )
            id_col = next(
                (c for c in df.columns if c.lower() in ["photo_id", "id"]), None
            )
            platform_col = next(
                (c for c in df.columns if c.lower() in ["platform", "source"]), None
            )
            h3_col = next((c for c in df.columns if c.lower() == "h3_cell"), None)

            if not (lat_col and lon_col) and not h3_col:
                print(f"Skipping {f}: Missing location columns.")
                continue

            # Deduplicate by photo ID if present
            if id_col:
                df = df.drop_duplicates(subset=[id_col])

            # Extract platform
            inferred_platform = "Flickr" if "flickr" in f.lower() else "Mapillary"
            if platform_col:
                df["Platform_Clean"] = (
                    df[platform_col].fillna(inferred_platform).astype(str)
                )
            else:
                df["Platform_Clean"] = inferred_platform

            # Compute/coarsen H3 cells in vectorized way
            if h3_col:
                df["H3_Clean"] = df[h3_col]
                # Coarsen if needed (using unique map is 100x faster than Map/Lambda row-by-row)
                sample_cell = (
                    df["H3_Clean"].dropna().iloc[0]
                    if not df["H3_Clean"].dropna().empty
                    else None
                )
                if sample_cell and h3.get_resolution(sample_cell) != args.res:
                    unique_cells = df["H3_Clean"].dropna().unique()
                    cell_map = {c: h3.cell_to_parent(c, args.res) for c in unique_cells}
                    df["H3_Clean"] = df["H3_Clean"].map(cell_map)
            else:
                # Compute from lat/lon in vectorized way
                df["H3_Clean"] = df.apply(
                    lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], args.res),
                    axis=1,
                )

            df = df.dropna(subset=["H3_Clean"])

            # Group by cell and platform and get counts
            gp = df.groupby(["H3_Clean", "Platform_Clean"]).size().unstack(fill_value=0)

            # Merge into global h3_stats
            for cell, row in gp.iterrows():
                if cell not in h3_stats:
                    h3_stats[cell] = {"total": 0, "platforms": {}}

                for plat, count in row.items():
                    if count > 0:
                        h3_stats[cell]["total"] += count
                        h3_stats[cell]["platforms"][plat] = (
                            h3_stats[cell]["platforms"].get(plat, 0) + count
                        )
                        total_images += count

        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not h3_stats:
        print("No valid data processed.")
        return

    print(f"Total unique images: {total_images}")
    print(f"Unique H3 cells at resolution {args.res}: {len(h3_stats)}")

    # 3. Create Interactive Map
    print("Generating Folium map...")
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB Positron")

    # Filter by min_count
    display_cells = {k: v for k, v in h3_stats.items() if v["total"] >= args.min_count}
    if not display_cells:
        print(f"No cells with at least {args.min_count} images.")
        return

    counts = [v["total"] for v in display_cells.values()]
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
            p_str = ", ".join([f"{k}: {v}" for k, v in stats["platforms"].items()])

            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [geojson_coords]},
                "properties": {
                    "cell": cell,
                    "count": stats["total"],
                    "platforms": p_str,
                    "log_count": np.log10(stats["total"]),
                },
            }
            features.append(feature)
        except Exception:
            continue

    geojson_data = {"type": "FeatureCollection", "features": features}

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
            localize=True,
        ),
    ).add_to(m)

    m.add_child(colormap)
    m.save(args.output)
    print(f"Successfully saved global occupancy map to: {args.output}")


if __name__ == "__main__":
    main()
