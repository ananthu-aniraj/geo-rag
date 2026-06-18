import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import os
import glob
import argparse
from tqdm import tqdm


def create_uncovered_shapefile(csv_dirs, output_shp, res=1.0):
    """
    Creates a shapefile representing all global areas NOT covered by the images in the CSVs.
    """
    print(f"Analyzing coverage with resolution {res} degrees...")

    # 1. Gather all CSVs
    csv_files = []
    for d in csv_dirs:
        if os.path.isdir(d):
            csv_files.extend(glob.glob(os.path.join(d, "*.csv")))
        elif os.path.isfile(d) and d.endswith(".csv"):
            csv_files.append(d)

    if not csv_files:
        print("No CSV files found.")
        return

    # Define grid
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)

    # 0 = uncovered, 1 = covered
    occupancy_grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.uint8)

    print(f"Reading {len(csv_files)} CSV files to determine coverage...")
    for f in tqdm(csv_files, desc="Processing CSVs"):
        try:
            # We only need lat/lon to save memory
            df = pd.read_csv(f)
            if df.empty:
                continue
                
            # Identify standard columns
            lat_col = next((c for c in df.columns if c.lower() in ['latitude', 'lat']), None)
            lon_col = next((c for c in df.columns if c.lower() in ['longitude', 'lon']), None)

            if not (lat_col and lon_col):
                continue

            chunk = df.dropna(subset=[lat_col, lon_col])

            # Simple binning
            lon_idx = np.digitize(chunk[lon_col], lon_bins) - 1
            lat_idx = np.digitize(chunk[lat_col], lat_bins) - 1

            # Filter valid indices
            valid = (lon_idx >= 0) & (lon_idx < len(lon_bins) - 1) & \
                    (lat_idx >= 0) & (lat_idx < len(lat_bins) - 1)

            occupancy_grid[lat_idx[valid], lon_idx[valid]] = 1

        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Find indices where occupancy is 0 (uncovered)
    uncovered_lat_idx, uncovered_lon_idx = np.where(occupancy_grid == 0)
    print(f"Found {len(uncovered_lat_idx)} uncovered grid cells.")

    print("Converting uncovered cells to polygons...")
    polygons = []
    for lat_i, lon_i in zip(uncovered_lat_idx, uncovered_lon_idx):
        min_lon = lon_bins[lon_i]
        max_lon = lon_bins[lon_i + 1]
        min_lat = lat_bins[lat_i]
        max_lat = lat_bins[lat_i + 1]
        polygons.append(box(min_lon, min_lat, max_lon, max_lat))

    # Create GeoDataFrame
    print(f"Creating GeoDataFrame with {len(polygons)} features...")
    gdf = gpd.GeoDataFrame({
        'geometry': polygons,
        'status': ['uncovered'] * len(polygons)
    }, crs="EPSG:4326")

    # Export to Shapefile
    print(f"Saving shapefile to {output_shp}...")
    gdf.to_file(output_shp, engine='pyogrio')
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a shapefile of uncovered global areas.")
    parser.add_argument("--dirs", nargs="+", required=True, help="List of directories containing CSV files.")
    parser.add_argument("--output", type=str, default="uncovered_areas.shp", help="Output shapefile path.")
    parser.add_argument("--res", type=float, default=1.0, help="Grid resolution in degrees.")
    args = parser.parse_args()

    create_uncovered_shapefile(args.dirs, args.output, res=args.res)
