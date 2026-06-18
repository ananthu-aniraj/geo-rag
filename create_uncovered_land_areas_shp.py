import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import os


def create_uncovered_shapefile(csv_path, output_shp, res=1.0):
    """
    Creates a shapefile representing all global areas NOT covered by the images in the CSV.

    Args:
        csv_path: Path to metadata_common_attributes.csv
        output_shp: Path for the resulting shapefile (.shp)
        res: Grid resolution in degrees (default 1.0 for efficient shapefile size)
    """
    print(f"Analyzing coverage with resolution {res} degrees...")

    # Define grid
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)

    # 0 = uncovered, 1 = covered
    occupancy_grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.uint8)

    chunksize = 1000000
    total_processed = 0

    print("Reading image coordinates to determine coverage...")
    try:
        reader = pd.read_csv(csv_path, usecols=['lat', 'lon'], chunksize=chunksize)
        for i, chunk in enumerate(reader):
            chunk = chunk.dropna(subset=['lat', 'lon'])

            # Simple binning to find occupied cells
            # We use digitize to find the indices quickly
            lon_idx = np.digitize(chunk['lon'], lon_bins) - 1
            lat_idx = np.digitize(chunk['lat'], lat_bins) - 1

            # Filter valid indices
            valid = (lon_idx >= 0) & (lon_idx < len(lon_bins) - 1) & \
                    (lat_idx >= 0) & (lat_idx < len(lat_bins) - 1)

            occupancy_grid[lat_idx[valid], lon_idx[valid]] = 1

            total_processed += len(chunk)
            if i % 2 == 0:
                print(f"  Processed {total_processed} rows...")

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Find indices where occupancy is 0 (uncovered)
    uncovered_lat_idx, uncovered_lon_idx = np.where(occupancy_grid == 0)
    print(f"Found {len(uncovered_lat_idx)} uncovered grid cells.")

    print("Converting uncovered cells to polygons...")
    polygons = []
    # We'll create a box for each empty cell
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
    # Note: This will create .shp, .shx, .dbf, etc.
    gdf.to_file(output_shp, engine='pyogrio')

    print("Success!")


if __name__ == "__main__":
    csv = "/user/aaniraj/home/Documents/Projects/data/global-streetscapes/train/platform.csv"
    out = "uncovered_areas.shp"

    if os.path.exists(csv):
        # 1.0 degree provides a good balance between detail and performance for a global SHP
        create_uncovered_shapefile(csv, out, res=1.0)
    else:
        print(f"Error: Could not find {csv}")
