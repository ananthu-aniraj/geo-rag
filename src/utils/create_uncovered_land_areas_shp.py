import argparse
import glob
import os
from collections import Counter

import geopandas as gpd
import h3
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import Polygon
from tqdm import tqdm

from src.utils.io import load_dataframe


def get_h3_polygon(cell):
    """Convert H3 cell to a Shapely Polygon."""
    coords = h3.cell_to_boundary(cell)
    # H3 returns (lat, lng), Shapely expects (lng, lat)
    # Handle antimeridian crossing, but ignore polar cells where longitudes naturally converge
    lngs = [c[1] for c in coords]
    lats = [c[0] for c in coords]

    is_polar = any(abs(lat) > 85.0 for lat in lats)
    if not is_polar and (max(lngs) - min(lngs) > 180):
        coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]

    return Polygon([[lng, lat] for lat, lng in coords])


def main():
    parser = argparse.ArgumentParser(description="Generate Uncovered Land Areas Shapefile using H3.")
    parser.add_argument("--csv_paths", nargs="+", required=True,
                        help="List of paths to CSV files or directories containing CSVs.")
    default_land_shp = "shapefiles/ne_10m_admin_0_countries.shp" if os.path.exists(
        "shapefiles/ne_10m_admin_0_countries.shp") else "ne_10m_admin_0_countries.shp"
    parser.add_argument("--land_shp", type=str, default=default_land_shp, help="Path to the base land shapefile.")
    parser.add_argument("--output", type=str, default="shapefiles/uncovered_land_areas_test.shp",
                        help="Output shapefile path.")
    parser.add_argument("--res", type=int, default=5, help="H3 resolution for covered areas.")
    parser.add_argument("--threshold", type=int, default=0,
                        help="Threshold for total number of images per H3 cell. Cells with counts <= threshold remain uncovered (default: 0).")
    args = parser.parse_args()

    # 1. Load Land
    print(f"Loading land shapefile from {args.land_shp}...")
    land_gdf = gpd.read_file(args.land_shp)

    # 2. Gather Covered Cells
    print(f"Processing data files to find covered H3 res {args.res} cells...")
    covered_cells_counter = Counter()

    data_files = []
    for path in args.csv_paths:
        if os.path.isdir(path):
            data_files.extend(glob.glob(os.path.join(path, "**/*.csv"), recursive=True))
            data_files.extend(glob.glob(os.path.join(path, "**/*.parquet"), recursive=True))
        else:
            data_files.append(path)

    for f in tqdm(data_files, desc="Reading Data Files"):
        try:
            if f.endswith('.parquet'):
                # Inspect columns using pyarrow schema
                available_cols = pq.ParquetFile(f).schema_arrow.names
                lat_col = next((c for c in available_cols if c.lower() in ['latitude', 'lat']), None)
                lon_col = next((c for c in available_cols if c.lower() in ['longitude', 'lon']), None)
                if lat_col and lon_col:
                    df = load_dataframe(f, columns=[lat_col, lon_col])
                else:
                    continue
            else:
                df = pd.read_csv(f, usecols=lambda c: c.lower() in ['latitude', 'lat', 'longitude', 'lon'])

            if df.empty:
                continue

            lat_col = next((c for c in df.columns if c.lower() in ['latitude', 'lat']), None)
            lon_col = next((c for c in df.columns if c.lower() in ['longitude', 'lon']), None)

            if not (lat_col and lon_col):
                continue

            for _, row in df.iterrows():
                try:
                    if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                        cell = h3.latlng_to_cell(float(row[lat_col]), float(row[lon_col]), args.res)
                        covered_cells_counter[cell] += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Filter cells by threshold
    covered_cells = {cell for cell, count in covered_cells_counter.items() if count > args.threshold}
    print(
        f"Found {len(covered_cells_counter):,} unique cells. Filtered to {len(covered_cells):,} cells with > {args.threshold} images.")

    if not covered_cells:
        print("No covered cells found. Saving unmodified land shapefile.")
        land_gdf.to_file(args.output)
        return

    # 3. Create Covered GeoDataFrame
    print("Generating polygons for covered cells...")
    polygons = [get_h3_polygon(c) for c in tqdm(covered_cells, desc="Creating Polygons")]
    covered_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    # Optional: Combine overlapping covered hexagons to speed up difference
    print("Unioning covered geometries...")
    covered_gdf.geometry = covered_gdf.geometry.make_valid().buffer(0)
    covered_geom = covered_gdf.union_all()

    # 4. Subtract Covered from Land
    print("Subtracting covered areas from land mass... (This may take a while!)")

    uncovered_geoms = land_gdf.geometry.difference(covered_geom)
    uncovered_gdf = gpd.GeoDataFrame(geometry=uncovered_geoms, crs="EPSG:4326")
    uncovered_gdf = uncovered_gdf[~uncovered_gdf.is_empty]

    print(f"Saving to {args.output}...")
    uncovered_gdf.to_file(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
