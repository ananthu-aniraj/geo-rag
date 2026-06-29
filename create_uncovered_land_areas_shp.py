import os
import glob
import pandas as pd
import h3
import geopandas as gpd
from shapely.geometry import Polygon
import argparse
from tqdm import tqdm


def get_h3_polygon(cell):
    """Convert H3 cell to a Shapely Polygon."""
    coords = h3.cell_to_boundary(cell)
    # H3 returns (lat, lng), Shapely expects (lng, lat)
    # Handle antimeridian crossing
    lngs = [c[1] for c in coords]
    if max(lngs) - min(lngs) > 180:
        coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]

    return Polygon([[lng, lat] for lat, lng in coords])


def main():
    parser = argparse.ArgumentParser(description="Generate Uncovered Land Areas Shapefile using H3.")
    parser.add_argument("--csv_paths", nargs="+", required=True, help="List of paths to CSV files or directories containing CSVs.")
    parser.add_argument("--land_shp", type=str, default="ne_10m_admin_0_countries.shp", help="Path to the base land shapefile.")
    parser.add_argument("--output", type=str, default="uncovered_land_areas.shp", help="Output shapefile path.")
    parser.add_argument("--res", type=int, default=5, help="H3 resolution for covered areas.")
    args = parser.parse_args()

    # 1. Load Land
    print(f"Loading land shapefile from {args.land_shp}...")
    land_gdf = gpd.read_file(args.land_shp)

    # 2. Gather Covered Cells
    print(f"Processing CSVs to find covered H3 res {args.res} cells...")
    covered_cells = set()

    csv_files = []
    for path in args.csv_paths:
        if os.path.isdir(path):
            csv_files.extend(glob.glob(os.path.join(path, "**/*.csv"), recursive=True))
        else:
            csv_files.append(path)

    for f in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(f, usecols=lambda c: c.lower() in ['latitude', 'lat', 'longitude', 'lon'])
            if df.empty: continue

            lat_col = next((c for c in df.columns if c.lower() in ['latitude', 'lat']), None)
            lon_col = next((c for c in df.columns if c.lower() in ['longitude', 'lon']), None)

            if not (lat_col and lon_col):
                continue

            for _, row in df.iterrows():
                try:
                    if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                        cell = h3.latlng_to_cell(float(row[lat_col]), float(row[lon_col]), args.res)
                        covered_cells.add(cell)
                except Exception:
                    continue
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Found {len(covered_cells)} unique covered cells at res {args.res}.")

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
