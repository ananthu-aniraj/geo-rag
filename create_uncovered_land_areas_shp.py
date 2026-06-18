import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import os
import glob
import argparse


def create_uncovered_land_shapefile(csv_paths, output_shp, res=1.0):
    """
    Creates a shapefile of Land areas NOT covered by images, excluding oceans.
    """
    print(f"1. Generating land mask at {res} degree resolution...")
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)

    try:
        # Using the standard naturalearth_lowres dataset
        # In newer versions of geopandas, this might need an explicit path or a different way to load
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'), engine='pyogrio')
        except AttributeError:
            # Fallback for newer geopandas versions where gpd.datasets is deprecated
            import pooch
            world_path = pooch.retrieve(
                url="https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
                known_hash="f63080e599e0486c8f621743666d3a95f9c4f1c1",
            )
            world = gpd.read_file(world_path, engine='pyogrio')
    except Exception as e:
        print(f"Error loading world map: {e}")
        return

    grid_polygons = []
    grid_coords = []
    for j in range(len(lat_bins) - 1):
        for i in range(len(lon_bins) - 1):
            grid_polygons.append(box(lon_bins[i], lat_bins[j], lon_bins[i + 1], lat_bins[j + 1]))
            grid_coords.append((j, i))

    all_grid = gpd.GeoDataFrame({'geometry': grid_polygons, 'coords': grid_coords}, crs="EPSG:4326")

    print("   Identifying land cells...")
    land_grid = gpd.sjoin(all_grid, world, how="inner", predicate="intersects")
    land_cells = set(land_grid['coords'])
    print(f"   Total land cells: {len(land_cells)}")

    print(f"2. Analyzing image coverage from {len(csv_paths)} files...")
    occupancy_grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.uint8)

    chunksize = 1000000
    for csv_path in csv_paths:
        print(f"   Processing {csv_path}...")
        try:
            # Peek at the columns to handle different naming conventions
            first_chunk = pd.read_csv(csv_path, nrows=5)
            col_map = {
                'lat': 'lat', 'latitude': 'lat', 'Latitude': 'lat',
                'lon': 'lon', 'longitude': 'lon', 'Longitude': 'lon'
            }
            available_cols = first_chunk.columns.tolist()
            use_cols = {}
            for k, v in col_map.items():
                if k in available_cols and v not in use_cols.values():
                    use_cols[k] = v
            
            if len(use_cols) < 2:
                print(f"      Warning: Could not find lat/lon columns in {csv_path}. Skipping.")
                continue

            reader = pd.read_csv(csv_path, usecols=list(use_cols.keys()), chunksize=chunksize)
            for i, chunk in enumerate(reader):
                chunk = chunk.rename(columns=use_cols)
                chunk = chunk.dropna(subset=['lat', 'lon'])
                lon_idx = np.digitize(chunk['lon'], lon_bins) - 1
                lat_idx = np.digitize(chunk['lat'], lat_bins) - 1
                valid = (lon_idx >= 0) & (lon_idx < len(lon_bins) - 1) & (lat_idx >= 0) & (lat_idx < len(lat_bins) - 1)
                occupancy_grid[lat_idx[valid], lon_idx[valid]] = 1
                if i % 5 == 0 and i > 0:
                    print(f"      Processed {i * chunksize + len(chunk)} rows...")
        except Exception as e:
            print(f"   Error processing {csv_path}: {e}")

    print("3. Filtering for uncovered land cells...")
    uncovered_land_polygons = []
    for (lat_i, lon_i) in land_cells:
        if occupancy_grid[lat_i, lon_i] == 0:
            uncovered_land_polygons.append(
                box(lon_bins[lon_i], lat_bins[lat_i], lon_bins[lon_i + 1], lat_bins[lat_i + 1]))

    if not uncovered_land_polygons:
        print("No uncovered land areas found!")
        return

    gdf_out = gpd.GeoDataFrame({
        'geometry': uncovered_land_polygons,
        'status': ['uncovered_land'] * len(uncovered_land_polygons)
    }, crs="EPSG:4326")

    print(f"4. Saving shapefile to {output_shp}...")
    gdf_out.to_file(output_shp, engine='pyogrio')
    print("Success! Data generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a shapefile of land areas not covered by images.")
    parser.add_argument("--dirs", nargs="+", required=True, help="List of directories containing CSVs.")
    parser.add_argument("--output", type=str, default="Projects/code/geo-rag/uncovered_land_areas.shp", help="Path to output shapefile.")
    parser.add_argument("--res", type=float, default=1.0, help="Resolution in degrees.")
    args = parser.parse_args()

    csv_files = []
    for d in args.dirs:
        if os.path.isdir(d):
            csv_files.extend(glob.glob(os.path.join(d, "*.csv")))
        elif os.path.isfile(d) and d.endswith(".csv"):
            csv_files.append(d)

    if not csv_files:
        print(f"No CSV files found in {args.dirs}")
    else:
        create_uncovered_land_shapefile(csv_files, args.output, res=args.res)
