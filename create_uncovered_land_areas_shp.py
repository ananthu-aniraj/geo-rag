import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import os

def create_uncovered_land_shapefile(csv_path, output_shp, res=1.0):
    """
    Creates a shapefile of Land areas NOT covered by images, excluding oceans.
    """
    print(f"1. Generating land mask at {res} degree resolution...")
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)
    
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'), engine='pyogrio')
    except Exception as e:
        print(f"Error loading world map: {e}")
        return

    grid_polygons = []
    grid_coords = []
    for j in range(len(lat_bins) - 1):
        for i in range(len(lon_bins) - 1):
            grid_polygons.append(box(lon_bins[i], lat_bins[j], lon_bins[i+1], lat_bins[j+1]))
            grid_coords.append((j, i))
            
    all_grid = gpd.GeoDataFrame({'geometry': grid_polygons, 'coords': grid_coords}, crs="EPSG:4326")
    
    print("   Identifying land cells...")
    land_grid = gpd.sjoin(all_grid, world, how="inner", predicate="intersects")
    land_cells = set(land_grid['coords'])
    print(f"   Total land cells: {len(land_cells)}")

    print(f"2. Analyzing image coverage from {csv_path}...")
    occupancy_grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.uint8)
    
    chunksize = 1000000
    try:
        reader = pd.read_csv(csv_path, usecols=['lat', 'lon'], chunksize=chunksize)
        for i, chunk in enumerate(reader):
            chunk = chunk.dropna(subset=['lat', 'lon'])
            lon_idx = np.digitize(chunk['lon'], lon_bins) - 1
            lat_idx = np.digitize(chunk['lat'], lat_bins) - 1
            valid = (lon_idx >= 0) & (lon_idx < len(lon_bins)-1) & (lat_idx >= 0) & (lat_idx < len(lat_bins)-1)
            occupancy_grid[lat_idx[valid], lon_idx[valid]] = 1
            if i % 2 == 0:
                print(f"   Processed {i*chunksize + len(chunk)} rows...")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("3. Filtering for uncovered land cells...")
    uncovered_land_polygons = []
    for (lat_i, lon_i) in land_cells:
        if occupancy_grid[lat_i, lon_i] == 0:
            uncovered_land_polygons.append(box(lon_bins[lon_i], lat_bins[lat_i], lon_bins[lon_i+1], lat_bins[lat_i+1]))
    
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
    csv = "Projects/code/geo-rag/metadata_common_attributes.csv"
    out = "Projects/code/geo-rag/uncovered_land_areas.shp"
    if os.path.exists(csv):
        create_uncovered_land_shapefile(csv, out, res=1.0)
    else:
        print(f"Missing input: {csv}")
