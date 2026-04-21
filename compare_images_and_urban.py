import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from shapely.geometry import Point
import os

def analyze_overlap(csv_path, shp_path, output_image, res=0.2):
    print(f"Loading urban areas from {shp_path}...")
    urban_areas = gpd.read_file(shp_path, engine='pyogrio')
    # Ensure CRS is WGS84 (lat/lon)
    if urban_areas.crs is None:
        urban_areas.set_crs(epsg=4326, inplace=True)
    else:
        urban_areas = urban_areas.to_crs(epsg=4326)

    # Prepare grids for image density
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)
    grid_urban = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.uint32)
    grid_non_urban = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.uint32)

    # Track which urban areas were hit
    urban_areas['hit_count'] = 0
    
    chunksize = 500000 # Smaller chunks for spatial join overhead
    total_processed = 0
    images_in_urban = 0
    images_outside_urban = 0

    print("Processing images and performing spatial join...")
    try:
        reader = pd.read_csv(csv_path, usecols=['lat', 'lon'], chunksize=chunksize)
        for i, chunk in enumerate(reader):
            chunk = chunk.dropna(subset=['lat', 'lon'])
            valid_mask = (chunk['lat'] >= -90) & (chunk['lat'] <= 90) & \
                         (chunk['lon'] >= -180) & (chunk['lon'] <= 180)
            chunk = chunk[valid_mask]
            
            # Convert chunk to GeoDataFrame
            geometry = [Point(xy) for xy in zip(chunk.lon, chunk.lat)]
            gdf_chunk = gpd.GeoDataFrame(chunk, geometry=geometry, crs="EPSG:4326")
            
            # Spatial join
            # This tells us which images fall into which urban polygons
            joined = gpd.sjoin(gdf_chunk, urban_areas, how="left", predicate="within")
            
            # Images with index_right are in urban areas
            in_urban_mask = joined['index_right'].notna()
            
            chunk_in_urban = chunk[in_urban_mask.values]
            chunk_outside_urban = chunk[~in_urban_mask.values]
            
            # Update grids
            if not chunk_in_urban.empty:
                h_u, _, _ = np.histogram2d(chunk_in_urban['lat'], chunk_in_urban['lon'], bins=[lat_bins, lon_bins])
                grid_urban += h_u.astype(np.uint32)
                
                # Update hit counts for urban areas
                hits = joined['index_right'].dropna().astype(int).value_counts()
                urban_areas.loc[hits.index, 'hit_count'] += hits.values

            if not chunk_outside_urban.empty:
                h_nu, _, _ = np.histogram2d(chunk_outside_urban['lat'], chunk_outside_urban['lon'], bins=[lat_bins, lon_bins])
                grid_non_urban += h_nu.astype(np.uint32)

            images_in_urban += len(chunk_in_urban)
            images_outside_urban += len(chunk_outside_urban)
            total_processed += len(chunk)
            print(f"  Processed {total_processed} rows... (Urban: {images_in_urban}, Outside: {images_outside_urban})")

    except Exception as e:
        print(f"Error during processing: {e}")
        return

    print("\nAnalysis Complete:")
    print(f"Total Images: {total_processed}")
    print(f"Images in Urban Areas: {images_in_urban} ({images_in_urban/total_processed*100:.1f}%)")
    print(f"Images outside Urban Areas: {images_outside_urban} ({images_outside_urban/total_processed*100:.1f}%)")
    
    urban_hit_count = (urban_areas['hit_count'] > 0).sum()
    print(f"Urban Areas with images: {urban_hit_count} / {len(urban_areas)}")

    # --- Visualization ---
    print("Generating comparison plot...")
    fig, ax = plt.subplots(figsize=(24, 12), facecolor='white')
    ax.set_facecolor('#e6f2ff') # Light blue for sea
    
    # Load world map for coastlines
    try:
        # Try loading with pyogrio engine
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'), engine='pyogrio')
        world.plot(ax=ax, color='#f5f5f5', edgecolor='#bcbcbc', linewidth=0.5) # Light gray for land
    except Exception as e:
        print(f"Warning: Could not load world coastlines: {e}")
        # If it fails, at least color the background white to indicate it's not finished
        ax.set_facecolor('white') 
    
    # 1. Plot images OUTSIDE urban areas (Red/Orange)
    gn_display = grid_non_urban.astype(float)
    gn_display[gn_display == 0] = np.nan
    ax.imshow(gn_display, origin='lower', extent=[-180, 180, -90, 90], 
              aspect='equal', cmap='YlOrRd', norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(grid_non_urban)),
              alpha=0.8, zorder=3)

    # 2. Plot images INSIDE urban areas (Green/Blue)
    gu_display = grid_urban.astype(float)
    gu_display[gu_display == 0] = np.nan
    ax.imshow(gu_display, origin='lower', extent=[-180, 180, -90, 90], 
              aspect='equal', cmap='winter', norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(grid_urban)),
              alpha=0.6, zorder=4)

    # 3. Plot Urban Areas without images (Light gray)
    no_hits = urban_areas[urban_areas['hit_count'] == 0]
    no_hits.plot(ax=ax, color='#aaaaaa', edgecolor='#888888', linewidth=0.3, alpha=0.5, zorder=2)

    plt.title('Common (Urban) vs Non-Common (Rural/Remote) Image Distribution', color='black', size=20)
    plt.xlabel('Longitude', color='black')
    plt.ylabel('Latitude', color='black')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    
    # Grid lines
    plt.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.2)
    
    # Legend proxy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=4, label='Intersection (Images in Urban Areas)'),
        Line2D([0], [0], color='orange', lw=4, label='Non-Common: Images (Rural/Remote)'),
        Line2D([0], [0], color='#cccccc', lw=4, label='Non-Common: Urban Areas (No Images)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', facecolor='white', framealpha=1)

    plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comparison map saved as '{output_image}'")

if __name__ == "__main__":
    csv = "Projects/code/geo-rag/metadata_common_attributes.csv"
    shp = "Projects/code/geo-rag/ne_10m_urban_areas.shp"
    out = "Projects/code/geo-rag/intersection_comparison_map.png"
    
    analyze_overlap(csv, shp, out)
