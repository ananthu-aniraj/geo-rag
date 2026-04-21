import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import folium
from folium.plugins import HeatMap

def create_maps(csv_path, output_image, output_html, res=0.2):
    """
    Creates a global occupancy map (static PNG and interactive HTML) from lat/lon data.
    """
    print(f"Starting processing of {csv_path} with resolution {res} degrees...")
    
    # Define grid boundaries
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)
    
    # Initialize 2D histogram array
    grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.uint32)
    
    chunksize = 1000000
    total_processed = 0
    
    try:
        for i, chunk in enumerate(pd.read_csv(csv_path, usecols=['lat', 'lon'], chunksize=chunksize)):
            chunk = chunk.dropna(subset=['lat', 'lon'])
            valid_mask = (chunk['lat'] >= -90) & (chunk['lat'] <= 90) & \
                         (chunk['lon'] >= -180) & (chunk['lon'] <= 180)
            chunk = chunk[valid_mask]
            
            h, _, _ = np.histogram2d(chunk['lat'], chunk['lon'], bins=[lat_bins, lon_bins])
            grid += h.astype(np.uint32)
            
            total_processed += len(chunk)
            print(f"Processed {total_processed} rows...")
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- 1. Static Map (PNG) ---
    print("Generating static plot...")
    plt.figure(figsize=(20, 10))
    display_grid = grid.astype(float)
    display_grid[display_grid == 0] = np.nan
    
    norm = mcolors.LogNorm(vmin=1, vmax=np.nanmax(display_grid))
    
    im = plt.imshow(display_grid, origin='lower', extent=[-180, 180, -90, 90], 
                    aspect='equal', cmap='viridis', norm=norm)
    
    plt.colorbar(im, label='Image Count (log scale)')
    plt.title(f'Global Occupancy Map (Resolution: {res} deg)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static map saved as '{output_image}'")

    # --- 2. Interactive Map (Folium HTML) ---
    print("Generating interactive Folium map...")
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
    
    # We use a heat map based on the binned data for efficiency
    # Extract non-zero bins to reduce data size for Folium
    y_indices, x_indices = np.where(grid > 0)
    
    # Create lat, lon, intensity triples
    # lat_bins[i] is the edge, so we use center of bin
    heat_data = []
    # Limit number of points for Folium if it's too high, but 0.2 res is manageable (~1.6M max, usually much less)
    # We'll take the top N most occupied bins or just all if reasonable
    max_folium_points = 50000 
    
    if len(y_indices) > max_folium_points:
        print(f"Too many occupied bins ({len(y_indices)}) for Folium. Subsampling to top {max_folium_points}...")
        # Get indices of top counts
        counts = grid[y_indices, x_indices]
        top_indices = np.argsort(counts)[-max_folium_points:]
        y_indices = y_indices[top_indices]
        x_indices = x_indices[top_indices]
        counts = counts[top_indices]
    else:
        counts = grid[y_indices, x_indices]

    for y_idx, x_idx, count in zip(y_indices, x_indices, counts):
        lat = lat_bins[y_idx] + res/2
        lon = lon_bins[x_idx] + res/2
        # Use log scale for intensity in Folium heatmap as well
        intensity = float(np.log10(count + 1))
        heat_data.append([lat, lon, intensity])
    
    HeatMap(heat_data, radius=10, blur=5, min_opacity=0.3).add_to(m)
    
    m.save(output_html)
    print(f"Interactive map saved as '{output_html}'")

if __name__ == "__main__":
    input_csv = "Projects/code/geo-rag/metadata_common_attributes.csv"
    output_png = "Projects/code/geo-rag/global_occupancy_map.png"
    output_html = "Projects/code/geo-rag/global_occupancy_map.html"
    
    if os.path.exists(input_csv):
        create_maps(input_csv, output_png, output_html, res=0.2)
    else:
        print(f"Error: Could not find {input_csv}")
