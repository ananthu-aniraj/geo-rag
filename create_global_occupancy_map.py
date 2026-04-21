import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def create_occupancy_map(csv_path, output_image, res=0.1):
    """
    Creates a global occupancy map from lat/lon data in a CSV file.
    
    Args:
        csv_path: Path to the metadata_common_attributes.csv
        output_image: Path to save the resulting plot.
        res: Grid resolution in degrees (default 0.1 deg).
    """
    print(f"Starting processing of {csv_path} with resolution {res} degrees...")
    
    # Define grid boundaries
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)
    
    # Initialize 2D histogram array
    # Note: np.histogram2d uses (len(x_bins)-1, len(y_bins)-1)
    grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.uint32)
    
    chunksize = 1000000
    total_processed = 0
    
    # Process the file in chunks to save memory
    try:
        for i, chunk in enumerate(pd.read_csv(csv_path, usecols=['lat', 'lon'], chunksize=chunksize)):
            # Drop NaN values if any
            chunk = chunk.dropna(subset=['lat', 'lon'])
            
            # Filter valid lat/lon
            valid_mask = (chunk['lat'] >= -90) & (chunk['lat'] <= 90) & \
                         (chunk['lon'] >= -180) & (chunk['lon'] <= 180)
            chunk = chunk[valid_mask]
            
            # Compute 2D histogram for this chunk
            # Note: histogram2d takes (x, y) where x is lon and y is lat for standard maps
            h, _, _ = np.histogram2d(chunk['lat'], chunk['lon'], bins=[lat_bins, lon_bins])
            
            # Add to main grid
            grid += h.astype(np.uint32)
            
            total_processed += len(chunk)
            print(f"Processed chunk {i+1}: {total_processed} rows handled so far...")
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Total points binned: {total_processed}")
    
    # Visualization
    print("Generating plot...")
    plt.figure(figsize=(20, 10))
    
    # Use logarithmic normalization for better visibility of low-density areas
    # Replace zeros with a very small value to avoid log(0)
    display_grid = grid.astype(float)
    display_grid[display_grid == 0] = np.nan
    
    norm = mcolors.LogNorm(vmin=1, vmax=np.nanmax(display_grid))
    
    # Plot the grid
    # origin='lower' ensures lat=-90 is at the bottom
    im = plt.imshow(display_grid, origin='lower', extent=[-180, 180, -90, 90], 
                    aspect='equal', cmap='viridis', norm=norm)
    
    plt.colorbar(im, label='Image Count (log scale)')
    plt.title(f'Global Occupancy Map (Resolution: {res} deg)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Save the plot
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Occupancy map saved as '{output_image}'")

if __name__ == "__main__":
    input_csv = "Projects/code/geo-rag/metadata_common_attributes.csv"
    output_png = "global_occupancy_map.png"
    
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}")
    else:
        create_occupancy_map(input_csv, output_png, res=0.2) # 0.2 deg for a bit faster processing and visibility
