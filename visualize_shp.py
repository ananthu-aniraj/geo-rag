import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys

def visualize_shapefile(shp_path, title=None):
    """
    Visualizes a shapefile on top of a global world map with blue sea.
    """
    if not os.path.exists(shp_path):
        print(f"Error: Shapefile {shp_path} not found.")
        return

    print(f"Loading shapefile: {shp_path}...")
    try:
        gdf = gpd.read_file(shp_path, engine='pyogrio')
        # Ensure CRS is 4326 for plotting consistency
        if gdf.crs and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
    except Exception as e:
        print(f"Error loading {shp_path}: {e}")
        return

    print("Generating map...")
    fig, ax = plt.subplots(figsize=(24, 12), facecolor='white')
    ax.set_facecolor('#e6f2ff') # Sea background
    
    # Load and plot world background for context
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'), engine='pyogrio')
        world.plot(ax=ax, color='#f5f5f5', edgecolor='#bcbcbc', linewidth=0.5, zorder=1)
    except Exception as e:
        print(f"Warning: Could not load background world map: {e}")
        ax.set_facecolor('white')

    # Plot the shapefile data
    # We use a bright red for visibility
    gdf.plot(ax=ax, color='#ff3333', alpha=0.7, edgecolor='none', zorder=2)
    
    # Styling
    plt_title = title if title else f"Visualization of {os.path.basename(shp_path)}"
    plt.title(plt_title, size=20, pad=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Global extent
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    plt.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.2)

    # Save output
    output_png = os.path.splitext(shp_path)[0] + ".png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Success! Map saved as '{output_png}'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_shp = sys.argv[1]
        visualize_shapefile(target_shp)
    else:
        # Default target if no argument provided
        default = "Projects/code/geo-rag/uncovered_land_areas.shp"
        visualize_shapefile(default, title="Global Uncovered Land Areas")
