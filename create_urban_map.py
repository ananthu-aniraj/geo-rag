import geopandas as gpd
import matplotlib.pyplot as plt
import os
import folium

def create_urban_map(shp_path, output_image, output_html):
    """
    Creates static and interactive maps of urban areas from a shapefile.
    """
    print(f"Loading shapefile: {shp_path}...")
    try:
        # Load the urban areas shapefile
        # Note: geopandas reads .shp and automatically picks up .dbf, .shx etc. in same dir
        urban_areas = gpd.read_file(shp_path, engine='pyogrio')
        print(f"Loaded {len(urban_areas)} urban areas.")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return

    # --- 1. Static Map (PNG) ---
    print("Generating static plot...")
    # Use a larger figure size for global detail
    fig, ax = plt.subplots(figsize=(24, 12), facecolor='black')
    ax.set_facecolor('black')
    
    # Plot the urban areas in a bright color to stand out on black
    urban_areas.plot(ax=ax, color='cyan', edgecolor='cyan', linewidth=0.5, alpha=0.8)
    
    # Styling
    plt.title('Natural Earth Global Urban Areas', color='white', size=20, pad=20)
    plt.xlabel('Longitude', color='white', size=14)
    plt.ylabel('Latitude', color='white', size=14)
    
    # Standard map extent
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    
    plt.xticks(np.arange(-180, 181, 30), color='white')
    plt.yticks(np.arange(-90, 91, 30), color='white')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Static map saved as '{output_image}'")

    # --- 2. Interactive Map (Folium HTML) ---
    print("Generating interactive Folium map...")
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
    
    # Simplify geometry for faster rendering in Folium if necessary
    # Urban areas can be very detailed; simplify slightly to keep HTML size manageable
    simplified_urban = urban_areas.copy()
    simplified_urban['geometry'] = simplified_urban['geometry'].simplify(0.01)
    
    # Convert to GeoJSON and add to map
    folium.GeoJson(
        simplified_urban,
        name="Urban Areas",
        style_function=lambda x: {
            'fillColor': '#00ffff',
            'color': '#00ffff',
            'weight': 1,
            'fillOpacity': 0.6,
        },
        tooltip=folium.GeoJsonTooltip(fields=['name_conve'], aliases=['Name:']) if 'name_conve' in simplified_urban.columns else None
    ).add_to(m)
    
    m.save(output_html)
    print(f"Interactive map saved as '{output_html}'")

if __name__ == "__main__":
    import numpy as np # Needed for ticks in script
    
    shp_file = "Projects/code/geo-rag/ne_10m_urban_areas.shp"
    output_png = "Projects/code/geo-rag/global_urban_areas.png"
    output_html = "Projects/code/geo-rag/global_urban_areas.html"
    
    if os.path.exists(shp_file):
        create_urban_map(shp_file, output_png, output_html)
    else:
        print(f"Error: Could not find {shp_file}")
