import folium
import pandas as pd
import math
import matplotlib.cm as cm
import matplotlib.colors as colors

# --- 1. Load the Attached Data ---
# Ensure the CSV file is in the same folder as this script
df = pd.read_csv("/user/aaniraj/home/Documents/Projects/data/cities688.csv")

# --- 2. Color Scale Setup (Logarithmic) ---
# We calculate the base-10 logarithm for the min and max image counts
min_log = math.log10(df['img_count'].min())
max_log = math.log10(df['img_count'].max())

# Create a colormap transitioning from faint yellow to deep red
colormap = colors.LinearSegmentedColormap.from_list("density", ["#ffeda0", "#feb24c", "#f03b20"])


def get_color(count):
    if count <= 0:
        return None
    # Normalize the logarithmic count between 0 and 1
    log_count = math.log10(count)
    norm_value = (log_count - min_log) / (max_log - min_log)
    rgba = colormap(norm_value)
    return colors.to_hex(rgba)


# --- 3. Initialize the Map ---
# Centered roughly on a global view (Latitude 20, Longitude 0) with a zoomed-out view
m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

# --- 4. Draw the 5x5 km Squares ---
print("Generating 5x5 km squares for 688 cities...")

for idx, row in df.iterrows():
    lat = row['city_lat']
    lon = row['city_lon']
    count = row['img_count']
    city_name = row['city']
    country = row['country']

    # Calculate the degree distances for a 5x5 km square
    lat_step = 5 / 111.32

    # Avoid math errors at extreme latitudes
    cos_lat = math.cos(math.radians(lat))
    if cos_lat == 0:
        cos_lat = 0.0001

    lon_step = 5 / (111.32 * cos_lat)

    # Center the square exactly over the city coordinates
    min_lat = lat - lat_step / 2
    max_lat = lat + lat_step / 2
    min_lon = lon - lon_step / 2
    max_lon = lon + lon_step / 2

    color = get_color(count)
    if not color:
        continue

    bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # Create an interactive hover tooltip
    tooltip_text = f"<b>{city_name}, {country}</b><br>Images Collected: {int(count):,}"

    # Draw the rectangle onto the map
    folium.Rectangle(
        bounds=bounds,
        color=color,
        weight=1,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        tooltip=tooltip_text
    ).add_to(m)

# --- 5. Save the Interactive Map ---
output_file = "city_occupancy_map.html"
m.save(output_file)
print(f"Finished! Interactive map saved as '{output_file}'")
