import os
import sys
import time
import argparse
import requests
import pandas as pd
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import folium
import branca.colormap as cm
import numpy as np


def geocode_location(location_name):
    """Resolve location to bounding box [min_lat, max_lat, min_lon, max_lon] using Nominatim."""
    print(f"Resolving location '{location_name}' using Nominatim Geocoding API...")
    url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json&limit=1"
    headers = {"User-Agent": "Geo-RAG-Dataset-Statistics"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200 and res.json():
            data = res.json()[0]
            bbox = [float(x) for x in data["boundingbox"]]  # [min_lat, max_lat, min_lon, max_lon]
            print(f" -> Found: {data['display_name']}")
            print(f" -> Bounding Box: Lat [{bbox[0]} to {bbox[1]}], Lon [{bbox[2]} to {bbox[3]}]")
            return bbox, data['display_name']
    except Exception as e:
        print(f"Warning: Geocoding failed: {e}")
    return None, None


def map_coordinates_to_regions(df, land_shp_path):
    """Map coordinates to countries and continents using a fast spatial join at H3 resolution 3."""
    if not os.path.exists(land_shp_path):
        print(f"Warning: Shapefile '{land_shp_path}' not found. Cannot map points to countries/continents.")
        df['continent'] = 'Unknown'
        df['country'] = 'Unknown'
        return df

    print("Mapping coordinates to countries/continents using H3 spatial index and shapefile...")
    t0 = time.time()

    # 1. Check if H3_Cell column exists
    if 'H3_Cell' in df.columns:
        unique_res11 = df['H3_Cell'].unique()
        # Convert unique res 11 cells to res 3 parents
        res11_to_res3 = {cell: h3.cell_to_parent(cell, 3) for cell in unique_res11}
        unique_res3 = list(set(res11_to_res3.values()))

        # Get centroids for the unique res 3 cells
        centroids = [h3.cell_to_latlng(cell) for cell in unique_res3]
        gdf_centroids = gpd.GeoDataFrame(
            {'h3_res3': unique_res3},
            geometry=gpd.points_from_xy([c[1] for c in centroids], [c[0] for c in centroids]),
            crs='EPSG:4326'
        )

        # Load countries shapefile
        countries = gpd.read_file(land_shp_path)
        col_mapping = {col: col.upper() for col in countries.columns if col.upper() in ['NAME', 'CONTINENT']}
        countries_subset = countries[list(col_mapping.keys()) + ['geometry']].rename(columns=col_mapping)

        # Spatial Join
        joined = gpd.sjoin(gdf_centroids, countries_subset, how='left', predicate='intersects')
        joined['CONTINENT'] = joined['CONTINENT'].fillna('Ocean / Unknown')
        joined['NAME'] = joined['NAME'].fillna('Ocean / Unknown')

        res3_to_continent = joined.set_index('h3_res3')['CONTINENT'].to_dict()
        res3_to_country = joined.set_index('h3_res3')['NAME'].to_dict()

        # Map back to main df
        res11_to_continent = {c11: res3_to_continent.get(c3, 'Ocean / Unknown') for c11, c3 in res11_to_res3.items()}
        res11_to_country = {c11: res3_to_country.get(c3, 'Ocean / Unknown') for c11, c3 in res11_to_res3.items()}

        df['continent'] = df['H3_Cell'].map(res11_to_continent).fillna('Ocean / Unknown')
        df['country'] = df['H3_Cell'].map(res11_to_country).fillna('Ocean / Unknown')
    else:
        # Fallback if H3_Cell column is not available: map unique coordinates
        print("H3_Cell column not found. Mapping using unique coordinate rounding fallback...")
        df_uniq = df[['Latitude', 'Longitude']].round(3).drop_duplicates().copy()
        df_uniq['h3_res3'] = df_uniq.apply(lambda r: h3.latlng_to_cell(r['Latitude'], r['Longitude'], 3), axis=1)

        unique_res3 = df_uniq['h3_res3'].unique().tolist()
        centroids = [h3.cell_to_latlng(cell) for cell in unique_res3]
        gdf_centroids = gpd.GeoDataFrame(
            {'h3_res3': unique_res3},
            geometry=gpd.points_from_xy([c[1] for c in centroids], [c[0] for c in centroids]),
            crs='EPSG:4326'
        )

        countries = gpd.read_file(land_shp_path)
        col_mapping = {col: col.upper() for col in countries.columns if col.upper() in ['NAME', 'CONTINENT']}
        countries_subset = countries[list(col_mapping.keys()) + ['geometry']].rename(columns=col_mapping)

        joined = gpd.sjoin(gdf_centroids, countries_subset, how='left', predicate='intersects')
        joined['CONTINENT'] = joined['CONTINENT'].fillna('Ocean / Unknown')
        joined['NAME'] = joined['NAME'].fillna('Ocean / Unknown')

        res3_to_continent = joined.set_index('h3_res3')['CONTINENT'].to_dict()
        res3_to_country = joined.set_index('h3_res3')['NAME'].to_dict()

        df_uniq['continent'] = df_uniq['h3_res3'].map(res3_to_continent)
        df_uniq['country'] = df_uniq['h3_res3'].map(res3_to_country)

        coord_map = df_uniq.set_index(['Latitude', 'Longitude'])[['continent', 'country']].to_dict('index')

        df_rounded = df[['Latitude', 'Longitude']].round(3)
        tuples = list(zip(df_rounded['Latitude'], df_rounded['Longitude']))
        df['continent'] = [coord_map.get(t, {'continent': 'Ocean / Unknown'})['continent'] for t in tuples]
        df['country'] = [coord_map.get(t, {'country': 'Ocean / Unknown'})['country'] for t in tuples]

    print(f" -> Mapping completed in {time.time() - t0:.2f}s.")
    return df


def load_dataset(file_path):
    """Load Parquet or CSV dataset efficiently by selecting only metadata columns."""
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' does not exist.")
        sys.exit(1)

    print(f"Loading dataset from '{file_path}'...")
    t0 = time.time()

    target_cols = [
        'Photo_ID', 'Platform', 'Latitude', 'Longitude',
        'Captured_At', 'Season', 'H3_Cell',
        'cluster_id', 'cluster_label', 'cluster_description',
        'parent_cluster_id', 'parent_cluster_label'
    ]

    if file_path.endswith('.csv'):
        # Check headers first
        sample_df = pd.read_csv(file_path, nrows=1)
        available_cols = sample_df.columns.tolist()
        cols_to_read = [c for c in target_cols if c in available_cols]
        df = pd.read_csv(file_path, usecols=cols_to_read, dtype={'Platform': str, 'Photo_ID': str})
    else:
        # Parquet
        parquet_file = pq.ParquetFile(file_path)
        available_cols = parquet_file.schema.names
        cols_to_read = [c for c in target_cols if c in available_cols]
        table = pq.read_table(file_path, columns=cols_to_read)
        df = table.to_pandas()

    print(f" -> Loaded {len(df)} records in {time.time() - t0:.2f}s.")
    return df


def ensure_time_of_day(df):
    """Classify Time of Day from Captured_At timestamp on the fly if missing."""
    if 'Time_Of_Day' in df.columns:
        df['Time_Of_Day'] = df['Time_Of_Day'].fillna('Unknown')
        return df

    if 'Captured_At' not in df.columns:
        df['Time_Of_Day'] = 'Unknown'
        return df

    print("Classifying Time of Day from Captured_At timestamps...")
    t0 = time.time()
    captured_series = df['Captured_At'].astype(str)

    try:
        hours = pd.to_numeric(captured_series.str[11:13], errors='coerce')
    except Exception:
        parsed_dates = pd.to_datetime(captured_series, errors='coerce', utc=True)
        hours = parsed_dates.dt.hour

    time_of_days = pd.Series(['Unknown'] * len(df), index=df.index, dtype=object)
    valid_hour_mask = hours.notna()

    time_of_days[valid_hour_mask & (hours >= 5) & (hours < 8)] = 'Dawn'
    time_of_days[valid_hour_mask & (hours >= 8) & (hours < 12)] = 'Morning'
    time_of_days[valid_hour_mask & (hours >= 12) & (hours < 17)] = 'Afternoon'
    time_of_days[valid_hour_mask & (hours >= 17) & (hours < 20)] = 'Dusk'
    time_of_days[valid_hour_mask & ((hours >= 20) | (hours < 5))] = 'Night'

    df['Time_Of_Day'] = time_of_days
    print(f" -> Classified Time of Day in {time.time() - t0:.2f}s.")
    return df


def generate_text_report(df, df_filtered, is_global, location_name):
    """Generate a formatted report string of the statistics, including definitions."""
    total_global = len(df)
    total_filtered = len(df_filtered)
    pct_global = (total_filtered / total_global) * 100 if total_global > 0 else 0

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append(f"📊 DATASET STATISTICS REPORT: {location_name.upper()}")
    lines.append("=" * 80)
    lines.append(f"📍 Location Status: {'Global Dataset' if is_global else 'Filtered Location'}")
    lines.append(f"📷 Total Images: {total_filtered:,} ({pct_global:.2f}% of global dataset)")
    lines.append("-" * 80)

    # 1. Platform Breakdown
    if 'Platform' in df_filtered.columns and total_filtered > 0:
        lines.append("\n🌐 PLATFORM BREAKDOWN:")
        platform_counts = df_filtered['Platform'].value_counts()
        for plat, count in platform_counts.items():
            pct = (count / total_filtered) * 100
            lines.append(f"  - {plat:<15}: {count:>10,} ({pct:>5.1f}%)")

    # 2. Continent Breakdown (only if global)
    if is_global and 'continent' in df_filtered.columns and total_filtered > 0:
        lines.append("\n🌍 CONTINENT BREAKDOWN:")
        continent_counts = df_filtered['continent'].value_counts()
        for cont, count in continent_counts.items():
            pct = (count / total_filtered) * 100
            lines.append(f"  - {cont:<15}: {count:>10,} ({pct:>5.1f}%)")

    # 3. Country Breakdown (if not global, or if it's a continent)
    if not is_global and 'country' in df_filtered.columns and total_filtered > 0:
        countries = df_filtered['country'].value_counts()
        if len(countries) > 1:
            lines.append("\n🏳️  TOP COUNTRIES IN THIS AREA:")
            for country, count in countries.head(10).items():
                pct = (count / total_filtered) * 100
                lines.append(f"  - {country:<25}: {count:>10,} ({pct:>5.1f}%)")

    # 4. Time of Day Breakdown
    if 'Time_Of_Day' in df_filtered.columns and total_filtered > 0:
        lines.append("\n⏰ TIME OF DAY DISTRIBUTION:")
        tod_counts = df_filtered['Time_Of_Day'].value_counts()
        for tod, count in tod_counts.items():
            pct = (count / total_filtered) * 100
            lines.append(f"  - {tod:<15}: {count:>10,} ({pct:>5.1f}%)")

    # 5. Season Breakdown
    if 'Season' in df_filtered.columns and total_filtered > 0:
        lines.append("\n🍂 SEASONAL DISTRIBUTION:")
        season_counts = df_filtered['Season'].value_counts()
        for season, count in season_counts.items():
            pct = (count / total_filtered) * 100
            lines.append(f"  - {season:<15}: {count:>10,} ({pct:>5.1f}%)")

    # 6. Top Cluster Labels (if present)
    if 'cluster_label' in df_filtered.columns and total_filtered > 0:
        lines.append("\n🏷️  TOP 10 SEMANTIC CLUSTER LABELS:")
        cluster_counts = df_filtered['cluster_label'].value_counts().head(10)
        for idx, (label, count) in enumerate(cluster_counts.items(), 1):
            pct = (count / total_filtered) * 100
            lines.append(f"  {idx:>2}. {label[:50]:<50}: {count:>8,} ({pct:>4.1f}%)")

    # 7. Coordinate extent
    if total_filtered > 0:
        min_lat = df_filtered['Latitude'].min()
        max_lat = df_filtered['Latitude'].max()
        min_lon = df_filtered['Longitude'].min()
        max_lon = df_filtered['Longitude'].max()
        lines.append("\n🌐 GEOGRAPHIC EXTENT:")
        lines.append(f"  - Latitude range : [{min_lat:.6f} to {max_lat:.6f}]")
        lines.append(f"  - Longitude range: [{min_lon:.6f} to {max_lon:.6f}]")

    # 8. Category Definitions
    lines.append("\n" + "-" * 80)
    lines.append("📖 DEFINITION OF CATEGORIES:")
    lines.append("-" * 80)
    lines.append("⏰ Time of Day Classifications:")
    lines.append("  - Dawn      : 05:00 to 07:59 (local time hour)")
    lines.append("  - Morning   : 08:00 to 11:59")
    lines.append("  - Afternoon : 12:00 to 16:59")
    lines.append("  - Dusk      : 17:00 to 19:59")
    lines.append("  - Night     : 20:00 to 04:59")
    lines.append("\n🍂 Seasonal Classifications (based on month & Latitude):")
    lines.append("  - Tropical Zone (Latitude between -23.5° and 23.5°):")
    lines.append("    * Wet Season: June, July, August, September")
    lines.append("    * Dry Season: October to May")
    lines.append("  - Northern Temperate/Polar Zone (Latitude > 23.5°):")
    lines.append("    * Spring: March, April, May")
    lines.append("    * Summer: June, July, August")
    lines.append("    * Autumn: September, October, November")
    lines.append("    * Winter: December, January, February")
    lines.append("  - Southern Temperate/Polar Zone (Latitude < -23.5°):")
    lines.append("    * Autumn: March, April, May")
    lines.append("    * Winter: June, July, August")
    lines.append("    * Spring: September, October, November")
    lines.append("    * Summer: December, January, February")
    lines.append("=" * 80 + "\n")

    return "\n".join(lines)


def generate_plots(df_filtered, is_global, location_name, output_path):
    """Generate a 2x2 multi-panel plot for visualization."""
    if len(df_filtered) == 0:
        print("Warning: No records found. Skipping plot generation.")
        return

    print("Generating statistics plots...")
    t0 = time.time()

    # Set modern style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Dataset Statistics: {location_name}", fontsize=18, fontweight='bold', y=0.98)

    # --- Subplot 1: Spatial Breakdown ---
    ax1 = axes[0, 0]
    plotted_counts = None
    if is_global:
        # Global: Continent breakdown
        plotted_counts = df_filtered['continent'].value_counts()
        sns.barplot(x=plotted_counts.values, y=plotted_counts.index, ax=ax1, palette="viridis", hue=plotted_counts.index, legend=False)
        ax1.set_title("Distribution by Continent", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Image Count")
    elif location_name.lower() in ['africa', 'europe', 'asia', 'north america', 'south america', 'oceania', 'antarctica', 'australia']:
        # Continent: Country breakdown
        counts = df_filtered['country'].value_counts()
        plotted_counts = counts.head(10)
        sns.barplot(x=plotted_counts.values, y=plotted_counts.index, ax=ax1, palette="viridis", hue=plotted_counts.index, legend=False)
        ax1.set_title(f"Top 10 Countries in {location_name}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Image Count")
    else:
        # Country/City: H3 cell breakdown
        if 'H3_Cell' in df_filtered.columns:
            counts = df_filtered['H3_Cell'].value_counts()
            plotted_counts = counts.head(10)
            sns.barplot(x=plotted_counts.values, y=plotted_counts.index, ax=ax1, palette="viridis", hue=plotted_counts.index, legend=False)
            ax1.set_title("Top 10 H3 Cells (Resolution 11)", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Image Count")
        else:
            ax1.text(0.5, 0.5, "H3_Cell column not available", ha='center', va='center')
            ax1.set_title("Spatial Breakdown (Unavailable)", fontsize=14, fontweight='bold')

    # Add text labels for horizontal bars in Subplot 1
    if plotted_counts is not None and len(plotted_counts) > 0:
        max_val = plotted_counts.values.max() if len(plotted_counts) > 0 else 1
        for i, v in enumerate(plotted_counts.values):
            # Print label slightly offset to the right of the bar
            ax1.text(v + (max_val * 0.01), i, f" {v:,}", va='center', fontweight='bold', fontsize=9)
        # Expand x limit slightly to prevent label clipping
        ax1.set_xlim(0, max_val * 1.15)

    # --- Subplot 2: Platform Breakdown ---
    ax2 = axes[0, 1]
    if 'Platform' in df_filtered.columns:
        counts = df_filtered['Platform'].value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax2, palette="muted", hue=counts.index, legend=False)
        ax2.set_title("Distribution by Platform", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Image Count")
        max_val = counts.values.max() if len(counts) > 0 else 1
        for i, v in enumerate(counts.values):
            ax2.text(i, v + (max_val * 0.01), f"{v:,}", ha='center', fontweight='bold', fontsize=9)
        ax2.set_ylim(0, max_val * 1.1)
    else:
        ax2.text(0.5, 0.5, "Platform column not available", ha='center', va='center')
        ax2.set_title("Platform Breakdown (Unavailable)", fontsize=14, fontweight='bold')

    # --- Subplot 3: Time of Day Distribution ---
    ax3 = axes[1, 0]
    if 'Time_Of_Day' in df_filtered.columns:
        tod_order = ['Dawn', 'Morning', 'Afternoon', 'Dusk', 'Night', 'Unknown']
        counts = df_filtered['Time_Of_Day'].value_counts()
        existing_order = [x for x in tod_order if x in counts.index]
        counts = counts.reindex(existing_order)

        sns.barplot(x=counts.index, y=counts.values, ax=ax3, palette="magma", hue=counts.index, legend=False)
        ax3.set_title("Distribution by Time of Day", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Image Count")
        max_val = counts.values.max() if len(counts) > 0 else 1
        for i, v in enumerate(counts.values):
            ax3.text(i, v + (max_val * 0.01), f"{v:,}", ha='center', fontweight='bold', fontsize=9)
        ax3.set_ylim(0, max_val * 1.1)
    else:
        ax3.text(0.5, 0.5, "Time of Day not available", ha='center', va='center')
        ax3.set_title("Time of Day Distribution (Unavailable)", fontsize=14, fontweight='bold')

    # --- Subplot 4: Season Distribution ---
    ax4 = axes[1, 1]
    if 'Season' in df_filtered.columns:
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter', 'Wet Season', 'Dry Season', 'Unknown']
        counts = df_filtered['Season'].value_counts()
        existing_order = [x for x in season_order if x in counts.index]
        counts = counts.reindex(existing_order)

        sns.barplot(x=counts.index, y=counts.values, ax=ax4, palette="coolwarm", hue=counts.index, legend=False)
        ax4.set_title("Distribution by Season", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Image Count")
        max_val = counts.values.max() if len(counts) > 0 else 1
        for i, v in enumerate(counts.values):
            ax4.text(i, v + (max_val * 0.01), f"{v:,}", ha='center', fontweight='bold', fontsize=9)
        ax4.set_ylim(0, max_val * 1.1)
    else:
        ax4.text(0.5, 0.5, "Season not available", ha='center', va='center')
        ax4.set_title("Season Distribution (Unavailable)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" -> Saved statistics plot to: {os.path.abspath(output_path)}")
    plt.close()


def generate_interactive_map(df_filtered, location_name, output_html_path):
    """Generate a multi-layered interactive Folium map centered on the filtered data."""
    if len(df_filtered) == 0:
        print("Warning: No records found. Skipping map generation.")
        return

    print("Generating interactive H3 map...")
    t0 = time.time()
    
    is_global = (location_name.lower() == "global dataset")
    
    # 1. Determine bounding box
    min_lat, max_lat = df_filtered['Latitude'].min(), df_filtered['Latitude'].max()
    min_lon, max_lon = df_filtered['Longitude'].min(), df_filtered['Longitude'].max()
    
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    
    # 2. Determine target resolution dynamically
    if lat_span > 100 or lon_span > 100:
        target_res = 4
    elif lat_span > 30 or lon_span > 30:
        target_res = 5
    elif lat_span > 5 or lon_span > 5:
        target_res = 6
    elif lat_span > 1 or lon_span > 1:
        target_res = 7
    else:
        target_res = 8
        
    print(f" -> Selected H3 resolution {target_res} based on latitude span ({lat_span:.2f}°) and longitude span ({lon_span:.2f}°)")
    
    # 3. Compute parent H3 cells at target resolution
    if 'H3_Cell' in df_filtered.columns:
        unique_res11 = df_filtered['H3_Cell'].unique()
        res11_to_parent = {c: h3.cell_to_parent(c, target_res) if h3.get_resolution(c) != target_res else c for c in unique_res11}
        df_filtered['map_h3'] = df_filtered['H3_Cell'].map(res11_to_parent)
    else:
        df_filtered['map_h3'] = df_filtered.apply(lambda r: h3.latlng_to_cell(r['Latitude'], r['Longitude'], target_res), axis=1)
        
    # Pre-aggregate breakdowns per H3 cell using fast groupby-unstack (highly optimized compared to looping/string-convs)
    platform_counts = pd.DataFrame()
    tod_counts = pd.DataFrame()
    season_counts = pd.DataFrame()
    
    if 'Platform' in df_filtered.columns:
        platform_counts = df_filtered.groupby(['map_h3', 'Platform']).size().unstack(fill_value=0)
    if 'Time_Of_Day' in df_filtered.columns:
        tod_counts = df_filtered.groupby(['map_h3', 'Time_Of_Day']).size().unstack(fill_value=0)
    if 'Season' in df_filtered.columns:
        season_counts = df_filtered.groupby(['map_h3', 'Season']).size().unstack(fill_value=0)
        
    # 4. Initialize Folium Map
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB Positron")
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    
    def get_clean_boundary(cell):
        coords = h3.cell_to_boundary(cell)
        lngs = [c[1] for c in coords]
        if max(lngs) - min(lngs) > 180:
            coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
        return coords

    def add_layer(category_col, value, layer_name, theme):
        if category_col:
            df_sub = df_filtered[df_filtered[category_col].astype(str).str.lower() == str(value).lower()]
        else:
            df_sub = df_filtered
            
        if len(df_sub) == 0:
            return
            
        counts = df_sub['map_h3'].value_counts()
        
        # Optimization: Apply a dynamic min_count filter to avoid map bloat from sparse/noisy cells (e.g. cells with 1 or 2 images)
        min_c = 1
        if len(counts) > 2000:
            if is_global:
                # For global maps, only show cells with >= 10 images in sublayers, or >= 5 in overall density
                min_c = 10 if category_col else 5
            else:
                # For regional maps, show cells with >= 3 images in sublayers, or >= 2 in overall density
                min_c = 3 if category_col else 2
        
        counts = counts[counts >= min_c]
        
        # Cap to keep Leaflet rendering smooth
        if len(counts) > 15000:
            counts = counts.head(15000)
            
        if len(counts) == 0:
            return
            
        min_val = np.log10(counts.min())
        max_val = np.log10(counts.max())
        if min_val == max_val:
            max_val += 0.1
            
        if theme == 'magma':
            colormap = cm.linear.magma.scale(min_val, max_val)
        elif theme == 'viridis':
            colormap = cm.linear.viridis.scale(min_val, max_val)
        elif theme == 'red':
            colormap = cm.linear.YlOrRd_09.scale(min_val, max_val)
        elif theme == 'blue':
            colormap = cm.linear.PuBu_09.scale(min_val, max_val)
        elif theme == 'green':
            colormap = cm.linear.YlGn_09.scale(min_val, max_val)
        elif theme == 'purple':
            colormap = cm.linear.RdPu_09.scale(min_val, max_val)
        elif theme == 'orange':
            colormap = cm.linear.YlOrBr_09.scale(min_val, max_val)
        else:
            colormap = cm.linear.YlOrRd_09.scale(min_val, max_val)
            
        features = []
        for cell, count in counts.items():
            try:
                boundary = get_clean_boundary(cell)
                geojson_coords = [[lng, lat] for lat, lng in boundary]
                geojson_coords.append(geojson_coords[0])
                
                properties = {
                    "cell": cell,
                    "count": int(count),
                    "log_count": float(np.log10(count))
                }
                
                # Enrich with pre-aggregated counts for the overall base layer
                if category_col is None:
                    if not platform_counts.empty and cell in platform_counts.index:
                        p_row = platform_counts.loc[cell]
                        properties["platforms"] = " | ".join([f"{k}: {v:,} ({v/count*100:.0f}%)" for k, v in p_row.items() if v > 0])
                    else:
                        properties["platforms"] = "N/A"
                        
                    if not tod_counts.empty and cell in tod_counts.index:
                        t_row = tod_counts.loc[cell]
                        properties["time_of_day"] = " | ".join([f"{k}: {v:,} ({v/count*100:.0f}%)" for k, v in t_row.items() if v > 0])
                    else:
                        properties["time_of_day"] = "N/A"
                        
                    if not season_counts.empty and cell in season_counts.index:
                        s_row = season_counts.loc[cell]
                        properties["seasons"] = " | ".join([f"{k}: {v:,} ({v/count*100:.0f}%)" for k, v in s_row.items() if v > 0])
                    else:
                        properties["seasons"] = "N/A"
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [geojson_coords]
                    },
                    "properties": properties
                }
                features.append(feature)
            except Exception:
                continue
                
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        fg = folium.FeatureGroup(name=layer_name, overlay=True, show=(category_col is None))
        
        def style_fn(f):
            log_c = f["properties"]["log_count"]
            color = colormap(log_c)
            return {
                "fillColor": color,
                "color": color,
                "weight": 1,
                "fillOpacity": 0.6,
            }
            
        if category_col is None:
            # Base layer has rich, unified tooltips
            tooltip_fields = ["cell", "count", "platforms", "time_of_day", "seasons"]
            tooltip_aliases = ["H3 Cell:", "Total Images:", "Platforms:", "Time of Day:", "Seasons:"]
        else:
            # Sublayers only show specific count
            tooltip_fields = ["cell", "count"]
            tooltip_aliases = ["H3 Cell:", "Count:"]
            
        folium.GeoJson(
            geojson_data,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True
            )
        ).add_to(fg)
        
        fg.add_to(m)

    # 1. Base Layer (always rendered with rich tooltip)
    add_layer(None, None, "Overall Image Density", "magma")
    
    # 2. Render sublayers only if not global to prevent massive browser lag and large file sizes (e.g. 70MB+)
    if not is_global:
        # Platform Layers
        if 'Platform' in df_filtered.columns:
            platforms = df_filtered['Platform'].dropna().unique()
            themes = {'flickr': 'blue', 'mapillary': 'green', 'inaturalist': 'viridis'}
            for plat in platforms:
                theme = themes.get(plat.lower(), 'red')
                add_layer('Platform', plat, f"Platform: {plat}", theme)
                
        # Time of Day Layers
        if 'Time_Of_Day' in df_filtered.columns:
            tods = df_filtered['Time_Of_Day'].dropna().unique()
            themes = {'dawn': 'purple', 'morning': 'green', 'afternoon': 'red', 'dusk': 'orange', 'night': 'magma'}
            for tod in tods:
                if tod == 'Unknown':
                    continue
                theme = themes.get(tod.lower(), 'red')
                add_layer('Time_Of_Day', tod, f"Time: {tod}", theme)
                
        # Season Layers
        if 'Season' in df_filtered.columns:
            seasons = df_filtered['Season'].dropna().unique()
            themes = {'spring': 'green', 'summer': 'red', 'autumn': 'orange', 'winter': 'blue', 'wet season': 'blue', 'dry season': 'red'}
            for season in seasons:
                if season == 'Unknown':
                    continue
                theme = themes.get(season.lower(), 'red')
                add_layer('Season', season, f"Season: {season}", theme)
                
        # Layer Control (only needed when multiple toggleable layers are rendered)
        folium.LayerControl(collapsed=False).add_to(m)
    
    # Ensure directory exists and save
    output_dir = os.path.dirname(os.path.abspath(output_html_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    m.save(output_html_path)
    print(f" -> Saved interactive map to: {os.path.abspath(output_html_path)}")


def main():
    parser = argparse.ArgumentParser(description="Generate and plot dataset statistics globally or for a specific location.")
    parser.add_argument("--input", type=str, default="full_pipeline_output/geo_space_deduplicated.parquet",
                        help="Path to the input Parquet or CSV dataset.")
    parser.add_argument("--location", type=str, default=None,
                        help="Name of location to filter by (continent, country, or city). If omitted, displays global stats.")
    parser.add_argument("--output_plot", type=str, default=None,
                        help="Path to save the generated plot (.png). Defaults to {location}_stats.png.")
    parser.add_argument("--output_text", type=str, default=None,
                        help="Path to save the text report/logs (.txt). Defaults to {location}_stats.txt.")
    parser.add_argument("--output_map", type=str, default=None,
                        help="Path to save the interactive HTML map. Defaults to {location}_map.html.")
    parser.add_argument("--land_shp", type=str, default="ne_10m_admin_0_countries.shp",
                        help="Path to the country shapefile for spatial region mapping.")
    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.input)

    # Classify time of day if needed
    df = ensure_time_of_day(df)

    # Map points to continent/country
    df = map_coordinates_to_regions(df, args.land_shp)

    # Handle filtering
    is_global = True
    location_name = "Global Dataset"
    df_filtered = df

    if args.location:
        loc_clean = args.location.strip().lower()
        continent_map = {
            'africa': 'Africa',
            'europe': 'Europe',
            'asia': 'Asia',
            'north america': 'North America',
            'south america': 'South America',
            'oceania': 'Oceania',
            'australia': 'Oceania',
            'antarctica': 'Antarctica'
        }

        # 1. Check if it's a continent
        if loc_clean in continent_map:
            continent_target = continent_map[loc_clean]
            df_filtered = df[df['continent'].str.lower() == continent_target.lower()]
            is_global = False
            location_name = continent_target
            print(f"Filtered dataset by continent '{continent_target}': kept {len(df_filtered):,} records.")

        # 2. Check if it matches a country in the dataset (case-insensitive)
        elif 'country' in df.columns and loc_clean in [c.lower() for c in df['country'].unique()]:
            actual_country = next(c for c in df['country'].unique() if c.lower() == loc_clean)
            df_filtered = df[df['country'].str.lower() == loc_clean]
            is_global = False
            location_name = actual_country
            print(f"Filtered dataset by country '{actual_country}': kept {len(df_filtered):,} records.")

        # 3. Fallback to Nominatim geocoding bounding box
        else:
            bbox, display_name = geocode_location(args.location)
            if bbox:
                min_lat, max_lat, min_lon, max_lon = bbox
                df_filtered = df[
                    (df['Latitude'] >= min_lat) & (df['Latitude'] <= max_lat) &
                    (df['Longitude'] >= min_lon) & (df['Longitude'] <= max_lon)
                ]
                is_global = False
                location_name = display_name if display_name else args.location
                print(f"Filtered dataset by bounding box for '{args.location}': kept {len(df_filtered):,} records.")
            else:
                print(f"Error: Could not resolve location '{args.location}'. Falling back to global statistics.")

    # Determine plot output path
    safe_loc = "".join([c if c.isalnum() else "_" for c in location_name]).strip("_").lower()
    
    if args.output_plot:
        plot_path = args.output_plot
    else:
        plot_path = f"{safe_loc}_stats.png"

    # Determine text output path
    if args.output_text:
        text_path = args.output_text
    else:
        text_path = f"{safe_loc}_stats.txt"

    # Determine map output path
    if args.output_map:
        map_path = args.output_map
    else:
        map_path = f"{safe_loc}_map.html"

    # Generate and print/save report
    report_str = generate_text_report(df, df_filtered, is_global, location_name)
    print(report_str)
    
    # Save text report to file
    try:
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f" -> Saved statistics report text to: {os.path.abspath(text_path)}")
    except Exception as e:
        print(f"Warning: Failed to save text report to '{text_path}': {e}")

    # Generate plots
    generate_plots(df_filtered, is_global, location_name, plot_path)

    # Generate interactive map
    generate_interactive_map(df_filtered, location_name, map_path)


if __name__ == "__main__":
    main()
