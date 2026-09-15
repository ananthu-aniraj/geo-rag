import argparse
import os
import sys
import time

import geopandas as gpd
import h3
import pandas as pd
import pyarrow.parquet as pq
import requests

from src.visualization.visualize_dataset_stats import (
    generate_grouped_platform_plot,
    generate_interactive_map,
    generate_plots,
    get_grouped_platform_series,
)


def geocode_location(location_name):
    """Resolve location to bounding box [min_lat, max_lat, min_lon, max_lon] using Nominatim."""
    print(f"Resolving location '{location_name}' using Nominatim Geocoding API...")
    url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json&limit=1"
    headers = {"User-Agent": "Geo-RAG-Dataset-Statistics"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200 and res.json():
            data = res.json()[0]
            bbox = [
                float(x) for x in data["boundingbox"]
            ]  # [min_lat, max_lat, min_lon, max_lon]
            print(f" -> Found: {data['display_name']}")
            print(
                f" -> Bounding Box: Lat [{bbox[0]} to {bbox[1]}], Lon [{bbox[2]} to {bbox[3]}]"
            )
            return bbox, data["display_name"]
    except Exception as e:
        print(f"Warning: Geocoding failed: {e}")
    return None, None


def map_coordinates_to_regions(
    df, land_shp_path, spatial_index_path=None, target_res=8
):
    """Map coordinates to countries and continents using H3 cells and nearest-land fallback."""
    if "country" in df.columns and "continent" in df.columns:
        if not df["country"].isna().any() and not (df["country"] == "Unknown").any():
            print(
                "Country and continent columns already present and populated. Skipping spatial region mapping."
            )
            return df

    if not os.path.exists(land_shp_path):
        print(
            f"Warning: Shapefile '{land_shp_path}' not found. Cannot map points to countries/continents."
        )
        if "continent" not in df.columns:
            df["continent"] = "Unknown"
        if "country" not in df.columns:
            df["country"] = "Unknown"
        return df

    print(
        f"Mapping coordinates to countries/continents using H3 resolution {target_res} cells & shapefile..."
    )
    t0 = time.time()

    # Load shapefile once
    countries = gpd.read_file(land_shp_path)
    col_mapping = {
        col: col.upper()
        for col in countries.columns
        if col.upper() in ["NAME", "CONTINENT"]
    }
    countries_subset = countries[list(col_mapping.keys()) + ["geometry"]].rename(
        columns=col_mapping
    )

    # 1. Determine unique res 5 H3 cells
    unique_target_res = set()

    # If pre-built spatial index exists, load res 5 query cells to accelerate/prime mapping
    if spatial_index_path and os.path.exists(spatial_index_path):
        print(f" -> Loading pre-built spatial index from '{spatial_index_path}'...")
        try:
            index_parquet = pq.ParquetFile(spatial_index_path)
            avail = index_parquet.schema_arrow.names
            if "resolution" in avail and "query_cell" in avail:
                table = pq.read_table(
                    spatial_index_path,
                    columns=["resolution", "query_cell"],
                    filters=[("resolution", "==", target_res)],
                )
                idx_query_cells = table.to_pandas()["query_cell"].dropna().unique()
                unique_target_res.update(idx_query_cells)
                print(
                    f" -> Found {len(idx_query_cells):,} pre-indexed H3 resolution {target_res} cells."
                )
        except Exception as e:
            print(f" -> Warning: Could not read spatial index: {e}")

    # Also extract cells from input df to ensure complete coverage
    cell_to_parent_cell = {}
    if "H3_Cell" in df.columns:
        unique_cells = df["H3_Cell"].dropna().unique()
        cell_to_parent_cell = {
            c: h3.cell_to_parent(c, target_res)
            if h3.get_resolution(c) >= target_res
            else c
            for c in unique_cells
        }
        unique_target_res.update(cell_to_parent_cell.values())
    else:
        print("H3_Cell column not found. Deriving H3 cells from coordinates...")
        df_coords = df[["Latitude", "Longitude"]].dropna().drop_duplicates()
        df_coords["h3_query_cell"] = [
            h3.latlng_to_cell(lat, lon, target_res)
            for lat, lon in zip(df_coords["Latitude"], df_coords["Longitude"])
        ]
        unique_target_res.update(df_coords["h3_query_cell"].unique())

    unique_query_cells = list(unique_target_res)
    print(
        f" -> Mapping {len(unique_query_cells):,} unique H3 resolution {target_res} cells against country boundaries..."
    )

    # Build GeoDataFrame of cell centroids
    centroids = [h3.cell_to_latlng(cell) for cell in unique_query_cells]
    gdf_centroids = gpd.GeoDataFrame(
        {"h3_cell": unique_query_cells},
        geometry=gpd.points_from_xy(
            [c[1] for c in centroids], [c[0] for c in centroids]
        ),
        crs="EPSG:4326",
    )

    # Step 1: Primary Spatial Join (Intersects)
    joined = gpd.sjoin(
        gdf_centroids, countries_subset, how="left", predicate="intersects"
    )

    # Identify coastal water cells or unmapped cells
    invalid_mask = joined["CONTINENT"].isna() | (
        joined["CONTINENT"] == "Seven seas (open ocean)"
    )
    unmatched = joined[invalid_mask].copy()
    matched = joined[~invalid_mask].copy()

    # Step 2: Nearest-Land Snapping for Coastal Water Cells (max 0.8 degrees / ~88 km)
    if len(unmatched) > 0:
        unmatched_clean = unmatched[["h3_cell", "geometry"]].copy()

        # Project both to EPSG:3857 (Web Mercator, units in meters) to calculate distances accurately and avoid warnings
        unmatched_projected = unmatched_clean.to_crs("EPSG:3857")
        countries_projected = countries_subset.to_crs("EPSG:3857")

        # 0.8 degrees is approx 88,800 meters
        nearest = gpd.sjoin_nearest(
            unmatched_projected, countries_projected, how="left", max_distance=88800
        )

        # Project back to EPSG:4326 (degrees) before concat to match matched CRS
        nearest = nearest.to_crs("EPSG:4326")

        nearest["CONTINENT"] = nearest["CONTINENT"].fillna("Ocean / Unknown")
        nearest["NAME"] = nearest["NAME"].fillna("Ocean / Unknown")
        nearest.loc[nearest["CONTINENT"] == "Seven seas (open ocean)", "CONTINENT"] = (
            "Ocean / Unknown"
        )
        nearest = nearest.drop_duplicates(subset=["h3_cell"])
        final_gdf = pd.concat([matched, nearest], ignore_index=True)
    else:
        final_gdf = joined

    query_cell_to_continent = final_gdf.set_index("h3_cell")["CONTINENT"].to_dict()
    query_cell_to_country = final_gdf.set_index("h3_cell")["NAME"].to_dict()

    # Assign mapped regions back to main dataframe
    if "H3_Cell" in df.columns:
        cell_to_continent = {
            c11: query_cell_to_continent.get(c_parent, "Ocean / Unknown")
            for c11, c_parent in cell_to_parent_cell.items()
        }
        cell_to_country = {
            c11: query_cell_to_country.get(c_parent, "Ocean / Unknown")
            for c11, c_parent in cell_to_parent_cell.items()
        }

        df["continent"] = df["H3_Cell"].map(cell_to_continent).fillna("Ocean / Unknown")
        df["country"] = df["H3_Cell"].map(cell_to_country).fillna("Ocean / Unknown")
    else:
        # Fallback coordinate mapping
        df_coords["continent"] = (
            df_coords["h3_query_cell"]
            .map(query_cell_to_continent)
            .fillna("Ocean / Unknown")
        )
        df_coords["country"] = (
            df_coords["h3_query_cell"]
            .map(query_cell_to_country)
            .fillna("Ocean / Unknown")
        )
        coord_map = df_coords.set_index(["Latitude", "Longitude"])[
            ["continent", "country"]
        ].to_dict("index")

        tuples = list(zip(df["Latitude"], df["Longitude"]))
        df["continent"] = [
            coord_map.get(t, {"continent": "Ocean / Unknown"})["continent"]
            for t in tuples
        ]
        df["country"] = [
            coord_map.get(t, {"country": "Ocean / Unknown"})["country"] for t in tuples
        ]

    print(f" -> High-quality region mapping completed in {time.time() - t0:.2f}s.")
    return df


def load_dataset(file_path):
    """Load Parquet or CSV dataset efficiently by selecting only metadata columns."""
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' does not exist.")
        sys.exit(1)

    print(f"Loading dataset from '{file_path}'...")
    t0 = time.time()

    target_cols = [
        "Photo_ID",
        "Platform",
        "Latitude",
        "Longitude",
        "Captured_At",
        "Season",
        "H3_Cell",
        "cluster_id",
        "cluster_label",
        "cluster_description",
        "parent_cluster_id",
        "parent_cluster_label",
        "Koppen_Code",
        "Koppen_Desc",
    ]

    if file_path.endswith(".csv"):
        # Check headers first
        sample_df = pd.read_csv(file_path, nrows=1)
        available_cols = sample_df.columns.tolist()
        cols_to_read = [c for c in target_cols if c in available_cols]
        df = pd.read_csv(
            file_path, usecols=cols_to_read, dtype={"Platform": str, "Photo_ID": str}
        )
    else:
        # Parquet
        from src.utils.io import load_dataset_with_clusters

        try:
            df = load_dataset_with_clusters(file_path, columns=target_cols)
        except Exception:
            parquet_file = pq.ParquetFile(file_path)
            available_cols = parquet_file.schema_arrow.names
            cols_to_read = [c for c in target_cols if c in available_cols]
            table = pq.read_table(file_path, columns=cols_to_read)
            df = table.to_pandas()

    print(f" -> Loaded {len(df)} records in {time.time() - t0:.2f}s.")
    if "Latitude" in df.columns:
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    if "Longitude" in df.columns:
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df


def ensure_time_of_day(df):
    """Classify Time of Day from Captured_At timestamp on the fly if missing."""
    if "Time_Of_Day" in df.columns:
        df["Time_Of_Day"] = df["Time_Of_Day"].fillna("Unknown")
        return df

    if "Captured_At" not in df.columns:
        df["Time_Of_Day"] = "Unknown"
        return df

    print("Classifying Time of Day from Captured_At timestamps...")
    t0 = time.time()
    captured_series = df["Captured_At"].astype(str)

    try:
        hours = pd.to_numeric(captured_series.str[11:13], errors="coerce")
    except Exception:
        parsed_dates = pd.to_datetime(captured_series, errors="coerce", utc=True)
        hours = parsed_dates.dt.hour

    time_of_days = pd.Series(["Unknown"] * len(df), index=df.index, dtype=object)
    valid_hour_mask = hours.notna()

    time_of_days[valid_hour_mask & (hours >= 5) & (hours < 8)] = "Dawn"
    time_of_days[valid_hour_mask & (hours >= 8) & (hours < 12)] = "Morning"
    time_of_days[valid_hour_mask & (hours >= 12) & (hours < 17)] = "Afternoon"
    time_of_days[valid_hour_mask & (hours >= 17) & (hours < 20)] = "Dusk"
    time_of_days[valid_hour_mask & ((hours >= 20) | (hours < 5))] = "Night"

    df["Time_Of_Day"] = time_of_days
    print(f" -> Classified Time of Day in {time.time() - t0:.2f}s.")
    return df


def generate_text_report(
    df, df_filtered, is_global, location_name, camera_trap_platforms=None
):
    """Generate a formatted report string of the statistics, including definitions."""
    total_global = len(df)
    total_filtered = len(df_filtered)
    pct_global = (total_filtered / total_global) * 100 if total_global > 0 else 0

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append(f"📊 DATASET STATISTICS REPORT: {location_name.upper()}")
    lines.append("=" * 80)
    lines.append(
        f"📍 Location Status: {'Global Dataset' if is_global else 'Filtered Location'}"
    )
    lines.append(
        f"📷 Total Images: {total_filtered:,} ({pct_global:.2f}% of global dataset)"
    )
    lines.append("-" * 80)

    # 1. Platform Breakdown
    if "Platform" in df_filtered.columns and total_filtered > 0:
        lines.append("\n🌐 PLATFORM BREAKDOWN:")
        platform_counts = df_filtered["Platform"].value_counts()
        for plat, count in platform_counts.items():
            pct = (count / total_filtered) * 100
            lines.append(f"  - {plat:<15}: {count:>10,} ({pct:>5.1f}%)")

        # Grouped Camera Trap Breakdown
        grouped_series = get_grouped_platform_series(
            df_filtered["Platform"], camera_trap_platforms=camera_trap_platforms
        )
        grouped_counts = grouped_series.value_counts()
        if "Camera Traps" in grouped_counts and len(grouped_counts) < len(
            platform_counts
        ):
            lines.append("\n📷 PLATFORM BREAKDOWN (CAMERA TRAPS GROUPED):")
            for plat, count in grouped_counts.items():
                pct = (count / total_filtered) * 100
                lines.append(f"  - {plat:<15}: {count:>10,} ({pct:>5.1f}%)")

    # 2. Continent Breakdown (only if global)
    if is_global and "continent" in df_filtered.columns and total_filtered > 0:
        lines.append("\n🌍 CONTINENT BREAKDOWN:")
        continent_counts = df_filtered["continent"].value_counts()
        for cont, count in continent_counts.items():
            pct = (count / total_filtered) * 100
            lines.append(f"  - {cont:<15}: {count:>10,} ({pct:>5.1f}%)")

    # 3. Country Breakdown (if not global, or if it's a continent)
    if not is_global and "country" in df_filtered.columns and total_filtered > 0:
        countries = df_filtered["country"].value_counts()
        if len(countries) > 1:
            lines.append("\n🏳️  TOP COUNTRIES IN THIS AREA:")
            for country, count in countries.head(10).items():
                pct = (count / total_filtered) * 100
                lines.append(f"  - {country:<25}: {count:>10,} ({pct:>5.1f}%)")

    # 4. Time of Day Breakdown
    if "Time_Of_Day" in df_filtered.columns and total_filtered > 0:
        lines.append("\n⏰ TIME OF DAY DISTRIBUTION:")
        tod_counts = df_filtered["Time_Of_Day"].value_counts()
        for tod, count in tod_counts.items():
            pct = (count / total_filtered) * 100
            lines.append(f"  - {tod:<15}: {count:>10,} ({pct:>5.1f}%)")

    # 5. Season Breakdown
    if "Season" in df_filtered.columns and total_filtered > 0:
        lines.append("\n🍂 SEASONAL DISTRIBUTION:")
        season_counts = df_filtered["Season"].value_counts()
        for season, count in season_counts.items():
            pct = (count / total_filtered) * 100
            lines.append(f"  - {season:<15}: {count:>10,} ({pct:>5.1f}%)")

    # 5b. Koppen-Geiger Climate Zone Breakdown
    if "Koppen_Code" in df_filtered.columns and total_filtered > 0:
        valid_koppen = df_filtered[
            df_filtered["Koppen_Code"].notna() & (df_filtered["Koppen_Code"] != "")
        ]
        if len(valid_koppen) > 0:
            lines.append("\n🌍 KÖPPEN-GEIGER CLIMATE ZONE DISTRIBUTION:")
            koppen_counts = (
                valid_koppen.groupby(["Koppen_Code", "Koppen_Desc"], observed=True)
                .size()
                .sort_values(ascending=False)
            )
            for (code, desc), count in koppen_counts.items():
                pct = (count / total_filtered) * 100
                lbl = f"{code} ({desc})"
                lines.append(f"  - {lbl:<55}: {count:>10,} ({pct:>5.1f}%)")

    # 6. Top Cluster Labels (if present)
    if "cluster_label" in df_filtered.columns and total_filtered > 0:
        lines.append("\n🏷️  TOP 10 SEMANTIC CLUSTER LABELS:")
        cluster_counts = df_filtered["cluster_label"].value_counts().head(10)
        for idx, (label, count) in enumerate(cluster_counts.items(), 1):
            pct = (count / total_filtered) * 100
            lines.append(f"  {idx:>2}. {label[:50]:<50}: {count:>8,} ({pct:>4.1f}%)")

    # 7. Coordinate extent
    if total_filtered > 0:
        min_lat = df_filtered["Latitude"].min()
        max_lat = df_filtered["Latitude"].max()
        min_lon = df_filtered["Longitude"].min()
        max_lon = df_filtered["Longitude"].max()
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
    lines.append(
        "\n🍂 Seasonal Classifications (Climate-Aware zoning via Köppen-Geiger, with Latitudinal fallbacks):"
    )
    lines.append("  - Desert / Dry Climates (BWh, BWk):")
    lines.append("    * Always classified as 'Dry Season' year-round.")
    lines.append("  - Tropical Savanna & Monsoon (Aw, Am):")
    lines.append("    * Northern Hemisphere Wet Season: June to September")
    lines.append("    * Southern Hemisphere Wet Season: November to April")
    lines.append("  - Mediterranean (Csa, Csb):")
    lines.append(
        "    * Northern Hemisphere Wet Season (rainy winter): December to February"
    )
    lines.append("    * Southern Hemisphere Wet Season (rainy winter): June to August")
    lines.append(
        "  - Standard Latitudinal Zones (Fallbacks if Köppen data is missing or other climate codes):"
    )
    lines.append("    * Tropical Zone (Latitudes between -23.5° and 23.5°):")
    lines.append("      + Wet Season: June, July, August, September")
    lines.append("      + Dry Season: October to May")
    lines.append("    * Temperate/Polar Zones (Latitudes > 23.5° or < -23.5°):")
    lines.append(
        "      + Spring / Summer / Autumn / Winter mapped dynamically based on hemisphere."
    )
    lines.append("=" * 80 + "\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and plot dataset statistics globally or for a specific location."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="full_pipeline_output/geo_space_deduplicated.parquet",
        help="Path to the input Parquet or CSV dataset.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Name of location to filter by (continent, country, or city). If omitted, displays global stats.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help="Path to save the generated plot (.png). Defaults to {location}_stats.png.",
    )
    parser.add_argument(
        "--output_grouped_plot",
        type=str,
        default=None,
        help="Path to save the separate grouped camera trap platforms plot (.png). Defaults to {location}_stats_grouped_platforms.png.",
    )
    parser.add_argument(
        "--group_camera_traps",
        action="store_true",
        help="If set, also group camera trap platforms into a single category in the main dashboard.",
    )
    parser.add_argument(
        "--camera_trap_platforms",
        nargs="+",
        default=None,
        help="List of camera trap platforms to group (e.g. iwildcam wildobs wildlife_insights snapshotusa).",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum number of images in an H3 cell to display on the map (default: 1).",
    )
    parser.add_argument(
        "--output_text",
        type=str,
        default=None,
        help="Path to save the text report/logs (.txt). Defaults to {location}_stats.txt.",
    )
    parser.add_argument(
        "--output_map",
        type=str,
        default=None,
        help="Path to save the interactive HTML map. Defaults to {location}_map.html.",
    )
    default_land_shp = (
        "shapefiles/ne_10m_admin_0_countries.shp"
        if os.path.exists("shapefiles/ne_10m_admin_0_countries.shp")
        else "ne_10m_admin_0_countries.shp"
    )
    parser.add_argument(
        "--land_shp",
        type=str,
        default=default_land_shp,
        help="Path to the country shapefile for spatial region mapping.",
    )
    parser.add_argument(
        "--spatial_index",
        type=str,
        default=None,
        help="Path to pre-built H3 spatial semantic index Parquet file.",
    )
    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.input)

    # Classify time of day if needed
    df = ensure_time_of_day(df)

    # Auto-detect pre-built spatial index if not provided
    spatial_index_path = args.spatial_index
    if not spatial_index_path and args.input:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        possible_index = os.path.join(input_dir, "geo_space_h3_semantic_index.parquet")
        if os.path.exists(possible_index):
            spatial_index_path = possible_index

    # Map points to continent/country
    df = map_coordinates_to_regions(
        df, args.land_shp, spatial_index_path=spatial_index_path
    )

    # Handle filtering
    is_global = True
    location_name = "Global Dataset"
    df_filtered = df

    if args.location:
        loc_clean = args.location.strip().lower()
        continent_map = {
            "africa": "Africa",
            "europe": "Europe",
            "asia": "Asia",
            "north america": "North America",
            "south america": "South America",
            "oceania": "Oceania",
            "australia": "Oceania",
            "antarctica": "Antarctica",
        }

        # 1. Check if it's a continent
        if loc_clean in continent_map:
            continent_target = continent_map[loc_clean]
            df_filtered = df[df["continent"].str.lower() == continent_target.lower()]
            is_global = False
            location_name = continent_target
            print(
                f"Filtered dataset by continent '{continent_target}': kept {len(df_filtered):,} records."
            )

        # 2. Check if it matches a country in the dataset (case-insensitive)
        elif "country" in df.columns and loc_clean in [
            c.lower() for c in df["country"].unique()
        ]:
            actual_country = next(
                c for c in df["country"].unique() if c.lower() == loc_clean
            )
            df_filtered = df[df["country"].str.lower() == loc_clean]
            is_global = False
            location_name = actual_country
            print(
                f"Filtered dataset by country '{actual_country}': kept {len(df_filtered):,} records."
            )

        # 3. Fallback to Nominatim geocoding bounding box
        else:
            bbox, display_name = geocode_location(args.location)
            if bbox:
                min_lat, max_lat, min_lon, max_lon = bbox
                df_filtered = df[
                    (df["Latitude"] >= min_lat)
                    & (df["Latitude"] <= max_lat)
                    & (df["Longitude"] >= min_lon)
                    & (df["Longitude"] <= max_lon)
                ]
                is_global = False
                location_name = display_name if display_name else args.location
                print(
                    f"Filtered dataset by bounding box for '{args.location}': kept {len(df_filtered):,} records."
                )
            else:
                print(
                    f"Error: Could not resolve location '{args.location}'. Falling back to global statistics."
                )

    # Determine plot output path
    safe_loc = (
        "".join([c if c.isalnum() else "_" for c in location_name]).strip("_").lower()
    )

    if args.output_plot:
        plot_path = args.output_plot
    else:
        plot_path = f"{safe_loc}_stats.png"

    # Determine grouped plot output path
    if args.output_grouped_plot:
        grouped_plot_path = args.output_grouped_plot
    else:
        base_p, ext_p = os.path.splitext(plot_path)
        grouped_plot_path = f"{base_p}_grouped_platforms{ext_p or '.png'}"

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
    report_str = generate_text_report(
        df,
        df_filtered,
        is_global,
        location_name,
        camera_trap_platforms=args.camera_trap_platforms,
    )
    print(report_str)

    # Save text report to file
    try:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f" -> Saved statistics report text to: {os.path.abspath(text_path)}")
    except Exception as e:
        print(f"Warning: Failed to save text report to '{text_path}': {e}")

    # Generate plots
    generate_plots(
        df_filtered,
        is_global,
        location_name,
        plot_path,
        group_camera_traps=args.group_camera_traps,
        camera_trap_platforms=args.camera_trap_platforms,
    )

    # Generate separate grouped camera trap platforms plot
    if "Platform" in df_filtered.columns:
        generate_grouped_platform_plot(
            df_filtered,
            location_name,
            grouped_plot_path,
            camera_trap_platforms=args.camera_trap_platforms,
        )

    # Generate interactive map
    generate_interactive_map(
        df_filtered, location_name, map_path, min_count=args.min_count
    )


if __name__ == "__main__":
    main()
