import os
import argparse
import pandas as pd
import requests


def geocode_location(location_name):
    """Resolve location to bounding box [min_lat, max_lat, min_lon, max_lon] (with continent overrides)."""
    loc_clean = location_name.strip().lower()

    # Continent geographic bounding box overrides (since geocoding databases often return truncated ranges for continent nodes)
    CONTINENT_BOUNDS = {
        "africa": [-35.0, 38.0, -26.0, 52.0],
        "europe": [35.0, 72.0, -25.0, 45.0],
        "asia": [1.0, 77.0, 26.0, 180.0],
        "north america": [7.0, 85.0, -168.0, -52.0],
        "south america": [-56.0, 13.0, -82.0, -34.0],
        "oceania": [-48.0, 0.0, 110.0, 180.0],
        "australia": [-48.0, -10.0, 110.0, 155.0],
        "antarctica": [-90.0, -60.0, -180.0, 180.0]
    }

    if loc_clean in CONTINENT_BOUNDS:
        bbox = CONTINENT_BOUNDS[loc_clean]
        print(f" -> Resolved using offline continent bounds for '{location_name}':")
        print(f" -> Bounding Box: Lat [{bbox[0]} to {bbox[1]}], Lon [{bbox[2]} to {bbox[3]}]")
        return bbox

    print(f"Resolving location '{location_name}' using Nominatim Geocoding API...")
    url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json&limit=1"
    headers = {"User-Agent": "Geo-RAG-Dataset-Filter"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200 and res.json():
            data = res.json()[0]
            bbox = [float(x) for x in data["boundingbox"]]  # [min_lat, max_lat, min_lon, max_lon]
            print(f" -> Found: {data['display_name']}")
            print(f" -> Bounding Box: Lat [{bbox[0]} to {bbox[1]}], Lon [{bbox[2]} to {bbox[3]}]")
            return bbox
    except Exception as e:
        print(f"Warning: Geocoding failed: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter the Geo-RAG Parquet/CSV dataset by location, date ranges, seasons, and times of day.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input clustered or deduplicated dataset (.parquet or .csv).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the filtered output dataset.")
    parser.add_argument("--location", type=str, default=None,
                        help="Place name to filter by (e.g. 'Rome', 'Tokyo'). Resolves to a bounding box.")
    parser.add_argument("--start_date", type=str, default=None,
                        help="Start date in YYYY-MM-DD or ISO 8601 format (inclusive).")
    parser.add_argument("--end_date", type=str, default=None,
                        help="End date in YYYY-MM-DD or ISO 8601 format (inclusive).")
    parser.add_argument("--season", type=str, default=None,
                        choices=['Spring', 'Summer', 'Autumn', 'Winter', 'Wet Season', 'Dry Season'],
                        help="Filter by season.")
    parser.add_argument("--time_of_day", type=str, default=None,
                        choices=['Dawn', 'Morning', 'Afternoon', 'Dusk', 'Night'], help="Filter by time of day.")
    parser.add_argument("--platform", type=str, default=None, help="Filter by platform (e.g. Flickr, Mapillary).")
    parser.add_argument("--cluster_id", type=int, default=None, help="Filter by specific cluster ID.")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_parquet(args.input)
    initial_len = len(df)
    print(f"Loaded {initial_len} records.")

    # 1. Location Filtering
    if args.location:
        bbox = geocode_location(args.location)
        if bbox:
            min_lat, max_lat, min_lon, max_lon = bbox
            df = df[
                (df['Latitude'] >= min_lat) & (df['Latitude'] <= max_lat) &
                (df['Longitude'] >= min_lon) & (df['Longitude'] <= max_lon)
                ]
            print(f" -> Location filter applied. Kept {len(df)} records.")
        else:
            print("Error: Could not resolve location. Aborting location filtering.")

    # 2. Date Range Filtering
    if args.start_date or args.end_date:
        if 'Captured_At' in df.columns:
            # Ensure we are checking non-null timestamps
            df = df.dropna(subset=['Captured_At'])
            if args.start_date:
                # Format start date to match ISO string comparison (string-wise works perfectly for ISO 8601)
                start_iso = args.start_date if 'T' in args.start_date else f"{args.start_date}T00:00:00Z"
                df = df[df['Captured_At'] >= start_iso]
            if args.end_date:
                end_iso = args.end_date if 'T' in args.end_date else f"{args.end_date}T23:59:59Z"
                df = df[df['Captured_At'] <= end_iso]
            print(f" -> Date filter applied. Kept {len(df)} records.")
        else:
            print("Warning: Column 'Captured_At' not found. Skipping date range filter.")

    # 3. Season Filtering
    if args.season:
        if 'Season' in df.columns:
            df = df[df['Season'].str.lower() == args.season.lower()]
            print(f" -> Season filter applied. Kept {len(df)} records.")
        else:
            print("Warning: Column 'Season' not found. Skipping season filter.")

    # 4. Time of Day Filtering
    if args.time_of_day:
        if 'Time_Of_Day' in df.columns:
            df = df[df['Time_Of_Day'].str.lower() == args.time_of_day.lower()]
            print(f" -> Time of day filter applied. Kept {len(df)} records.")
        else:
            print("Warning: Column 'Time_Of_Day' not found. Skipping time of day filter.")

    # 5. Platform Filtering
    if args.platform:
        if 'Platform' in df.columns:
            df = df[df['Platform'].str.lower() == args.platform.lower()]
            print(f" -> Platform filter applied. Kept {len(df)} records.")

    # 6. Cluster ID Filtering
    if args.cluster_id is not None:
        if 'cluster_id' in df.columns:
            df = df[df['cluster_id'] == args.cluster_id]
            print(f" -> Cluster ID filter applied. Kept {len(df)} records.")

    # Save output
    print(f"Saving filtered dataset ({len(df)} records) to {args.output}...")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.output.endswith('.csv'):
        df.to_csv(args.output, index=False)
    else:
        df.to_parquet(args.output, index=False)
    print("Filtering complete!")


if __name__ == "__main__":
    main()
