import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import requests

from src.utils.io import (
    get_core_base_name,
    load_dataframe,
    load_dataset_with_clusters,
    save_dataframe,
)

# Continent bounding box overrides for fallback geocoding
CONTINENT_BOUNDS = {
    "africa": [-35.0, 38.0, -26.0, 52.0],
    "europe": [35.0, 72.0, -25.0, 45.0],
    "asia": [1.0, 77.0, 26.0, 180.0],
    "north america": [7.0, 85.0, -168.0, -52.0],
    "south america": [-56.0, 13.0, -82.0, -34.0],
    "oceania": [-48.0, 0.0, 110.0, 180.0],
    "australia": [-48.0, -10.0, 110.0, 155.0],
    "antarctica": [-90.0, -60.0, -180.0, 180.0],
}

# Known continent names for direct column matching
KNOWN_CONTINENTS = {
    "africa": "Africa",
    "europe": "Europe",
    "asia": "Asia",
    "north america": "North America",
    "south america": "South America",
    "oceania": "Oceania",
    "australia": "Oceania",
    "antarctica": "Antarctica",
}

# Common country name aliases mapped to standardized dataset names
COUNTRY_ALIASES = {
    "usa": "United States of America",
    "us": "United States of America",
    "united states": "United States of America",
    "america": "United States of America",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "uae": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "south korea": "South Korea",
    "korea": "South Korea",
    "czech republic": "Czechia",
    "russia": "Russia",
    "russian federation": "Russia",
    "the netherlands": "Netherlands",
    "holland": "Netherlands",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
}


def parse_timestamps_series(ts_raw):
    """
    Parses timestamps from various formats into a UTC datetime Series.
    Supports ISO 8601, mixed date formats, EXIF colon-delimited dates (YYYY:MM:DD),
    and Unix epoch numeric timestamps (seconds or milliseconds), matching standardize_timestamps.py.
    """
    if ts_raw is None or len(ts_raw) == 0:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    # If already datetime with UTC
    if pd.api.types.is_datetime64_any_dtype(ts_raw):
        if hasattr(ts_raw.dt, "tz") and ts_raw.dt.tz is None:
            return ts_raw.dt.tz_localize("UTC")
        elif hasattr(ts_raw.dt, "tz") and ts_raw.dt.tz is not None:
            return ts_raw.dt.tz_convert("UTC")
        return ts_raw

    result = pd.Series(pd.NaT, index=ts_raw.index, dtype="datetime64[ns, UTC]")

    # Identify Unix epoch numeric timestamps vs. string representations
    numeric_ts = pd.to_numeric(ts_raw, errors="coerce")
    is_numeric = numeric_ts.notna() & (numeric_ts > 1e8)

    if is_numeric.any():
        is_ms_mask = is_numeric & (numeric_ts > 5e10)
        is_s_mask = is_numeric & ~is_ms_mask

        if is_ms_mask.any():
            parsed_ms = pd.to_datetime(
                numeric_ts[is_ms_mask], unit="ms", utc=True, errors="coerce"
            )
            result.loc[is_ms_mask] = parsed_ms

        if is_s_mask.any():
            parsed_s = pd.to_datetime(
                numeric_ts[is_s_mask], unit="s", utc=True, errors="coerce"
            )
            result.loc[is_s_mask] = parsed_s

    is_string = ts_raw.notna() & ~is_numeric
    if is_string.any():
        str_vals = ts_raw[is_string].astype(str).str.strip()
        # Fix EXIF colon dates e.g. "2020:05:12 14:30:00" -> "2020-05-12 14:30:00"
        has_colon_date = str_vals.str.match(r"^\d{4}:\d{2}:\d{2}")
        if has_colon_date.any():
            str_vals.loc[has_colon_date] = str_vals[has_colon_date].str.replace(
                ":", "-", n=2
            )

        parsed_str = pd.to_datetime(str_vals, errors="coerce", utc=True, format="mixed")
        result.loc[is_string] = parsed_str

    return result


def geocode_location(location_name):
    """Resolve location to bounding box [min_lat, max_lat, min_lon, max_lon] (with continent overrides)."""
    loc_clean = location_name.strip().lower()

    if loc_clean in CONTINENT_BOUNDS:
        bbox = CONTINENT_BOUNDS[loc_clean]
        print(f" -> Resolved using offline continent bounds for '{location_name}':")
        print(
            f" -> Bounding Box: Lat [{bbox[0]} to {bbox[1]}], Lon [{bbox[2]} to {bbox[3]}]"
        )
        return bbox

    print(f"Resolving location '{location_name}' using Nominatim Geocoding API...")
    url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json&limit=1"
    headers = {"User-Agent": "Geo-RAG-Dataset-Filter/2.0"}
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
            return bbox
    except Exception as e:
        print(f"Warning: Geocoding failed: {e}")
    return None


def parse_bbox_string(bbox_str):
    """Parses a comma- or space-separated bounding box: min_lat, max_lat, min_lon, max_lon."""
    try:
        cleaned = bbox_str.replace(",", " ").strip()
        parts = [float(x) for x in cleaned.split() if x]
        if len(parts) == 4:
            return parts
    except Exception:
        pass
    return None


def resolve_output_path(
    output_arg, input_path, location=None, country=None, continent=None
):
    """
    Determines the output file path. If output_arg is omitted or a directory,
    names the file based on the location/country/continent (e.g. 'rome.parquet').
    """
    ext = os.path.splitext(input_path)[1].lower()
    if not ext:
        ext = ".parquet"

    # Derive location slug
    name_source = None
    if location:
        name_source = location
    elif country:
        name_source = country.split(",")[0].strip()
    elif continent:
        name_source = continent

    if name_source:
        loc_slug = re.sub(r"[^a-zA-Z0-9]+", "_", name_source.strip().lower()).strip("_")
    else:
        in_base = os.path.splitext(os.path.basename(input_path))[0]
        loc_slug = f"{in_base}_filtered"

    default_filename = f"{loc_slug}{ext}"

    if not output_arg:
        # Default: save in same directory as input
        input_dir = os.path.dirname(os.path.abspath(input_path))
        return os.path.join(input_dir, default_filename)

    # If output_arg is an existing directory or ends with a separator
    if (
        os.path.isdir(output_arg)
        or output_arg.endswith(os.sep)
        or output_arg.endswith("/")
    ):
        return os.path.join(output_arg, default_filename)

    # If output_arg does not have an extension, add input extension
    base, out_ext = os.path.splitext(output_arg)
    if not out_ext:
        return f"{output_arg}{ext}"

    return output_arg


def resolve_and_slice_embeddings(
    input_path,
    df_filtered,
    kept_indices,
    representation_type="cls",
    precision="float32",
):
    """
    Finds and slices companion embeddings (.npy) for the filtered records.
    Uses memory-mapping and companion keys index if available to avoid copying the full matrix into RAM.
    """
    if len(df_filtered) == 0:
        return None

    # Case A: Embedded 'embedding' column already present in df
    if "embedding" in df_filtered.columns:
        print(" -> Found embedded 'embedding' column in dataset.")
        return np.vstack(df_filtered["embedding"].values)

    # Case B: Decoupled companion embeddings (.npy)
    db_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]
    core_name = get_core_base_name(base_name)

    candidate_names = [
        f"{base_name}_{representation_type}_embeddings.npy",
        f"{core_name}_{representation_type}_embeddings.npy",
        f"{base_name}_embeddings.npy",
        f"{core_name}_embeddings.npy",
    ]
    npy_path = None
    for name in candidate_names:
        p = os.path.join(db_dir, name)
        if os.path.exists(p):
            npy_path = p
            break

    if not npy_path:
        # Wildcard fallback search
        pattern = os.path.join(db_dir, f"{core_name}*.npy")
        matches = glob.glob(pattern)
        if matches:
            preferred = [
                m for m in matches if representation_type in os.path.basename(m)
            ]
            npy_path = preferred[0] if preferred else matches[0]

    if not npy_path or not os.path.exists(npy_path):
        print(
            f"Warning: No companion embeddings found matching '{core_name}' in '{db_dir}'. Skipping embeddings."
        )
        return None

    print(f" -> Found companion embeddings matrix: {npy_path}")
    dtype = np.float16 if precision == "float16" else np.float32

    # Check for companion keys file
    keys_path = npy_path.replace(".npy", ".keys.parquet")
    has_photo_key = "photo_key" in df_filtered.columns
    has_platform_id = (
        "Platform" in df_filtered.columns and "Photo_ID" in df_filtered.columns
    )

    if os.path.exists(keys_path) and (has_photo_key or has_platform_id):
        print(f" -> Slicing embeddings via companion keys index: {keys_path}")
        master_keys = pd.Index(
            pd.read_parquet(keys_path, columns=["photo_key"])["photo_key"]
            .astype(str)
            .str.lower()
        )
        if has_photo_key:
            query_keys = df_filtered["photo_key"].astype(str).str.lower().values
        else:
            query_keys = (
                df_filtered["Platform"].astype(str).str.lower()
                + "_"
                + df_filtered["Photo_ID"].astype(str)
            ).values

        if master_keys.is_unique:
            indices = master_keys.get_indexer(query_keys)
        else:
            pos_series = pd.Series(np.arange(len(master_keys)), index=master_keys)
            pos_series = pos_series[~pos_series.index.duplicated(keep="first")]
            indices = pos_series.reindex(query_keys, fill_value=-1).values

        valid_mask = indices >= 0
        mmap_emb = np.load(npy_path, mmap_mode="r")
        safe_indices = np.clip(indices, 0, len(mmap_emb) - 1)
        sliced_emb = mmap_emb[safe_indices].astype(dtype)
        if not valid_mask.all():
            print(
                f"Warning: Found {np.sum(~valid_mask):,} keys missing from embeddings index. Zero-filling..."
            )
            sliced_emb[~valid_mask] = 0.0
        return sliced_emb

    # Check for embedding_idx in df
    if "embedding_idx" in df_filtered.columns:
        print(" -> Slicing embeddings via 'embedding_idx' mapping...")
        indices = df_filtered["embedding_idx"].values
        mmap_emb = np.load(npy_path, mmap_mode="r")
        safe_indices = np.clip(indices, 0, len(mmap_emb) - 1)
        sliced_emb = mmap_emb[safe_indices].astype(dtype)
        return sliced_emb

    # Sequential row index slicing
    print(" -> Slicing embeddings using sequential row indices...")
    mmap_emb = np.load(npy_path, mmap_mode="r")
    safe_indices = np.clip(kept_indices, 0, len(mmap_emb) - 1)
    sliced_emb = mmap_emb[safe_indices].astype(dtype)
    return sliced_emb


def main():
    parser = argparse.ArgumentParser(
        description="Filter Geo-RAG Parquet/CSV datasets by hierarchical location, date ranges, seasons, climate, and time of day."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset (.parquet or .csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the filtered output dataset (or output directory). "
        "If omitted or a directory, the filename is automatically set to the location name (e.g. 'rome.parquet').",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Geographic name to filter by (continent, country, or specific city/place). "
        "Directly filters by 'continent' or 'country' if matched; otherwise falls back to Nominatim city bounding box.",
    )
    parser.add_argument(
        "--continent",
        type=str,
        default=None,
        help="Explicit continent filter (e.g. 'Europe', 'North America', 'Asia'). Case-insensitive.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Explicit country filter (e.g. 'Italy', 'France', or comma-separated 'Italy, France'). Case-insensitive.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Direct bounding box: 'min_lat,max_lat,min_lon,max_lon'. Offline, skips Nominatim.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD or ISO 8601 format (inclusive).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD or ISO 8601 format (inclusive).",
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        choices=["Spring", "Summer", "Autumn", "Winter", "Wet Season", "Dry Season"],
        help="Filter by season.",
    )
    parser.add_argument(
        "--time_of_day",
        type=str,
        default=None,
        choices=["Dawn", "Morning", "Afternoon", "Dusk", "Night"],
        help="Filter by time of day.",
    )
    parser.add_argument(
        "--koppen",
        type=str,
        default=None,
        help="Filter by Köppen climate code (e.g. 'Csa', 'Dfb', 'BWh'). Case-insensitive.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Filter by platform (e.g. Flickr, Mapillary, iNaturalist).",
    )
    parser.add_argument(
        "--cluster_id",
        type=int,
        default=None,
        help="Filter by specific cluster ID.",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Extract and save companion embeddings (.npy + .keys.parquet) matching the filtered dataset.",
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        default="cls",
        choices=["cls", "cls_avg_patch", "avg_patch"],
        help="Embedding representation type to slice and save (default: cls).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Precision for saved companion embeddings (default: float32).",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        df = load_dataframe(args.input)
    else:
        df = load_dataset_with_clusters(args.input)

    initial_len = len(df)
    print(f"Loaded {initial_len:,} records.")

    # Track original sequential indices for 1-to-1 embedding slicing if needed
    df = df.copy()
    df["__orig_idx__"] = np.arange(initial_len)

    # Normalize timestamp column name if alternative naming is used (matching standardize_timestamps.py)
    col_map = {
        "captured_at": "Captured_At",
        "Date_Observed": "Captured_At",
        "observed_on_string": "Captured_At",
        "datetime_local": "Captured_At",
    }
    rename_dict = {
        k: v
        for k, v in col_map.items()
        if k in df.columns and "Captured_At" not in df.columns
    }
    if rename_dict:
        print(f"Renaming timestamp columns to standard 'Captured_At': {rename_dict}")
        df = df.rename(columns=rename_dict)

    # 1. Hierarchical Geographic Filtering
    # 1a. Explicit Continent Filter
    if args.continent:
        target_cont = args.continent.strip().lower()
        if "continent" in df.columns:
            df = df[df["continent"].astype(str).str.lower() == target_cont]
            print(
                f" -> Explicit continent filter ('{args.continent}') applied. Kept {len(df):,} records."
            )
        else:
            # Fall back to continent bounding box
            if target_cont in CONTINENT_BOUNDS:
                min_lat, max_lat, min_lon, max_lon = CONTINENT_BOUNDS[target_cont]
                df = df[
                    (df["Latitude"] >= min_lat)
                    & (df["Latitude"] <= max_lat)
                    & (df["Longitude"] >= min_lon)
                    & (df["Longitude"] <= max_lon)
                ]
                print(
                    f" -> Applied continent bounding box for '{args.continent}'. Kept {len(df):,} records."
                )
            else:
                print(
                    f"Warning: Continent column missing and no bounds for '{args.continent}'."
                )

    # 1b. Explicit Country Filter
    if args.country:
        raw_countries = [c.strip() for c in args.country.split(",") if c.strip()]
        if "country" in df.columns:
            # Map aliases
            resolved_countries = []
            for c in raw_countries:
                resolved = COUNTRY_ALIASES.get(c.lower(), c)
                resolved_countries.append(resolved.lower())

            df = df[df["country"].astype(str).str.lower().isin(resolved_countries)]
            print(
                f" -> Explicit country filter ({raw_countries}) applied. Kept {len(df):,} records."
            )
        else:
            print(
                "Warning: Column 'country' not found. Attempting bounding box geocoding for countries..."
            )
            bbox_masks = []
            for c in raw_countries:
                c_bbox = geocode_location(c)
                if c_bbox:
                    b_min_lat, b_max_lat, b_min_lon, b_max_lon = c_bbox
                    m = (
                        (df["Latitude"] >= b_min_lat)
                        & (df["Latitude"] <= b_max_lat)
                        & (df["Longitude"] >= b_min_lon)
                        & (df["Longitude"] <= b_max_lon)
                    )
                    bbox_masks.append(m)
            if bbox_masks:
                combined_mask = bbox_masks[0]
                for m in bbox_masks[1:]:
                    combined_mask |= m
                df = df[combined_mask]
                print(
                    f" -> Country bounding box filter applied. Kept {len(df):,} records."
                )

    # 1c. Explicit Bounding Box Filter
    if args.bbox:
        parsed_bbox = parse_bbox_string(args.bbox)
        if parsed_bbox:
            min_lat, max_lat, min_lon, max_lon = parsed_bbox
            df = df[
                (df["Latitude"] >= min_lat)
                & (df["Latitude"] <= max_lat)
                & (df["Longitude"] >= min_lon)
                & (df["Longitude"] <= max_lon)
            ]
            print(
                f" -> Bounding box [{min_lat}, {max_lat}, {min_lon}, {max_lon}] applied. Kept {len(df):,} records."
            )
        else:
            print(
                f"Error: Invalid --bbox format '{args.bbox}'. Expected 'min_lat,max_lat,min_lon,max_lon'."
            )

    # 1d. Smart Location Filter (Hierarchical: Continent -> Country -> City/Region Bounding Box)
    if args.location:
        loc_str = args.location.strip()
        loc_lower = loc_str.lower()
        applied_location = False

        # Step 1: Check if location is a known continent
        if loc_lower in KNOWN_CONTINENTS and "continent" in df.columns:
            matched_continent = KNOWN_CONTINENTS[loc_lower]
            df = df[
                df["continent"].astype(str).str.lower() == matched_continent.lower()
            ]
            print(
                f" -> Location '{loc_str}' recognized as continent. Filtered directly on 'continent' == '{matched_continent}'. Kept {len(df):,} records."
            )
            applied_location = True

        # Step 2: Check if location is a known country (or alias)
        if not applied_location and "country" in df.columns:
            matched_country = COUNTRY_ALIASES.get(loc_lower, None)
            if not matched_country:
                # Check directly against distinct countries in the dataset
                distinct_countries = {
                    c.lower(): c for c in df["country"].dropna().unique() if c
                }
                if loc_lower in distinct_countries:
                    matched_country = distinct_countries[loc_lower]

            if matched_country:
                df = df[
                    df["country"].astype(str).str.lower() == matched_country.lower()
                ]
                print(
                    f" -> Location '{loc_str}' recognized as country. Filtered directly on 'country' == '{matched_country}'. Kept {len(df):,} records."
                )
                applied_location = True

        # Step 3: Specific city, landmark, or region -> Geocode bounding box
        if not applied_location:
            print(
                f" -> Location '{loc_str}' identified as specific city/region. Resolving geographic bounding box..."
            )
            bbox = geocode_location(loc_str)
            if bbox:
                min_lat, max_lat, min_lon, max_lon = bbox
                df = df[
                    (df["Latitude"] >= min_lat)
                    & (df["Latitude"] <= max_lat)
                    & (df["Longitude"] >= min_lon)
                    & (df["Longitude"] <= max_lon)
                ]
                print(
                    f" -> City bounding box filter applied. Kept {len(df):,} records."
                )
            else:
                print(
                    f"Error: Could not resolve location '{loc_str}'. Aborting location filtering."
                )

    # 2. Date Range Filtering
    if args.start_date or args.end_date:
        if "Captured_At" in df.columns:
            ts_series = parse_timestamps_series(df["Captured_At"])
            valid_ts_mask = ts_series.notna()
            date_mask = valid_ts_mask.copy()

            if args.start_date:
                start_dt = pd.to_datetime(args.start_date, utc=True)
                date_mask &= ts_series >= start_dt
            if args.end_date:
                end_dt = pd.to_datetime(args.end_date, utc=True)
                # If date format is YYYY-MM-DD (length 10), cover until the end of that day
                if len(args.end_date.strip()) == 10:
                    end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
                date_mask &= ts_series <= end_dt

            df = df[date_mask]
            print(f" -> Date filter applied. Kept {len(df):,} records.")
        else:
            print(
                "Warning: Column 'Captured_At' not found. Skipping date range filter."
            )

    # 3. Season Filtering
    if args.season:
        if "Season" in df.columns:
            df = df[df["Season"].astype(str).str.lower() == args.season.lower()]
            print(f" -> Season filter applied. Kept {len(df):,} records.")
        else:
            print("Warning: Column 'Season' not found. Skipping season filter.")

    # 4. Time of Day Filtering
    if args.time_of_day:
        if "Time_Of_Day" in df.columns:
            df = df[
                df["Time_Of_Day"].astype(str).str.lower() == args.time_of_day.lower()
            ]
            print(f" -> Time of day filter applied. Kept {len(df):,} records.")
        else:
            print(
                "Warning: Column 'Time_Of_Day' not found. Skipping time of day filter."
            )

    # 5. Köppen Climate Filtering
    if args.koppen:
        if "Koppen_Code" in df.columns:
            df = df[
                df["Koppen_Code"].astype(str).str.lower() == args.koppen.strip().lower()
            ]
            print(
                f" -> Köppen climate filter ('{args.koppen}') applied. Kept {len(df):,} records."
            )
        else:
            print(
                "Warning: Column 'Koppen_Code' not found. Skipping Köppen climate filter."
            )

    # 6. Platform Filtering
    if args.platform:
        if "Platform" in df.columns:
            df = df[df["Platform"].astype(str).str.lower() == args.platform.lower()]
            print(f" -> Platform filter applied. Kept {len(df):,} records.")
        else:
            print("Warning: Column 'Platform' not found. Skipping platform filter.")

    # 7. Cluster ID Filtering
    if args.cluster_id is not None:
        if "cluster_id" in df.columns:
            df = df[df["cluster_id"] == args.cluster_id]
            print(f" -> Cluster ID filter applied. Kept {len(df):,} records.")
        else:
            print("Warning: Column 'cluster_id' not found. Skipping cluster ID filter.")

    # Extract sequential kept indices before dropping helper column
    kept_indices = df["__orig_idx__"].values
    df = df.drop(columns=["__orig_idx__"])

    # Resolve output path (automatically named after location if omitted or a directory)
    output_path = resolve_output_path(
        args.output,
        args.input,
        location=args.location,
        country=args.country,
        continent=args.continent,
    )

    # 8. Slicing Companion Embeddings
    if args.save_embeddings:
        print(
            f"Extracting companion embeddings ({args.representation_type}, {args.precision})..."
        )
        sliced_embs = resolve_and_slice_embeddings(
            input_path=args.input,
            df_filtered=df,
            kept_indices=kept_indices,
            representation_type=args.representation_type,
            precision=args.precision,
        )
        if sliced_embs is not None:
            df["embedding"] = list(sliced_embs)
            print(
                f" -> Successfully sliced companion embeddings of shape {sliced_embs.shape}."
            )

    # Save output dataset
    print(f"Saving filtered dataset ({len(df):,} records) to {output_path}...")
    save_dataframe(
        df,
        output_path,
        representation_type=args.representation_type,
        precision=args.precision,
    )
    print("Filtering complete!")


if __name__ == "__main__":
    main()
