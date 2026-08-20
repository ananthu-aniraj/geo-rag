import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import requests
from shapely.geometry import Point, box as shapely_box
from tqdm import tqdm

from src.utils.io import get_core_base_name, load_dataframe, save_dataframe
from src.utils.licensing import FLICKR_LICENSE_MAP

FLICKR_API_KEY = "FLICKR_API_KEY_PLACEHOLDER"
FLICKR_DELAY = 1.1


def fetch_flickr_individual_license(photo_id):
    """Queries individual photo info to get its license code."""
    url = (
        f"https://www.flickr.com/services/rest/"
        f"?method=flickr.photos.getInfo"
        f"&api_key={FLICKR_API_KEY}"
        f"&photo_id={photo_id}"
        f"&format=json"
        f"&nojsoncallback=1"
    )
    try:
        time.sleep(FLICKR_DELAY)
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("stat") == "ok":
                license_code = data.get("photo", {}).get("license", "0")
                return str(photo_id), str(license_code)
    except Exception:
        pass
    return str(photo_id), None


def fetch_flickr_bbox_licenses(bbox_str):
    """Fetches photo IDs and license codes for a bounding box in bulk (up to 500 per call)."""
    results = {}

    # Priority 1: Outdoors (2). Priority 2: Unlabelled (0).
    for geo_context in [2, 0]:
        page = 1
        total_pages = 1
        while page <= total_pages:
            url = (
                f"https://www.flickr.com/services/rest/"
                f"?method=flickr.photos.search"
                f"&api_key={FLICKR_API_KEY}"
                f"&bbox={bbox_str}"
                f"&has_geo=1"
                f"&geo_context={geo_context}"
                f"&extras=license"
                f"&per_page=250"
                f"&page={page}"
                f"&format=json"
                f"&nojsoncallback=1"
            )
            try:
                time.sleep(FLICKR_DELAY)
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("stat") == "ok":
                        # Limit to Page 1 as scraped content fits within 100 limit per box
                        total_pages = 1

                        photos = data.get("photos", {}).get("photo", [])
                        if not photos:
                            break

                        for p in photos:
                            pid = str(p.get("id"))
                            lic = p.get("license")
                            if lic is not None:
                                results[pid] = str(lic)

                        page += 1
                    else:
                        break
                else:
                    break
            except Exception:
                break
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing Flickr image licenses in the Parquet dataset."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input Parquet or CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file. Overwrites input if omitted.",
    )
    parser.add_argument(
        "--log_dirs",
        nargs="+",
        default=None,
        help="List of log directories containing flickr_completed_boxes_chunk_*.txt files for spatial join optimization.",
    )
    args = parser.parse_args()

    # Determine input and output paths
    resolved_input = args.input

    if os.path.exists(args.input) and args.input.endswith(".parquet"):
        pf = pq.ParquetFile(args.input)
        schema_names = pf.schema_arrow.names
        if "cluster_id" in schema_names and (
            "Latitude" not in schema_names or "Longitude" not in schema_names
        ):
            db_dir = os.path.dirname(os.path.abspath(args.input))
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            core_name = get_core_base_name(base_name)
            for fallback in [
                f"{core_name}_cleaned.parquet",
                f"{core_name}_deduplicated.parquet",
                f"{core_name}.parquet",
            ]:
                fallback_path = os.path.join(db_dir, fallback)
                if os.path.exists(fallback_path):
                    resolved_input = fallback_path
                    print(
                        f" -> Input is a decoupled sidecar. Resolving base metadata to: {resolved_input}"
                    )
                    break

    out_path = args.output if args.output else resolved_input

    print(f"Loading dataset from {resolved_input}...")
    df = load_dataframe(resolved_input)

    # Ensure License column exists
    if "License" not in df.columns:
        df["License"] = None
    df["License"] = (
        df["License"]
        .astype(str)
        .replace({"nan": None, "None": None, "<NA>": None, "": None})
    )

    modified = False

    # --- 1. Scan and Load Offline Dataset CSV Licenses from args.log_dirs ---
    offline_licenses = {}
    if args.log_dirs:
        # Split any space-separated strings (shell-quoting issue helper)
        actual_log_dirs = []
        for d in args.log_dirs:
            actual_log_dirs.extend([x.strip() for x in d.split() if x.strip()])

        # Build Photo_ID to Platform map from main dataset
        pid_to_platform = dict(
            zip(df["Photo_ID"].astype(str), df["Platform"].astype(str).str.lower())
        )

        print("Scanning --log_dirs for CSV files to load per-image licenses...")
        for d in actual_log_dirs:
            if not os.path.exists(d):
                continue
            csv_files = glob.glob(
                os.path.join(d, "**/*.csv"), recursive=True
            ) + glob.glob(os.path.join(d, "*.csv"))
            csv_files = list(set(csv_files))
            for csv_file in csv_files:
                try:
                    # Read CSV, only loading columns that match Photo_ID/License (case-insensitive)
                    df_csv = pd.read_csv(
                        csv_file,
                        usecols=lambda col: col.lower() in ["photo_id", "license"],
                    )
                    df_csv.columns = [c.lower() for c in df_csv.columns]
                    if "photo_id" in df_csv.columns and "license" in df_csv.columns:
                        df_csv = df_csv.dropna(subset=["photo_id", "license"])
                        if not df_csv.empty:
                            pids = df_csv["photo_id"].astype(str).tolist()
                            lics = df_csv["license"].astype(str).tolist()
                            for p, l in zip(pids, lics):
                                plat = pid_to_platform.get(p, "")
                                if plat == "flickr":
                                    clean_code = l.split(".")[0].strip()
                                    if clean_code in FLICKR_LICENSE_MAP:
                                        l = FLICKR_LICENSE_MAP[clean_code]
                                offline_licenses[p] = l
                except Exception:
                    pass
        if offline_licenses:
            print(f"Loaded {len(offline_licenses):,} licenses from offline CSV files.")
            is_missing = df["License"].isna() | (
                df["License"].astype(str).str.strip() == ""
            )
            df_missing_pids = df.loc[is_missing, "Photo_ID"].astype(str)
            mapped_offline = df_missing_pids.map(offline_licenses)
            if mapped_offline.notna().any():
                df["License"] = df["License"].combine_first(mapped_offline)
                modified = True
                print(
                    f" -> Backfilled {mapped_offline.notna().sum()} records using offline CSV maps."
                )

    # --- 2. Identify Flickr rows that STILL have missing/empty licenses ---
    is_flickr = df["Platform"].astype(str).str.lower() == "flickr"
    is_missing = (
        df["License"].isna()
        | (df["License"] == "")
        | (df["License"].astype(str).str.strip() == "")
        | (df["License"].astype(str) == "nan")
    )

    df_missing = df[is_flickr & is_missing].copy()
    flickr_licenses = {}

    if len(df_missing) > 0:
        print(f"Found {len(df_missing):,} Flickr records lacking license codes.")
        flickr_ids = set(df_missing["Photo_ID"].astype(str).tolist())

        # --- Option A: Spatial Join Bounding Box Lookup (Bulk Search) ---
        if args.log_dirs:
            # Split any space-separated strings (shell-quoting issue helper)
            actual_dirs = []
            for d in args.log_dirs:
                actual_dirs.extend([x.strip() for x in d.split() if x.strip()])

            print("Parsing completed bounding boxes log files...")
            bboxes = set()
            for log_dir in actual_dirs:
                pattern = os.path.join(log_dir, "*completed_boxes*.txt")
                log_files = glob.glob(pattern)
                print(f"  -> Found {len(log_files)} log files in: {log_dir}")
                for f in log_files:
                    try:
                        with open(f, "r") as fh:
                            for line in fh:
                                line_clean = line.replace("\x00", "").strip()
                                if line_clean:
                                    parts = line_clean.split(",")
                                    if len(parts) == 4:
                                        try:
                                            [float(x) for x in parts]
                                            bboxes.add(line_clean)
                                        except ValueError:
                                            continue
                    except Exception as fe:
                        print(f"  -> Warning reading log file {f}: {fe}")
            print(f"Loaded {len(bboxes):,} unique completed bounding boxes.")

            active_bboxes = []
            box_to_photos = {}

            if bboxes:
                try:
                    print(
                        "Running spatial join to associate missing coordinates with bounding boxes..."
                    )

                    points = [
                        Point(lon, lat)
                        for lat, lon in zip(
                            df_missing["Latitude"], df_missing["Longitude"]
                        )
                    ]
                    gdf_points = gpd.GeoDataFrame(
                        {"Photo_ID": df_missing["Photo_ID"].astype(str)},
                        geometry=points,
                        crs="EPSG:4326",
                    )

                    boxes_list = []
                    box_strs = []
                    for b_str in bboxes:
                        parts = [float(x) for x in b_str.split(",")]
                        boxes_list.append(
                            shapely_box(parts[0], parts[1], parts[2], parts[3])
                        )
                        box_strs.append(b_str)

                    gdf_boxes = gpd.GeoDataFrame(
                        {"bbox_str": box_strs}, geometry=boxes_list, crs="EPSG:4326"
                    )

                    joined = gpd.sjoin(
                        gdf_boxes, gdf_points, how="inner", predicate="intersects"
                    )
                    box_to_photos = (
                        joined.groupby("bbox_str")["Photo_ID"].apply(set).to_dict()
                    )
                    active_bboxes = sorted(
                        box_to_photos.keys(),
                        key=lambda b: len(box_to_photos[b]),
                        reverse=True,
                    )
                    print(
                        f"Filtered and sorted to {len(active_bboxes)} active bounding boxes containing missing Flickr images."
                    )
                except Exception as se:
                    print(
                        f"Spatial join optimization failed or geopandas not available: {se}"
                    )
                    print("Falling back to scanning all discovered bounding boxes...")
                    active_bboxes = list(bboxes)
                    box_to_photos = {}

                if active_bboxes:
                    print("Running optimized bulk license search on active boxes...")
                    for bbox in tqdm(active_bboxes, desc="Bulk Scan Flickr BBoxes"):
                        if box_to_photos and bbox in box_to_photos:
                            box_photos = box_to_photos[bbox]
                            needed_photos = box_photos - set(flickr_licenses.keys())
                            if not needed_photos:
                                continue

                        res_box = fetch_flickr_bbox_licenses(bbox)
                        for pid, lic_code in res_box.items():
                            if pid in flickr_ids:
                                flickr_licenses[pid] = lic_code

                        if len(flickr_licenses) >= len(flickr_ids):
                            print(
                                "\nAll missing Flickr licenses successfully backfilled! Terminating early..."
                            )
                            break

                print(
                    f"Retrieved {len(flickr_licenses)} Flickr licenses using bulk search."
                )

        # --- Option B: Fallback Individual Queries ---
        remaining_ids = list(flickr_ids - set(flickr_licenses.keys()))
        if remaining_ids:
            if args.log_dirs:
                print(
                    f"\nBulk search left {len(remaining_ids)} photos un-retrieved. Fetching individually..."
                )
            else:
                print(
                    f"\n[WARNING] No --log_dirs provided. Querying Flickr API individually for {len(remaining_ids)} photos."
                )
                print(
                    f"This will take approximately {len(remaining_ids) / 3000:.1f} hours due to Flickr's API limits."
                )
                print(
                    "Provide --log_dirs with your completed box text files to speed this up by 250x."
                )

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(fetch_flickr_individual_license, pid): pid
                    for pid in remaining_ids
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Fetch Flickr License (1-by-1)",
                ):
                    pid, lic_code = future.result()
                    if lic_code is not None:
                        flickr_licenses[pid] = lic_code
    else:
        print("🎉 No missing Flickr licenses found to fetch from API.")

    # --- Save back to dataset ---
    # Standardize License column type and NaNs first
    if "License" not in df.columns:
        df["License"] = None
    df["License"] = (
        df["License"]
        .astype(str)
        .replace({"nan": None, "None": None, "<NA>": None, "": None})
    )

    # Re-apply platform-wide blanket licenses for non-Flickr platforms
    platform_lower = df["Platform"].astype(str).str.lower()

    # Mapillary
    mapillary_mask = (platform_lower == "mapillary") & df["License"].isna()
    if mapillary_mask.any():
        df.loc[mapillary_mask, "License"] = "CC BY-SA 4.0"
        modified = True
        print(
            f" -> Backfilled {mapillary_mask.sum()} Mapillary records with 'CC BY-SA 4.0'"
        )

    # KartaView
    kartaview_mask = (platform_lower == "kartaview") & df["License"].isna()
    if kartaview_mask.any():
        df.loc[kartaview_mask, "License"] = "CC BY-SA 4.0"
        modified = True
        print(
            f" -> Backfilled {kartaview_mask.sum()} KartaView records with 'CC BY-SA 4.0'"
        )

    # iWildCam
    iwildcam_mask = (platform_lower == "iwildcam") & df["License"].isna()
    if iwildcam_mask.any():
        df.loc[iwildcam_mask, "License"] = "CDLA-Permissive-1.0"
        modified = True
        print(
            f" -> Backfilled {iwildcam_mask.sum()} iWildCam records with 'CDLA-Permissive-1.0'"
        )

    # iNaturalist
    inat_mask = (
        platform_lower.str.contains("inaturalist") | (platform_lower == "inat")
    ) & df["License"].isna()
    if inat_mask.any():
        df.loc[inat_mask, "License"] = "CC BY-NC 4.0"
        modified = True
        print(
            f" -> Backfilled {inat_mask.sum()} iNaturalist records with 'CC BY-NC 4.0'"
        )

    if flickr_licenses:
        print("\nMapping retrieved Flickr license codes back to dataset...")
        # Build mapping series
        flickr_map = df["Photo_ID"].astype(str).map(flickr_licenses)
        # Convert numeric codes in flickr_map to standard text labels
        clean_mapped = flickr_map.astype(str).str.split(".").str[0]
        mapped_labels = clean_mapped.map(FLICKR_LICENSE_MAP)
        flickr_map = mapped_labels.fillna(flickr_map)

        df["License"] = df["License"].combine_first(flickr_map)
        modified = True

    # Set any remaining empty or null licenses to the default: 'All Rights Reserved'
    null_or_empty_mask = (
        df["License"].isna()
        | (df["License"].astype(str).str.strip() == "")
        | (df["License"].astype(str) == "nan")
    )
    if null_or_empty_mask.any():
        print(
            f"Setting {null_or_empty_mask.sum():,} remaining empty licenses to default 'All Rights Reserved'..."
        )
        df.loc[null_or_empty_mask, "License"] = "All Rights Reserved"
        modified = True

    if modified:
        print(f"Saving updated database to: {out_path}")
        save_dataframe(df, out_path)
        print("Backfill completed successfully!")
    else:
        print("\nNo licenses needed backfilling; database is already fully up to date.")


if __name__ == "__main__":
    main()
