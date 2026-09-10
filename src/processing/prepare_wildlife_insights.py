"""
Script to prepare and filter camera trap metadata from Wildlife Insights (e.g., Snapshot USA 2024).

This script performs stratified temporal sampling (across seasons and times of day)
for each camera deployment, prioritizes wildlife detections over blanks, enforces
burst sequence and date diversity, filters out human detections and fuzzed GPS coordinates,
and formats the resulting dataset for direct ingestion by process_scraped_data.py.
"""

import argparse
import glob
import os
import re
import time
from collections import defaultdict

import pandas as pd


def get_season(month: int) -> str:
    """Maps a calendar month (1-12) to meteorological season."""
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "fall"


def get_time_of_day(
    hour: int, mode: str = "day_night", day_start: int = 7, day_end: int = 19
) -> str:
    """
    Classifies hour of day into sensor modalities / periods.
    Modes:
      - 'day_night': 'day' (color daylight sensor) vs 'night' (infrared monochrome)
      - '4_period': 'dawn', 'day', 'dusk', 'night'
    """
    if mode == "4_period":
        if 5 <= hour < 7:
            return "dawn"
        elif 7 <= hour < 17:
            return "day"
        elif 17 <= hour < 19:
            return "dusk"
        else:
            return "night"
    else:
        if day_start <= hour < day_end:
            return "day"
        else:
            return "night"


def load_projects(projects_path: str) -> pd.DataFrame:
    """Loads projects.csv if available to attach project name, licensing, and citation metadata."""
    if not projects_path or not os.path.exists(projects_path):
        return pd.DataFrame()
    print(f"Loading project metadata from: {projects_path}")
    try:
        df_proj = pd.read_csv(projects_path)
        proj_cols = [
            c
            for c in [
                "project_id",
                "project_name",
                "project_short_name",
                "metadata_license",
                "image_license",
                "data_citation",
            ]
            if c in df_proj.columns
        ]
        if proj_cols:
            df_proj = df_proj[proj_cols].drop_duplicates(subset=["project_id"])
            for _, row in df_proj.iterrows():
                p_id = row.get("project_id", "")
                p_name = row.get("project_name", "")
                p_short = row.get("project_short_name", "")
                m_lic = row.get("metadata_license", "")
                i_lic = row.get("image_license", "")
                print(
                    f" -> Found project [{p_id}]: {p_name} ({p_short}) | Image License: {i_lic}, Metadata License: {m_lic}"
                )
            return df_proj
    except Exception as e:
        print(f" -> Warning: Failed to load projects.csv: {e}")
    return pd.DataFrame()


def load_deployments(
    deployments_path: str,
    projects_path: str = None,
    exclude_fuzzed: bool = True,
    only_functioning: bool = True,
) -> pd.DataFrame:
    """
    Loads deployments.csv, extracts high-precision functioning camera coordinates,
    and returns a DataFrame indexed by deployment_id.
    """
    if not os.path.exists(deployments_path):
        raise FileNotFoundError(f"Deployments file not found at: {deployments_path}")

    print(f"Loading deployments from: {deployments_path}")
    df_dep = pd.read_csv(deployments_path)
    initial_count = len(df_dep)
    print(f" -> Found {initial_count} total deployments.")

    # Filter invalid GPS
    valid_mask = df_dep["latitude"].notna() & df_dep["longitude"].notna()
    df_dep = df_dep[valid_mask].copy()

    # Filter fuzzed coordinates
    if exclude_fuzzed and "fuzzed" in df_dep.columns:
        fuzzed_count = (df_dep["fuzzed"] == True).sum()  # noqa: E712
        df_dep = df_dep[df_dep["fuzzed"] == False]  # noqa: E712
        if "deployment_fuzzed" in df_dep.columns:
            df_dep = df_dep[df_dep["deployment_fuzzed"] == False]  # noqa: E712
        print(
            f" -> Excluded {fuzzed_count} fuzzed deployments to preserve GPS precision."
        )

    # Filter functioning cameras
    if only_functioning and "camera_functioning" in df_dep.columns:
        func_mask = (
            df_dep["camera_functioning"].astype(str).str.strip().str.lower()
            == "camera functioning"
        )
        non_func_count = (~func_mask).sum()
        df_dep = df_dep[func_mask]
        print(f" -> Excluded {non_func_count} non-functioning deployments.")

    # Optionally attach project metadata if projects.csv is found
    if not projects_path:
        default_proj = os.path.join(os.path.dirname(deployments_path), "projects.csv")
        if os.path.exists(default_proj):
            projects_path = default_proj

    if projects_path and os.path.exists(projects_path):
        df_proj = load_projects(projects_path)
        if not df_proj.empty and "project_id" in df_dep.columns:
            df_dep = df_dep.merge(df_proj, on="project_id", how="left")

    print(f" -> {len(df_dep)} valid deployments retained.")

    keep_cols = [
        "deployment_id",
        "latitude",
        "longitude",
        "placename",
        "subproject_name",
        "feature_type",
        "camera_functioning",
        "camera_id",
        "project_id",
        "project_name",
        "project_short_name",
        "metadata_license",
        "image_license",
        "data_citation",
    ]
    avail_cols = [c for c in keep_cols if c in df_dep.columns]
    return df_dep[avail_cols].drop_duplicates(subset=["deployment_id"])


def natural_sort_key(s: str):
    """Sort strings with embedded numbers naturally (e.g., images_2.csv before images_10.csv)."""
    match = re.search(r"(\d+)", os.path.basename(s))
    return int(match.group(1)) if match else s


def prepare_wildlife_insights(
    data_dir: str,
    deployments_path: str = None,
    projects_path: str = None,
    images_glob: str = None,
    output_path: str = None,
    save_csv: bool = True,
    platform_name: str = "wildlife_insights",
    samples_per_bin: int = 2,
    max_images_per_camera: int = 4,
    tod_mode: str = "day_night",
    day_start: int = 7,
    day_end: int = 19,
    exclude_humans: bool = True,
    exclude_fuzzed: bool = True,
    only_functioning: bool = True,
    chunksize: int = 250000,
) -> pd.DataFrame:
    """
    Main pipeline to process chunked or single Wildlife Insights image CSVs, stratify per camera,
    and save a standardized dataset for process_scraped_data.py.
    """
    t_start = time.time()

    if not deployments_path:
        deployments_path = os.path.join(data_dir, "deployments.csv")
    if not projects_path:
        default_proj = os.path.join(data_dir, "projects.csv")
        if os.path.exists(default_proj):
            projects_path = default_proj
    if not output_path:
        clean_name = re.sub(r"[^\w\-]", "_", platform_name.strip().lower())
        output_path = os.path.join(data_dir, f"{clean_name}_filtered.parquet")

    # 1. Load valid deployments
    df_deployments = load_deployments(
        deployments_path,
        projects_path=projects_path,
        exclude_fuzzed=exclude_fuzzed,
        only_functioning=only_functioning,
    )
    valid_dep_ids = set(df_deployments["deployment_id"].dropna().unique())

    # 2. Discover image files (supports both chunked images_*.csv and single images.csv)
    if not images_glob:
        chunked = sorted(
            glob.glob(os.path.join(data_dir, "images_*.csv")), key=natural_sort_key
        )
        single = os.path.join(data_dir, "images.csv")
        if chunked:
            img_files = chunked
        elif os.path.exists(single):
            img_files = [single]
        else:
            raise FileNotFoundError(
                f"No image CSV files found in: {data_dir} (searched for images_*.csv and images.csv)"
            )
    else:
        img_files = sorted(glob.glob(images_glob), key=natural_sort_key)
        if not img_files:
            raise FileNotFoundError(f"No image CSV files found matching: {images_glob}")
    print(f"Found {len(img_files)} image CSV file(s) to scan.")

    # Determine default image license (fallback to projects.csv's image_license if available, else 'CC-BY')
    default_img_license = "CC-BY"
    if "image_license" in df_deployments.columns:
        proj_lics = df_deployments["image_license"].dropna().unique()
        if len(proj_lics) > 0 and str(proj_lics[0]).strip():
            default_img_license = str(proj_lics[0]).strip()

    # 3. Stratified Candidate Collector
    # cam_bins[dep_id][(season, tod)] = list of candidate records
    cam_bins = defaultdict(lambda: defaultdict(list))

    load_cols = [
        "image_id",
        "deployment_id",
        "sequence_id",
        "filename",
        "location",
        "timestamp",
        "common_name",
        "species",
        "genus",
        "family",
        "order",
        "class",
        "license",
        "fuzzed",
    ]

    total_rows_scanned = 0
    total_valid_rows = 0

    for file_idx, fpath in enumerate(img_files, 1):
        f_name = os.path.basename(fpath)
        t_f = time.time()
        file_valid_count = 0

        # Read first line to inspect available columns
        sample_df = pd.read_csv(fpath, nrows=1)
        actual_cols = [c for c in load_cols if c in sample_df.columns]

        for chunk in pd.read_csv(
            fpath, usecols=actual_cols, chunksize=chunksize, low_memory=False
        ):
            total_rows_scanned += len(chunk)

            # Filter valid deployments
            chunk = chunk[chunk["deployment_id"].isin(valid_dep_ids)]
            if chunk.empty:
                continue

            # Drop missing locations or timestamps
            chunk = chunk.dropna(subset=["location", "timestamp"])
            if chunk.empty:
                continue

            # Exclude fuzzed images if column present
            if exclude_fuzzed and "fuzzed" in chunk.columns:
                chunk = chunk[chunk["fuzzed"] == False]  # noqa: E712
                if chunk.empty:
                    continue

            # Exclude humans and calibration tags
            if exclude_humans and "common_name" in chunk.columns:
                cname_lower = chunk["common_name"].astype(str).str.lower()
                human_mask = cname_lower.str.contains(
                    "human"
                ) | cname_lower.str.contains("calibration")
                chunk = chunk[~human_mask]
                if chunk.empty:
                    continue

            # Parse timestamps
            dts = pd.to_datetime(chunk["timestamp"], errors="coerce")
            valid_ts_mask = dts.notna()
            if not valid_ts_mask.any():
                continue

            chunk = chunk[valid_ts_mask]
            dts = dts[valid_ts_mask]

            seasons = dts.dt.month.map(get_season)
            tods = dts.dt.hour.map(
                lambda h: get_time_of_day(
                    h, mode=tod_mode, day_start=day_start, day_end=day_end
                )
            )
            dates = dts.dt.date

            # Extract standard fields
            dep_vals = chunk["deployment_id"].tolist()
            img_vals = chunk["image_id"].tolist()
            if "sequence_id" in chunk.columns:
                seq_vals = (
                    chunk["sequence_id"].fillna(chunk["image_id"]).astype(str).tolist()
                )
            else:
                seq_vals = [str(x) for x in img_vals]
            fname_vals = (
                chunk["filename"].tolist()
                if "filename" in chunk.columns
                else [""] * len(chunk)
            )
            loc_vals = chunk["location"].tolist()
            ts_vals = chunk["timestamp"].tolist()
            cname_vals = (
                chunk["common_name"].tolist()
                if "common_name" in chunk.columns
                else [""] * len(chunk)
            )
            species_vals = (
                chunk["species"].tolist()
                if "species" in chunk.columns
                else [""] * len(chunk)
            )
            genus_vals = (
                chunk["genus"].tolist()
                if "genus" in chunk.columns
                else [""] * len(chunk)
            )
            family_vals = (
                chunk["family"].tolist()
                if "family" in chunk.columns
                else [""] * len(chunk)
            )
            order_vals = (
                chunk["order"].tolist()
                if "order" in chunk.columns
                else [""] * len(chunk)
            )
            class_vals = (
                chunk["class"].tolist()
                if "class" in chunk.columns
                else [""] * len(chunk)
            )
            lic_vals = (
                chunk["license"].tolist()
                if "license" in chunk.columns
                else ["CC0"] * len(chunk)
            )

            for (
                dep,
                img_id,
                seq_id,
                fname,
                loc,
                ts,
                cname,
                sp,
                gen,
                fam,
                ordr,
                cls_name,
                lic,
                season,
                tod,
                dt_val,
                dt_obj,
            ) in zip(
                dep_vals,
                img_vals,
                seq_vals,
                fname_vals,
                loc_vals,
                ts_vals,
                cname_vals,
                species_vals,
                genus_vals,
                family_vals,
                order_vals,
                class_vals,
                lic_vals,
                seasons,
                tods,
                dates,
                dts,
            ):
                file_valid_count += 1
                slots = cam_bins[dep]
                bin_key = (season, tod)
                current_in_bin = slots[bin_key]

                # Format timestamp to ISO 8601 string
                iso_ts = dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
                is_animal = bool(cname and str(cname).strip().lower() != "blank")

                # Diversity check: avoid duplicate sequences within the entire camera
                existing_seqs = {r["sequence_id"] for b in slots.values() for r in b}
                if str(seq_id) in existing_seqs and len(current_in_bin) > 0:
                    continue

                existing_dates = {r["date"] for r in current_in_bin}

                candidate_record = {
                    "Photo_ID": str(img_id),
                    "Platform": platform_name,
                    "Image_URL": str(loc),
                    "Captured_At": iso_ts,
                    "License": str(lic)
                    if pd.notna(lic) and str(lic).strip()
                    else default_img_license,
                    "photo_key": f"{platform_name}_{img_id}",
                    "deployment_id": dep,
                    "sequence_id": str(seq_id),
                    "filename": str(fname),
                    "common_name": str(cname) if pd.notna(cname) else "",
                    "species": str(sp) if pd.notna(sp) else "",
                    "genus": str(gen) if pd.notna(gen) else "",
                    "family": str(fam) if pd.notna(fam) else "",
                    "order": str(ordr) if pd.notna(ordr) else "",
                    "class": str(cls_name) if pd.notna(cls_name) else "",
                    "season": season,
                    "time_of_day": tod,
                    "date": dt_val,
                    "is_animal": is_animal,
                }

                if len(current_in_bin) < samples_per_bin:
                    # Append if new date or first item in this bin
                    if dt_val not in existing_dates or len(current_in_bin) == 0:
                        current_in_bin.append(candidate_record)
                else:
                    # Slot is full: upgrade Blank with Animal if available
                    if is_animal:
                        for idx_r, r in enumerate(current_in_bin):
                            if not r["is_animal"] and dt_val != r["date"]:
                                current_in_bin[idx_r] = candidate_record
                                break

        total_valid_rows += file_valid_count
        print(
            f" [{file_idx}/{len(img_files)}] {f_name}: matched {file_valid_count:,} records in {time.time() - t_f:.2f}s"
        )

    print(f"\nScanned {total_rows_scanned:,} total rows across {len(img_files)} files.")
    print(f"Captured candidate images for {len(cam_bins)} unique camera deployments.")

    # 4. Final Balanced Selection per Camera
    selected_records = []
    for dep, bins in cam_bins.items():
        cam_selected = []

        # Pass 1: Select 1 best candidate per bin (animals preferred)
        for b_key, candidates in bins.items():
            if candidates and len(cam_selected) < max_images_per_camera:
                sorted_cands = sorted(
                    candidates, key=lambda x: 0 if x["is_animal"] else 1
                )
                cam_selected.append(sorted_cands[0])

        # Pass 2: Fill remaining camera quota with second candidates
        for b_key, candidates in bins.items():
            if len(candidates) > 1 and len(cam_selected) < max_images_per_camera:
                sorted_cands = sorted(
                    candidates, key=lambda x: 0 if x["is_animal"] else 1
                )
                cam_selected.append(sorted_cands[1])

        # Drop internal helper 'is_animal' and 'date' before saving
        for rec in cam_selected:
            rec.pop("is_animal", None)
            rec.pop("date", None)

        selected_records.extend(cam_selected)

    df_selected = pd.DataFrame(selected_records)
    print(f"\nTotal selected images after stratification: {len(df_selected):,}")

    # 5. Join GPS coordinates & deployment metadata
    df_result = df_selected.merge(
        df_deployments,
        on="deployment_id",
        how="inner",
    )
    # Standardize column casing
    df_result["Latitude"] = pd.to_numeric(df_result["latitude"], errors="coerce")
    df_result["Longitude"] = pd.to_numeric(df_result["longitude"], errors="coerce")
    df_result.drop(columns=["latitude", "longitude"], inplace=True)

    # Ensure required columns are ordered at the front for process_scraped_data.py
    required_cols = [
        "Photo_ID",
        "Platform",
        "Latitude",
        "Longitude",
        "Image_URL",
        "Captured_At",
        "License",
        "photo_key",
    ]
    extra_cols = [c for c in df_result.columns if c not in required_cols]
    final_cols = required_cols + extra_cols
    df_result = df_result[final_cols]

    # 6. Save outputs
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving filtered metadata to Parquet: {output_path}")
    df_result.to_parquet(output_path, index=False, compression="zstd")

    if save_csv:
        csv_path = os.path.splitext(output_path)[0] + ".csv"
        print(f"Saving filtered metadata to CSV: {csv_path}")
        df_result.to_csv(csv_path, index=False)

    total_time = time.time() - t_start
    print("\n" + "=" * 50)
    print("Wildlife Insights Preparation Complete")
    print("=" * 50)
    print(f"Total cameras processed:     {df_result['deployment_id'].nunique():,}")
    print(f"Total filtered images:       {len(df_result):,}")
    print(
        f"Images per camera (mean):    {len(df_result) / df_result['deployment_id'].nunique():.2f}"
    )
    print(f"Total execution time:        {total_time:.2f}s")
    print("\nSeason Distribution:")
    print(df_result["season"].value_counts().to_string())
    print("\nTime of Day Distribution:")
    print(df_result["time_of_day"].value_counts().to_string())
    print("\nTop 10 Taxa / Common Names:")
    print(df_result["common_name"].value_counts().head(10).to_string())
    print("=" * 50)

    return df_result


def main():
    parser = argparse.ArgumentParser(
        description="Filter and prepare Wildlife Insights (Snapshot USA or custom exports) camera trap metadata."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/user/aaniraj/home/Documents/Projects/data/wildlife-insights_a884a154-046e-41bd-85c8-d4e76cba3435_all-platform-data",
        help="Directory containing deployments.csv and images.csv (or images_*.csv).",
    )
    parser.add_argument(
        "--deployments_file",
        type=str,
        default=None,
        help="Path to deployments.csv (defaults to <data_dir>/deployments.csv).",
    )
    parser.add_argument(
        "--projects_file",
        type=str,
        default=None,
        help="Path to projects.csv (defaults to <data_dir>/projects.csv if present).",
    )
    parser.add_argument(
        "--images_glob",
        type=str,
        default=None,
        help="Glob pattern or path for image CSVs (defaults to <data_dir>/images_*.csv or <data_dir>/images.csv).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for filtered Parquet (defaults to <data_dir>/<platform_name>_filtered.parquet).",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Disable saving companion CSV copy.",
    )
    parser.add_argument(
        "--platform_name",
        type=str,
        default="wildlife_insights",
        help="Platform identifier (default: wildlife_insights).",
    )
    parser.add_argument(
        "--samples_per_bin",
        type=int,
        default=2,
        help="Candidate images to collect per (season, time_of_day) bin (default: 2).",
    )
    parser.add_argument(
        "--max_images_per_camera",
        type=int,
        default=4,
        help="Maximum images to select per camera deployment (default: 4).",
    )
    parser.add_argument(
        "--tod_mode",
        type=str,
        choices=["day_night", "4_period"],
        default="day_night",
        help="Time-of-day binning mode: 'day_night' (default) or '4_period'.",
    )
    parser.add_argument(
        "--day_start",
        type=int,
        default=7,
        help="Hour of day (0-23) when daytime sensor starts (default: 7).",
    )
    parser.add_argument(
        "--day_end",
        type=int,
        default=19,
        help="Hour of day (0-23) when daytime sensor ends (default: 19).",
    )
    parser.add_argument(
        "--include_humans",
        action="store_true",
        help="Allow human and camera trapper images (default: excluded).",
    )
    parser.add_argument(
        "--include_fuzzed",
        action="store_true",
        help="Allow fuzzed coordinate deployments and images (default: excluded).",
    )
    parser.add_argument(
        "--include_non_functioning",
        action="store_true",
        help="Allow camera deployments that experienced failures (default: excluded).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250000,
        help="Chunk size for reading CSV files (default: 250,000).",
    )

    args = parser.parse_args()

    prepare_wildlife_insights(
        data_dir=args.data_dir,
        deployments_path=args.deployments_file,
        projects_path=args.projects_file,
        images_glob=args.images_glob,
        output_path=args.output_path,
        save_csv=not args.no_csv,
        platform_name=args.platform_name,
        samples_per_bin=args.samples_per_bin,
        max_images_per_camera=args.max_images_per_camera,
        tod_mode=args.tod_mode,
        day_start=args.day_start,
        day_end=args.day_end,
        exclude_humans=not args.include_humans,
        exclude_fuzzed=not args.include_fuzzed,
        only_functioning=not args.include_non_functioning,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
