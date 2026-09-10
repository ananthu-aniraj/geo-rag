"""
WildObs (National Wildlife Camera Database of Australia) Scraper & Pipeline Preparer.

Connects to the WildObs REST API (https://camdbapi.wildobs.org.au/find), discovers open &
shareable camera trap projects following Camtrap DP and Frictionless Data Package specifications,
streams public media records, joins camera station coordinates from deployments and taxonomic
classifications from observations, enforces per-camera quotas and temporal stratification,
and outputs standardized Geo-RAG Parquet and CSV files for direct ingestion by process_scraped_data.py.
"""

import argparse
import concurrent.futures
import os
import sys
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

API_BASE_URL = "https://camdbapi.wildobs.org.au/find"
DEFAULT_PLATFORM = "wildobs"


def load_env_credentials(env_path: str = ".env") -> Optional[str]:
    """Attempts to find and load WILDOBS_API_KEY from environment or .env file."""
    # 1. Check current environment variables
    key = os.environ.get("WILDOBS_API_KEY") or os.environ.get("WILDOBSR_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Check local .env file
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k in ("WILDOBS_API_KEY", "WILDOBSR_API_KEY") and v:
                            os.environ[k] = v
                            return v
        except Exception:
            pass

    return None


def get_api_session(api_key: str) -> requests.Session:
    """Creates a configured requests Session with authentication and exponential backoff retries."""
    session = requests.Session()
    session.headers.update(
        {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "GeoRAG-WildObsScraper/1.0",
        }
    )
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def query_wildobs(
    session: requests.Session,
    collection: str,
    filter_dict: Optional[Dict[str, Any]] = None,
    sort_dict: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    timeout: int = 45,
) -> List[Dict[str, Any]]:
    """
    Executes a query against the WildObs MongoDB REST API.
    Returns the list of document results.
    """
    body: Dict[str, Any] = {"collection": collection}
    if filter_dict:
        body["filter"] = filter_dict
    if sort_dict:
        body["sort"] = sort_dict
    if limit is not None:
        body["limit"] = limit

    try:
        response = session.post(API_BASE_URL, json=body, timeout=timeout)
        if response.status_code != 200:
            print(
                f" [!] API query error for collection '{collection}': HTTP {response.status_code}"
            )
            return []
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return data[0].get("results", [])
        return []
    except Exception as e:
        print(f" [!] Request failed for collection '{collection}': {e}")
        return []


def is_shareable_project(proj_doc: Dict[str, Any]) -> bool:
    """
    Evaluates WildObs data-governance sharing rules (the 'Three-Gate Rule'):
    1. WildObsMetadata.tabularSharingPreference == 'open'
    2. bibliographicCitation contains a persistent RAiD identifier ('https://raid.org/')
    3. bibliographicCitation does NOT contain 'DEMO'
    """
    meta_block = proj_doc.get("WildObsMetadata") or {}
    pref = meta_block.get("tabularSharingPreference", "").lower()
    citation = proj_doc.get("bibliographicCitation", "")

    if isinstance(citation, list) and citation:
        citation_str = str(citation[0])
    else:
        citation_str = str(citation)

    has_open_pref = pref == "open"
    has_valid_raid = "https://raid.org/" in citation_str and "DEMO" not in citation_str
    return has_open_pref and has_valid_raid


def get_all_projects(
    session: requests.Session, only_shareable: bool = True
) -> List[Dict[str, Any]]:
    """Retrieves all project data packages from the metadata collection."""
    results = query_wildobs(session, collection="metadata")
    if not only_shareable:
        return results
    return [p for p in results if is_shareable_project(p)]


def extract_media_license(proj_doc: Dict[str, Any]) -> str:
    """Extracts media license string (e.g. 'CC-BY-4.0') from project metadata."""
    licenses = proj_doc.get("licenses") or []
    for lic in licenses:
        if isinstance(lic, dict):
            if lic.get("scope") == "media" and lic.get("name"):
                return lic["name"]
    for lic in licenses:
        if isinstance(lic, dict) and lic.get("name"):
            return lic["name"]
    return "CC-BY-4.0"


def parse_timestamp(val: Any) -> Optional[pd.Timestamp]:
    """Safely extracts a UTC datetime from MongoDB BSON or ISO string representations."""
    if isinstance(val, dict) and "$date" in val:
        val = val["$date"]
    if pd.isna(val) or not val:
        return None
    try:
        return pd.to_datetime(val, utc=True)
    except Exception:
        return None


def get_season_southern(month: int) -> str:
    """Maps calendar month to meteorological season in the Southern Hemisphere (Australia)."""
    if month in (12, 1, 2):
        return "summer"
    elif month in (3, 4, 5):
        return "fall"
    elif month in (6, 7, 8):
        return "winter"
    else:
        return "spring"


def get_time_of_day(
    hour: int, mode: str = "day_night", day_start: int = 7, day_end: int = 19
) -> str:
    """Classifies hour of day into daylight vs infrared night sensor modalities."""
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
        return "day" if day_start <= hour < day_end else "night"


def fetch_project_deployments(
    session: requests.Session, project_id: str
) -> Dict[str, Dict[str, Any]]:
    """Loads and indexes all deployments for a project by deploymentID."""
    deps = query_wildobs(
        session, collection="deployments", filter_dict={"projectName": project_id}
    )
    dep_dict = {}
    for d in deps:
        dep_id = d.get("deploymentID")
        lat = d.get("latitude")
        lon = d.get("longitude")
        if dep_id and lat is not None and lon is not None:
            try:
                lat_f = float(lat)
                lon_f = float(lon)
                if -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0:
                    dep_dict[dep_id] = {
                        "latitude": lat_f,
                        "longitude": lon_f,
                        "locationName": d.get("locationName", "") or "",
                        "locationID": d.get("locationID", "") or "",
                        "habitat": d.get("habitat", "") or "",
                        "featureType": d.get("featureType", "") or "",
                        "cameraModel": d.get("cameraModel", "") or "",
                        "cameraID": d.get("cameraID", "") or "",
                    }
            except (ValueError, TypeError):
                continue
    return dep_dict


def fetch_project_observations(
    session: requests.Session,
    project_id: str,
    only_animals: bool = True,
    target_taxa: Optional[List[str]] = None,
    batch_size: int = 5000,
) -> Dict[str, Dict[str, Any]]:
    """
    Loads all observations for a project and indexes them by observationID.
    Paginates using observationID cursor to handle large projects.
    """
    obs_dict = {}
    last_id = None
    taxa_set = {t.strip().lower() for t in target_taxa} if target_taxa else None

    filt: Dict[str, Any] = {"projectName": project_id}
    if only_animals:
        filt["observationType"] = "animal"

    while True:
        curr_filt = filt.copy()
        if last_id is not None:
            curr_filt["observationID"] = {"$gt": last_id}

        batch = query_wildobs(
            session,
            collection="observations",
            filter_dict=curr_filt,
            sort_dict={"observationID": 1},
            limit=batch_size,
        )
        if not batch:
            break

        for o in batch:
            obs_id = o.get("observationID")
            if not obs_id:
                continue

            sp_name = o.get("scientificName", "") or ""
            if taxa_set and sp_name.lower() not in taxa_set:
                continue

            obs_dict[obs_id] = {
                "scientificName": sp_name,
                "observationType": o.get("observationType", "") or "",
                "count": o.get("count", 1) or 1,
                "classifiedBy": o.get("classifiedBy", "") or "",
                "eventID": o.get("eventID", "") or obs_id,
            }

        last_id = batch[-1].get("observationID")
        if len(batch) < batch_size or not last_id:
            break

    return obs_dict


def stream_project_public_media(
    session: requests.Session,
    project_id: str,
    batch_size: int = 5000,
):
    """
    Generator yielding batches of public media records for a given project
    using cursor pagination on mediaID.
    """
    last_id = None
    while True:
        filt: Dict[str, Any] = {
            "projectName": project_id,
            "filePublic": True,
        }
        if last_id is not None:
            filt["mediaID"] = {"$gt": last_id}

        batch = query_wildobs(
            session,
            collection="media",
            filter_dict=filt,
            sort_dict={"mediaID": 1},
            limit=batch_size,
        )
        if not batch:
            break

        yield batch

        last_id = batch[-1].get("mediaID")
        if len(batch) < batch_size or not last_id:
            break


def scrape_wildobs_project(
    session: requests.Session,
    project_doc: Dict[str, Any],
    platform_name: str = DEFAULT_PLATFORM,
    max_images_per_camera: int = 4,
    samples_per_bin: int = 2,
    tod_mode: str = "day_night",
    day_start: int = 7,
    day_end: int = 19,
    only_animals: bool = True,
    target_taxa: Optional[List[str]] = None,
    batch_size: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Scrapes, joins, and temporally stratifies media records for a single WildObs project.
    """
    project_id = project_doc["id"]
    project_title = project_doc.get("title", "") or project_id
    media_license = extract_media_license(project_doc)
    citation = project_doc.get("bibliographicCitation", "")
    if isinstance(citation, list) and citation:
        citation_str = str(citation[0])
    else:
        citation_str = str(citation)

    print(f"\n[{project_id}] Loading deployments and observations...")
    dep_map = fetch_project_deployments(session, project_id)
    if not dep_map:
        print(f" -> No valid coordinate deployments found for {project_id}. Skipping.")
        return []

    obs_map = fetch_project_observations(
        session,
        project_id,
        only_animals=only_animals,
        target_taxa=target_taxa,
        batch_size=batch_size,
    )
    print(
        f" -> {len(dep_map)} deployments | {len(obs_map)} filtered observations indexed."
    )

    # cam_bins[dep_id][(season, tod)] = list of candidate records
    cam_bins: Dict[str, Dict[Tuple[str, str], List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    total_media_scanned = 0
    total_media_matched = 0

    print(f"[{project_id}] Streaming public media records...")
    for batch in stream_project_public_media(
        session, project_id, batch_size=batch_size
    ):
        total_media_scanned += len(batch)

        for m in batch:
            dep_id = m.get("deploymentID")
            file_path = m.get("filePath")
            obs_id = m.get("observationID")
            ts_raw = m.get("timestamp")
            media_id = m.get("mediaID")

            # Validate basic fields and URL
            if not dep_id or dep_id not in dep_map:
                continue
            if not file_path or not file_path.startswith("http"):
                continue
            if not media_id:
                continue

            # Parse timestamp
            dt_obj = parse_timestamp(ts_raw)
            if dt_obj is None:
                continue

            # Taxonomic check
            obs_info = obs_map.get(obs_id) if obs_id else None
            is_animal = False
            sp_name = ""
            obs_type = ""
            obj_count = 1
            event_id = obs_id or media_id

            if obs_info:
                obs_type = obs_info.get("observationType", "")
                sp_name = obs_info.get("scientificName", "")
                obj_count = obs_info.get("count", 1)
                event_id = obs_info.get("eventID", obs_id)
                is_animal = obs_type == "animal"
            elif only_animals:
                # If only_animals requested and observation not matching/absent, skip
                continue

            # Temporal binning (Southern Hemisphere)
            season = get_season_southern(dt_obj.month)
            tod = get_time_of_day(
                dt_obj.hour, mode=tod_mode, day_start=day_start, day_end=day_end
            )
            date_val = dt_obj.date()
            iso_ts = dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

            dep_info = dep_map[dep_id]
            bin_key = (season, tod)
            slots = cam_bins[dep_id]
            current_in_bin = slots[bin_key]

            # Sequence/event deduplication across this camera
            existing_events = {r["event_id"] for b in slots.values() for r in b}
            if event_id in existing_events and len(current_in_bin) > 0:
                continue

            existing_dates = {r["date"] for r in current_in_bin}

            candidate_record = {
                "Photo_ID": str(media_id),
                "Platform": platform_name,
                "Latitude": dep_info["latitude"],
                "Longitude": dep_info["longitude"],
                "Image_URL": str(file_path),
                "Captured_At": iso_ts,
                "License": media_license,
                "photo_key": f"{platform_name}_{media_id}",
                "deployment_id": dep_id,
                "location_id": dep_info["locationID"],
                "location_name": dep_info["locationName"],
                "habitat": dep_info["habitat"],
                "feature_type": dep_info["featureType"],
                "scientific_name": sp_name,
                "observation_type": obs_type,
                "individual_count": obj_count,
                "project_id": project_id,
                "project_title": project_title,
                "data_citation": citation_str,
                "season": season,
                "time_of_day": tod,
                "date": date_val,
                "event_id": event_id,
                "is_animal": is_animal,
            }

            if len(current_in_bin) < samples_per_bin:
                if date_val not in existing_dates or len(current_in_bin) == 0:
                    current_in_bin.append(candidate_record)
                    total_media_matched += 1
            else:
                # Upgrade blank with animal detection if possible
                if is_animal:
                    for idx_r, r in enumerate(current_in_bin):
                        if not r["is_animal"] and date_val != r["date"]:
                            current_in_bin[idx_r] = candidate_record
                            break

    # Balanced selection per camera deployment
    selected_records = []
    for dep_id, bins in cam_bins.items():
        cam_selected = []

        # Pass 1: 1 best candidate per bin (animals preferred)
        for b_key, candidates in bins.items():
            if candidates and len(cam_selected) < max_images_per_camera:
                sorted_cands = sorted(
                    candidates, key=lambda x: 0 if x["is_animal"] else 1
                )
                cam_selected.append(sorted_cands[0])

        # Pass 2: Fill remaining camera quota with second candidate
        for b_key, candidates in bins.items():
            if len(candidates) > 1 and len(cam_selected) < max_images_per_camera:
                sorted_cands = sorted(
                    candidates, key=lambda x: 0 if x["is_animal"] else 1
                )
                cam_selected.append(sorted_cands[1])

        for rec in cam_selected:
            rec.pop("is_animal", None)
            rec.pop("date", None)
            rec.pop("event_id", None)
            selected_records.append(rec)

    print(
        f" -> Scanned {total_media_scanned:,} media records | Selected {len(selected_records):,} stratified images across {len(cam_bins)} cameras."
    )
    return selected_records


def download_single_image(
    item: Dict[str, Any], output_dir: str, timeout: int = 15
) -> Tuple[bool, str]:
    """Downloads a single image from WildObs static storage."""
    url = item["Image_URL"]
    photo_id = item["Photo_ID"]
    ext = os.path.splitext(url.split("?")[0])[-1] or ".jpg"
    target_path = os.path.join(output_dir, f"{photo_id}{ext}")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        return True, target_path

    os.makedirs(output_dir, exist_ok=True)
    temp_path = f"{target_path}.tmp_{os.getpid()}_{threading.get_ident()}"
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code == 200:
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            os.replace(temp_path, target_path)
            return True, target_path
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, ""
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, ""


def batch_download_images(
    records: List[Dict[str, Any]], output_dir: str, threads: int = 16
):
    """Downloads images in parallel for offline use."""
    print(
        f"\nDownloading {len(records):,} images to: {output_dir} using {threads} threads..."
    )
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(download_single_image, rec, output_dir): rec
            for rec in records
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Downloading WildObs imagery",
        ):
            ok, local_path = future.result()
            rec = futures[future]
            if ok:
                success_count += 1
                rec["Image_Location"] = local_path

    print(
        f"Download complete: {success_count}/{len(records)} images saved successfully."
    )


def print_project_summary(projects: List[Dict[str, Any]], session: requests.Session):
    """Displays a clean summary table of all projects available on WildObs."""
    print("\n" + "=" * 110)
    print(f"{'WildObs Project ID':<48} | {'Sharing':<8} | {'RAiD':<5} | {'Title'}")
    print("=" * 110)
    for p in projects:
        pid = p.get("id", "")
        title = p.get("title", "") or ""
        pref = p.get("WildObsMetadata", {}).get("tabularSharingPreference", "unknown")
        cit = p.get("bibliographicCitation", "")
        cit_str = cit[0] if isinstance(cit, list) and cit else str(cit)
        has_raid = (
            "Yes" if "https://raid.org/" in cit_str and "DEMO" not in cit_str else "No"
        )
        print(f"{pid:<48} | {pref:<8} | {has_raid:<5} | {title[:42]}")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape and prepare camera trap data from the WildObs Australian database."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="WildObs API Key (defaults to WILDOBS_API_KEY from environment or .env).",
    )
    parser.add_argument(
        "--project_ids",
        type=str,
        nargs="+",
        default=None,
        help="One or more specific WildObs project IDs to scrape.",
    )
    parser.add_argument(
        "--all_open",
        action="store_true",
        help="Scrape all open & shareable projects in the database.",
    )
    parser.add_argument(
        "--list_projects",
        action="store_true",
        help="List all accessible projects in the database and exit.",
    )
    parser.add_argument(
        "--max_images_per_camera",
        type=int,
        default=4,
        help="Maximum images to select per camera deployment (default: 4).",
    )
    parser.add_argument(
        "--samples_per_bin",
        type=int,
        default=2,
        help="Maximum images per (season, time_of_day) bin (default: 2).",
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
        help="Hour of day (0-23) when daylight sensor begins (default: 7).",
    )
    parser.add_argument(
        "--day_end",
        type=int,
        default=19,
        help="Hour of day (0-23) when daylight sensor ends (default: 19).",
    )
    parser.add_argument(
        "--include_blanks",
        action="store_true",
        help="Allow blank/unclassified camera triggers (default: animals only).",
    )
    parser.add_argument(
        "--taxa",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific binomial scientific names (e.g. 'Alectura lathami').",
    )
    parser.add_argument(
        "--platform_name",
        type=str,
        default=DEFAULT_PLATFORM,
        help="Platform identifier for Geo-RAG schema (default: wildobs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./wildobs_outputs/wildobs_filtered.parquet",
        help="Output path for filtered Parquet (defaults to ./wildobs_outputs/wildobs_filtered.parquet).",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Disable saving companion CSV copy.",
    )
    parser.add_argument(
        "--download_images",
        action="store_true",
        help="Download image binaries locally to an offline image directory.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory to save downloaded images (defaults to <out_dir>/images).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of threads for concurrent image downloading (default: 16).",
    )

    args = parser.parse_args()

    # 1. Resolve API key
    api_key = args.api_key or load_env_credentials()
    if not api_key:
        print(" [!] Error: No WildObs API key found.")
        print("     Set WILDOBS_API_KEY in your .env file or pass --api_key <KEY>.")
        print(
            "     You can create a free API key at: https://dashboard.wildobs.org.au/"
        )
        sys.exit(1)

    session = get_api_session(api_key)

    # 2. List projects mode
    if args.list_projects:
        print("Querying WildObs projects catalog...")
        all_projs = get_all_projects(session, only_shareable=False)
        print_project_summary(all_projs, session)
        sys.exit(0)

    # 3. Resolve target projects
    target_projs = []
    if args.project_ids:
        # User specified explicit projects
        all_projs_dict = {
            p["id"]: p for p in get_all_projects(session, only_shareable=False)
        }
        for pid in args.project_ids:
            if pid in all_projs_dict:
                target_projs.append(all_projs_dict[pid])
            else:
                print(
                    f" [!] Warning: Project '{pid}' was not found in WildObs metadata."
                )
    elif args.all_open:
        target_projs = get_all_projects(session, only_shareable=True)
    else:
        # Default: scrape all open shareable projects
        target_projs = get_all_projects(session, only_shareable=True)

    if not target_projs:
        print(" [!] No matching shareable projects to process.")
        sys.exit(1)

    print(f"Discovered {len(target_projs)} target project(s) to scrape from WildObs:")
    for p in target_projs:
        print(f" - {p['id']} ({p.get('title', '')[:50]}...)")

    # 4. Scrape and stratify
    all_selected = []
    for p_doc in target_projs:
        p_records = scrape_wildobs_project(
            session=session,
            project_doc=p_doc,
            platform_name=args.platform_name,
            max_images_per_camera=args.max_images_per_camera,
            samples_per_bin=args.samples_per_bin,
            tod_mode=args.tod_mode,
            day_start=args.day_start,
            day_end=args.day_end,
            only_animals=not args.include_blanks,
            target_taxa=args.taxa,
        )
        all_selected.extend(p_records)

    if not all_selected:
        print(" [!] No media records matched the filtering criteria.")
        sys.exit(0)

    # 5. Optional image downloading
    if args.download_images:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        img_dir = args.image_dir or os.path.join(out_dir, "images")
        batch_download_images(all_selected, img_dir, threads=args.threads)

    # 6. Format and save
    df_result = pd.DataFrame(all_selected)

    # Ensure required columns at front
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

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nSaving {len(df_result):,} records to Parquet: {args.output}")
    df_result.to_parquet(args.output, index=False, compression="zstd")

    if not args.no_csv:
        csv_path = os.path.splitext(args.output)[0] + ".csv"
        print(f"Saving companion CSV: {csv_path}")
        df_result.to_csv(csv_path, index=False)

    print("\n" + "=" * 55)
    print("WildObs Dataset Scraping & Stratification Complete")
    print("=" * 55)
    print(f"Total projects processed:    {df_result['project_id'].nunique():,}")
    print(f"Total cameras processed:     {df_result['deployment_id'].nunique():,}")
    print(f"Total filtered images:       {len(df_result):,}")
    print(
        f"Images per camera (mean):    {len(df_result) / df_result['deployment_id'].nunique():.2f}"
    )
    print("\nSeason Distribution:")
    print(df_result["season"].value_counts().to_string())
    print("\nTime of Day Distribution:")
    print(df_result["time_of_day"].value_counts().to_string())
    print("\nTop 10 Taxa / Species:")
    print(df_result["scientific_name"].value_counts().head(10).to_string())
    print("=" * 55)


if __name__ == "__main__":
    main()
