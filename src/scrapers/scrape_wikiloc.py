"""
Wikiloc Trail & Waypoint Scraper.

Crawls targeted geographic regions and trails on Wikiloc using Playwright with stealth,
extracts high-resolution geolocated waypoint photos and rich trail metadata from embedded
JSON-LD schemas, downloads verified image binaries locally with browser headers, and saves
standardized Geo-RAG Parquet and companion CSV datasets.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

from src.utils.io import is_valid_image_file, save_dataframe

DEFAULT_PLATFORM = "wikiloc"
DEFAULT_ACTIVITY = "hiking"
DEFAULT_CHROME_PATH = "/usr/bin/google-chrome"
WIKILOC_BASE_URL = "https://www.wikiloc.com"

# Standard browser headers required for Wikiloc CDN image streaming
CDN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.wikiloc.com/",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "sec-ch-ua": '"Chromium";v="146", "Not A(Brand";v="24", "Google Chrome";v="146"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
    "sec-fetch-dest": "image",
    "sec-fetch-mode": "no-cors",
    "sec-fetch-site": "cross-site",
}


def find_chrome_executable(user_path: Optional[str] = None) -> Optional[str]:
    """Detects Google Chrome or Chromium executable on the system."""
    candidates = [
        user_path,
        os.environ.get("CHROME_PATH"),
        DEFAULT_CHROME_PATH,
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def get_image_session() -> requests.Session:
    """Configures a thread-safe requests session with retries for image downloads."""
    session = requests.Session()
    session.headers.update(CDN_HEADERS)
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Transnational biomes and remote physical regions mapped to their constituent Wikiloc regional slugs
BIOME_PRESETS: Dict[str, List[str]] = {
    "sahara": [
        "morocco/draa-tafilalet",
        "algeria/tamanrasset",
        "mauritania/adrar",
        "egypt/new-valley",
        "tunisia/tataouine",
    ],
    "sahara-desert": [
        "morocco/draa-tafilalet",
        "algeria/tamanrasset",
        "mauritania/adrar",
        "egypt/new-valley",
        "tunisia/tataouine",
    ],
    "patagonia": [
        "chile/magallanes-y-antartica-chilena",
        "argentina/tierra-del-fuego",
        "argentina/santa-cruz",
        "chile/aysen",
    ],
    "outback": [
        "australia/western-australia",
        "australia/northern-territory",
        "australia/south-australia",
    ],
    "australian-outback": [
        "australia/western-australia",
        "australia/northern-territory",
        "australia/south-australia",
    ],
    "atacama": [
        "chile/antofagasta",
        "chile/atacama",
        "bolivia/potosi",
    ],
    "atacama-desert": [
        "chile/antofagasta",
        "chile/atacama",
        "bolivia/potosi",
    ],
    "himalayas": [
        "nepal",
        "india/himachal-pradesh",
        "india/ladakh",
        "india/uttarakhand",
    ],
    "arctic": [
        "greenland",
        "norway/svalbard-and-jan-mayen",
        "norway/troms-og-finnmark",
        "iceland",
    ],
    "alps": [
        "switzerland/valais",
        "france/rhone-alpes",
        "italy/trentino-alto-adige",
        "austria/tyrol",
    ],
    "andes": [
        "peru/ancash",
        "peru/cusco",
        "bolivia/potosi",
        "argentina/mendoza",
    ],
    "gobi": [
        "mongolia/umnugovi",
        "mongolia/dornogovi",
    ],
    "gobi-desert": [
        "mongolia/umnugovi",
        "mongolia/dornogovi",
    ],
    "namib": [
        "namibia/erongo",
        "namibia/hardap",
    ],
    "namib-desert": [
        "namibia/erongo",
        "namibia/hardap",
    ],
    "pyrenees": [
        "spain/aragon/huesca",
        "spain/catalonia/lleida",
        "france/midi-pyrenees",
    ],
}

COUNTRY_ALIASES: Dict[str, str] = {
    "espana": "spain",
    "estados-unidos": "united-states",
    "united-states-of-america": "united-states",
    "deutschland": "germany",
    "italia": "italy",
    "osterreich": "austria",
    "schweiz": "switzerland",
    "suisse": "switzerland",
    "svizzera": "switzerland",
    "kalaallit-nunaat": "greenland",
}


def slugify(text: str) -> str:
    """Converts a natural language place name into a lowercase, hyphenated slug without accents."""
    norm = (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("utf-8")
    )
    norm = re.sub(r"[^\w\s-]", "", norm).strip().lower()
    return re.sub(r"[-\s]+", "-", norm)


def resolve_query_via_nominatim(query: str) -> Optional[str]:
    """Resolves natural language queries (landmarks, parks, ranges) via OpenStreetMap Nominatim into a Wikiloc slug."""
    headers = {"User-Agent": "GeoRAG-WikilocResolver/1.0"}
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(query)}&format=json&addressdetails=1&limit=1"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data:
                addr = data[0].get("address", {})
                country = addr.get("country", "")
                state = (
                    addr.get("state")
                    or addr.get("region")
                    or addr.get("province")
                    or addr.get("state_district")
                    or addr.get("county")
                    or ""
                )
                if country:
                    c_slug = slugify(country)
                    c_slug = COUNTRY_ALIASES.get(c_slug, c_slug)
                    s_slug = slugify(state)
                    # Clean common prefixes like 'region-de-', 'province-of-', 'state-of-'
                    s_slug = re.sub(
                        r"^(?:region-de-|provincia-de-|state-of-|shire-of-)",
                        "",
                        s_slug,
                    )
                    if s_slug:
                        return f"{c_slug}/{s_slug}"
                    return c_slug
    except Exception:
        pass
    return None


def resolve_regional_inputs(inputs: List[str]) -> List[str]:
    """
    Expands biome presets (e.g. 'sahara' -> 5 desert regions), resolves natural landmarks
    via Nominatim (e.g. 'Karijini' -> 'australia/western-australia'), or preserves explicit slugs.
    """
    resolved: List[str] = []
    for inp in inputs:
        clean = inp.strip()
        if not clean:
            continue
        key = slugify(clean)

        # 1. Biome preset expansion
        if key in BIOME_PRESETS:
            expanded = BIOME_PRESETS[key]
            print(
                f" -> Biome preset match: '{clean}' automatically expanded to {len(expanded)} regional slugs:"
            )
            for ex in expanded:
                print(f"      * {ex}")
            resolved.extend(expanded)
            continue

        # 2. Direct slug or full URL
        if "/" in clean or clean.startswith("http"):
            resolved.append(clean)
            continue

        # 3. Geocode natural place name via Nominatim
        print(f" -> Geocoding location query: '{clean}'...")
        geo_slug = resolve_query_via_nominatim(clean)
        if geo_slug:
            print(f"    Resolved '{clean}' -> '{geo_slug}'")
            resolved.append(geo_slug)
        else:
            # Fallback to direct slug
            resolved.append(key)

    return list(dict.fromkeys(resolved))


def format_region_url(region_or_url: str, activity: str = DEFAULT_ACTIVITY) -> str:
    """Normalizes an input string to a full canonical Wikiloc regional trail URL."""
    clean = region_or_url.strip()
    if clean.startswith("http://") or clean.startswith("https://"):
        return clean
    clean = clean.lstrip("/")
    if not clean.startswith("trails/"):
        clean = f"trails/{activity}/{clean}"
    return f"{WIKILOC_BASE_URL}/{clean}"


def scrape_trail_links_for_region(
    page,
    region_url: str,
    max_pages: int = 3,
    delay: float = 2.0,
) -> List[str]:
    """Navigates through regional listing pages and collects unique trail URLs."""
    trail_urls: List[str] = []
    seen: set = set()

    print(f"\nScanning regional directory: {region_url}")
    for page_num in range(1, max_pages + 1):
        target_url = region_url if page_num == 1 else f"{region_url}?page={page_num}"
        try:
            time.sleep(delay)
            page.goto(target_url, wait_until="domcontentloaded", timeout=35000)
            page.wait_for_timeout(2500)

            soup = BeautifulSoup(page.content(), "html.parser")
            page_trails = []

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                # Matches trail links with trail ID numeric suffix (e.g. /hiking-trails/...-12060871)
                if (
                    ("/hiking-trails/" in href or "/trails/" in href)
                    and re.search(r"-\d+$", href)
                    and not href.endswith("/trails/hiking")
                ):
                    full_url = urljoin(WIKILOC_BASE_URL, href)
                    if full_url not in seen:
                        seen.add(full_url)
                        page_trails.append(full_url)

            if not page_trails:
                print(f" -> Page {page_num}: No more trails discovered.")
                break

            trail_urls.extend(page_trails)
            print(
                f" -> Page {page_num}: Discovered {len(page_trails)} trails (Total: {len(trail_urls)})"
            )

        except Exception as e:
            print(f" -> Error scanning page {page_num} ({target_url}): {e}")
            break

    return trail_urls


def scrape_trail_details(
    page,
    trail_url: str,
    max_photos: Optional[int] = None,
    delay: float = 1.5,
) -> List[Dict[str, Any]]:
    """Loads a trail detail page, parses metadata and extracts geolocated waypoint photos."""
    records: List[Dict[str, Any]] = []
    try:
        time.sleep(delay)
        page.goto(trail_url, wait_until="networkidle", timeout=35000)
        page.wait_for_timeout(1500)

        soup = BeautifulSoup(page.content(), "html.parser")

        # 1. Extract Trail ID
        m = re.search(r"-(\d+)(?:$|\?)", trail_url)
        trail_id = m.group(1) if m else "unknown"

        # 2. Extract Title & Description
        og_title = soup.find("meta", property="og:title")
        title = (
            og_title.get("content", "")
            if og_title
            else (soup.title.string.strip() if soup.title else "")
        )
        title = re.sub(r"\s*\|\s*Wikiloc.*$", "", title).strip()

        og_desc = soup.find("meta", property="og:description")
        desc = og_desc.get("content", "").strip() if og_desc else ""

        # 3. Extract Breadcrumbs (Country, Region, Locality)
        country, region, locality = "", "", ""
        for s in soup.find_all("script", type="application/ld+json"):
            if not s.string:
                continue
            try:
                d = json.loads(s.string)
                if isinstance(d, dict) and d.get("@type") == "BreadcrumbList":
                    items = d.get("itemListElement", [])
                    for it in items:
                        pos = it.get("position")
                        name = it.get("item", {}).get("name", "")
                        name = re.sub(r"(?i)The best .* trails in\s*", "", name).strip()
                        if pos == 1:
                            country = name
                        elif pos == 2:
                            region = name
                        elif pos == 3:
                            locality = name
            except Exception:
                pass

        # 4. Extract Date
        date_str = ""
        date_el = soup.find("time") or soup.find(
            class_=lambda c: c and "date" in c.lower()
        )
        if date_el:
            date_str = date_el.get("datetime", "") or date_el.text.strip()

        # 5. Extract Geolocated Waypoint Photos from JSON-LD
        photo_idx = 0
        seen_images = set()

        for s in soup.find_all("script", type="application/ld+json"):
            if not s.string:
                continue
            try:
                d = json.loads(s.string)
                if isinstance(d, dict) and d.get("@type") == "Landform":
                    img_url = d.get("image")
                    geo = d.get("geo", {})
                    lat = geo.get("latitude")
                    lon = geo.get("longitude")
                    wp_name = d.get("name", "").strip()

                    if (
                        img_url
                        and lat is not None
                        and lon is not None
                        and img_url not in seen_images
                    ):
                        seen_images.add(img_url)
                        photo_idx += 1

                        # Derive stable photo ID from URL (e.g. 7406729Master.jpg -> 7406729)
                        pid_match = re.search(r"/(\d+)(?:Master)?\.[a-zA-Z]+$", img_url)
                        photo_id_part = (
                            pid_match.group(1) if pid_match else str(photo_idx)
                        )
                        photo_id = f"{trail_id}_{photo_id_part}"

                        records.append(
                            {
                                "Photo_ID": photo_id,
                                "Platform": DEFAULT_PLATFORM,
                                "Latitude": float(lat),
                                "Longitude": float(lon),
                                "Image_URL": img_url,
                                "Title": wp_name or title,
                                "Caption": desc,
                                "Country": country,
                                "Region": region,
                                "Locality": locality,
                                "Date": date_str,
                                "Trail_ID": trail_id,
                                "Trail_URL": trail_url,
                                "License": "All Rights Reserved (Wikiloc)",
                                "photo_key": f"{DEFAULT_PLATFORM}_{photo_id}",
                            }
                        )
                        if max_photos and len(records) >= max_photos:
                            break
            except Exception:
                pass

    except Exception as e:
        print(f" -> Error reading trail {trail_url}: {e}")

    return records


def download_single_image(
    record: Dict[str, Any],
    session: requests.Session,
    output_dir: str,
    timeout: int = 15,
) -> Tuple[bool, str]:
    """Streams and verifies a single image binary to disk with Wikiloc CDN headers."""
    photo_id = str(record.get("Photo_ID", "")).strip()
    img_url = record.get("Image_URL", "")
    if not photo_id or not img_url:
        return False, ""

    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, f"{photo_id}.jpg")

    # If file already exists and is valid, return immediately
    if os.path.isfile(target_path) and is_valid_image_file(target_path):
        return True, target_path

    temp_path = f"{target_path}.tmp_{os.getpid()}_{threading.get_ident()}_{time.time()}"
    try:
        r = session.get(img_url, timeout=timeout, stream=True)
        if r.status_code == 200:
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

            if is_valid_image_file(temp_path):
                os.replace(temp_path, target_path)
                return True, target_path
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False, ""
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, ""
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, ""


def batch_download_images(
    records: List[Dict[str, Any]],
    output_dir: str,
    base_dir: Optional[str] = None,
    threads: int = 8,
) -> List[Dict[str, Any]]:
    """
    Downloads images in parallel using the warm browser-header session.
    Returns only the records where the image was successfully downloaded and verified.
    """
    print(
        f"\nDownloading {len(records):,} images to: {output_dir} using {threads} threads..."
    )
    session = get_image_session()
    successful_records = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(download_single_image, rec, session, output_dir): rec
            for rec in records
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Downloading Wikiloc images",
        ):
            rec = futures[future]
            ok, local_path = future.result()
            if ok:
                if base_dir:
                    try:
                        rec["Image_Location"] = "./" + os.path.relpath(
                            local_path, base_dir
                        )
                    except Exception:
                        rec["Image_Location"] = local_path
                else:
                    rec["Image_Location"] = local_path

                photo_id = str(rec.get("Photo_ID", "")).strip()
                rec["file_name"] = f"{photo_id}.jpg"
                successful_records.append(rec)

    print(
        f" -> Download complete: {len(successful_records):,} / {len(records):,} images saved successfully."
    )
    return successful_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape geolocated trails and waypoint imagery from Wikiloc for remote regions."
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=None,
        help="One or more regional paths (e.g. 'chile/magallanes-y-antartica-chilena' 'spain/aragon/huesca').",
    )
    parser.add_argument(
        "--urls",
        type=str,
        nargs="+",
        default=None,
        help="Explicit listing or trail URLs to scrape directly.",
    )
    parser.add_argument(
        "--activity",
        type=str,
        default=DEFAULT_ACTIVITY,
        help=f"Trail activity type (default: {DEFAULT_ACTIVITY}).",
    )
    parser.add_argument(
        "--max_pages_per_region",
        type=int,
        default=3,
        help="Maximum listing pages to scrape per region (default: 3).",
    )
    parser.add_argument(
        "--max_trails",
        type=int,
        default=None,
        help="Maximum number of trails to process in total.",
    )
    parser.add_argument(
        "--max_photos_per_trail",
        type=int,
        default=20,
        help="Maximum waypoint photos to extract per trail (default: 20).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between page loads (default: 2.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/scraped/wikiloc/wikiloc_trails.parquet",
        help="Path to output Parquet file (defaults to data/scraped/wikiloc/wikiloc_trails.parquet).",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Disable saving companion CSV file.",
    )
    parser.add_argument(
        "--download_images",
        action="store_true",
        default=True,
        help="Download image binaries immediately to disk (default: True).",
    )
    parser.add_argument(
        "--no_download_images",
        action="store_false",
        dest="download_images",
        help="Skip image binary downloading and save metadata only.",
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
        default=8,
        help="Worker threads for concurrent image downloads (default: 8).",
    )
    parser.add_argument(
        "--chrome_path",
        type=str,
        default=None,
        help="Path to Google Chrome or Chromium executable.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve Chrome Executable
    chrome_bin = find_chrome_executable(args.chrome_path)
    if not chrome_bin:
        print(" [!] Error: Google Chrome / Chromium executable not found.")
        print("     Install Google Chrome or specify its path via --chrome_path.")
        sys.exit(1)

    print("=" * 65)
    print("Wikiloc Remote Region Trail & Waypoint Scraper")
    print("=" * 65)
    print(f"Browser binary: {chrome_bin}")
    print(f"Activity:       {args.activity}")
    print(f"Output path:    {args.output}")

    # 2. Build target list of URLs to visit
    target_listing_urls = []
    direct_trail_urls = []

    if args.urls:
        for u in args.urls:
            if re.search(r"-\d+$", u.strip()):
                direct_trail_urls.append(u.strip())
            else:
                target_listing_urls.append(u.strip())

    if args.regions:
        resolved_regions = resolve_regional_inputs(args.regions)
        for r in resolved_regions:
            target_listing_urls.append(format_region_url(r, args.activity))

    if not target_listing_urls and not direct_trail_urls:
        # Default to iconic remote trekking region: Patagonia (Chile & Argentina)
        default_regions = [
            "chile/magallanes-y-antartica-chilena",
            "argentina/tierra-del-fuego",
        ]
        print(
            f"No regions or URLs specified. Defaulting to: {', '.join(default_regions)}"
        )
        resolved_defaults = resolve_regional_inputs(default_regions)
        for r in resolved_defaults:
            target_listing_urls.append(format_region_url(r, args.activity))

    # 3. Launch Playwright Stealth Browser
    all_trail_urls = list(direct_trail_urls)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome_bin,
            headless=args.headless,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = browser.new_context(
            user_agent=CDN_HEADERS["User-Agent"],
            locale="en-US",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        stealth = Stealth()
        stealth.apply_stealth_sync(page)

        # 3a. Discover trail links from listing directories
        for listing_url in target_listing_urls:
            if args.max_trails and len(all_trail_urls) >= args.max_trails:
                break
            found = scrape_trail_links_for_region(
                page=page,
                region_url=listing_url,
                max_pages=args.max_pages_per_region,
                delay=args.delay,
            )
            all_trail_urls.extend(found)

        # Deduplicate trail URLs
        all_trail_urls = list(dict.fromkeys(all_trail_urls))
        if args.max_trails:
            all_trail_urls = all_trail_urls[: args.max_trails]

        print(
            f"\nFound {len(all_trail_urls):,} total trail(s) to scrape for waypoints & imagery."
        )
        if not all_trail_urls:
            print("No trails found. Exiting.")
            browser.close()
            sys.exit(0)

        # 3b. Scrape each trail for geolocated waypoint photos
        all_records: List[Dict[str, Any]] = []
        for trail_url in tqdm(all_trail_urls, desc="Scraping trail waypoints"):
            trail_records = scrape_trail_details(
                page=page,
                trail_url=trail_url,
                max_photos=args.max_photos_per_trail,
                delay=args.delay,
            )
            all_records.extend(trail_records)

        browser.close()

    print(f"\nExtracted {len(all_records):,} total geolocated photo records.")
    if not all_records:
        print("No photo records extracted. Exiting.")
        sys.exit(0)

    # 4. Immediate image downloading and verification
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.download_images:
        img_dir = args.image_dir or os.path.join(out_dir, "images")
        all_records = batch_download_images(
            records=all_records,
            output_dir=img_dir,
            base_dir=out_dir,
            threads=args.threads,
        )

    if not all_records:
        print(" [!] No images were successfully downloaded. Exiting.")
        sys.exit(0)

    # 5. Format DataFrame and Save
    df = pd.DataFrame(all_records)

    # Standardize column order matching Geo-RAG schema
    leading_cols = [
        "Photo_ID",
        "Platform",
        "Latitude",
        "Longitude",
        "Image_URL",
        "Image_Location",
        "file_name",
        "Title",
        "Caption",
        "Country",
        "Region",
        "Locality",
        "Date",
        "License",
        "Trail_ID",
        "Trail_URL",
        "photo_key",
    ]
    cols = [c for c in leading_cols if c in df.columns] + [
        c for c in df.columns if c not in leading_cols
    ]
    df = df[cols]

    print(f"\nSaving {len(df):,} verified records to Parquet: {args.output}")
    save_dataframe(df, args.output)

    if not args.no_csv:
        csv_path = os.path.splitext(args.output)[0] + ".csv"
        print(f"Saving companion CSV: {csv_path}")
        df.to_csv(csv_path, index=False)

    print("\n" + "=" * 65)
    print("Wikiloc Scraping & Local Backup Complete")
    print("=" * 65)
    print(f"Total trails processed:   {df['Trail_ID'].nunique():,}")
    print(f"Total offline photos:     {len(df):,}")
    print(f"Countries covered:        {df['Country'].unique().tolist()}")
    print(f"Regions covered:          {df['Region'].unique().tolist()}")
    print(f"Parquet dataset:          {args.output}")
    if args.download_images:
        print(f"Images directory:         {img_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
