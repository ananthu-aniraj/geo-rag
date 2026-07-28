import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse

import h3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm


def extract_category_name(url):
    """Extract and unquote Wikimedia Commons category name from its URL."""
    if not isinstance(url, str):
        return None
    match = re.search(r'Category:(.+)$', url)
    if match:
        cat = urllib.parse.unquote(match.group(1))
        return cat.replace('_', ' ').strip()
    return None


def parse_wkt_coords(coords_str):
    """Parse WKT Point format 'Point(lon lat)' from Wikidata to (lat, lon)."""
    if not isinstance(coords_str, str):
        return None
    match = re.search(r'(?i)Point\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', coords_str)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lat, lon
    return None


def fetch_wikimedia_timestamps(image_titles):
    """Fetch upload timestamps for a list of Wikimedia Commons file titles in batches of 50."""
    timestamps = {}
    titles_list = list(image_titles)

    full_titles = []
    title_to_orig = {}
    for t in titles_list:
        ft = t if t.startswith("File:") else f"File:{t}"
        full_titles.append(ft)
        title_to_orig[ft] = t

    url = "https://commons.wikimedia.org/w/api.php"
    headers = {
        "User-Agent": "Geo-RAG-Landmark-Sampler/1.0 (aaniraj@home)",
    }

    batch_size = 50
    print(f"Fetching upload timestamps from MediaWiki API for {len(full_titles):,} images in batches of 50...")
    for i in tqdm(range(0, len(full_titles), batch_size), desc="Fetching timestamps"):
        batch = full_titles[i:i+batch_size]
        titles_str = "|".join(batch)
        params = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "timestamp",
            "titles": titles_str,
            "format": "json"
        }
        try:
            res = requests.get(url, params=params, headers=headers, timeout=15)
            if res.status_code == 200:
                data = res.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    title = page_info.get("title")
                    imageinfo = page_info.get("imageinfo", [])
                    if title and imageinfo:
                        ts = imageinfo[0].get("timestamp")
                        orig_title = title_to_orig.get(title)
                        if orig_title and ts:
                            timestamps[orig_title] = ts
            time.sleep(0.1)  # Polite delay
        except Exception:
            pass

    return timestamps


def query_wikidata_batch(categories):
    """Query Wikidata SPARQL endpoint for a batch of category strings."""
    escaped_cats = []
    for cat in categories:
        escaped = cat.replace('"', '\\"')
        escaped_cats.append(f'"{escaped}"')
    
    cats_str = " ".join(escaped_cats)
    
    query = f"""
    SELECT ?commons_cat ?coords WHERE {{
      VALUES ?commons_cat {{ {cats_str} }}
      ?item wdt:P373 ?commons_cat .
      ?item wdt:P625 ?coords .
    }}
    """
    
    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "Geo-RAG-Landmark-Sampler/1.0 (aaniraj@home)",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, data={'query': query, 'format': 'json'}, headers=headers, timeout=45)
            if response.status_code == 200:
                data = response.json()
                results = {}
                for binding in data.get('results', {}).get('bindings', []):
                    cat = binding.get('commons_cat', {}).get('value')
                    coords = binding.get('coords', {}).get('value')
                    if cat and coords:
                        results[cat] = coords
                return results
            elif response.status_code == 429:
                # Rate limit hit, sleep and retry
                time.sleep(5 * (attempt + 1))
            else:
                break
        except Exception:
            time.sleep(2 * (attempt + 1))
            
    return {}


def main():
    parser = argparse.ArgumentParser(description="Sample Google Landmarks v2 dataset in under-represented zones.")
    parser.add_argument("--dataset_parquet", type=str, default="full_pipeline_output/geo_space_cleaned.parquet",
                        help="Path to our current dataset parquet file.")
    parser.add_argument("--landmarks_csv", type=str, required=True,
                        help="Path to index_image_to_landmark.csv.")
    parser.add_argument("--category_csv", type=str, required=True,
                        help="Path to index_label_to_category.csv.")
    parser.add_argument("--attribution_csv", type=str, default=None,
                        help="Optional path to train_attribution.csv to resolve precise image licenses.")
    parser.add_argument("--fetch_timestamps", action="store_true",
                        help="Fetch exact upload timestamps from MediaWiki API for the sampled subset.")
    parser.add_argument("--clean_csv", type=str, default=None,
                        help="Optional path to train_clean.csv to filter out noisy/unclean image IDs.")
    parser.add_argument("--cache_path", type=str, default="category_coords_cache.json",
                        help="Path to coordinates JSON cache.")
    parser.add_argument("--output_csv", type=str, default="google_landmarks_sampled.csv",
                        help="Path to output sampled CSV.")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Maximum total images to sample. Set to 0 or negative to sample all available matching images without limit.")
    parser.add_argument("--max_per_cell", type=int, default=10,
                        help="Maximum images to sample from a single H3 cell.")
    parser.add_argument("--max_per_landmark", type=int, default=3,
                        help="Maximum images to sample from a single unique landmark.")
    parser.add_argument("--density_threshold", type=int, default=5,
                        help="Maximum existing images in cell to consider it under-represented.")
    parser.add_argument("--target_h3_res", type=int, default=5,
                        help="H3 resolution for spatial density calculation.")
    args = parser.parse_args()

    # 1. Load existing dataset coordinates and construct H3 cell frequencies
    print(f"Loading existing dataset from '{args.dataset_parquet}'...")
    if not os.path.exists(args.dataset_parquet):
        print(f"Error: Dataset Parquet '{args.dataset_parquet}' not found.")
        sys.exit(1)
        
    try:
        # Load only spatial columns
        parquet_file = pq.ParquetFile(args.dataset_parquet)
        available_cols = parquet_file.schema_arrow.names
        read_cols = []
        if 'H3_Cell' in available_cols:
            read_cols.append('H3_Cell')
        if 'Latitude' in available_cols:
            read_cols.append('Latitude')
        if 'Longitude' in available_cols:
            read_cols.append('Longitude')
            
        df_existing = pd.read_parquet(args.dataset_parquet, columns=read_cols)
        
        # Calculate cell coverage counts
        cell_counts = {}
        if 'H3_Cell' in df_existing.columns:
            # Map existing cells to the target resolution
            for cell in df_existing['H3_Cell'].dropna():
                try:
                    res = h3.get_resolution(cell)
                    target_cell = h3.cell_to_parent(cell, args.target_h3_res) if res > args.target_h3_res else cell
                    cell_counts[target_cell] = cell_counts.get(target_cell, 0) + 1
                except Exception:
                    continue
        else:
            print("H3_Cell column missing. Resolving cells from coordinates...")
            for lat, lon in zip(df_existing['Latitude'].dropna(), df_existing['Longitude'].dropna()):
                try:
                    cell = h3.latlng_to_cell(float(lat), float(lon), args.target_h3_res)
                    cell_counts[cell] = cell_counts.get(cell, 0) + 1
                except Exception:
                    continue
                    
        print(f"Found {len(cell_counts):,} populated spatial cells in our dataset.")
    except Exception as e:
        print(f"Error loading existing dataset: {e}")
        sys.exit(1)

    # 2. Load and parse category mapping
    print(f"Loading category mapping from '{args.category_csv}'...")
    if not os.path.exists(args.category_csv):
        print(f"Error: Category CSV '{args.category_csv}' not found.")
        sys.exit(1)
        
    df_cat = pd.read_csv(args.category_csv)
    df_cat['clean_name'] = df_cat['category'].apply(extract_category_name)
    
    # 3. Handle geocoding cache
    coords_cache = {}
    if os.path.exists(args.cache_path):
        print(f"Loading geocoded coordinates cache from '{args.cache_path}'...")
        try:
            with open(args.cache_path, 'r', encoding='utf-8') as f:
                coords_cache = json.load(f)
            print(f"Loaded {len(coords_cache):,} cached category locations.")
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")

    # Find categories to query
    uncached_cats = [c for c in df_cat['clean_name'].dropna().unique() if c not in coords_cache]
    
    if uncached_cats:
        print(f"Querying Wikidata SPARQL endpoint for {len(uncached_cats):,} uncached categories in batches...")
        batch_size = 1000
        for idx in tqdm(range(0, len(uncached_cats), batch_size), desc="Wikidata Geocoding"):
            batch = uncached_cats[idx:idx + batch_size]
            results = query_wikidata_batch(batch)
            coords_cache.update(results)
            
            # Rate limiting sleep
            time.sleep(0.5)
            
            # Periodically save cache progress
            if (idx // batch_size) % 10 == 0:
                try:
                    with open(args.cache_path, 'w', encoding='utf-8') as f:
                        json.dump(coords_cache, f, indent=2)
                except Exception:
                    pass

        # Save final cache
        try:
            with open(args.cache_path, 'w', encoding='utf-8') as f:
                json.dump(coords_cache, f, indent=2)
            print(f"Saved updated cache containing {len(coords_cache):,} locations.")
        except Exception as e:
            print(f"Warning: Failed to save final cache: {e}")

    # Map categories to coordinates
    landmark_coords = {}
    for _, row in df_cat.iterrows():
        l_id = row['landmark_id']
        name = row['clean_name']
        if name in coords_cache:
            parsed = parse_wkt_coords(coords_cache[name])
            if parsed:
                landmark_coords[l_id] = {
                    'latitude': parsed[0],
                    'longitude': parsed[1],
                    'category_url': row['category']
                }

    # 3.5 Load clean image IDs list if provided
    clean_ids = set()
    if args.clean_csv:
        if os.path.exists(args.clean_csv):
            print(f"Loading clean image IDs from '{args.clean_csv}'...")
            try:
                df_clean = pd.read_csv(args.clean_csv)
                for imgs_str in df_clean['images'].dropna():
                    clean_ids.update(imgs_str.split())
                print(f"Loaded {len(clean_ids):,} clean image IDs.")
            except Exception as e:
                print(f"Warning: Failed to load clean CSV: {e}")
        else:
            print(f"Warning: Clean CSV '{args.clean_csv}' not found. No clean filter will be applied.")

    # 4. Load images-to-landmark index and perform sampling
    print(f"Loading image-to-landmark indexing from '{args.landmarks_csv}'...")
    if not os.path.exists(args.landmarks_csv):
        print(f"Error: Landmarks indexing CSV '{args.landmarks_csv}' not found.")
        sys.exit(1)
        
    # Read in chunks to remain memory efficient
    sampled_records = []
    cell_sampled_counts = {}
    landmark_sampled_counts = {}
    total_sampled = 0

    chunksize = 100000
    for chunk in pd.read_csv(args.landmarks_csv, chunksize=chunksize):
        if args.max_images > 0 and total_sampled >= args.max_images:
            break
            
        # Filter for rows that have mapped landmark coordinates
        valid_chunk = chunk[chunk['landmark_id'].isin(landmark_coords.keys())].copy()
        
        # Add coords columns
        valid_chunk['latitude'] = valid_chunk['landmark_id'].map(lambda x: landmark_coords[x]['latitude'])
        valid_chunk['longitude'] = valid_chunk['landmark_id'].map(lambda x: landmark_coords[x]['longitude'])
        valid_chunk['category_url'] = valid_chunk['landmark_id'].map(lambda x: landmark_coords[x]['category_url'])
        
        # Calculate target H3 cell
        def get_h3(row):
            try:
                return h3.latlng_to_cell(row['latitude'], row['longitude'], args.target_h3_res)
            except Exception:
                return None
                
        valid_chunk['h3_cell'] = valid_chunk.apply(get_h3, axis=1)
        valid_chunk = valid_chunk.dropna(subset=['h3_cell'])
        
        # Evaluate occupancy mapping and filter
        for _, row in valid_chunk.iterrows():
            if args.max_images > 0 and total_sampled >= args.max_images:
                break

            img_id = str(row['id'])
            if clean_ids and img_id not in clean_ids:
                continue

            cell = row['h3_cell']
            # Get existing image counts in this cell
            existing_count = cell_counts.get(cell, 0)
            
            # If the cell is under-represented in our dataset
            if existing_count < args.density_threshold:
                sampled_in_cell = cell_sampled_counts.get(cell, 0)
                landmark_id = row['landmark_id']
                sampled_for_landmark = landmark_sampled_counts.get(landmark_id, 0)
                
                if sampled_in_cell < args.max_per_cell and sampled_for_landmark < args.max_per_landmark:
                    # Get cleaned landmark label category name
                    cat_name = "Unknown Landmark"
                    if 'clean_name' in df_cat.columns:
                        matches = df_cat[df_cat['landmark_id'] == landmark_id]['clean_name'].values
                        if len(matches) > 0 and pd.notna(matches[0]):
                            cat_name = str(matches[0])

                    # Construct relative path in GLDv2 nested format: ./images/a/b/c/abcdef.jpg
                    rel_img_path = f"./images/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"

                    # Sample this record matching iWildCam metadata schema exactly
                    sampled_records.append({
                        'Photo_ID': img_id,
                        'Platform': 'GoogleLandmarks',
                        'Latitude': row['latitude'],
                        'Longitude': row['longitude'],
                        'Image_URL': str(row['url']) if 'url' in valid_chunk.columns else '',
                        'Image_Location': rel_img_path,
                        'Captured_At': '',
                        'location': str(landmark_id),
                        'category_id': str(landmark_id),
                        'category_name': cat_name,
                        'file_name': f"{img_id}.jpg",
                        'seq_id': img_id,
                        'License': 'CC BY-SA 4.0',
                        'h3_cell': cell
                    })
                    cell_sampled_counts[cell] = sampled_in_cell + 1
                    landmark_sampled_counts[landmark_id] = sampled_for_landmark + 1
                    total_sampled += 1

    # 5. Output sampled dataset
    if sampled_records:
        df_sampled = pd.DataFrame(sampled_records)

        # Optionally resolve precise licenses and direct URLs from train_attribution.csv using a streaming lookup
        id_to_license = {}
        id_to_wiki_url = {}
        if args.attribution_csv:
            if os.path.exists(args.attribution_csv):
                print(f"Resolving precise image licenses and direct Wikimedia URLs from '{args.attribution_csv}' using streaming lookup...")
                sampled_ids_set = set(df_sampled['Photo_ID'].tolist())

                # Stream the attribution file in chunks to keep memory usage low
                attr_chunksize = 250000
                try:
                    for chunk in pd.read_csv(args.attribution_csv, chunksize=attr_chunksize, usecols=['id', 'license', 'title']):
                        matched_chunk = chunk[chunk['id'].isin(sampled_ids_set)]
                        for _, row_attr in matched_chunk.iterrows():
                            img_id = str(row_attr['id'])
                            # Clean up license text: e.g., "CC BY-SA 3.0(http://...)" -> "CC BY-SA 3.0"
                            raw_lic = str(row_attr['license'])
                            clean_lic = raw_lic.split('(')[0].replace('-', ' ').strip()
                            id_to_license[img_id] = clean_lic

                            # Construct direct Wikimedia Commons image URL
                            title_val = str(row_attr['title'])
                            if title_val and title_val != 'nan':
                                filename = title_val[5:] if title_val.startswith("File:") else title_val
                                filename = filename.replace(" ", "_")
                                quoted_filename = urllib.parse.quote(filename)
                                md5_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()
                                direct_url = f"https://upload.wikimedia.org/wikipedia/commons/{md5_hash[0]}/{md5_hash[:2]}/{quoted_filename}"
                                id_to_wiki_url[img_id] = direct_url
                except Exception as e:
                    print(f"Warning: Failed to parse attribution file: {e}")

                # Map resolved licenses back to dataframe
                if id_to_license:
                    df_sampled['License'] = df_sampled['Photo_ID'].map(lambda x: id_to_license.get(x, 'CC BY-SA 4.0'))
                    print(f" -> Successfully resolved precise licenses for {len(id_to_license):,} images.")
            else:
                print(f"Warning: Attribution CSV '{args.attribution_csv}' not found. Defaulting to 'CC BY-SA 4.0'.")

        # Map Wikimedia URLs (will be empty string if not resolved)
        if id_to_wiki_url:
            df_sampled['Image_URL'] = df_sampled['Photo_ID'].map(lambda x: id_to_wiki_url.get(x, ''))

        # Optionally fetch upload timestamps from MediaWiki API
        if args.fetch_timestamps and id_to_wiki_url:
            id_to_title = {}
            for img_id, wiki_url in id_to_wiki_url.items():
                quoted_fn = os.path.basename(wiki_url)
                fn = urllib.parse.unquote(quoted_fn)
                id_to_title[img_id] = f"File:{fn}"

            titles_to_fetch = set(id_to_title.values())
            title_to_ts = fetch_wikimedia_timestamps(titles_to_fetch)

            id_to_ts = {}
            for img_id, title_val in id_to_title.items():
                if title_val in title_to_ts:
                    id_to_ts[img_id] = title_to_ts[title_val]

            if id_to_ts:
                df_sampled['Captured_At'] = df_sampled['Photo_ID'].map(lambda x: id_to_ts.get(x, ''))
                print(f" -> Successfully resolved upload timestamps for {len(id_to_ts):,} images.")

        df_sampled.to_csv(args.output_csv, index=False)
        print(f"\nSampling Complete!")
        print(f" -> Sampled {len(df_sampled):,} Google Landmarks images in under-represented zones.")
        print(f" -> Cover {df_sampled['h3_cell'].nunique():,} unique under-represented H3 cells.")
        print(f" -> Output saved to: {os.path.abspath(args.output_csv)}")
    else:
        print("No matching Google Landmarks records found in under-represented zones.")


if __name__ == "__main__":
    main()
