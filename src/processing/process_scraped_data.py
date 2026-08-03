import argparse
import glob
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import h3
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel
from urllib3.util import Retry

from src.utils.io import get_parquet_writer, load_dataframe, save_dataframe
from src.utils.licensing import FLICKR_LICENSE_MAP

# TIPSv2 specific transform
tips_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'

# Global connection pooled session configuration for thread-safe high-throughput downloads
http_session = requests.Session()
http_session.headers.update({
    "User-Agent": "Geo-RAG-Scraper-Pipeline/1.0 (aaniraj@home; contact: aaniraj@home.com)"
})
_adapter = HTTPAdapter(
    pool_connections=128,
    pool_maxsize=128,
    max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
)
http_session.mount("https://", _adapter)
http_session.mount("http://", _adapter)


def clean_photo_id(val):
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    val_str = val_str.removesuffix('.0')
    return val_str


def standardize_timestamps_vectorized(ts_raw):
    """Standardizes a Pandas Series of timestamps (numeric or string) to ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)."""
    # Initialize output Series with None as default
    standardized = pd.Series([None] * len(ts_raw), index=ts_raw.index, dtype=object)

    # Identify Unix epoch numeric timestamps vs. string representations
    numeric_ts = pd.to_numeric(ts_raw, errors='coerce')
    is_numeric = numeric_ts.notna() & (numeric_ts > 1e8)

    # Parse Unix numeric timestamps
    if is_numeric.any():
        is_ms_mask = is_numeric & (numeric_ts > 5e10)
        is_s_mask = is_numeric & ~is_ms_mask

        if is_ms_mask.any():
            parsed_ms = pd.to_datetime(numeric_ts[is_ms_mask], unit='ms', utc=True, errors='coerce')
            try:
                valid_ms = (parsed_ms.dt.year >= 1) & (parsed_ms.dt.year <= 9999)
            except Exception:
                valid_ms = pd.Series(True, index=parsed_ms.index)

            formatted_ms = pd.Series([None] * len(parsed_ms), index=parsed_ms.index, dtype=object)
            if valid_ms.any():
                formatted_ms[valid_ms] = parsed_ms[valid_ms].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            standardized.loc[is_ms_mask] = formatted_ms

        if is_s_mask.any():
            parsed_s = pd.to_datetime(numeric_ts[is_s_mask], unit='s', utc=True, errors='coerce')
            try:
                valid_s = (parsed_s.dt.year >= 1) & (parsed_s.dt.year <= 9999)
            except Exception:
                valid_s = pd.Series(True, index=parsed_s.index)

            formatted_s = pd.Series([None] * len(parsed_s), index=parsed_s.index, dtype=object)
            if valid_s.any():
                formatted_s[valid_s] = parsed_s[valid_s].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            standardized.loc[is_s_mask] = formatted_s

    # Parse string representations
    is_string = ts_raw.notna() & ~is_numeric
    if is_string.any():
        str_vals = ts_raw[is_string].astype(str).str.strip()
        has_colon_date = str_vals.str.match(r'^\d{4}:\d{2}:\d{2}')
        if has_colon_date.any():
            str_vals.loc[has_colon_date] = str_vals[has_colon_date].str.replace(':', '-', n=2)

        parsed_str = pd.to_datetime(str_vals, errors='coerce', utc=True, format='mixed')
        try:
            valid_str = (parsed_str.dt.year >= 1) & (parsed_str.dt.year <= 9999)
        except Exception:
            valid_str = pd.Series(True, index=parsed_str.index)

        formatted_str = pd.Series([None] * len(parsed_str), index=parsed_str.index, dtype=object)
        if valid_str.any():
            formatted_str[valid_str] = parsed_str[valid_str].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        standardized.loc[is_string] = formatted_str

    # Fallback to original strings if parsing failed but was not null/empty
    ts_series = ts_raw.astype(str).str.strip()
    is_parsed = standardized.notna()
    invalid_mask = ~is_parsed & ts_raw.notna() & (ts_raw != '')
    standardized[invalid_mask] = ts_series[invalid_mask]

    return standardized


def download_image(url, photo_id=None, platform=None, offline_dirs=None):
    """Downloads an image using the global connection pool session and returns a resized PIL Image."""
    try:
        from src.utils.io import resolve_offline_image_path
        
        # Check if url is a local path first
        if not (url.startswith("http://") or url.startswith("https://") or url.startswith(
                "mapillary://") or url.startswith("kartaview://")):
            if os.path.exists(url):
                img = Image.open(url).convert("RGB")
                img_resized = img.resize((448, 448))
                img.close()
                return img_resized
            
        if offline_dirs:
            resolved_path = resolve_offline_image_path(url, offline_dirs, photo_id, platform)
            if resolved_path:
                img = Image.open(resolved_path).convert("RGB")
                img_resized = img.resize((448, 448))
                img.close()
                return img_resized
            return None

        if url.startswith("mapillary://"):
            orig_id = url.split("://")[1]
            api_url = f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
            res = http_session.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
            else:
                return None
        elif url.startswith("kartaview://"):
            orig_id = url.split("://")[1]
            api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
            res = http_session.get(api_url, timeout=10)
            if res.status_code == 200:
                data = res.json().get("result", {}).get("data", {})
                url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")
            else:
                return None

        if not url:
            return None

        response = http_session.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_resized = img.resize((448, 448))
            img.close()
            return img_resized
    except Exception:
        pass
    return None


def get_tips_embeddings(images, model, device, batch_size=32):
    """Computes TIPSv2 embeddings for a list of PIL images in batches."""
    if not images:
        return None

    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            # Batch transform and stack
            batch_tensors = torch.stack([tips_transform(img) for img in batch]).to(device)
            features = model.encode_image(batch_tensors).cls_token
            all_features.append(features.squeeze(1).cpu().numpy())

    return np.concatenate(all_features, axis=0)


def process_cell(cell_id, metadata_list, model, device, sim_threshold, executor, text_features=None,
                 existing_items=None, cell_chunk_size=128, tips_batch_size=32, macro_idx=-1, sky_idx=-1,
                 offline_dirs=None):
    """Filters indoor images (Flickr only) and deduplicates images within an H3 cell in chunks."""
    results = existing_items.copy() if existing_items else []
    processed_embeddings = [item['embedding'] for item in results]

    from functools import partial
    download_fn = partial(download_image, offline_dirs=offline_dirs)

    # Process new images in chunks to limit peak memory usage
    for chunk_start in range(0, len(metadata_list), cell_chunk_size):
        chunk_metadata = metadata_list[chunk_start: chunk_start + cell_chunk_size]
        urls = [m['Image_URL'] for m in chunk_metadata]
        pids = [m['Photo_ID'] for m in chunk_metadata]
        plats = [m.get('Platform') for m in chunk_metadata]

        # Download images in parallel for this chunk
        imgs = list(executor.map(download_fn, urls, pids, plats))

        valid_indices = [i for i, img in enumerate(imgs) if img is not None]
        if not valid_indices:
            continue

        valid_imgs = [imgs[i] for i in valid_indices]

        # Compute embeddings for this chunk using configured tips_batch_size
        all_embeddings = get_tips_embeddings(valid_imgs, model, device, batch_size=tips_batch_size)

        # Explicitly close PIL images immediately to free RAM
        for img in valid_imgs:
            try:
                img.close()
            except Exception:
                pass

        if all_embeddings is None:
            continue

        # Matrix multiply for indoor/outdoor zero-shot classification
        if text_features is not None:
            emb_norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            emb_norms[emb_norms == 0] = 1.0
            norm_embeddings = all_embeddings / emb_norms

            text_norms = np.linalg.norm(text_features, axis=1, keepdims=True)
            text_norms[text_norms == 0] = 1.0
            norm_text = text_features / text_norms

            all_io_sims = np.dot(norm_embeddings, norm_text.T)

        for i, idx in enumerate(valid_indices):
            metadata = chunk_metadata[idx]
            embedding = all_embeddings[i]

            # Zero-shot filtering (Indoor, Macro, and Sky/Flying)
            if text_features is not None:
                sims = all_io_sims[i]
                best_class = np.argmax(sims)

                # 1. Flickr & Wikimedia/GoogleLandmarks Indoor Filter (always Class 0)
                if metadata['Platform'] in ['Flickr', 'Wikimedia', 'GoogleLandmarks'] and best_class == 0:
                    continue

                # 2. iNaturalist-only Macro/Close-up Filter
                if macro_idx != -1 and best_class == macro_idx:
                    if str(metadata['Platform']).lower() == 'inaturalist':
                        continue

                # 3. iNaturalist-only Sky/Flying Filter
                if sky_idx != -1 and best_class == sky_idx:
                    if str(metadata['Platform']).lower() == 'inaturalist':
                        continue

            # Deduplication check
            is_duplicate = False
            if processed_embeddings:
                curr_norm = embedding / (np.linalg.norm(embedding) or 1.0)
                kept_embs = np.array(processed_embeddings)
                kept_norms = np.linalg.norm(kept_embs, axis=1, keepdims=True)
                kept_norms[kept_norms == 0] = 1.0
                norm_kept = kept_embs / kept_norms

                sims = np.dot(norm_kept, curr_norm)
                if np.any(sims > sim_threshold):
                    is_duplicate = True

            if not is_duplicate:
                processed_embeddings.append(embedding)
                item = metadata.copy()
                item['embedding'] = embedding
                results.append(item)

    return results


def stream_update_parquet(input_path, output_path, df_new, active_cells):
    """
    Reads input_path in row groups, filters out rows belonging to active_cells,
    and writes the remaining inactive rows along with df_new to output_path.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow

    # Dynamically upgrade schema to include License if it is present in the new df
    # but not in the existing parquet database
    has_license_in_new = "License" in df_new.columns if df_new is not None else False
    has_license_in_existing = "License" in schema.names

    if has_license_in_new and not has_license_in_existing:
        new_fields = list(schema)
        new_fields.append(pa.field("License", pa.string()))
        schema = pa.schema(new_fields)

    tmp_output = f"{output_path}.tmp_stream"
    try:
        active_arr = pa.array(list(active_cells))
        with get_parquet_writer(tmp_output, schema) as writer:
            # 1. Stream copy inactive rows from original parquet
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)

                # Append a null column for License if we upgraded the schema
                if has_license_in_new and "License" not in table.column_names:
                    null_col = pa.array([None] * len(table), type=pa.string())
                    table = table.append_column("License", null_col)

                h3_col = table["H3_Cell"]
                mask = pc.invert(pc.is_in(h3_col, value_set=active_arr))
                filtered_table = table.filter(mask)
                if len(filtered_table) > 0:
                    writer.write_table(filtered_table)

            # 2. Write the new/updated active rows
            if df_new is not None and not df_new.empty:
                df_new_aligned = df_new.copy()
                for name in schema.names:
                    if name not in df_new_aligned.columns:
                        df_new_aligned[name] = None
                df_new_aligned = df_new_aligned[schema.names]

                new_table = pa.Table.from_pandas(df_new_aligned, schema=schema, preserve_index=False)
                writer.write_table(new_table)

        if os.path.exists(tmp_output):
            os.replace(tmp_output, output_path)
    except Exception as e:
        if os.path.exists(tmp_output):
            try:
                os.remove(tmp_output)
            except Exception:
                pass
        raise e


def save_checkpoint(final_data, processed_cells, checkpoint_path, checkpoint_meta_path, resume_from=None,
                    active_cells=None):
    """Saves the intermediate state to checkpoint files atomically."""
    tmp_path = f"{checkpoint_path}.tmp"
    tmp_meta_path = f"{checkpoint_meta_path}.tmp"
    try:
        # Convert final_data to DataFrame and save to tmp parquet
        if not final_data:
            df = pd.DataFrame(
                columns=['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'H3_Cell', 'embedding',
                         'Captured_At', 'License'])
        else:
            df = pd.DataFrame(final_data)

        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

        if resume_from and os.path.exists(resume_from) and active_cells:
            stream_update_parquet(resume_from, tmp_path, df, active_cells)
        else:
            save_dataframe(df, tmp_path)

        # Save processed cells to tmp meta
        with open(tmp_meta_path, 'wb') as f:
            pickle.dump(processed_cells, f)

        # Atomic rename
        if os.path.exists(tmp_path):
            os.replace(tmp_path, checkpoint_path)
        if os.path.exists(tmp_meta_path):
            os.replace(tmp_meta_path, checkpoint_meta_path)
        print(f"\nCheckpoint saved: {len(final_data)} images kept, {len(processed_cells)} cells processed.")
    except Exception as e:
        print(f"\nError saving checkpoint: {e}")


def load_and_preprocess_csv(f, offline_dirs=None):
    """Loads a single CSV file, normalizes column names, and converts Mapillary/KartaView URLs."""
    try:
        df = pd.read_csv(f, dtype=str)
        if df.empty:
            return None

        # Check if this is the iWildCam CSV
        is_iwildcam = 'iwildcam' in f.lower() or (
                    'Platform' in df.columns and df['Platform'].astype(str).str.lower().eq('iwildcam').any())

        if 'uuid' in df.columns and 'source' in df.columns and 'orig_id' in df.columns:
            df['Platform'] = df['source']
            df['Latitude'] = df['lat']
            df['Longitude'] = df['lon']
            df['Photo_ID'] = df['orig_id']
            df['Captured_At'] = df['datetime_local'] if 'datetime_local' in df.columns else None

            urls = df['url'] if 'url' in df.columns else [None] * len(df)
            df['Image_URL'] = [
                f"mapillary://{oid}" if str(src).lower() == 'mapillary'
                else (f"kartaview://{oid}" if str(src).lower() == 'kartaview' else url)
                for oid, src, url in zip(df['orig_id'], df['source'], urls)
            ]
        elif is_iwildcam:
            platform = 'iWildCam'
            col_map = {
                'latitude': 'Latitude',
                'longitude': 'Longitude',
                'photo_id': 'Photo_ID',
                'ID': 'Photo_ID',
                'captured_at': 'Captured_At',
                'Captured_At': 'Captured_At',
                'Image_Location': 'Image_URL',
                'Image_URL': 'Image_URL'
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df['Platform'] = platform
        else:
            if 'inaturalist' in f.lower():
                platform = 'iNaturalist'
            elif 'flickr' in f.lower():
                platform = 'Flickr'
            elif 'iwildcam' in f.lower():
                platform = 'iWildCam'
            else:
                platform = 'Mapillary'

            col_map = {
                'latitude': 'Latitude',
                'longitude': 'Longitude',
                'image_url': 'Image_URL',
                'Image_Location': 'Image_URL',
                'photo_id': 'Photo_ID',
                'ID': 'Photo_ID',
                'captured_at': 'Captured_At',
                'Captured_At': 'Captured_At',
                'Date_Observed': 'Captured_At',
                'observed_on_string': 'Captured_At',
                'license': 'License',
                'License': 'License'
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if 'Platform' not in df.columns:
                df['Platform'] = platform

        # Determine if the file is from an offline directory
        is_offline = False
        if offline_dirs:
            abs_f = os.path.abspath(f)
            for od in offline_dirs:
                if abs_f.startswith(os.path.abspath(od)):
                    is_offline = True
                    break

        # Resolve any local image paths (Image_Location or Image_URL) to absolute paths relative to the CSV's directory
        image_col = 'Image_URL' if 'Image_URL' in df.columns else (
            'Image_Location' if 'Image_Location' in df.columns else None)
        if image_col:
            csv_dir = os.path.dirname(os.path.abspath(f))

            def resolve_offline_path(loc):
                if not isinstance(loc, str):
                    return loc
                if is_offline:
                    if os.path.isabs(loc):
                        return loc
                    # Try direct join
                    p = os.path.abspath(os.path.join(csv_dir, loc))
                    if os.path.exists(p):
                        return p
                    # Try with "train" subdir (backward compatibility for flat iWildCam)
                    p_train = os.path.abspath(os.path.join(csv_dir, "train", os.path.basename(loc)))
                    if os.path.exists(p_train):
                        return p_train
                    return p
                else:
                    return loc

            df['Image_URL'] = [resolve_offline_path(loc) for loc in df[image_col]]

        required_cols = ['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'Captured_At', 'License']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # Populate standard platform licenses if missing
        platform_lower = df['Platform'].astype(str).str.lower()
        mapillary_mask = (platform_lower == 'mapillary') & (df['License'].isna() | (df['License'] == ''))
        if mapillary_mask.any():
            df.loc[mapillary_mask, 'License'] = 'CC BY-SA 4.0'

        kartaview_mask = (platform_lower == 'kartaview') & (df['License'].isna() | (df['License'] == ''))
        if kartaview_mask.any():
            df.loc[kartaview_mask, 'License'] = 'CC BY-SA 4.0'

        iwildcam_mask = (platform_lower == 'iwildcam') & (df['License'].isna() | (df['License'] == ''))
        if iwildcam_mask.any():
            df.loc[iwildcam_mask, 'License'] = 'CDLA-Permissive-1.0'

        inat_mask = (platform_lower.str.contains('inaturalist') | (platform_lower == 'inat')) & (
                    df['License'].isna() | (df['License'] == ''))
        if inat_mask.any():
            df.loc[inat_mask, 'License'] = 'CC BY-NC 4.0'

        # Map Flickr numeric indexes to human-readable strings
        flickr_mask = (platform_lower == 'flickr') & df['License'].notna() & (df['License'] != '')
        if flickr_mask.any():
            # Clean keys to handle potential floats like '4.0'
            clean_keys = df.loc[flickr_mask, 'License'].astype(str).str.split('.').str[0]
            mapped = clean_keys.map(FLICKR_LICENSE_MAP)
            df.loc[flickr_mask, 'License'] = mapped.fillna(df.loc[flickr_mask, 'License'])

        df = df[required_cols].copy()
        df['License'] = df['License'].astype(str).replace({'nan': None, 'None': None, '<NA>': None})
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error reading {f}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Consolidate, Filter, and Deduplicate Geo-Scraped Data using TIPSv2.")
    parser.add_argument("--dirs", nargs="+", required=True, help="List of directories containing chunked CSVs.")
    parser.add_argument("--save_path", type=str, default=".", help="Directory to save output files and data.")
    parser.add_argument("--output_name", type=str, default="geo_embedding_space", help="Base name for output files.")
    parser.add_argument("--h3_res", type=int, default=11, help="H3 resolution (~25m).")
    parser.add_argument("--sim_threshold", type=float, default=0.95, help="TIPSv2 cosine similarity threshold.")
    parser.add_argument("--no_filter", action="store_true", help="Disable Flickr indoor/outdoor filtering.")
    parser.add_argument("--filter_macro", action="store_true",
                        help="Filter out macro/close-up photos of leaves, flowers, bark, and insects using zero-shot embeddings.")
    parser.add_argument("--filter_sky", action="store_true",
                        help="Filter out photos of the sky, clouds, and flying objects (birds/insects in flight) for iNaturalist.")
    parser.add_argument("--limit_cells", type=int, default=0, help="Limit number of cells to process (for testing).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a previously generated .pkl or parquet file to resume from.")
    parser.add_argument("--checkpoint_interval", type=int, default=1800,
                        help="Interval in seconds to save checkpoints (0 to disable).")
    parser.add_argument("--cell_chunk_size", type=int, default=128,
                        help="Number of images within a cell to download/process in a chunk.")
    parser.add_argument("--tips_batch_size", type=int, default=32, help="Batch size for TIPSv2 embedding inference.")
    parser.add_argument("--offline_dataset_dirs", type=str, nargs="*", default=None,
                        help="Base directories containing offline dataset CSV indexes and images folders.")
    args = parser.parse_args()

    # 1. Gather all CSVs
    csv_files = []
    for d in args.dirs:
        csv_files.extend(glob.glob(os.path.join(d, "*.csv")))

    if args.offline_dataset_dirs:
        for d in args.offline_dataset_dirs:
            offline_csvs = glob.glob(os.path.join(d, "*.csv"))
            print(f"Found {len(offline_csvs)} CSV files in offline directory '{d}'.")
            csv_files.extend(offline_csvs)

    print(f"Found {len(csv_files)} total CSV files to process.")

    # 2. Check for resume files
    df_existing = None
    seen_keys = set()
    active_cells = set()

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from existing data: {args.resume_from}")
        if args.resume_from.endswith('.pkl'):
            with open(args.resume_from, 'rb') as f:
                existing_data = pickle.load(f)
            df_existing = pd.DataFrame(existing_data)
            del existing_data  # Free list from RAM
            existing_embeddings = np.vstack(df_existing['embedding'].values).astype(np.float32)

            seen_keys = set(zip(df_existing['Platform'], df_existing['Photo_ID'].apply(clean_photo_id)))

            # Retroactively clean existing URLs to virtual format if they are Mapillary/KartaView (vectorized)
            if not df_existing.empty and 'Image_URL' in df_existing.columns:
                platforms = df_existing['Platform'].to_numpy()
                photo_ids = df_existing['Photo_ID'].to_numpy()
                image_urls = df_existing['Image_URL'].to_numpy()
                df_existing['Image_URL'] = [
                    f"mapillary://{pid}" if str(plat).lower() == 'mapillary'
                    else (f"kartaview://{pid}" if str(plat).lower() == 'kartaview' else url)
                    for plat, pid, url in zip(platforms, photo_ids, image_urls)
                ]
        else:
            # Load minimal metadata columns only (uses ~50MB RAM even for millions of rows)
            print("Loading existing dataset metadata using PyArrow...")
            df_existing = load_dataframe(args.resume_from, columns=['Photo_ID', 'Platform', 'H3_Cell'])
            seen_keys = set(zip(df_existing['Platform'], df_existing['Photo_ID'].apply(clean_photo_id)))

        # Retroactively clean existing URLs to virtual format if they are Mapillary/KartaView (vectorized)
        if not df_existing.empty and 'Image_URL' in df_existing.columns:
            platforms = df_existing['Platform'].to_numpy()
            photo_ids = df_existing['Photo_ID'].to_numpy()
            image_urls = df_existing['Image_URL'].to_numpy()
            df_existing['Image_URL'] = [
                f"mapillary://{pid}" if str(plat).lower() == 'mapillary'
                else (f"kartaview://{pid}" if str(plat).lower() == 'kartaview' else url)
                for plat, pid, url in zip(platforms, photo_ids, image_urls)
            ]
            df_existing['Latitude'] = pd.to_numeric(df_existing['Latitude'], errors='coerce')
            df_existing['Longitude'] = pd.to_numeric(df_existing['Longitude'], errors='coerce')
        print(f"Loaded {len(df_existing)} existing images across {df_existing['H3_Cell'].nunique()} cells.")

    # Read CSVs in parallel using ThreadPoolExecutor
    all_dfs = []
    if csv_files:
        print(f"Reading {len(csv_files)} CSV files in parallel...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(load_and_preprocess_csv, f, offline_dirs=args.offline_dataset_dirs) for f in
                       csv_files]
            for fut in tqdm(as_completed(futures), total=len(csv_files), desc="Reading CSVs"):
                res = fut.result()
                if res is not None:
                    all_dfs.append(res)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame(
            columns=['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'Captured_At', 'License'])
    all_dfs = []  # Free memory

    # Vectorized H3 cell computation and filtering
    if not df_all.empty:
        # Standardize timestamps across all platforms (Flickr, Mapillary, iNaturalist) in a vectorized way
        df_all['Captured_At'] = standardize_timestamps_vectorized(df_all['Captured_At'])
        df_all['Latitude'] = pd.to_numeric(df_all['Latitude'], errors='coerce')
        df_all['Longitude'] = pd.to_numeric(df_all['Longitude'], errors='coerce')

        # Vectorized H3 cell calculation
        lats = df_all['Latitude'].to_numpy()
        lons = df_all['Longitude'].to_numpy()
        h3_res = args.h3_res
        df_all['H3_Cell'] = [
            h3.latlng_to_cell(float(lat), float(lon), h3_res) if pd.notna(lat) and pd.notna(lon) else None
            for lat, lon in zip(lats, lons)
        ]

        df_all = df_all.dropna(subset=['H3_Cell', 'Photo_ID'])
        df_all['Photo_ID'] = df_all['Photo_ID'].apply(clean_photo_id)

        # Convert any raw Mapillary/KartaView URLs to virtual URIs to prevent CDN expiration (vectorized)
        platforms = df_all['Platform'].to_numpy()
        photo_ids = df_all['Photo_ID'].to_numpy()
        image_urls = df_all['Image_URL'].to_numpy()
        df_all['Image_URL'] = [
            f"mapillary://{pid}" if str(plat).lower() == 'mapillary'
            else (f"kartaview://{pid}" if str(plat).lower() == 'kartaview' else url)
            for plat, pid, url in zip(platforms, photo_ids, image_urls)
        ]

        df_all = df_all.drop_duplicates(subset=['Platform', 'Photo_ID'])
        if seen_keys:
            df_all['temp_key'] = list(zip(df_all['Platform'], df_all['Photo_ID']))
            df_all = df_all[~df_all['temp_key'].isin(seen_keys)]
            df_all = df_all.drop(columns=['temp_key'])

    print(f"Total NEW raw images: {len(df_all)}")

    new_cells = set(df_all['H3_Cell'].unique()) if not df_all.empty else set()
    existing_cells = set(df_existing['H3_Cell'].unique()) if df_existing is not None else set()
    all_cells = new_cells | existing_cells
    active_cells = new_cells

    # Load embeddings only for active cells (saves massive RAM for millions of rows)
    df_existing_active = None
    existing_embeddings = None
    if args.resume_from and os.path.exists(args.resume_from) and not args.resume_from.endswith('.pkl') and active_cells:
        print("Loading active cell embeddings only using PyArrow Dataset...")
        t0 = time.time()
        import pyarrow.dataset as ds
        dataset = ds.dataset(args.resume_from, format="parquet")
        active_cells_list = list(active_cells)

        filter_expr = ds.field("H3_Cell").isin(active_cells_list)
        cols_to_load = ['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'H3_Cell', 'Captured_At', 'Image_URL',
                        'embedding']
        if 'License' in dataset.schema.names:
            cols_to_load.append('License')
        table_active = dataset.to_table(
            filter=filter_expr,
            columns=cols_to_load
        )
        df_existing_active = table_active.to_pandas()

        if len(df_existing_active) > 0:
            chunked_arr = table_active['embedding']
            dim = len(chunked_arr.chunk(0)[0].as_py())
            existing_embeddings = np.empty((len(df_existing_active), dim), dtype=np.float32)
            current_row = 0
            for chunk in chunked_arr.chunks:
                chunk_len = len(chunk)
                flat_chunk = chunk.flatten().to_numpy()
                existing_embeddings[current_row:current_row + chunk_len] = flat_chunk.reshape(chunk_len, dim)
                current_row += chunk_len

            # Retroactively clean existing URLs to virtual format if they are Mapillary/KartaView (vectorized)
            if 'Image_URL' in df_existing_active.columns:
                platforms = df_existing_active['Platform'].to_numpy()
                photo_ids = df_existing_active['Photo_ID'].to_numpy()
                image_urls = df_existing_active['Image_URL'].to_numpy()
                df_existing_active['Image_URL'] = [
                    f"mapillary://{pid}" if str(plat).lower() == 'mapillary'
                    else (f"kartaview://{pid}" if str(plat).lower() == 'kartaview' else url)
                    for plat, pid, url in zip(platforms, photo_ids, image_urls)
                ]
        else:
            existing_embeddings = None
        print(f" -> Loaded {len(df_existing_active):,} active existing embeddings in {time.time() - t0:.2f}s.")
    elif df_existing is not None and not df_existing.empty:
        df_existing_active = df_existing[df_existing['H3_Cell'].isin(active_cells)].copy()

    print(
        f"Total H3 cells: {len(all_cells)} ({len(active_cells)} active with new data, {len(all_cells) - len(active_cells)} inactive/skipped)")

    # 3. Load TIPSv2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TIPSv2 model on {device}...")
    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
    model.eval().to(device)

    # Pre-compute text features for Zero-Shot filtering
    text_features = None
    macro_idx = -1
    sky_idx = -1
    if not args.no_filter:
        print("Pre-computing zero-shot filter text embeddings...")
        with torch.no_grad():
            prompts = ["An indoor scene", "An outdoor landscape or street view"]
            if args.filter_macro:
                prompts.append(
                    "A close-up macro photo of a single leaf, plant petal, flower, insect, mushroom, or tree bark")
                macro_idx = len(prompts) - 1
            if args.filter_sky:
                prompts.append(
                    "A photo of the sky, a bird flying in the air, an insect in flight, an airplane, or a close-up of a cloud with no ground visible")
                sky_idx = len(prompts) - 1
            text_features = model.encode_text(prompts).cpu().numpy()

    # 4. Process and Deduplicate
    checkpoint_path = os.path.join(args.save_path, f"{args.output_name}_checkpoint.parquet")
    checkpoint_meta_path = os.path.join(args.save_path, f"{args.output_name}_checkpoint_meta.pkl")

    final_data = []
    processed_cells = set()

    if args.checkpoint_interval > 0 and os.path.exists(checkpoint_path) and os.path.exists(checkpoint_meta_path):
        print(f"Found checkpoint files: {checkpoint_path}")
        print(
            "Resuming from checkpoint. (To start fresh, delete these checkpoint files or run with --checkpoint_interval 0)")
        try:
            df_ckpt = load_dataframe(checkpoint_path)
            # Filter df_ckpt to only include active cells for final_data
            df_ckpt_active = df_ckpt[df_ckpt['H3_Cell'].isin(active_cells)]
            final_data = df_ckpt_active.to_dict('records')
            with open(checkpoint_meta_path, 'rb') as f:
                processed_cells = pickle.load(f)
            print(f"Loaded {len(final_data)} images from checkpoint. {len(processed_cells)} cells already processed.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch/resume_from.")
            final_data = []
            processed_cells = set()

    cells_to_process = list(active_cells)
    if args.limit_cells > 0:
        cells_to_process = cells_to_process[:args.limit_cells]
        print(f"Limiting to {args.limit_cells} cells for testing.")

    last_checkpoint_time = time.time()

    print("Grouping metadata by H3 cell...")
    from collections import defaultdict
    new_metadata_dict = defaultdict(list)
    if not df_all.empty:
        pids = df_all['Photo_ID'].to_numpy()
        plats = df_all['Platform'].to_numpy()
        lats = df_all['Latitude'].to_numpy()
        lons = df_all['Longitude'].to_numpy()
        urls = df_all['Image_URL'].to_numpy()
        caps = df_all['Captured_At'].to_numpy()
        cells = df_all['H3_Cell'].to_numpy()

        for pid, plat, lat, lon, url, cap, cell in zip(pids, plats, lats, lons, urls, caps, cells):
            new_metadata_dict[cell].append({
                'Photo_ID': pid,
                'Platform': plat,
                'Latitude': lat,
                'Longitude': lon,
                'Image_URL': url,
                'Captured_At': cap,
                'H3_Cell': cell
            })

    existing_items_dict = defaultdict(list)
    if df_existing_active is not None and not df_existing_active.empty:
        pids = df_existing_active['Photo_ID'].to_numpy()
        plats = df_existing_active['Platform'].to_numpy()
        lats = df_existing_active['Latitude'].to_numpy()
        lons = df_existing_active['Longitude'].to_numpy()
        urls = df_existing_active['Image_URL'].to_numpy()
        caps = df_existing_active['Captured_At'].to_numpy()
        cells = df_existing_active['H3_Cell'].to_numpy()

        if 'existing_embeddings' in locals() and existing_embeddings is not None:
            if args.resume_from.endswith('.pkl'):
                active_indices = df_existing_active.index.values
                embs = existing_embeddings[active_indices]
            else:
                embs = existing_embeddings
        else:
            embs = [None] * len(df_existing_active)

        for pid, plat, lat, lon, url, cap, cell, emb in zip(pids, plats, lats, lons, urls, caps, cells, embs):
            existing_items_dict[cell].append({
                'Photo_ID': pid,
                'Platform': plat,
                'Latitude': lat,
                'Longitude': lon,
                'Image_URL': url,
                'Captured_At': cap,
                'H3_Cell': cell,
                'embedding': emb
            })

    with ThreadPoolExecutor(max_workers=64) as executor:
        for cell in tqdm(cells_to_process, desc="Processing cells"):
            if cell in processed_cells:
                continue

            new_metadata = new_metadata_dict.get(cell, [])
            existing_items = existing_items_dict.get(cell, [])

            # If there's no new data for this cell, just keep the existing data
            if not new_metadata:
                final_data.extend(existing_items)
                processed_cells.add(cell)
                continue

            deduped = process_cell(cell, new_metadata, model, device, args.sim_threshold, executor,
                                   text_features, existing_items,
                                   cell_chunk_size=args.cell_chunk_size,
                                   tips_batch_size=args.tips_batch_size,
                                   macro_idx=macro_idx,
                                   sky_idx=sky_idx,
                                   offline_dirs=args.offline_dataset_dirs)
            final_data.extend(deduped)
            processed_cells.add(cell)

            # Periodic checkpoint saving
            if args.checkpoint_interval > 0:
                current_time = time.time()
                if current_time - last_checkpoint_time > args.checkpoint_interval:
                    save_checkpoint(final_data, processed_cells, checkpoint_path, checkpoint_meta_path,
                                    resume_from=args.resume_from, active_cells=active_cells)
                    last_checkpoint_time = current_time

    # Save a final checkpoint upon loop completion so that raw data is never lost if saving fails
    if args.checkpoint_interval > 0:
        print("\nSaving final completed checkpoint...")
        save_checkpoint(final_data, processed_cells, checkpoint_path, checkpoint_meta_path,
                        resume_from=args.resume_from, active_cells=active_cells)

    # 5. Save Results
    if not final_data:
        print("No data processed successfully.")
        return

    os.makedirs(args.save_path, exist_ok=True)
    out_df = pd.DataFrame(final_data)
    if 'Captured_At' in out_df.columns:
        out_df['Captured_At'] = standardize_timestamps_vectorized(out_df['Captured_At'])
    out_df['Latitude'] = pd.to_numeric(out_df['Latitude'], errors='coerce')
    out_df['Longitude'] = pd.to_numeric(out_df['Longitude'], errors='coerce')
    csv_path = os.path.join(args.save_path, f"{args.output_name}.csv")
    parquet_path = os.path.join(args.save_path, f"{args.output_name}.parquet")

    # If resume_from exists, we write the final parquet by streaming
    if args.resume_from and os.path.exists(args.resume_from) and not args.resume_from.endswith('.pkl'):
        print("Writing final Parquet database using streaming update...")
        # 1. Parquet stream update
        stream_update_parquet(args.resume_from, parquet_path, out_df, active_cells)
    else:
        # Save Full Data to Parquet (High-performance binary storage)
        save_dataframe(out_df, parquet_path)

    # Clean up checkpoint files on successful completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass
    if os.path.exists(checkpoint_meta_path):
        try:
            os.remove(checkpoint_meta_path)
        except Exception:
            pass

    print("\nProcessing Complete!")
    print(f"Unique images kept: {len(final_data)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Parquet saved to: {parquet_path}")


if __name__ == "__main__":
    main()
