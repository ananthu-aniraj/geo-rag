import os
import time
import glob
import pandas as pd
import torch
import h3
import pickle
import argparse
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel
from torchvision import transforms

# TIPSv2 specific transform
tips_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


def clean_photo_id(val):
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.endswith('.0'):
        val_str = val_str[:-2]
    return val_str


def standardize_timestamp(ts):
    """Standardizes various timestamp formats from Flickr, Mapillary, and iNaturalist to ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)."""
    if pd.isna(ts) or not ts:
        return None
    ts_str = str(ts).strip()
    try:
        # pd.to_datetime handles timezone-aware, custom strings, and timestamps gracefully
        dt = pd.to_datetime(ts_str, errors='coerce', utc=True)
        if pd.notna(dt):
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        pass
    return ts_str


def download_image(url):
    """Downloads an image and returns a PIL Image object."""
    try:
        if url.startswith("mapillary://"):
            orig_id = url.split("://")[1]
            api_url = f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
            res = requests.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
            else:
                return None
        elif url.startswith("kartaview://"):
            orig_id = url.split("://")[1]
            api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
            res = requests.get(api_url, timeout=10)
            if res.status_code == 200:
                data = res.json().get("result", {}).get("data", {})
                url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")
            else:
                return None

        if not url:
            return None

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
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
            batch = images[i : i + batch_size]
            # Batch transform and stack
            batch_tensors = torch.stack([tips_transform(img) for img in batch]).to(device)
            features = model.encode_image(batch_tensors).cls_token
            all_features.append(features.squeeze(1).cpu().numpy())

    return np.concatenate(all_features, axis=0)


def process_cell(cell_id, metadata_list, model, device, sim_threshold, executor, text_features=None, existing_items=None, cell_chunk_size=128, tips_batch_size=32):
    """Filters indoor images (Flickr only) and deduplicates images within an H3 cell in chunks."""
    results = existing_items.copy() if existing_items else []
    processed_embeddings = [item['embedding'] for item in results]

    # Process new images in chunks to limit peak memory usage
    for chunk_start in range(0, len(metadata_list), cell_chunk_size):
        chunk_metadata = metadata_list[chunk_start : chunk_start + cell_chunk_size]
        urls = [m['Image_URL'] for m in chunk_metadata]
        
        # Download images in parallel for this chunk
        imgs = list(executor.map(download_image, urls))
        
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

            # Zero-shot filtering (Indoor and Macro/Close-up)
            if text_features is not None:
                sims = all_io_sims[i]
                is_macro_enabled = (text_features.shape[0] == 3)
                
                if is_macro_enabled:
                    best_class = np.argmax(sims)
                    # Flickr-only Indoor Filter (Class 0)
                    if metadata['Platform'] == 'Flickr' and best_class == 0:
                        continue
                    # iNaturalist-only Macro/Close-up Filter (Class 2)
                    if str(metadata['Platform']).lower() == 'inaturalist' and best_class == 2:
                        continue
                else:
                    # Standard Flickr-only Indoor Filter
                    if metadata['Platform'] == 'Flickr' and sims[0] > sims[1]:
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


def save_checkpoint(final_data, processed_cells, checkpoint_path, checkpoint_meta_path):
    """Saves the intermediate state to checkpoint files atomically."""
    tmp_path = f"{checkpoint_path}.tmp"
    tmp_meta_path = f"{checkpoint_meta_path}.tmp"
    try:
        # Convert final_data to DataFrame and save to tmp parquet
        if not final_data:
            df = pd.DataFrame(columns=['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'H3_Cell', 'embedding', 'Captured_At'])
        else:
            df = pd.DataFrame(final_data)
        df.to_parquet(tmp_path, index=False)
        
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
    parser.add_argument("--limit_cells", type=int, default=0, help="Limit number of cells to process (for testing).")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a previously generated .pkl or parquet file to resume from.")
    parser.add_argument("--checkpoint_interval", type=int, default=1800, help="Interval in seconds to save checkpoints (0 to disable).")
    parser.add_argument("--cell_chunk_size", type=int, default=128, help="Number of images within a cell to download/process in a chunk.")
    parser.add_argument("--tips_batch_size", type=int, default=32, help="Batch size for TIPSv2 embedding inference.")
    args = parser.parse_args()

    # 1. Gather all CSVs
    csv_files = []
    for d in args.dirs:
        csv_files.extend(glob.glob(os.path.join(d, "*.csv")))

    print(f"Found {len(csv_files)} CSV files.")

    # 2. Aggregating Metadata & Handling Resume
    df_existing = None
    seen_photo_ids = set()

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from existing data: {args.resume_from}")
        if args.resume_from.endswith('.pkl'):
            with open(args.resume_from, 'rb') as f:
                existing_data = pickle.load(f)
            df_existing = pd.DataFrame(existing_data)
            del existing_data  # Free list from RAM
        else:
            df_existing = pd.read_parquet(args.resume_from)
        
        seen_photo_ids = set(df_existing['Photo_ID'].apply(clean_photo_id))
        
        # Retroactively clean existing URLs to virtual format if they are Mapillary/KartaView
        if not df_existing.empty and 'Image_URL' in df_existing.columns:
            df_existing['Image_URL'] = df_existing.apply(
                lambda r: f"mapillary://{r['Photo_ID']}" if str(r['Platform']).lower() == 'mapillary'
                else (f"kartaview://{r['Photo_ID']}" if str(r['Platform']).lower() == 'kartaview' else r['Image_URL']),
                axis=1
            )
        print(f"Loaded {len(df_existing)} existing images across {df_existing['H3_Cell'].nunique()} cells.")

    all_dfs = []
    for f in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(f)
            if df.empty: continue

            if 'uuid' in df.columns and 'source' in df.columns and 'orig_id' in df.columns:
                df['Platform'] = df['source']
                df['Latitude'] = df['lat']
                df['Longitude'] = df['lon']
                df['Photo_ID'] = df['orig_id']
                df['Captured_At'] = df['datetime_local'] if 'datetime_local' in df.columns else None
                df['Image_URL'] = df.apply(lambda r: f"mapillary://{r['orig_id']}" if str(r['source']).lower() == 'mapillary' else (f"kartaview://{r['orig_id']}" if str(r['source']).lower() == 'kartaview' else (r['url'] if 'url' in r else None)), axis=1)
            else:
                if 'inaturalist' in f.lower():
                    platform = 'iNaturalist'
                elif 'flickr' in f.lower():
                    platform = 'Flickr'
                else:
                    platform = 'Mapillary'
                
                col_map = {
                    'latitude': 'Latitude', 
                    'longitude': 'Longitude', 
                    'image_url': 'Image_URL', 
                    'photo_id': 'Photo_ID', 
                    'ID': 'Photo_ID', 
                    'captured_at': 'Captured_At', 
                    'Captured_At': 'Captured_At',
                    'Date_Observed': 'Captured_At',
                    'observed_on_string': 'Captured_At'
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                if 'Platform' not in df.columns:
                    df['Platform'] = platform

            required_cols = ['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'Captured_At']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            df = df[required_cols]
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame(columns=['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL'])
    all_dfs = []  # Free memory

    # Vectorized H3 cell computation and filtering
    if not df_all.empty:
        # Standardize timestamps across all platforms (Flickr, Mapillary, iNaturalist)
        df_all['Captured_At'] = df_all['Captured_At'].apply(standardize_timestamp)
        df_all['H3_Cell'] = df_all.apply(lambda r: h3.latlng_to_cell(float(r['Latitude']), float(r['Longitude']), args.h3_res) if pd.notna(r['Latitude']) and pd.notna(r['Longitude']) else None, axis=1)
        df_all = df_all.dropna(subset=['H3_Cell', 'Photo_ID'])
        df_all['Photo_ID'] = df_all['Photo_ID'].apply(clean_photo_id)
        
        # Convert any raw Mapillary/KartaView URLs to virtual URIs to prevent CDN expiration
        df_all['Image_URL'] = df_all.apply(
            lambda r: f"mapillary://{r['Photo_ID']}" if str(r['Platform']).lower() == 'mapillary' 
            else (f"kartaview://{r['Photo_ID']}" if str(r['Platform']).lower() == 'kartaview' else r['Image_URL']),
            axis=1
        )
        
        df_all = df_all.drop_duplicates(subset=['Photo_ID'])
        if seen_photo_ids:
            df_all = df_all[~df_all['Photo_ID'].isin(seen_photo_ids)]

    print(f"Total NEW raw images: {len(df_all)}")
    
    new_cells = set(df_all['H3_Cell'].unique()) if not df_all.empty else set()
    existing_cells = set(df_existing['H3_Cell'].unique()) if df_existing is not None else set()
    all_cells = new_cells | existing_cells
    print(f"Total H3 cells to verify/process: {len(all_cells)}")

    # 3. Load TIPSv2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TIPSv2 model on {device}...")
    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
    model.eval().to(device)

    # Pre-compute text features for Zero-Shot filtering
    text_features = None
    if not args.no_filter:
        print("Pre-computing zero-shot filter text embeddings...")
        with torch.no_grad():
            prompts = ["An indoor scene", "An outdoor landscape or street view"]
            if args.filter_macro:
                prompts.append("A close-up macro photo of a animal, single leaf, plant petal, flower, insect, mushroom, or tree bark")
            text_features = model.encode_text(prompts).cpu().numpy()

    # 4. Process and Deduplicate
    checkpoint_path = os.path.join(args.save_path, f"{args.output_name}_checkpoint.parquet")
    checkpoint_meta_path = os.path.join(args.save_path, f"{args.output_name}_checkpoint_meta.pkl")
    
    final_data = []
    processed_cells = set()
    
    if args.checkpoint_interval > 0 and os.path.exists(checkpoint_path) and os.path.exists(checkpoint_meta_path):
        print(f"Found checkpoint files: {checkpoint_path}")
        print("Resuming from checkpoint. (To start fresh, delete these checkpoint files or run with --checkpoint_interval 0)")
        try:
            df_ckpt = pd.read_parquet(checkpoint_path)
            final_data = df_ckpt.to_dict('records')
            with open(checkpoint_meta_path, 'rb') as f:
                processed_cells = pickle.load(f)
            print(f"Loaded {len(final_data)} images from checkpoint. {len(processed_cells)} cells already processed.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch/resume_from.")
            final_data = []
            processed_cells = set()

    cells_to_process = list(all_cells)
    if args.limit_cells > 0:
        cells_to_process = cells_to_process[:args.limit_cells]
        print(f"Limiting to {args.limit_cells} cells for testing.")

    last_checkpoint_time = time.time()
    
    print("Grouping metadata by H3 cell...")
    new_metadata_dict = {}
    if not df_all.empty:
        new_metadata_dict = {cell: grp.to_dict('records') for cell, grp in df_all.groupby('H3_Cell')}
        
    existing_items_dict = {}
    if df_existing is not None:
        existing_items_dict = {cell: grp.to_dict('records') for cell, grp in df_existing.groupby('H3_Cell')}

    with ThreadPoolExecutor(max_workers=20) as executor:
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
                                   tips_batch_size=args.tips_batch_size)
            final_data.extend(deduped)
            processed_cells.add(cell)
            
            # Periodic checkpoint saving
            if args.checkpoint_interval > 0:
                current_time = time.time()
                if current_time - last_checkpoint_time > args.checkpoint_interval:
                    save_checkpoint(final_data, processed_cells, checkpoint_path, checkpoint_meta_path)
                    last_checkpoint_time = current_time

    # 5. Save Results
    if not final_data:
        print("No data processed successfully.")
        return

    os.makedirs(args.save_path, exist_ok=True)
    out_df = pd.DataFrame(final_data)
    csv_path = os.path.join(args.save_path, f"{args.output_name}.csv")
    parquet_path = os.path.join(args.save_path, f"{args.output_name}.parquet")

    # Save CSV without embeddings (for human readability)
    cols_to_drop = [c for c in ['embedding', 'patch_embedding'] if c in out_df.columns]
    out_df.drop(columns=cols_to_drop).to_csv(csv_path, index=False)

    # Save Full Data to Parquet (High-performance binary storage)
    out_df.to_parquet(parquet_path, index=False)

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

    print(f"\nProcessing Complete!")
    print(f"Unique images kept: {len(final_data)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Parquet saved to: {parquet_path}")


if __name__ == "__main__":
    main()
