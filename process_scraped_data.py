import os
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

def get_tips_embeddings(images, model, device):
    """Computes TIPSv2 embeddings for a list of PIL images."""
    if not images:
        return None
    
    embeddings = []
    with torch.no_grad():
        for img in images:
            img_tensor = tips_transform(img).unsqueeze(0).to(device)
            features = model.encode_image(img_tensor).cls_token.cpu().numpy()
            embeddings.append(features[0])
    
    return np.array(embeddings)

def process_cell(cell_id, metadata_list, model, device, sim_threshold, text_features=None):
    """Filters indoor images (Flickr only) and deduplicates images within an H3 cell."""
    # 1. Download all images in cell
    urls = [m['Image_URL'] for m in metadata_list]
    with ThreadPoolExecutor(max_workers=5) as executor:
        imgs = list(executor.map(download_image, urls))
    
    valid_indices = [i for i, img in enumerate(imgs) if img is not None]
    if not valid_indices:
        return []
    
    valid_imgs = [imgs[i] for i in valid_indices]
    all_embeddings = get_tips_embeddings(valid_imgs, model, device)
    
    if all_embeddings is None:
        return []

    # 2. Conditional Filtering and Deduplication
    final_indices = []
    processed_embeddings = []

    for i, idx in enumerate(valid_indices):
        metadata = metadata_list[idx]
        embedding = all_embeddings[i]
        
        # Flickr-only Indoor/Outdoor Filter
        if metadata['Platform'] == 'Flickr' and text_features is not None:
            sims = cosine_similarity(embedding.reshape(1, -1), text_features)[0]
            # sims[0] is Indoor, sims[1] is Outdoor
            if sims[0] > sims[1]:
                continue # Skip indoor Flickr image
        
        # Deduplication check against already kept images in this cell
        is_duplicate = False
        for j_emb in processed_embeddings:
            sim = cosine_similarity(embedding.reshape(1, -1), j_emb.reshape(1, -1))[0][0]
            if sim > sim_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_indices.append(idx)
            processed_embeddings.append(embedding)
    
    results = []
    for i, orig_idx in enumerate(final_indices):
        item = metadata_list[orig_idx].copy()
        item['embedding'] = processed_embeddings[i]
        results.append(item)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Consolidate, Filter, and Deduplicate Geo-Scraped Data using TIPSv2.")
    parser.add_argument("--dirs", nargs="+", required=True, help="List of directories containing chunked CSVs.")
    parser.add_argument("--output_name", type=str, default="geo_embedding_space", help="Base name for output files.")
    parser.add_argument("--h3_res", type=int, default=11, help="H3 resolution (~25m).")
    parser.add_argument("--sim_threshold", type=float, default=0.95, help="TIPSv2 cosine similarity threshold.")
    parser.add_argument("--no_filter", action="store_true", help="Disable Flickr indoor/outdoor filtering.")
    parser.add_argument("--limit_cells", type=int, default=0, help="Limit number of cells to process (for testing).")
    args = parser.parse_args()

    # 1. Gather all CSVs
    csv_files = []
    for d in args.dirs:
        csv_files.extend(glob.glob(os.path.join(d, "*.csv")))
    
    print(f"Found {len(csv_files)} CSV files.")
    
    # 2. Aggregate Metadata
    h3_buckets = {}
    total_raw_images = 0
    seen_photo_ids = set()

    for f in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(f)
            if df.empty: continue

            if 'uuid' in df.columns and 'source' in df.columns and 'orig_id' in df.columns:
                df['Platform'] = df['source']
                df['Latitude'] = df['lat']
                df['Longitude'] = df['lon']
                df['Photo_ID'] = df['orig_id']

                def make_url(row):
                    src = str(row['source']).lower()
                    if src == 'mapillary':
                        return f"mapillary://{row['orig_id']}"
                    elif src == 'kartaview':
                        return f"kartaview://{row['orig_id']}"
                    return row['url']

                df['Image_URL'] = df.apply(make_url, axis=1)
            else:
                # Normalize columns
                platform = 'Flickr' if 'flickr' in f.lower() else 'Mapillary'

                # Ensure standard names
                col_map = {
                    'latitude': 'Latitude', 'longitude': 'Longitude', 
                    'image_url': 'Image_URL', 'photo_id': 'Photo_ID',
                    'ID': 'Photo_ID'
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                # If Platform column doesn't exist, use inferred platform
                if 'Platform' not in df.columns:
                    df['Platform'] = platform

            for _, row in df.iterrows():
                try:
                    photo_id = str(row['Photo_ID'])
                    if photo_id in seen_photo_ids:
                        continue

                    lat, lon = float(row['Latitude']), float(row['Longitude'])
                    cell = h3.latlng_to_cell(lat, lon, args.h3_res)

                    item = {
                        'Photo_ID': photo_id,
                        'Platform': row['Platform'],
                        'Latitude': lat,
                        'Longitude': lon,
                        'Image_URL': row['Image_URL'],
                        'H3_Cell': cell
                    }

                    if cell not in h3_buckets:
                        h3_buckets[cell] = []
                    h3_buckets[cell].append(item)
                    seen_photo_ids.add(photo_id)
                    total_raw_images += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Total raw images: {total_raw_images}")
    print(f"Unique H3 cells: {len(h3_buckets)}")

    # 3. Load TIPSv2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TIPSv2 model on {device}...")
    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
    model.eval().to(device)

    # Pre-compute text features for Zero-Shot filtering
    text_features = None
    if not args.no_filter:
        print("Pre-computing indoor/outdoor text embeddings...")
        with torch.no_grad():
            prompts = ["An indoor scene", "An outdoor landscape or street view"]
            text_features = model.encode_text(prompts).cpu().numpy()

    # 4. Process and Deduplicate
    final_data = []
    cells_to_process = list(h3_buckets.keys())
    if args.limit_cells > 0:
        cells_to_process = cells_to_process[:args.limit_cells]
        print(f"Limiting to {args.limit_cells} cells for testing.")

    for cell in tqdm(cells_to_process, desc="Deduplicating cells"):
        deduped = process_cell(cell, h3_buckets[cell], model, device, args.sim_threshold, text_features)
        final_data.extend(deduped)

    # 5. Save Results
    if not final_data:
        print("No data processed successfully.")
        return

    out_df = pd.DataFrame(final_data)
    csv_path = f"{args.output_name}.csv"
    pkl_path = f"{args.output_name}.pkl"

    # Save CSV without embeddings
    out_df.drop(columns=['embedding']).to_csv(csv_path, index=False)
    
    # Save Full Data (including embeddings) to Pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"\nProcessing Complete!")
    print(f"Unique images kept: {len(final_data)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Pickle saved to: {pkl_path}")

if __name__ == "__main__":
    main()
