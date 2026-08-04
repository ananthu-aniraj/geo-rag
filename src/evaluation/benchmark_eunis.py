import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from requests.adapters import HTTPAdapter
from shapely.geometry import Point
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoModel,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from urllib3.util import Retry

from src.evaluation.metrics import compute_ap, compute_precision_at_k, compute_rr
from src.utils.spatial_overlays import get_crs_transformer, lookup_raster_pixel

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'
DISCARD_CLASSES = {2, 12, 20, 43, 80, 83, 102, 127}  # sky, person, car, sign, bus, truck, van, bike

# Global connection pooled session for thread-safe downloads
http_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=64,
    pool_maxsize=64,
    max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
)
http_session.mount("https://", _adapter)
http_session.mount("http://", _adapter)


def download_image(url, photo_id=None, image_size=448):
    """Downloads an image using the global connection pool and returns a resized PIL Image."""
    try:
        if url.startswith("mapillary://") or (photo_id and "fbcdn.net" in url):
            orig_id = str(photo_id) if photo_id else url.split("://")[1]
            api_url = f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
            res = http_session.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
        if not url:
            return None
        res = http_session.get(url, timeout=10)
        if res.status_code == 200:
            img = Image.open(BytesIO(res.content)).convert("RGB")
            img_resized = img.resize((image_size, image_size))
            img.close()
            return img_resized
    except Exception:
        pass
    return None


def load_eunis_mapping_from_dbf(dbf_path):
    """Loads EUNIS value mapping dynamically from the associated DBF file if it exists."""
    mapping = {}
    if os.path.exists(dbf_path):
        try:
            print(f"Loading EUNIS classification mapping from local DBF: {dbf_path}...")
            # Geopandas can read DBF files directly via read_file
            gdf = gpd.read_file(dbf_path)
            val_col = next((c for c in gdf.columns if c.lower() == "value"), None)
            label_col = next((c for c in gdf.columns if c.lower() in ["maes_l2", "maes_level2", "class_name"]), None)

            if val_col and label_col:
                for _, row in gdf.iterrows():
                    val = int(float(str(row[val_col])))
                    label = str(row[label_col]).strip()
                    if label and label.lower() != 'none':
                        mapping[val] = label
                print(f"Loaded {len(mapping)} dynamic class mappings from DBF.")
        except Exception as e:
            print(f"Warning: Failed to load DBF attribute mapping: {e}")

    return mapping


from src.utils.spatial_overlays import get_eunis_label


def encode_image_value_attention(model_image, img):
    """Extracts spatial value features from model vision encoder using the MaskCLIP values trick."""
    B, _, H, W = img.shape
    P = model_image.patch_size if hasattr(model_image, 'patch_size') else 14
    new_H = math.ceil(H / P) * P
    new_W = math.ceil(W / P) * P

    if (H, W) != (new_H, new_W):
        img = F.interpolate(img, size=(new_H, new_W), mode='bicubic', align_corners=False)

    B, _, h_i, w_i = img.shape
    x = model_image.prepare_tokens_with_masks(img)

    num_register = getattr(model_image, 'num_register_tokens', 1)
    all_blocks = list(model_image.blocks)
    for i, blk in enumerate(all_blocks):
        if i < len(all_blocks) - 1:
            x = blk(x)
        else:
            x_normed = blk.norm1(x)
            b_dim, n_dim, c_dim = x_normed.shape
            qkv = (
                blk.attn.qkv(x_normed)
                .reshape(b_dim, n_dim, 3, blk.attn.num_heads, c_dim // blk.attn.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            v = qkv[2]
            v_out = v.transpose(1, 2).reshape(b_dim, n_dim, c_dim)
            v_out = blk.attn.proj(v_out)
            v_out = blk.ls1(v_out)
            x_val = v_out + x

            y_val = blk.norm2(x_val)
            y_val = blk.ls2(blk.mlp(y_val))
            x_val = x_val + y_val

    x_val = model_image.norm(x_val)
    patch_tokens = x_val[:, 1 + num_register:, :]
    blocks_patches = patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
    return blocks_patches


def main():
    parser = argparse.ArgumentParser(description="EUNIS Ecosystem Map Representation Semantic Retrieval Benchmark.")
    parser.add_argument("--csv_path", type=str, default="./full_pipeline_output/geo_space_deduplicated.csv",
                        help="Path to CSV containing geolocated images.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV containing geolocated images (alias for --csv_path).")
    parser.add_argument("--countries_shp", type=str, default="shapefiles/ne_10m_admin_0_countries.shp",
                        help="Path to Natural Earth countries shapefile to filter for Europe.")
    parser.add_argument("--raster", type=str,
                        default="/user/aaniraj/home/Documents/Projects/data/eea_r_3035_100_m_ecosystem-types-terrestrial-r_p_2012_v03_r01/eea_r_3035_100_m_etm-terrestrial-r_2012_v3-1_r00.tif",
                        help="Path to EUNIS Ecosystem map GeoTIFF.")

    parser.add_argument("--num_queries", type=int, default=100, help="Number of query evaluations to run.")
    parser.add_argument("--num_database", type=int, default=500, help="Number of database images to search against.")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU batch size for feature extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tips_model_path", type=str, default=None,
                        help="Path to the official TIPSv2 model checkpoint (.npz). If None, uses Hugging Face 'google/tipsv2-b14'.")
    parser.add_argument("--tips_model_variant", type=str, default="B", choices=["S", "B", "L", "So400m", "g"],
                        help="Variant of the official TIPSv2 model.")
    parser.add_argument("--tips_low_res", action="store_true", help="Set image resolution to 224px instead of 448px.")
    parser.add_argument("--output_report", type=str, default="./benchmark_results/eunis_report.txt",
                        help="Path to write the report summary.")
    parser.add_argument("--output_csv", type=str, default="./benchmark_results/eunis_results.csv",
                        help="Path to write detailed query CSV results.")
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else args.csv_path

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Open the EUNIS Ecosystem map
    if not os.path.exists(args.raster):
        print(f"Error: EUNIS Ecosystem raster map '{args.raster}' not found.")
        sys.exit(1)

    print(f"Opening EUNIS Ecosystem raster: {args.raster}...")
    try:
        raster_dataset = rasterio.open(args.raster)
    except Exception as e:
        print(f"Error opening raster: {e}")
        sys.exit(1)

    # Load dynamic attribute mappings from associated DBF if available
    dbf_path = args.raster.replace(".tif", ".tif.vat.dbf")
    dynamic_mapping = load_eunis_mapping_from_dbf(dbf_path)

    # 2. Load CSV
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        sys.exit(1)

    print(f"Loading image CSV metadata: {csv_path}...")
    df = pd.read_csv(csv_path)

    # Identify key columns in CSV
    lat_col = next((col for col in df.columns if col.lower() in ["latitude", "lat"]), None)
    lon_col = next((col for col in df.columns if col.lower() in ["longitude", "lon", "lng"]), None)
    url_col = next((col for col in df.columns if col.lower() in ["image_url", "url"]), None)
    id_col = next((col for col in df.columns if col.lower() in ["photo_id", "id"]), None)

    if not lat_col or not lon_col:
        print("Error: Could not locate Latitude/Longitude columns in the CSV.")
        sys.exit(1)

    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    # Perform European filtering
    continent_col = next((col for col in df.columns if col.lower() == "continent"), None)
    if continent_col:
        print("Filtering European coordinates using the CSV 'Continent' column...")
        df = df[df[continent_col].astype(str).str.lower() == "europe"].reset_index(drop=True)
        print(f"Keep {len(df)} records inside Europe.")
    elif args.countries_shp and os.path.exists(args.countries_shp):
        print(f"Filtering European coordinates using shapefile: {args.countries_shp}...")
        try:
            countries_gdf = gpd.read_file(args.countries_shp)
            europe_gdf = countries_gdf[countries_gdf['CONTINENT'] == 'Europe']

            # Convert df to GeoDataFrame
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

            if europe_gdf.crs != geo_df.crs:
                europe_gdf = europe_gdf.to_crs(geo_df.crs)

            joined = gpd.sjoin(geo_df, europe_gdf, how="inner", predicate="intersects")
            df = pd.DataFrame(joined.drop(columns="geometry")).reset_index(drop=True)
            print(f"Keep {len(df)} records inside Europe.")
        except Exception as e:
            print(f"Warning: Failed to filter Europe using shapefile: {e}. Falling back to raster bounds check.")
    else:
        print("Neither 'Continent' column in CSV nor countries shapefile found. Falling back to raster bounds check.")

    if len(df) == 0:
        print("Error: No coordinates left inside Europe.")
        return

    # Coordinate transformer from WGS84 (EPSG:4326) to Raster CRS (often EPSG:3035)
    print(f"Raster CRS: {raster_dataset.crs}. Setting up coordinate transformer...")
    transformer, has_axis_order = get_crs_transformer(raster_dataset.crs)

    matched_images = []
    print("Mapping coordinates to EUNIS Ecosystem classes...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Spatial Query"):
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])

            # Sample pixel value using unified lookup
            pixel_val = lookup_raster_pixel(lat, lon, raster_dataset, transformer, has_axis_order)

            # Skip nodata / empty values
            if pixel_val is None or pixel_val == raster_dataset.nodata or pixel_val <= 0:
                continue

            eunis_class = get_eunis_label(pixel_val, dynamic_mapping)
            if eunis_class == "Unknown":
                continue

            matched_images.append({
                "url": str(row[url_col]) if url_col else "",
                "photo_id": str(row[id_col]) if id_col else None,
                "lat": lat,
                "lon": lon,
                "eunis_class": eunis_class
            })
        except Exception:
            continue

    raster_dataset.close()

    if not matched_images:
        print("Error: No images mapped successfully to EUNIS ecosystem categories.")
        return

    print(f"Successfully mapped {len(matched_images):,} records to EUNIS categories.")

    # Group matched images by EUNIS category
    images_by_class = {}
    for item in matched_images:
        cls = item["eunis_class"]
        if cls not in images_by_class:
            images_by_class[cls] = []
        images_by_class[cls].append(item)

    sorted_classes = sorted(images_by_class.keys())
    for cls in sorted_classes:
        random.shuffle(images_by_class[cls])

    # Interleave categories to ensure query diversity
    selected_images = []
    max_images_in_any_class = max(len(images_by_class[cls]) for cls in sorted_classes)

    for i in range(max_images_in_any_class):
        for cls in sorted_classes:
            if i < len(images_by_class[cls]):
                selected_images.append(images_by_class[cls][i])

    # Partition into Queries and Database
    total_needed = args.num_queries + args.num_database
    if len(selected_images) < total_needed:
        print(f"Warning: Only {len(selected_images)} matched images available. Adjusting query/database split.")
        if len(selected_images) <= args.num_queries:
            args.num_queries = max(1, len(selected_images) // 5)
        args.num_database = len(selected_images) - args.num_queries

    selected_images = selected_images[:args.num_queries + args.num_database]

    # Download images in parallel
    print(f"Downloading {len(selected_images)} images for benchmarking in parallel...")
    images_dict = {}
    image_size = 224 if args.tips_low_res else 448
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(download_image, item['url'], item.get('photo_id'), image_size): idx
            for idx, item in enumerate(selected_images)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            idx = futures[future]
            img = future.result()
            if img:
                images_dict[idx] = img

    print(f"Successfully downloaded {len(images_dict)} images.")
    if len(images_dict) < 10:
        print("Error: Too few images successfully downloaded to run benchmark.")
        return

    # Keep only downloaded items
    active_indices = sorted(list(images_dict.keys()))
    selected_images = [selected_images[idx] for idx in active_indices]
    for idx, img in enumerate([images_dict[k] for k in active_indices]):
        selected_images[idx]['img'] = img

    queries_meta = selected_images[:args.num_queries]
    database_meta = selected_images[args.num_queries:]

    print(f"Final evaluation split: {len(queries_meta)} Queries, {len(database_meta)} Database.")
    if len(queries_meta) == 0 or len(database_meta) == 0:
        print("Error: Empty query or database split. Adjust your parameters.")
        return

    # 3. Initialize Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing models on {device}...")

    # Load TIPSv2 Model (either official checkpoint or Hugging Face)
    if args.tips_model_path:
        print(f"Loading official TIPSv2 checkpoint from: {args.tips_model_path}...")
        from src.models import tips_image_encoder as image_encoder

        model_def = {
            'S': image_encoder.vit_small,
            'B': image_encoder.vit_base,
            'L': image_encoder.vit_large,
            'So400m': image_encoder.vit_so400m,
            'g': image_encoder.vit_giant2,
        }[args.tips_model_variant]

        ffn_layer = 'swiglu' if args.tips_model_variant == 'g' else 'mlp'

        checkpoint = dict(np.load(args.tips_model_path, allow_pickle=False))
        for key in checkpoint:
            checkpoint[key] = torch.tensor(checkpoint[key])

        image_size = 224 if args.tips_low_res else 448
        model = model_def(
            img_size=image_size,
            patch_size=14,
            ffn_layer=ffn_layer,
            block_chunks=0,
            init_values=1.0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
        model.load_state_dict(checkpoint)
        tipsv2 = model.eval().to(device)
    else:
        print("Loading Hugging Face google/tipsv2-b14 model...")
        tipsv2 = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).eval().to(device)
        image_size = 448

    print("Loading SegFormer model...")
    seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").eval().to(
        device)

    # 4. Setup representations dynamically
    if args.tips_model_path:
        representations = {
            "TIPSv2 1st CLS": {"query": [], "db": []},
            "TIPSv2 2nd CLS": {"query": [], "db": []},
            "TIPSv2 Average Patch": {"query": [], "db": []},
            "TIPSv2 Seg-Masked": {"query": [], "db": []}
        }
    else:
        representations = {
            "TIPSv2 CLS": {"query": [], "db": []},
            "TIPSv2 Average Patch": {"query": [], "db": []},
            "TIPSv2 Seg-Masked": {"query": [], "db": []}
        }

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    grid_size = 16 if (args.tips_model_path and args.tips_low_res) else 32
    num_patches = grid_size * grid_size

    def extract_features_batch(metadata_list, split_key):
        print(f"Extracting features for {len(metadata_list)} images ({split_key} split)...")
        for i in tqdm(range(0, len(metadata_list), args.batch_size), desc=f"Extraction ({split_key})"):
            batch_meta = metadata_list[i: i + args.batch_size]
            batch_imgs = []

            for item in batch_meta:
                try:
                    img = item['img']
                    if img.size != (image_size, image_size):
                        img = img.resize((image_size, image_size))
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load image '{item.get('url', '')}': {e}")

            if not batch_imgs:
                continue

            # A. SegFormer segmentation masks
            inputs = seg_processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = seg_model(**inputs)
            logits = torch.nn.functional.interpolate(outputs.logits, size=(image_size, image_size), mode="bilinear",
                                                     align_corners=False)
            pred_masks = logits.argmax(dim=1).cpu().numpy()
            
            # Free SegFormer GPU memory
            del inputs, outputs, logits

            # B. TIPSv2 feature extraction
            img_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
            with torch.no_grad():
                is_local = bool(args.tips_model_path)
                from src.models.vision_model_inference import (
                    extract_benchmark_features_single_pass,
                )
                cls_out, patch_tokens_vals = extract_benchmark_features_single_pass(tipsv2, img_tensors, is_local=is_local)
                patch_tokens_vals = patch_tokens_vals.reshape(len(batch_imgs), num_patches, -1)

                if is_local:
                    first_cls, second_cls = cls_out
                    representations["TIPSv2 1st CLS"][split_key].extend(first_cls)
                    representations["TIPSv2 2nd CLS"][split_key].extend(second_cls)
                else:
                    cls_tokens = cls_out
                    representations["TIPSv2 CLS"][split_key].extend(cls_tokens)
            del img_tensors

            # Process patch poolings for each image in batch
            for idx in range(len(batch_imgs)):
                patch_tokens = patch_tokens_vals[idx]  # (num_patches, D)
                pred_mask = pred_masks[idx]

                # TIPSv2 Average Patch
                avg_patch = np.mean(patch_tokens, axis=0)
                representations["TIPSv2 Average Patch"][split_key].append(avg_patch)

                # TIPSv2 Seg-Masked
                keep_mask = np.ones_like(pred_mask, dtype=float)
                for c in DISCARD_CLASSES:
                    keep_mask[pred_mask == c] = 0.0

                # Downsample keep mask to grid_size x grid_size patch resolution
                patch_weights = np.zeros((grid_size, grid_size))
                for r in range(grid_size):
                    for c in range(grid_size):
                        patch_weights[r, c] = np.mean(keep_mask[r * 14:(r + 1) * 14, c * 14:(c + 1) * 14])
                patch_weights_flat = patch_weights.flatten()[:, np.newaxis]  # (num_patches, 1)

                masked_patch_sum = np.sum(patch_tokens * patch_weights_flat, axis=0)
                masked_patch_weight_sum = np.sum(patch_weights_flat)
                if masked_patch_weight_sum > 0:
                    masked_avg_embed = (masked_patch_sum / (masked_patch_weight_sum + 1e-9))
                else:
                    masked_avg_embed = avg_patch

                representations["TIPSv2 Seg-Masked"][split_key].append(masked_avg_embed)

    extract_features_batch(queries_meta, "query")
    extract_features_batch(database_meta, "db")

    # 5. Retrieval & Similarity Benchmarking
    expanded_representations = {}
    for rep_name, splits in list(representations.items()):
        expanded_representations[f"{rep_name} (FP32)"] = splits
        
        q_fp16 = [v.astype(np.float16).astype(np.float32) for v in splits["query"]]
        db_fp16 = [v.astype(np.float16).astype(np.float32) for v in splits["db"]]
        expanded_representations[f"{rep_name} (FP16)"] = {
            "query": q_fp16,
            "db": db_fp16
        }
    representations = expanded_representations

    results = {}
    detailed_rows = []

    for rep_name, splits in representations.items():
        q_vectors = np.array(splits["query"])
        db_vectors = np.array(splits["db"])

        if len(q_vectors) == 0 or len(db_vectors) == 0:
            print(f"Skipping {rep_name} due to empty features.")
            continue

        # L2 Normalize vectors
        q_norms = np.linalg.norm(q_vectors, axis=1, keepdims=True) + 1e-9
        db_norms = np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-9
        q_vectors_norm = q_vectors / q_norms
        db_vectors_norm = db_vectors / db_norms

        results[rep_name] = {
            "p@1": 0.0,
            "p@5": 0.0,
            "p@10": 0.0,
            "map@10": 0.0,
            "mrr@10": 0.0
        }

        # Compute all similarities in a single batched operation: (Q, D_dim) x (D_dim, DB) -> (Q, DB)
        sim_matrix = np.dot(q_vectors_norm, db_vectors_norm.T)
        # Retrieve top 10 database indices for all queries at once
        top_indices = np.argsort(-sim_matrix, axis=1)[:, :10]

        # Query loop
        for q_idx in range(len(queries_meta)):
            q_item = queries_meta[q_idx]
            q_label = q_item["eunis_class"]
            sorted_db_indices = top_indices[q_idx]

            retrieved_items = [database_meta[idx] for idx in sorted_db_indices[:10]]
            retrieved_labels = [item["eunis_class"] for item in retrieved_items]

            # P@1
            p1 = compute_precision_at_k(retrieved_labels, q_label, k=1)
            results[rep_name]["p@1"] += p1

            # P@5
            p5 = compute_precision_at_k(retrieved_labels, q_label, k=5)
            results[rep_name]["p@5"] += p5

            # P@10
            p10 = compute_precision_at_k(retrieved_labels, q_label, k=10)
            results[rep_name]["p@10"] += p10

            # mAP@10
            ap = compute_ap(retrieved_labels, q_label, k=10)
            results[rep_name]["map@10"] += ap

            # MRR@10
            rr = compute_rr(retrieved_labels, q_label, k=10)
            results[rep_name]["mrr@10"] += rr

            detailed_rows.append({
                "Query_Image": q_item["filename"],
                "Representation": rep_name,
                "Ground_Truth": q_label,
                "Top_1_Retrieved": retrieved_items[0]["filename"],
                "Top_1_Label": retrieved_labels[0],
                "P@1": p1,
                "P@5": p5,
                "P@10": p10,
                "AP@10": ap,
                "RR@10": rr
            })

        # Normalize metrics by query count
        valid_queries = len(queries_meta)
        results[rep_name]["p@1"] = (results[rep_name]["p@1"] / valid_queries) * 100.0
        results[rep_name]["p@5"] = (results[rep_name]["p@5"] / valid_queries) * 100.0
        results[rep_name]["p@10"] = (results[rep_name]["p@10"] / valid_queries) * 100.0
        results[rep_name]["map@10"] = (results[rep_name]["map@10"] / valid_queries) * 100.0
        results[rep_name]["mrr@10"] = (results[rep_name]["mrr@10"] / valid_queries) * 100.0

    # 5. Compile and Print Report
    report_lines = []
    report_lines.append("EUNIS Ecosystem Map Representation Semantic Retrieval Report")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"- Queries count: {len(queries_meta)} images")
    report_lines.append(f"- Database count: {len(database_meta)} images")
    report_lines.append("")

    print("\n" + "=" * 95)
    print("                    EUNIS ECOSYSTEM MAP SEMANTIC RETRIEVAL REPORT")
    print("=" * 95)

    row_format = "{:<24} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}"
    header = row_format.format("Representation", "P@1 (%)", "P@5 (%)", "P@10 (%)", "mAP@10 (%)", "MRR@10 (%)")
    print(header)
    print("-" * 90)

    report_lines.append(header)
    report_lines.append("-" * 90)

    for rep_name in results.keys():
        metrics = results[rep_name]
        row_str = row_format.format(
            rep_name,
            f"{metrics['p@1']:.1f}%",
            f"{metrics['p@5']:.1f}%",
            f"{metrics['p@10']:.1f}%",
            f"{metrics['map@10']:.1f}%",
            f"{metrics['mrr@10']:.1f}%"
        )
        print(row_str)
        report_lines.append(row_str)

    print("=" * 95)

    # Save TXT report summary
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"\nBenchmark report saved successfully to: {os.path.abspath(args.output_report)}")

    # Save detailed CSV results
    if detailed_rows:
        df_detailed = pd.DataFrame(detailed_rows)
        df_detailed.to_csv(args.output_csv, index=False)
        print(f"Detailed query results saved to: {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
