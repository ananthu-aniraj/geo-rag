import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import h3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoModel,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from src.evaluation.metrics import compute_ap, compute_precision_at_k, compute_rr
from src.models import tips_image_encoder as image_encoder
from src.models.vision_model_inference import (
    extract_benchmark_features_single_pass,
)

# Environmental Zones 2025 (version 2.0 / Metzger 2025) Value Mapping
from src.utils.io import download_image
from src.utils.spatial_overlays import (
    get_crs_transformer,
    get_environmental_zone_label as get_zone_label,
    lookup_raster_pixel,
)

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'
DISCARD_CLASSES = {2, 12, 20, 43, 80, 83, 102, 127}  # sky, person, car, sign, bus, truck, van, bike


def main():
    parser = argparse.ArgumentParser(
        description="Environmental Zones of Europe Representation Semantic Retrieval Benchmark.")
    parser.add_argument("--csv_path", type=str, default="./full_pipeline_output/geo_space_deduplicated.csv",
                        help="Path to CSV containing geolocated images.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV containing geolocated images (alias for --csv_path).")
    parser.add_argument("--countries_shp", type=str, default="shapefiles/ne_10m_admin_0_countries.shp",
                        help="Path to Natural Earth countries shapefile to filter for Europe.")
    parser.add_argument("--raster", type=str,
                        default="/user/aaniraj/home/Documents/Projects/data/environmental_zones/eea_r_3035_100_m_EnvZ-Metzger_2025_v1_r00.tif",
                        help="Path to the Environmental Zones GeoTIFF.")

    parser.add_argument("--num_queries", type=int, default=100, help="Number of query evaluations to run.")
    parser.add_argument("--num_database", type=int, default=500, help="Number of database images to search against.")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU batch size for feature extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--offline_dataset_dirs", type=str, nargs="*", default=None,
                        help="Base directories containing offline dataset images and CSV folders.")
    parser.add_argument("--tips_model_path", type=str, default=None,
                        help="Path to the official TIPSv2 model checkpoint (.npz). If None, uses Hugging Face 'google/tipsv2-b14'.")
    parser.add_argument("--tips_model_variant", type=str, default="B", choices=["S", "B", "L", "So400m", "g"],
                        help="Variant of the official TIPSv2 model.")
    parser.add_argument("--tips_low_res", action="store_true", help="Set image resolution to 224px instead of 448px.")
    parser.add_argument("--output_report", type=str, default="./benchmark_results/env_zones_report.txt",
                        help="Path to write the report summary.")
    parser.add_argument("--output_csv", type=str, default="./benchmark_results/env_zones_results.csv",
                        help="Path to write detailed query CSV results.")
    parser.add_argument("--query_platform", type=str, default=None,
                        help="Filter query images to only use this platform (e.g. 'flickr').")
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else args.csv_path

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Open the Environmental Zones map
    if not os.path.exists(args.raster):
        print(f"Error: Environmental Zones raster map '{args.raster}' not found.")
        sys.exit(1)

    print(f"Opening Environmental Zones raster: {args.raster}...")
    try:
        raster_dataset = rasterio.open(args.raster)
    except Exception as e:
        print(f"Error opening raster: {e}")
        sys.exit(1)

    # 2. Load CSV
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        sys.exit(1)

    if csv_path.endswith('.parquet'):
        print(f"Loading Parquet database from: {csv_path}...")
        schema = pq.read_schema(csv_path)
        cols = [c for c in ["Photo_ID", "Platform", "Latitude", "Longitude", "Image_URL", "continent", "License"] if c in schema.names]
        df = pq.read_table(csv_path, columns=cols).to_pandas()
    else:
        print(f"Loading CSV database from: {csv_path}...")
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
    else:
        print("Warning: 'Continent' column not found in database. Proceeding without European filtering.")

    if len(df) == 0:
        print("Error: No coordinates left inside Europe.")
        return

    # Coordinate transformer from WGS84 (EPSG:4326) to Raster CRS (EPSG:3035)
    print(f"Raster CRS: {raster_dataset.crs}. Setting up coordinate transformer...")
    transformer, has_axis_order = get_crs_transformer(raster_dataset.crs)

    matched_images = []
    print("Mapping coordinates to Environmental Zones...")
    platform_col = next((col for col in df.columns if col.lower() == "platform"), None)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Spatial Query"):
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])

            # Sample pixel value using unified lookup
            pixel_val = lookup_raster_pixel(lat, lon, raster_dataset, transformer, has_axis_order)

            # Skip nodata / empty values
            if pixel_val is None or pixel_val == raster_dataset.nodata or pixel_val <= 0:
                continue

            zone_class = get_zone_label(pixel_val)
            if zone_class == "Unknown":
                continue

            # Extract platform
            plat_val = str(row[platform_col]).strip().lower() if platform_col else ""
            if not plat_val:
                url_val = str(row[url_col]).lower() if url_col else ""
                if "flickr" in url_val:
                    plat_val = "flickr"
                elif "mapillary" in url_val:
                    plat_val = "mapillary"
                elif "openstreetcam" in url_val or "kartaview" in url_val:
                    plat_val = "kartaview"
                elif "inaturalist" in url_val:
                    plat_val = "inaturalist"
                else:
                    plat_val = "unknown"

            matched_images.append({
                "url": str(row[url_col]) if url_col else "",
                "photo_id": str(row[id_col]) if id_col else None,
                "lat": lat,
                "lon": lon,
                "env_zone": zone_class,
                "platform": plat_val
            })
        except Exception:
            continue

    raster_dataset.close()

    if not matched_images:
        print("Error: No images mapped successfully to Environmental Zones.")
        return

    print("Partitioning queries and database geographically to prevent spatial autocorrelation/sequence leakage...")
    
    # 1. Map each image to an H3 Resolution 4 parent block (~11,000 km2)
    for item in matched_images:
        try:
            h3_res4 = h3.latlng_to_cell(float(item["lat"]), float(item["lon"]), 4)
        except Exception:
            h3_res4 = "unknown"
        item["parent_block"] = h3_res4

    # 2. Perform Greedy Block Stratification to ensure full class coverage
    class_to_blocks = {}
    for item in matched_images:
        blk = item["parent_block"]
        if blk == "unknown":
            continue
        cls = item["env_zone"]
        if cls not in class_to_blocks:
            class_to_blocks[cls] = set()
        class_to_blocks[cls].add(blk)

    sorted_classes = sorted(class_to_blocks.keys(), key=lambda c: len(class_to_blocks[c]))
    query_blocks = set()
    database_blocks = set()
    random.seed(args.seed)

    for cls in sorted_classes:
        blocks = list(class_to_blocks[cls])
        random.shuffle(blocks)
        
        assigned_q = sum(1 for b in blocks if b in query_blocks)
        assigned_db = sum(1 for b in blocks if b in database_blocks)
        
        total_class_blocks = len(blocks)
        target_q = max(1, int(total_class_blocks * 0.20)) if total_class_blocks > 1 else 0
        
        unassigned_blocks = [b for b in blocks if b not in query_blocks and b not in database_blocks]
        
        needed_q = max(0, target_q - assigned_q)
        needed_db = max(0, 1 - assigned_db) if total_class_blocks > 1 else 0
        
        for b in unassigned_blocks:
            if needed_q > 0:
                query_blocks.add(b)
                needed_q -= 1
            elif needed_db > 0:
                database_blocks.add(b)
                needed_db -= 1
            else:
                if random.random() < 0.20:
                    query_blocks.add(b)
                else:
                    database_blocks.add(b)

    # Allocate any remaining unassigned blocks in the dataset
    all_blocks = set(item["parent_block"] for item in matched_images if item["parent_block"] != "unknown")
    unassigned_all = all_blocks - query_blocks - database_blocks
    for b in unassigned_all:
        if random.random() < 0.20:
            query_blocks.add(b)
        else:
            database_blocks.add(b)

    query_candidates = [item for item in matched_images if item["parent_block"] in query_blocks]
    database_candidates = [item for item in matched_images if item["parent_block"] in database_blocks]
    
    unique_blocks = sorted(list(all_blocks))
    
    print(f" -> Found {len(unique_blocks)} unique H3 blocks.")
    print(f" -> Query block pool: {len(query_blocks)} blocks ({len(query_candidates):,} images)")
    print(f" -> Database block pool: {len(database_blocks)} blocks ({len(database_candidates):,} images)")

    # Group candidates by Zone category
    query_by_class = {}
    for item in query_candidates:
        cls = item["env_zone"]
        if cls not in query_by_class:
            query_by_class[cls] = []
        query_by_class[cls].append(item)

    db_by_class = {}
    for item in database_candidates:
        cls = item["env_zone"]
        if cls not in db_by_class:
            db_by_class[cls] = []
        db_by_class[cls].append(item)

    sorted_classes = sorted(list(set(item["env_zone"] for item in matched_images)))
    for cls in sorted_classes:
        if cls in query_by_class:
            random.shuffle(query_by_class[cls])
        if cls in db_by_class:
            random.shuffle(db_by_class[cls])

    # Enforce platform-specific queries
    q_plat = args.query_platform.lower() if args.query_platform else None

    def matches_query_plat(item):
        if not q_plat:
            return True
        plat = item.get("platform", "").lower()
        if plat == q_plat:
            return True
        url = item.get("url", "").lower()
        if q_plat in url:
            return True
        return False

    # Interleave to build query pool (only matching query platform)
    query_pool = []
    class_q_lists = {cls: [item for item in query_by_class.get(cls, []) if matches_query_plat(item)] for cls in sorted_classes}
    max_q_len = max(len(lst) for lst in class_q_lists.values()) if class_q_lists else 0
    for i in range(max_q_len):
        for cls in sorted_classes:
            if i < len(class_q_lists[cls]):
                query_pool.append(class_q_lists[cls][i])

    # Interleave to build database pool (all images in database blocks)
    db_pool = []
    class_db_lists = {cls: db_by_class.get(cls, []) for cls in sorted_classes}
    max_db_len = max(len(lst) for lst in class_db_lists.values()) if class_db_lists else 0
    for i in range(max_db_len):
        for cls in sorted_classes:
            if i < len(class_db_lists[cls]):
                db_pool.append(class_db_lists[cls][i])

    # Select queries from the query pool
    if q_plat and len(query_pool) < args.num_queries:
        print(
            f"Warning: Only {len(query_pool)} query candidates available for platform '{q_plat}' in query blocks. Adjusting --num_queries.")
        args.num_queries = len(query_pool)

    queries_selection = query_pool[:args.num_queries]
    database_selection = db_pool[:args.num_database]

    print(
        f"Prereq selection sizes: {len(queries_selection)} Queries ({q_plat if q_plat else 'any'}), {len(database_selection)} Database.")

    # 1. Download query images in parallel
    print(f"Downloading {len(queries_selection)} query images for benchmarking in parallel...")
    queries_dict = {}
    image_size = 224 if args.tips_low_res else 448
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(
                download_image,
                item['url'],
                photo_id=item.get('photo_id'),
                platform=item.get('platform'),
                offline_dirs=args.offline_dataset_dirs,
                image_size=image_size
            ): idx
            for idx, item in enumerate(queries_selection)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Queries"):
            idx = futures[future]
            img = future.result()
            if img:
                queries_dict[idx] = img

    active_query_indices = sorted(list(queries_dict.keys()))
    queries_meta = [queries_selection[idx] for idx in active_query_indices]
    for idx, img in enumerate([queries_dict[k] for k in active_query_indices]):
        queries_meta[idx]['img'] = img

    print(f"Successfully downloaded {len(queries_meta)} query images.")
    if len(queries_meta) == 0:
        print("Error: No query images successfully downloaded.")
        return

    # Keep database metadata selection without downloading yet (streamed in chunks later)
    database_meta = database_selection
    print(f"Final evaluation split: {len(queries_meta)} Queries, {len(database_meta)} Database (processing DB in chunks).")

    # 3. Initialize Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing models on {device}...")

    # Load TIPSv2 Model (either official checkpoint or Hugging Face)
    if args.tips_model_path:
        print(f"Loading official TIPSv2 checkpoint from: {args.tips_model_path}...")
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
            "TIPSv2 1st CLS + Average Patch": {"query": [], "db": []},
            "TIPSv2 2nd CLS + Average Patch": {"query": [], "db": []},
            "TIPSv2 Seg-Masked": {"query": [], "db": []}
        }
    else:
        representations = {
            "TIPSv2 CLS": {"query": [], "db": []},
            "TIPSv2 Average Patch": {"query": [], "db": []},
            "TIPSv2 CLS + Average Patch": {"query": [], "db": []},
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
                cls_out, patch_tokens_vals = extract_benchmark_features_single_pass(tipsv2, img_tensors,
                                                                                    is_local=is_local)
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

                # Concatenated CLS + Average Patch
                if is_local:
                    first_cls_token = first_cls[idx]
                    second_cls_token = second_cls[idx]
                    representations["TIPSv2 1st CLS + Average Patch"][split_key].append(np.concatenate([first_cls_token, avg_patch]))
                    representations["TIPSv2 2nd CLS + Average Patch"][split_key].append(np.concatenate([second_cls_token, avg_patch]))
                else:
                    cls_token = cls_tokens[idx]
                    representations["TIPSv2 CLS + Average Patch"][split_key].append(np.concatenate([cls_token, avg_patch]))

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

    # Extract query features and free RAM immediately
    extract_features_batch(queries_meta, "query")
    for item in queries_meta:
        if 'img' in item:
            if hasattr(item['img'], 'close'):
                item['img'].close()
            item.pop('img', None)

    # 4.5 Download and extract database features in chunks to protect RAM
    db_chunk_size = 1000
    downloaded_database_meta = []
    print(f"Processing database of {len(database_meta)} images in chunks of {db_chunk_size}...")
    for chunk_start in range(0, len(database_meta), db_chunk_size):
        chunk_meta = database_meta[chunk_start : chunk_start + db_chunk_size]
        print(f"\n--- Processing database chunk {chunk_start // db_chunk_size + 1} ({chunk_start} to {chunk_start + len(chunk_meta)}) ---")

        db_dict = {}
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(
                    download_image,
                    item['url'],
                    photo_id=item.get('photo_id'),
                    platform=item.get('platform'),
                    offline_dirs=args.offline_dataset_dirs,
                    image_size=image_size
                ): idx
                for idx, item in enumerate(chunk_meta)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading DB Chunk"):
                idx = futures[future]
                img = future.result()
                if img:
                    db_dict[idx] = img

        active_db_indices = sorted(list(db_dict.keys()))
        chunk_downloaded = [chunk_meta[idx] for idx in active_db_indices]
        for idx, img in enumerate([db_dict[k] for k in active_db_indices]):
            chunk_downloaded[idx]['img'] = img

        if len(chunk_downloaded) > 0:
            extract_features_batch(chunk_downloaded, "db")
            downloaded_database_meta.extend(chunk_downloaded)

            # Immediately free RAM for this chunk's images
            for item in chunk_downloaded:
                if 'img' in item:
                    if hasattr(item['img'], 'close'):
                        item['img'].close()
                    item.pop('img', None)

    # Update database_meta to match ONLY the successfully downloaded images
    database_meta = downloaded_database_meta
    print(f"Successfully processed features for {len(database_meta)} database images.")

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
            q_label = q_item["env_zone"]
            sorted_db_indices = top_indices[q_idx]

            retrieved_items = [database_meta[idx] for idx in sorted_db_indices[:10]]
            retrieved_labels = [item["env_zone"] for item in retrieved_items]

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

            q_id = f"{q_item.get('platform', 'unknown')}_{q_item.get('photo_id', '')}" if q_item.get('photo_id') else q_item["url"]
            r_id = f"{retrieved_items[0].get('platform', 'unknown')}_{retrieved_items[0].get('photo_id', '')}" if retrieved_items[0].get('photo_id') else retrieved_items[0]["url"]

            detailed_rows.append({
                "Query_Image": q_id,
                "Representation": rep_name,
                "Ground_Truth": q_label,
                "Top_1_Retrieved": r_id,
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
    report_lines.append("Environmental Zones of Europe Image Representation Semantic Retrieval Report")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"- Queries count: {len(queries_meta)} images")
    report_lines.append(f"- Database count: {len(database_meta)} images")
    report_lines.append("")

    print("\n" + "=" * 95)
    print("                    ENVIRONMENTAL ZONES SEMANTIC RETRIEVAL REPORT")
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
