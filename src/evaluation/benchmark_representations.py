import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import faiss
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from requests.adapters import HTTPAdapter
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoModel,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from urllib3.util import Retry

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'
DISCARD_CLASSES = {2, 12, 20, 43, 80, 83, 102, 127}  # sky, person, car, sign, bus, truck, van, bike
use_gpu = torch.cuda.is_available()

# Global connection pooled session for thread-safe downloads
http_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=64,
    pool_maxsize=64,
    max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
)
http_session.mount("https://", _adapter)
http_session.mount("http://", _adapter)


def download_image(url, photo_id=None):
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
            img_resized = img.resize((448, 448))
            img.close()
            return img_resized
    except Exception:
        pass
    return None


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


def snap_mask_with_pca(seg_keep, pca_rgb, k_segments=4, use_gpu=True):
    """Refines/snaps a segment keep-mask to high-resolution visual boundaries from the upscaled PCA map."""
    h, w, c = pca_rgb.shape
    pixels = pca_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    
    # Use high-speed FAISS K-Means (GPU/CPU) if available
    d = 3
    kmeans = faiss.Kmeans(d, k_segments, niter=10, verbose=False, gpu=use_gpu, seed=42)
    kmeans.train(pixels)
    _, labels_flat = kmeans.index.search(pixels, 1)
    labels = labels_flat.ravel().reshape(h, w)
    discard_mask = 1.0 - seg_keep
    snapped_keep = np.ones_like(seg_keep)
    
    for r in range(k_segments):
        segment_mask = (labels == r)
        segment_size = np.sum(segment_mask)
        if segment_size == 0:
            continue
        overlap = np.sum(segment_mask & (discard_mask == 1.0)) / segment_size
        if overlap >= 0.65:
            snapped_keep[segment_mask] = 0.0
            
    return snapped_keep


def haversine_distance(lat1, lon1, lat2, lon2):
    """Computes distance between coordinates in kilometers."""
    deg_to_rad = np.pi / 180.0
    phi1 = lat1 * deg_to_rad
    phi2 = lat2 * deg_to_rad
    dphi = (lat2 - lat1) * deg_to_rad
    dlambda = (lon2 - lon1) * deg_to_rad

    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371.0 * c


def main():
    parser = argparse.ArgumentParser(description="Representation & Layout Retrieval Benchmarking Suite.")
    parser.add_argument("--csv_path", type=str, default="./full_pipeline_output/geo_space_deduplicated.csv", help="Path to database CSV.")
    parser.add_argument("--num_images", type=int, default=300, help="Total database images to download and index.")
    parser.add_argument("--num_queries", type=int, default=50, help="Number of query evaluations to run.")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU batch size for feature extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading metadata from {args.csv_path}...")
    if not os.path.exists(args.csv_path):
        # Look in other potential directory output locations
        alt_paths = ["./full_pipeline_output/geo_space_cleaned.csv", "full_pipeline_output/geo_space_cleaned.csv"]
        for path in alt_paths:
            if os.path.exists(path):
                args.csv_path = path
                break
        else:
            print("Error: Database CSV not found. Please run spatial deduplication first.")
            return

    df = pd.read_csv(args.csv_path, dtype=str)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
    print(f"Loaded {len(df)} database records.")

    # Sample a balanced subset of Flickr and Mapillary images
    flickr_df = df[df['Platform'].str.lower() == 'flickr']
    mapillary_df = df[df['Platform'].str.lower() == 'mapillary']
    
    half_size = args.num_images // 2
    if len(flickr_df) < half_size or len(mapillary_df) < half_size:
        print("Warning: Insufficient platform records for balanced sampling. Using random sampling.")
        sampled_df = df.sample(min(args.num_images, len(df)), random_state=args.seed).reset_index(drop=True)
    else:
        sampled_df = pd.concat([
            flickr_df.sample(half_size, random_state=args.seed),
            mapillary_df.sample(half_size, random_state=args.seed)
        ]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"Downloading {len(sampled_df)} images for benchmarking in parallel...")
    images = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(download_image, row['Image_URL'], row.get('Photo_ID')): idx
            for idx, row in sampled_df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            idx = futures[future]
            img = future.result()
            if img:
                images[idx] = img

    print(f"Successfully downloaded {len(images)} images.")
    if len(images) < 10:
        print("Error: Too few images successfully downloaded to run benchmark.")
        return

    # Keep only downloaded items
    active_indices = sorted(list(images.keys()))
    sampled_df = sampled_df.iloc[active_indices].reset_index(drop=True)
    # Re-map images dictionary to new indices (0 to len-1)
    images = {i: images[idx] for i, idx in enumerate(active_indices)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing vision models on {device}...")
    seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").eval().to(device)
    tipsv2 = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).eval().to(device)

    print("Loading AnyUp model for feature upsampling...")
    try:
        from anyup import anyup_multi_backbone
        anyup_model = anyup_multi_backbone(use_natten=False, pretrained=True, device=device).eval()
        print(" -> AnyUp loaded successfully.")
    except Exception as e:
        print(f" -> [WARNING] Failed to load AnyUp model locally: {e}. Trying torch.hub loader...")
        try:
            anyup_model = torch.hub.load('wimmerth/anyup', 'anyup_multi_backbone', use_natten=False, pretrained=True, device=device).eval()
            print(" -> AnyUp loaded successfully via torch.hub.")
        except Exception as e_hub:
            print(f" -> [WARNING] Torch Hub AnyUp load failed: {e_hub}. Falling back to standard resizing.")
            anyup_model = None

    # Setup 150 ADE20K text embeddings for TIPSv2 zero-shot mapping
    ade_labels = [seg_model.config.id2label[i] for i in range(150)]
    
    with torch.no_grad():
        ade_text_embeds = tipsv2.encode_text(ade_labels)
        ade_text_embeds_np = ade_text_embeds.cpu().numpy()
        ade_text_embeds_norm = ade_text_embeds_np / (np.linalg.norm(ade_text_embeds_np, axis=1, keepdims=True) + 1e-9)

    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    # Dictionaries to hold our feature extractions
    cls_embeds = []
    unmasked_avg_embeds = []
    seg_masked_embeds = []
    tips_ade_embeds = []
    anyup_snapped_embeds = []
    tips_ade_keep_masks_list = [] # Save for masked PCA histogram generation
    raw_patch_tokens_list = [] # Used for Global PCA fitting
    seg_viz_samples = [] # Hold first 3 samples for segmentation mask visualization

    print("Extracting spatial features and pre-computing keep-masks in batches...")
    batch_size = args.batch_size
    for i in tqdm(range(0, len(images), batch_size), desc="Feature Extraction"):
        batch_keys = list(range(i, min(i + batch_size, len(images))))
        batch_imgs = [images[k] for k in batch_keys]

        # 1. Segformer keep masks
        inputs = seg_processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = seg_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(448, 448), mode="bilinear", align_corners=False)
        pred_masks = logits.argmax(dim=1).cpu().numpy()

        # 2. TIPSv2 feature extractions
        img_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            out = tipsv2.encode_image(img_tensors)
            cls_tokens = out.cls_token.cpu().numpy()
            if cls_tokens.ndim == 3:
                cls_tokens = cls_tokens.squeeze(1)
            
            patch_tokens_vals = encode_image_value_attention(tipsv2.vision_encoder, img_tensors)
            patch_tokens_vals = patch_tokens_vals.reshape(len(batch_keys), 1024, -1).cpu().numpy()

        # Process each image in batch
        for batch_i in range(len(batch_keys)):
            pred_mask = pred_masks[batch_i]
            cls_token = cls_tokens[batch_i]
            patch_tokens = patch_tokens_vals[batch_i]
            raw_patch_tokens_list.append(patch_tokens)

            # Segformer mask
            keep_mask = np.ones_like(pred_mask, dtype=float)
            for c in DISCARD_CLASSES:
                keep_mask[pred_mask == c] = 0.0
            
            # Map keep mask down to 32x32 patch resolution
            patch_weights = np.zeros((32, 32))
            for r in range(32):
                for c in range(32):
                    patch_weights[r, c] = np.mean(keep_mask[r*14:(r+1)*14, c*14:(c+1)*14])
            patch_weights_flat = patch_weights.flatten()[:, np.newaxis]

            # Zero-shot 150 ADE20K segmentation for TIPSv2
            norm_patches = patch_tokens / (np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-9)
            patch_ade_sim = np.dot(norm_patches, ade_text_embeds_norm.T)
            best_ade_idx = np.argmax(patch_ade_sim, axis=1) # 0 to 149
            
            tips_ade_keep_mask = np.ones(1024, dtype=float)
            for c in DISCARD_CLASSES:
                tips_ade_keep_mask[best_ade_idx == c] = 0.0
            tips_ade_keep_mask_flat = tips_ade_keep_mask[:, np.newaxis]

            # Averages
            simple_avg = np.mean(patch_tokens, axis=0)
            
            total_seg_weight = np.sum(patch_weights_flat)
            seg_avg = np.sum(patch_tokens * patch_weights_flat, axis=0) / total_seg_weight if total_seg_weight > 0 else simple_avg

            total_tips_ade_weight = np.sum(tips_ade_keep_mask_flat)
            tips_ade_avg = np.sum(patch_tokens * tips_ade_keep_mask_flat, axis=0) / total_tips_ade_weight if total_tips_ade_weight > 0 else simple_avg

            # Normalizations
            cls_norm = cls_token / (np.linalg.norm(cls_token) + 1e-9)
            simple_norm = simple_avg / (np.linalg.norm(simple_avg) + 1e-9)
            seg_norm = seg_avg / (np.linalg.norm(seg_avg) + 1e-9)
            tips_ade_norm = tips_ade_avg / (np.linalg.norm(tips_ade_avg) + 1e-9)

            # Store representations
            cls_embeds.append(cls_norm)
            unmasked_avg_embeds.append(simple_norm)
            seg_masked_embeds.append(seg_norm)
            tips_ade_embeds.append(tips_ade_norm)
            tips_ade_keep_masks_list.append(tips_ade_keep_mask)

            # Compute GPU local PCA (q=3) of patches for visual inspection and AnyUp mask snapping
            with torch.no_grad():
                patches_t = torch.tensor(patch_tokens, device=device)
                patches_centered = patches_t - patches_t.mean(dim=0)
                _, _, V_local = torch.pca_lowrank(patches_centered, q=3, center=False)
                proj_local = torch.matmul(patches_centered, V_local) # [1024, 3]
                p_min = proj_local.min(dim=0, keepdim=True).values
                p_max = proj_local.max(dim=0, keepdim=True).values
                p_norm = (proj_local - p_min) / (p_max - p_min + 1e-9)
                
                # Upscale using AnyUp (Option B) if available
                pca_3d = p_norm.reshape(1, 32, 32, 3).permute(0, 3, 1, 2)
                if anyup_model is not None:
                    try:
                        # img_tensors is shape [B, 3, 448, 448] (range [0,1])
                        hr_img_t = img_tensors[batch_i:batch_i+1]
                        # Normalize to ImageNet mean & std for AnyUp vision blocks
                        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                        hr_img_norm = (hr_img_t - mean) / std
                        
                        hr_pca_t = anyup_model(hr_img_norm, pca_3d).squeeze(0).permute(1, 2, 0) # [448, 448, 3]
                        h_min = hr_pca_t.min()
                        h_max = hr_pca_t.max()
                        hr_pca_t = (hr_pca_t - h_min) / (h_max - h_min + 1e-9)
                        pca_rgb = (hr_pca_t.cpu().numpy() * 255).astype(np.uint8)
                        
                        # Generate snapped mask
                        snapped_keep = snap_mask_with_pca(keep_mask, pca_rgb, k_segments=4)
                    except Exception as e:
                        pca_rgb = (p_norm.cpu().numpy() * 255).astype(np.uint8).reshape(32, 32, 3)
                        snapped_keep = keep_mask
                else:
                    pca_rgb = (p_norm.cpu().numpy() * 255).astype(np.uint8).reshape(32, 32, 3)
                    snapped_keep = keep_mask

            # Downscale snapped_keep (448x448) to 32x32 to mask the 32x32 patch tokens
            if pca_rgb.shape[:2] == (448, 448):
                try:
                    snapped_pil = Image.fromarray((snapped_keep * 255).astype(np.uint8)).resize((32, 32), resample=Image.BILINEAR)
                    snapped_32x32 = (np.array(snapped_pil) > 127).astype(float)
                except Exception:
                    snapped_32x32 = keep_mask.reshape(32, 32)
            else:
                snapped_32x32 = keep_mask.reshape(32, 32)

            snapped_keep_flat = snapped_32x32.reshape(-1, 1)
            total_snapped_weight = np.sum(snapped_keep_flat)
            snapped_avg = np.sum(patch_tokens * snapped_keep_flat, axis=0) / total_snapped_weight if total_snapped_weight > 0 else simple_avg
            snapped_norm = snapped_avg / (np.linalg.norm(snapped_avg) + 1e-9)
            anyup_snapped_embeds.append(snapped_norm)

            # Store first 3 processed images for segmentation mask and local PCA visualization
            if len(seg_viz_samples) < 3:
                seg_viz_samples.append({
                    "img": images[batch_keys[batch_i]],
                    "seg_keep": keep_mask,
                    "tips_ade_keep": tips_ade_keep_mask.reshape(32, 32),
                    "pca_rgb": pca_rgb,
                    "snapped_keep": snapped_keep
                })

    # Convert to NumPy matrices for rapid distance computations
    CLS_MAT = np.vstack(cls_embeds)
    UNMASKED_MAT = np.vstack(unmasked_avg_embeds)
    SEG_MASK_MAT = np.vstack(seg_masked_embeds)
    TIPS_ADE_MAT = np.vstack(tips_ade_embeds)
    ANYUP_SNAPPED_MAT = np.vstack(anyup_snapped_embeds)

    # 3. Fit Global PCA dynamically to extract aligned layout signatures on the GPU
    print("\nFitting Global PCA model on patch token sample using GPU...")
    # Convert all raw patches to a single tensor on GPU/device
    all_patches_tensor = torch.tensor(np.stack(raw_patch_tokens_list), device=device) # Shape [N, 1024, 768]
    N_imgs, P_cnt, D_dim = all_patches_tensor.shape
    all_patches_flat = all_patches_tensor.reshape(-1, D_dim) # Shape [N * 1024, 768]

    # Sample patches for PCA fitting
    sample_indices = torch.randperm(all_patches_flat.shape[0], device=device)[:min(50000, all_patches_flat.shape[0])]
    sampled_patches = all_patches_flat[sample_indices]

    # Perform low-rank SVD / PCA on GPU
    # torch.pca_lowrank center=True automatically centers each column
    with torch.no_grad():
        mean_vector = sampled_patches.mean(dim=0)
        # We project using V from PCA lowrank
        U, S, V = torch.pca_lowrank(sampled_patches, q=16, center=True, niter=3)
        # V has shape [768, 16] (principal axes/loadings)
    print(" -> Global PCA fitted on GPU (16 components).")

    # Generate layout signature histograms on GPU
    print("Generating Global PCA spatial layout histograms (masked only) on GPU...")
    masked_pca_histograms = []
    with torch.no_grad():
        for idx_img, patch_tokens in enumerate(all_patches_tensor): # shape [1024, 768]
            # Center and project patches
            centered = patch_tokens - mean_vector
            projected = torch.matmul(centered, V) # shape [1024, 16]
            dominant_components = torch.argmax(projected, dim=1) # shape [1024]
            
            # Masked PCA histogram (ignoring discarded patches)
            keep_mask_np = tips_ade_keep_masks_list[idx_img] # shape [1024]
            keep_mask_t = torch.tensor(keep_mask_np, device=device)
            masked_dominant = dominant_components[keep_mask_t == 1.0]
            if len(masked_dominant) > 0:
                m_hist = torch.bincount(masked_dominant, minlength=16).float()
                hist_masked = m_hist / m_hist.sum()
            else:
                hist_masked = torch.zeros(16, device=device)
            masked_pca_histograms.append(hist_masked.cpu().numpy())
            
    MASKED_PCA_HIST_MAT = np.vstack(masked_pca_histograms) # Shape [N, 16]

    # 4. Concatenated representations (L2 normalized for cosine similarity matching)
    CLS_TIPS_ADE_CONCAT = np.concatenate([CLS_MAT, TIPS_ADE_MAT], axis=1)
    CLS_TIPS_ADE_CONCAT = CLS_TIPS_ADE_CONCAT / (np.linalg.norm(CLS_TIPS_ADE_CONCAT, axis=1, keepdims=True) + 1e-9)

    CLS_UNMASKED_CONCAT = np.concatenate([CLS_MAT, UNMASKED_MAT], axis=1)
    CLS_UNMASKED_CONCAT = CLS_UNMASKED_CONCAT / (np.linalg.norm(CLS_UNMASKED_CONCAT, axis=1, keepdims=True) + 1e-9)

    # 5. Hybrid Land Use representation (Textured Scenery + Layout Composition)
    HYBRID_LAND_USE = np.concatenate([TIPS_ADE_MAT, MASKED_PCA_HIST_MAT], axis=1)

    # --- Run Benchmarking Suite ---
    print(f"\nRunning comparative benchmarks over {args.num_queries} queries...")
    query_indices = np.random.choice(len(sampled_df), min(args.num_queries, len(sampled_df)), replace=False)
    
    # Structure to hold metrics
    representations = {
        "Global CLS": CLS_MAT,
        "Unmasked Patch Average": UNMASKED_MAT,
        "Segformer-Masked Average": SEG_MASK_MAT,
        "TIPSv2 ADE20K-Masked Average": TIPS_ADE_MAT,
        "AnyUp-PCA Snapped Mask Average": ANYUP_SNAPPED_MAT,
        "CLS + TIPSv2 ADE20K-Masked (Concat)": CLS_TIPS_ADE_CONCAT,
        "CLS + Unmasked Average (Concat)": CLS_UNMASKED_CONCAT,
        "Hybrid Land Use Signature": HYBRID_LAND_USE
    }

    results = {rep: {"distances": [], "top_10_distances": [], "r_1_5km": 0, "r_5_5km": 0, "r_1_50km": 0, "r_5_50km": 0, "country_match": 0, "cross_platform_success": 0, "cross_platform_total": 0} for rep in representations}

    for q_idx in query_indices:
        q_row = sampled_df.iloc[q_idx]
        q_lat, q_lon = q_row['Latitude'], q_row['Longitude']
        q_plat = str(q_row['Platform']).lower()
        q_country = q_row.get('country', 'Unknown')

        # Create database search pool (exclude the query image itself to prevent trivial top-1 matching)
        db_mask = np.ones(len(sampled_df), dtype=bool)
        db_mask[q_idx] = False

        db_df = sampled_df[db_mask]

        # Calculate haversine distances to all database items
        db_distances = haversine_distance(q_lat, q_lon, db_df['Latitude'].values, db_df['Longitude'].values)

        for rep_name, representation_matrix in representations.items():
            # Get representation vectors
            q_vector = representation_matrix[q_idx]
            db_vectors = representation_matrix[db_mask]

            if "Hybrid" in rep_name:
                # Custom blended distance metric for hybrid signatures
                q_sem = q_vector[:768]
                q_hist = q_vector[768:]
                
                db_sem = db_vectors[:, :768]
                db_hist = db_vectors[:, 768:]
                
                # Cosine distance for semantic features
                cos_sim = np.dot(db_sem, q_sem)
                d_sem = 1.0 - cos_sim
                
                # L1 distance for composition histograms (max possible L1 distance is 2.0)
                d_hist = np.sum(np.abs(db_hist - q_hist), axis=1) / 2.0
                
                # Blend: 70% semantic weight, 30% structural weight
                blended_dist = 0.7 * d_sem + 0.3 * d_hist
                sorted_indices = np.argsort(blended_dist)
            else:
                # Use Cosine Similarity for semantic embeddings
                similarities = np.dot(db_vectors, q_vector) # vectors are already normalized
                # Sort in descending order (larger similarity = more similar)
                sorted_indices = np.argsort(similarities)[::-1]

            # Get top retrieved database items
            top_1_dist = db_distances[sorted_indices[0]]
            top_5_dists = db_distances[sorted_indices[:5]]
            top_10_dists = db_distances[sorted_indices[:10]]
            top_1_row = db_df.iloc[sorted_indices[0]]

            results[rep_name]["distances"].append(top_1_dist)
            results[rep_name]["top_10_distances"].append(top_10_dists)

            # Recall metrics
            if top_1_dist <= 5.0:
                results[rep_name]["r_1_5km"] += 1
            if np.any(top_5_dists <= 5.0):
                results[rep_name]["r_5_5km"] += 1

            if top_1_dist <= 50.0:
                results[rep_name]["r_1_50km"] += 1
            if np.any(top_5_dists <= 50.0):
                results[rep_name]["r_5_50km"] += 1

            # Country Metadata Overlap
            if q_country != 'Unknown' and top_1_row.get('country', 'Unknown') == q_country:
                results[rep_name]["country_match"] += 1

            # Cross-Platform Retrieval Success
            # Count how many queries have a cross-platform neighbor within 50km in the database pool
            has_cross_platform_nearby = False
            cross_db_indices = np.where(db_df['Platform'].str.lower() != q_plat)[0]
            if len(cross_db_indices) > 0:
                cross_db_dists = db_distances[cross_db_indices]
                if np.any(cross_db_dists <= 50.0):
                    has_cross_platform_nearby = True

            if has_cross_platform_nearby:
                results[rep_name]["cross_platform_total"] += 1
                # Check if the top-1 retrieved cross-platform image is within 50km
                # (filter retrieved list for items from the opposite platform)
                sorted_db_plats = db_df['Platform'].iloc[sorted_indices].str.lower().values
                sorted_db_dists = db_distances[sorted_indices]
                
                first_cross_idx = np.where(sorted_db_plats != q_plat)[0][0]
                if sorted_db_dists[first_cross_idx] <= 50.0:
                    results[rep_name]["cross_platform_success"] += 1

    # --- Print Benchmark Report ---
    print("\n" + "="*80)
    print("                   GEOGRAPHIC RETRIEVAL BENCHMARK REPORT")
    print("="*80)
    print(f"Database Size: {len(sampled_df)} images (50% Flickr / 50% Mapillary)")
    print(f"Evaluation Queries: {len(query_indices)} diverse samples")
    print("-"*80)

    # Format output table
    row_format = "{:<32} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}"
    print(row_format.format("Representation", "Median Err", "R@1 (5km)", "R@5 (5km)", "R@1 (50km)", "Cross-Plat"))
    print(row_format.format("", "(km)", "(%)", "(%)", "(%)", "Recall (%)"))
    print("-" * 80)

    report_rows = []
    for rep_name, metrics in results.items():
        median_err = np.median(metrics["distances"])
        r_1_5 = (metrics["r_1_5km"] / len(query_indices)) * 100
        r_5_5 = (metrics["r_5_5km"] / len(query_indices)) * 100
        r_1_50 = (metrics["r_1_50km"] / len(query_indices)) * 100
        
        cross_plat_recall = 0.0
        if metrics["cross_platform_total"] > 0:
            cross_plat_recall = (metrics["cross_platform_success"] / metrics["cross_platform_total"]) * 100

        print(row_format.format(
            rep_name,
            f"{median_err:.2f}",
            f"{r_1_5:.1f}%",
            f"{r_5_5:.1f}%",
            f"{r_1_50:.1f}%",
            f"{cross_plat_recall:.1f}%" if metrics["cross_platform_total"] > 0 else "N/A"
        ))
        
        report_rows.append({
            "name": rep_name,
            "median_err": median_err,
            "r_1_5": r_1_5,
            "r_5_5": r_5_5,
            "r_1_50": r_1_50,
            "cross_plat": cross_plat_recall
        })

    print("="*80)

    # Delete the old markdown report if it exists
    artifact_path = "/user/aaniraj/home/.gemini/antigravity-cli/brain/842fbd6a-4490-43d7-9738-f5007612ce34/benchmark_report.md"
    if os.path.exists(artifact_path):
        try:
            os.remove(artifact_path)
            print(f"\nRemoved old markdown report: {artifact_path}")
        except Exception:
            pass

    # --- Generate Visualization Plots ---
    output_dir = os.path.dirname(args.csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    import matplotlib.pyplot as plt
    
    # Plot 1: Recall Curve and Error CDF
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left Plot: Recall Curve (K=1 to 10)
    for rep_name, metrics in results.items():
        k_values = list(range(1, 11))
        recalls = []
        top_10_dist_array = np.vstack(metrics["top_10_distances"]) # Shape [num_queries, 10]
        for k in k_values:
            success = np.any(top_10_dist_array[:, :k] <= 50.0, axis=1) # using 50km threshold
            recalls.append(np.mean(success) * 100)
        axes[0].plot(k_values, recalls, marker='o', label=rep_name)
    axes[0].set_title("Recall@K Curve (within 50 km)")
    axes[0].set_xlabel("K (Number of retrieved images)")
    axes[0].set_ylabel("Recall (%)")
    axes[0].set_xticks(range(1, 11))
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=8)

    # Right Plot: Error Cumulative Distribution Function (CDF)
    for rep_name, metrics in results.items():
        sorted_errors = np.sort(metrics["distances"])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        axes[1].plot(sorted_errors, cdf, label=rep_name)
    axes[1].set_title("Error Cumulative Distribution Function (CDF)")
    axes[1].set_xlabel("Top-1 Geodesic Error (km)")
    axes[1].set_ylabel("Queries Resolved (%)")
    axes[1].set_xlim(0, 100) # focus on errors within 100km
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "benchmark_metrics_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved metrics comparison plot to: {plot_path}")

    # Plot 2: Qualitative Query Sample Comparison Grid
    if len(query_indices) > 0:
        q_idx = query_indices[0]
        q_row = sampled_df.iloc[q_idx]
        q_img = images[q_idx]
        
        reps_to_visualize = list(representations.keys())
        
        fig2, axes2 = plt.subplots(len(reps_to_visualize), 4, figsize=(16, 3 * len(reps_to_visualize)))
        db_mask = np.ones(len(sampled_df), dtype=bool)
        db_mask[q_idx] = False
        db_df = sampled_df[db_mask]
        db_distances = haversine_distance(q_row['Latitude'], q_row['Longitude'], db_df['Latitude'].values, db_df['Longitude'].values)
        
        for row_i, rep_name in enumerate(reps_to_visualize):
            if rep_name not in representations:
                continue
            representation_matrix = representations[rep_name]
            q_vector = representation_matrix[q_idx]
            db_vectors = representation_matrix[db_mask]
            
            if "PCA" in rep_name:
                distances = np.sum(np.abs(db_vectors - q_vector), axis=1)
                sorted_indices = np.argsort(distances)
            else:
                similarities = np.dot(db_vectors, q_vector)
                sorted_indices = np.argsort(similarities)[::-1]
                
            # Column 0: Query image
            axes2[row_i, 0].imshow(q_img)
            axes2[row_i, 0].set_title(f"Query ({q_row['Platform']})", fontsize=10)
            axes2[row_i, 0].axis("off")
            
            # Columns 1, 2, 3: Top-3 retrieved images
            for col_j in range(1, 4):
                retrieved_db_idx = sorted_indices[col_j - 1]
                retrieved_img_idx = db_df.index[retrieved_db_idx]
                retrieved_img = images[retrieved_img_idx]
                retrieved_row = db_df.iloc[retrieved_db_idx]
                dist_err = db_distances[retrieved_db_idx]
                
                # Format a readable, wrapped title name for subplots
                clean_title = rep_name.replace(" (L1 Dist)", "").replace(" (Concat)", "\n(Concat)")
                
                axes2[row_i, col_j].imshow(retrieved_img)
                axes2[row_i, col_j].set_title(
                    f"{clean_title} Top-{col_j}\n({retrieved_row['Platform']}) Err: {dist_err:.1f}km",
                    fontsize=8
                )
                axes2[row_i, col_j].axis("off")
                
        plt.tight_layout()
        grid_plot_path = os.path.join(output_dir, "benchmark_retrieval_examples.png")
        plt.savefig(grid_plot_path, dpi=150)
        plt.close()
        print(f"Saved qualitative retrieval example grid to: {grid_plot_path}")

    # Plot 3: Segmentation Mask Diagnostics
    if len(seg_viz_samples) > 0:
        fig3, axes3 = plt.subplots(len(seg_viz_samples), 5, figsize=(20, 4 * len(seg_viz_samples)))
        if len(seg_viz_samples) == 1:
            axes3 = np.expand_dims(axes3, axis=0) # ensure 2D array shape
            
        for idx_s, sample in enumerate(seg_viz_samples):
            # Column 0: Original image
            axes3[idx_s, 0].imshow(sample["img"])
            axes3[idx_s, 0].set_title(f"Image {idx_s + 1}", fontsize=10)
            axes3[idx_s, 0].axis("off")
            
            # Column 1: Segformer Keep Mask
            seg_keep_rgb = np.zeros((448, 448, 3), dtype=np.uint8)
            seg_keep_rgb[sample["seg_keep"] == 1.0] = [34, 139, 34]    # Keep: Forest Green
            seg_keep_rgb[sample["seg_keep"] == 0.0] = [178, 34, 34]   # Discard: Firebrick Red
            axes3[idx_s, 1].imshow(seg_keep_rgb)
            axes3[idx_s, 1].set_title("Segformer ADE150 Mask\n(Green=Keep, Red=Discard)", fontsize=9)
            axes3[idx_s, 1].axis("off")
            
            # Column 2: TIPSv2 ADE20K-Masked Keep Mask
            ade_keep_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            ade_keep_rgb[sample["tips_ade_keep"] == 1.0] = [34, 139, 34]    # Keep: Green
            ade_keep_rgb[sample["tips_ade_keep"] == 0.0] = [178, 34, 34]   # Discard: Red
            # Upsample for better visual layout
            ade_keep_upsampled = Image.fromarray(ade_keep_rgb).resize((448, 448), resample=Image.NEAREST)
            axes3[idx_s, 2].imshow(ade_keep_upsampled)
            axes3[idx_s, 2].set_title("TIPSv2 ADE150 Zero-Shot Mask\n(Green=Keep, Red=Discard)", fontsize=9)
            axes3[idx_s, 2].axis("off")
            
            # Column 3: TIPSv2 Unsupervised Local PCA Component Map
            pca_rgb_np = sample["pca_rgb"]
            if pca_rgb_np.shape[:2] == (448, 448):
                axes3[idx_s, 3].imshow(pca_rgb_np)
                axes3[idx_s, 3].set_title("TIPSv2 + AnyUp Local PCA\n(PC1/PC2/PC3 mapped to R/G/B)", fontsize=9)
            else:
                pca_upsampled = Image.fromarray(pca_rgb_np).resize((448, 448), resample=Image.NEAREST)
                axes3[idx_s, 3].imshow(pca_upsampled)
                axes3[idx_s, 3].set_title("TIPSv2 Local PCA Projection\n(PC1/PC2/PC3 mapped to R/G/B)", fontsize=9)
            axes3[idx_s, 3].axis("off")
            
            # Column 4: AnyUp-PCA Snapped Mask (Refined keep-mask)
            snapped_keep_rgb = np.zeros((448, 448, 3), dtype=np.uint8)
            snapped_keep_rgb[sample["snapped_keep"] == 1.0] = [34, 139, 34]    # Keep: Forest Green
            snapped_keep_rgb[sample["snapped_keep"] == 0.0] = [178, 34, 34]   # Discard: Firebrick Red
            axes3[idx_s, 4].imshow(snapped_keep_rgb)
            axes3[idx_s, 4].set_title("AnyUp-PCA Snapped Mask\n(Green=Keep, Red=Discard)", fontsize=9)
            axes3[idx_s, 4].axis("off")
            
        plt.tight_layout()
        seg_plot_path = os.path.join(output_dir, "benchmark_segmentation_masks.png")
        plt.savefig(seg_plot_path, dpi=150)
        plt.close()
        print(f"Saved segmentation visualization masks to: {seg_plot_path}")


if __name__ == "__main__":
    main()
