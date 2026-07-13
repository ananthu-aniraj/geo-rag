import os
import argparse
import pandas as pd
import numpy as np
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoModel, SegformerImageProcessor, SegformerForSemanticSegmentation
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration Defaults
DEFAULT_CSV_PATH = "./full_pipeline_output/geo_space_deduplicated.csv"
DEFAULT_OUTPUT_DIR = "./retrieval_exps"
MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'
DISCARD_CLASSES = {2, 12, 20, 43, 80, 83, 102, 127}  # sky, person, car, sign, bus, truck, van, bike


def download_image(url, photo_id=None):
    try:
        if url.startswith("mapillary://") or (photo_id and "fbcdn.net" in url):
            orig_id = str(photo_id) if photo_id else url.split("://")[1]
            api_url = f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
            res = requests.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
        if not url: return None
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return Image.open(BytesIO(res.content)).convert("RGB")
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run local retrieval comparison and save images into organized directories.")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, help="Path to the lightweight CSV dataset.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Base directory to save downloaded images.")
    parser.add_argument("--query_idx", type=int, default=0, help="Row index of the query image in the sampled set (or platform subset).")
    parser.add_argument("--num_images", type=int, default=30, help="Number of diverse images to sample from the CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--query_platform", type=str, default=None, choices=["flickr", "mapillary"],
                        help="Choose platform for the query image (selects query_idx index from this platform's subset).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading metadata from {args.csv_path}...")
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows.")

    # Select diverse images representing different platforms/regions
    print(f"Selecting {args.num_images} diverse streetscape photos...")
    sampled_df = pd.concat([
        df[df['Platform'].str.lower() == 'mapillary'].sample(args.num_images // 2, random_state=args.seed),
        df[df['Platform'].str.lower() == 'flickr'].sample(args.num_images // 2, random_state=args.seed)
    ]).reset_index(drop=True)

    # Resolve query image row index
    if args.query_idx < 0:
        print("Error: --query_idx must be >= 0")
        return

    query_row_idx = args.query_idx
    if args.query_platform:
        platform_indices = sampled_df[sampled_df['Platform'].str.lower() == args.query_platform.lower()].index.tolist()
        if not platform_indices:
            print(f"Error: No images found for platform '{args.query_platform}' in the sampled set.")
            return
        if args.query_idx >= len(platform_indices):
            print(f"Error: --query_idx must be between 0 and {len(platform_indices) - 1} when filtering for platform '{args.query_platform}'")
            return
        query_row_idx = platform_indices[args.query_idx]
    else:
        if args.query_idx >= len(sampled_df):
            print(f"Error: --query_idx must be between 0 and {len(sampled_df) - 1}")
            return

    # Download in parallel
    print("Downloading images in parallel...")
    images = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(download_image, row['Image_URL'], row.get('Photo_ID')): idx
            for idx, row in sampled_df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloads"):
            idx = futures[future]
            img = future.result()
            if img:
                images[idx] = img

    print(f"Successfully downloaded {len(images)} out of {args.num_images} images.")
    if len(images) < 5:
        print("Not enough images downloaded to perform test. Aborting.")
        return

    if query_row_idx not in images:
        print(f"Error: Query image index {query_row_idx} failed to download. Please try another seed or index.")
        return

    # Load Models
    print("\nLoading Segformer and TIPSv2 on device:", device)
    seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").eval().to(
        device)
    tipsv2 = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).eval().to(device)

    # Compute Embeddings for downloaded images
    print("\nExtracting [CLS], [Simple-Average], [Background-Average], [CLS+Simple-Avg Concat], and [CLS+BG-Avg Concat] embeddings...")
    cls_embeddings = {}
    bg_embeddings = {}
    simple_embeddings = {}
    concat_simple_embeddings = {}
    concat_bg_embeddings = {}
    diagnostic_images = {}
    valid_indices = []

    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    for idx, img in tqdm(images.items(), desc="Inference"):
        img_resized = img.resize((448, 448))

        # 1. Segformer masking
        inputs = seg_processor(images=img_resized, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = seg_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(448, 448), mode="bilinear", align_corners=False)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        keep_mask = np.ones_like(pred_mask, dtype=float)
        for c in DISCARD_CLASSES:
            keep_mask[pred_mask == c] = 0.0

        patch_size = 14
        grid_size = 32
        patch_weights = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                patch_weights[r, c] = np.mean(
                    keep_mask[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size])
        patch_weights_flat = patch_weights.flatten()[:, np.newaxis]

        # Generate side-by-side diagnostic visualization (Original + binary Keep/Discard mask)
        binary_mask_visual = np.zeros((448, 448, 3), dtype=np.uint8)
        binary_mask_visual[keep_mask == 1.0] = [0, 200, 0]   # Keep (Green)
        binary_mask_visual[keep_mask == 0.0] = [200, 0, 0]   # Discard (Red)

        diag_img = Image.new('RGB', (896, 448))
        diag_img.paste(img_resized, (0, 0))
        diag_img.paste(Image.fromarray(binary_mask_visual), (448, 0))
        diagnostic_images[idx] = diag_img

        # 2. TIPSv2 Extraction
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = tipsv2.encode_image(img_tensor)
            cls_token = out.cls_token.squeeze().cpu().numpy()
            patch_tokens = out.patch_tokens.squeeze(0).cpu().numpy()

        # Background average
        total_weight = np.sum(patch_weights_flat)
        bg_avg = np.sum(patch_tokens * patch_weights_flat, axis=0) / total_weight if total_weight > 0 else np.mean(
            patch_tokens, axis=0)

        # Simple average of all patch tokens (unmasked)
        simple_avg = np.mean(patch_tokens, axis=0)

        # Concat CLS and Simple-Average (L2 normalized individually, then combined and normalized)
        cls_norm = cls_token / (np.linalg.norm(cls_token) + 1e-9)
        simple_norm = simple_avg / (np.linalg.norm(simple_avg) + 1e-9)
        concat_simple = np.concatenate([cls_norm, simple_norm], axis=0)
        concat_simple /= np.linalg.norm(concat_simple) + 1e-9

        # Concat CLS and Background-Average (L2 normalized individually, then combined and normalized)
        bg_norm = bg_avg / (np.linalg.norm(bg_avg) + 1e-9)
        concat_bg = np.concatenate([cls_norm, bg_norm], axis=0)
        concat_bg /= np.linalg.norm(concat_bg) + 1e-9

        # L2 normalize individual embeddings
        cls_token /= np.linalg.norm(cls_token) + 1e-9
        bg_avg /= np.linalg.norm(bg_avg) + 1e-9
        simple_avg /= np.linalg.norm(simple_avg) + 1e-9

        cls_embeddings[idx] = cls_token
        bg_embeddings[idx] = bg_avg
        simple_embeddings[idx] = simple_avg
        concat_simple_embeddings[idx] = concat_simple
        concat_bg_embeddings[idx] = concat_bg
        valid_indices.append(idx)

    # Perform retrieval matching
    query_row = sampled_df.iloc[query_row_idx]
    print(f"\n--- QUERY IMAGE DETAILS ---")
    print(f"Index: {query_row_idx}")
    print(f"Platform: {query_row['Platform']}")
    print(f"Lat/Lon: {query_row['Latitude']:.4f}, {query_row['Longitude']:.4f}")
    print(f"URL: {query_row['Image_URL']}")

    cls_similarities = {}
    bg_similarities = {}
    simple_similarities = {}
    concat_simple_similarities = {}
    concat_bg_similarities = {}
    q_cls = cls_embeddings[query_row_idx]
    q_bg = bg_embeddings[query_row_idx]
    q_simple = simple_embeddings[query_row_idx]
    q_concat_simple = concat_simple_embeddings[query_row_idx]
    q_concat_bg = concat_bg_embeddings[query_row_idx]

    for idx in valid_indices:
        if idx == query_row_idx: continue
        cls_similarities[idx] = np.dot(cls_embeddings[idx], q_cls)
        bg_similarities[idx] = np.dot(bg_embeddings[idx], q_bg)
        simple_similarities[idx] = np.dot(simple_embeddings[idx], q_simple)
        concat_simple_similarities[idx] = np.dot(concat_simple_embeddings[idx], q_concat_simple)
        concat_bg_similarities[idx] = np.dot(concat_bg_embeddings[idx], q_concat_bg)

    # Sort results
    top_cls = sorted(cls_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    top_bg = sorted(bg_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    top_simple = sorted(simple_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    top_concat_simple = sorted(concat_simple_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    top_concat_bg = sorted(concat_bg_similarities.items(), key=lambda x: x[1], reverse=True)[:3]

    # Create organized directories
    exp_dir = os.path.join(args.output_dir, f"exp_query_{query_row_idx}")
    query_dir = os.path.join(exp_dir, "query")
    cls_dir = os.path.join(exp_dir, "cls")
    simple_dir = os.path.join(exp_dir, "simple_avg")
    bg_dir = os.path.join(exp_dir, "bg_avg")
    concat_simple_dir = os.path.join(exp_dir, "cls_simple_concat")
    concat_bg_dir = os.path.join(exp_dir, "cls_bg_concat")

    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(cls_dir, exist_ok=True)
    os.makedirs(simple_dir, exist_ok=True)
    os.makedirs(bg_dir, exist_ok=True)
    os.makedirs(concat_simple_dir, exist_ok=True)
    os.makedirs(concat_bg_dir, exist_ok=True)

    # Save query image
    query_path = os.path.join(query_dir, "query_image.png")
    images[query_row_idx].save(query_path)
    # Save query diagnostic
    query_diag_path = os.path.join(query_dir, "query_segmentation.png")
    diagnostic_images[query_row_idx].save(query_diag_path)
    print(f"\nSaved query image and mask to: {query_dir}")

    print("\n🏆 === TOP 3 LOCAL MATCHES USING STANDARD [CLS] === 🏆")
    for rank, (idx, sim) in enumerate(top_cls, 1):
        row = sampled_df.iloc[idx]
        print(
            f" {rank}. Similarity: {sim:.4f} | Lat/Lon: {row['Latitude']:.4f}, {row['Longitude']:.4f} | URL: {row['Image_URL']}")
        # Save image under cls/
        match_path = os.path.join(cls_dir, f"match_{rank}_sim_{sim:.4f}.png")
        images[idx].save(match_path)
        print(f"    -> Saved match to: {match_path}")

    print("\n🏆 === TOP 3 LOCAL MATCHES USING [SIMPLE-AVERAGE] === 🏆")
    for rank, (idx, sim) in enumerate(top_simple, 1):
        row = sampled_df.iloc[idx]
        print(
            f" {rank}. Similarity: {sim:.4f} | Lat/Lon: {row['Latitude']:.4f}, {row['Longitude']:.4f} | URL: {row['Image_URL']}")
        # Save image under simple_avg/ (no masks)
        match_path = os.path.join(simple_dir, f"match_{rank}_sim_{sim:.4f}.png")
        images[idx].save(match_path)
        print(f"    -> Saved match to: {match_path}")

    print("\n🏆 === TOP 3 LOCAL MATCHES USING [BACKGROUND-AVERAGE] === 🏆")
    for rank, (idx, sim) in enumerate(top_bg, 1):
        row = sampled_df.iloc[idx]
        print(
            f" {rank}. Similarity: {sim:.4f} | Lat/Lon: {row['Latitude']:.4f}, {row['Longitude']:.4f} | URL: {row['Image_URL']}")
        # Save image under bg_avg/
        match_path = os.path.join(bg_dir, f"match_{rank}_sim_{sim:.4f}.png")
        images[idx].save(match_path)
        # Save diagnostic mask under bg_avg/
        match_diag_path = os.path.join(bg_dir, f"match_{rank}_sim_{sim:.4f}_segmentation.png")
        diagnostic_images[idx].save(match_diag_path)
        print(f"    -> Saved match to: {match_path}")

    print("\n🏆 === TOP 3 LOCAL MATCHES USING [CLS + SIMPLE-AVG CONCAT] === 🏆")
    for rank, (idx, sim) in enumerate(top_concat_simple, 1):
        row = sampled_df.iloc[idx]
        print(
            f" {rank}. Similarity: {sim:.4f} | Lat/Lon: {row['Latitude']:.4f}, {row['Longitude']:.4f} | URL: {row['Image_URL']}")
        # Save image under cls_simple_concat/
        match_path = os.path.join(concat_simple_dir, f"match_{rank}_sim_{sim:.4f}.png")
        images[idx].save(match_path)
        print(f"    -> Saved match to: {match_path}")

    print("\n🏆 === TOP 3 LOCAL MATCHES USING [CLS + BG-AVG CONCAT] === 🏆")
    for rank, (idx, sim) in enumerate(top_concat_bg, 1):
        row = sampled_df.iloc[idx]
        print(
            f" {rank}. Similarity: {sim:.4f} | Lat/Lon: {row['Latitude']:.4f}, {row['Longitude']:.4f} | URL: {row['Image_URL']}")
        # Save image under cls_bg_concat/
        match_path = os.path.join(concat_bg_dir, f"match_{rank}_sim_{sim:.4f}.png")
        images[idx].save(match_path)
        print(f"    -> Saved match to: {match_path}")

    print(f"\nExperiment output saved successfully to: {exp_dir}")


if __name__ == "__main__":
    main()
