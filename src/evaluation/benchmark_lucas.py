import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoModel,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

DISCARD_CLASSES = {2, 12, 20, 43, 80, 83, 102, 127}  # sky, person, car, sign, bus, truck, van, bike

def load_lucas_metadata(csv_path):
    """Loads LUCAS metadata and maps the 8-digit point ID to its land cover, land use, and EUNIS class labels."""
    if not os.path.exists(csv_path):
        print(f"Error: Metadata file '{csv_path}' not found.")
        sys.exit(1)

    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    metadata = {}
    
    for _, row in df.iterrows():
        im_id = str(row['im_id']).strip()
        match = re.search(r'(\d{8})', im_id)
        if match:
            clean_id = match.group(1)
        else:
            clean_id = im_id
            
        metadata[clean_id] = {
            'im_id': im_id,
            'lc_label': row.get('lc_label', ''),
            'lu_label': row.get('lu_label', ''),
            'eunis_class': row.get('eunis_class', '')
        }
        
    return metadata, df


def get_class_lists(df):
    """Retrieves class lists from lucas_class_mapping if available, falling back to unique values in the metadata CSV."""
    lc_list, lu_list, eunis_list = [], [], []
    try:
        from src.utils import lucas_class_mapping
        lc_list = list(lucas_class_mapping.lc1_class_mapping.values())
        lu_list = list(lucas_class_mapping.lu1_class_mapping.values())
        eunis_list = list(lucas_class_mapping.eunis_mapping.values())
        print("Successfully loaded class lists from lucas_class_mapping.")
    except ImportError:
        print("Warning: lucas_class_mapping not found or could not be imported. Extracting unique classes from metadata CSV...")
        if 'lc_label' in df.columns:
            lc_list = sorted(df['lc_label'].dropna().unique().tolist())
        if 'lu_label' in df.columns:
            lu_list = sorted(df['lu_label'].dropna().unique().tolist())
        if 'eunis_class' in df.columns:
            eunis_list = sorted(df['eunis_class'].dropna().unique().tolist())
            
    return lc_list, lu_list, eunis_list


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


from src.evaluation.metrics import compute_ap, compute_rr, compute_precision_at_k


def main():
    parser = argparse.ArgumentParser(description="LUCAS Representation Semantic Retrieval Benchmarking Suite.")
    parser.add_argument("--csv", type=str, default="/user/aaniraj/home/Documents/Projects/data/LUCAS2018/Sen4Map_Metadata_test.csv",
                        help="Path to the Sen4Map_Metadata_test.csv file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing LUCAS images.")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of query evaluations to run.")
    parser.add_argument("--num_database", type=int, default=500, help="Number of database images to search against (0 for all remaining).")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU batch size for feature extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tips_model_path", type=str, default=None,
                        help="Path to the official TIPSv2 model checkpoint (.npz). If None, uses Hugging Face 'google/tipsv2-b14'.")
    parser.add_argument("--tips_model_variant", type=str, default="B", choices=["S", "B", "L", "So400m", "g"],
                        help="Variant of the official TIPSv2 model.")
    parser.add_argument("--tips_low_res", action="store_true", help="Set image resolution to 224px instead of 448px.")
    parser.add_argument("--output_report", type=str, default="./benchmark_results/lucas_report.txt", help="Path to write the report summary.")
    parser.add_argument("--output_csv", type=str, default="./benchmark_results/lucas_results.csv", help="Path to write detailed query CSV results.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Load metadata and scan for local images
    metadata_dict, df = load_lucas_metadata(args.csv)
    lc_list, lu_list, eunis_list = get_class_lists(df)

    print(f"Scanning for images in {args.img_dir}...")
    matched_images = []
    for root, _, files in os.walk(args.img_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                basename = os.path.splitext(filename)[0]
                match = re.search(r'(\d{8})', basename)
                if match:
                    point_num = match.group(1)
                    dir_match = re.search(rf'{point_num}([WENS])', basename, re.IGNORECASE)
                    direction = dir_match.group(1).upper() if dir_match else "Unknown"
                    
                    if point_num in metadata_dict:
                        matched_images.append({
                            'path': os.path.join(root, filename),
                            'filename': filename,
                            'point_id': point_num,
                            'direction': direction,
                            'lc_label': metadata_dict[point_num]['lc_label'],
                            'lu_label': metadata_dict[point_num]['lu_label'],
                            'eunis_class': metadata_dict[point_num]['eunis_class']
                        })

    if not matched_images:
        print("Error: No matching images found in the provided directory.")
        return

    print(f"Found {len(matched_images):,} matched image files in directory.")

    # Group matched images by point ID to preserve geographical diversity in sampling
    images_by_point = {}
    for item in matched_images:
        pt_id = item['point_id']
        if pt_id not in images_by_point:
            images_by_point[pt_id] = []
        images_by_point[pt_id].append(item)

    sorted_points = sorted(images_by_point.keys())
    random.shuffle(sorted_points)

    # Interleave images across different points to maximize variety
    selected_images = []
    max_dirs = max(len(images_by_point[pt]) for pt in sorted_points)
    
    for dir_idx in range(max_dirs):
        for pt in sorted_points:
            if dir_idx < len(images_by_point[pt]):
                selected_images.append(images_by_point[pt][dir_idx])

    # Determine subset counts
    total_needed = args.num_queries + (args.num_database if args.num_database > 0 else (len(selected_images) - args.num_queries))
    if len(selected_images) < total_needed:
        print(f"Warning: Only {len(selected_images)} matched images available. Adjusting query/database split.")
        total_needed = len(selected_images)
        
    selected_images = selected_images[:total_needed]
    
    # Split into Queries and Database
    queries_meta = selected_images[:args.num_queries]
    database_meta = selected_images[args.num_queries:]
    
    print(f"Split data into: {len(queries_meta)} Queries and {len(database_meta)} Database images.")
    if len(queries_meta) == 0 or len(database_meta) == 0:
        print("Error: Empty query or database split. Adjust your parameters.")
        return

    # 2. Initialize Models
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
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").eval().to(device)

    # 3. Setup representations dynamically
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
            batch_meta = metadata_list[i : i + args.batch_size]
            batch_imgs = []
            
            for item in batch_meta:
                try:
                    img = Image.open(item['path']).convert("RGB")
                    img_resized = img.resize((image_size, image_size))
                    batch_imgs.append(img_resized)
                except Exception as e:
                    print(f"Warning: Failed to load image '{item['path']}': {e}")
            
            if not batch_imgs:
                continue

            # A. SegFormer segmentation masks
            inputs = seg_processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = seg_model(**inputs)
            logits = torch.nn.functional.interpolate(outputs.logits, size=(image_size, image_size), mode="bilinear", align_corners=False)
            pred_masks = logits.argmax(dim=1).cpu().numpy()

            # B. TIPSv2 feature extraction
            img_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
            with torch.no_grad():
                if args.tips_model_path:
                    # Official model forward returns: first_cls_token, second_cls_token, patch_tokens
                    first_cls_token, second_cls_token, patch_tokens = tipsv2(img_tensors)
                    
                    first_cls = first_cls_token.cpu().numpy()
                    if first_cls.ndim == 3:
                        first_cls = first_cls.squeeze(1)
                        
                    second_cls = second_cls_token.cpu().numpy()
                    if second_cls.ndim == 3:
                        second_cls = second_cls.squeeze(1)
                        
                    representations["TIPSv2 1st CLS"][split_key].extend(first_cls)
                    representations["TIPSv2 2nd CLS"][split_key].extend(second_cls)
                else:
                    out = tipsv2.encode_image(img_tensors)
                    cls_tokens = out.cls_token.cpu().numpy()
                    if cls_tokens.ndim == 3:
                        cls_tokens = cls_tokens.squeeze(1)
                    representations["TIPSv2 CLS"][split_key].extend(cls_tokens)

                # Extract patch tokens
                vision_encoder = tipsv2 if args.tips_model_path else tipsv2.vision_encoder
                patch_tokens_vals = encode_image_value_attention(vision_encoder, img_tensors)
                patch_tokens_vals = patch_tokens_vals.reshape(len(batch_imgs), num_patches, -1).cpu().numpy()

            # Process patch poolings for each image in batch
            for idx in range(len(batch_imgs)):
                patch_tokens = patch_tokens_vals[idx] # (num_patches, D)
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
                        patch_weights[r, c] = np.mean(keep_mask[r*14:(r+1)*14, c*14:(c+1)*14])
                patch_weights_flat = patch_weights.flatten()[:, np.newaxis] # (num_patches, 1)

                masked_patch_sum = np.sum(patch_tokens * patch_weights_flat, axis=0)
                masked_patch_weight_sum = np.sum(patch_weights_flat)
                if masked_patch_weight_sum > 0:
                    masked_avg_embed = (masked_patch_sum / (masked_patch_weight_sum + 1e-9))
                else:
                    masked_avg_embed = avg_patch
                
                representations["TIPSv2 Seg-Masked"][split_key].append(masked_avg_embed)

    extract_features_batch(queries_meta, "query")
    extract_features_batch(database_meta, "db")

    # 4. Retrieval & Similarity Benchmarking
    label_types = ["lc_label", "lu_label", "eunis_class"]
    label_names = {"lc_label": "Land Cover", "lu_label": "Land Use", "eunis_class": "EUNIS Class"}
    
    results = {}
    detailed_rows = []
    
    for rep_name, splits in representations.items():
        q_vectors = np.array(splits["query"])
        db_vectors = np.array(splits["db"])

        if len(q_vectors) == 0 or len(db_vectors) == 0:
            print(f"Skipping {rep_name} due to empty features.")
            continue

        # L2 Normalize vectors for Cosine Similarity via dot product
        q_norms = np.linalg.norm(q_vectors, axis=1, keepdims=True) + 1e-9
        db_norms = np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-9
        q_vectors_norm = q_vectors / q_norms
        db_vectors_norm = db_vectors / db_norms

        results[rep_name] = {}
        for l_type in label_types:
            results[rep_name][l_type] = {
                "p@1": 0.0,
                "p@5": 0.0,
                "p@10": 0.0,
                "map@10": 0.0,
                "mrr@10": 0.0
            }

        # Query loop
        for q_idx in range(len(queries_meta)):
            q_vec = q_vectors_norm[q_idx]
            q_item = queries_meta[q_idx]
            
            # Compute cosine similarities to all DB images
            similarities = np.dot(db_vectors_norm, q_vec)
            sorted_db_indices = np.argsort(similarities)[::-1]

            for l_type in label_types:
                q_label = q_item[l_type]
                if pd.isna(q_label) or q_label == '':
                    continue
                
                retrieved_items = [database_meta[idx] for idx in sorted_db_indices[:10]]
                retrieved_labels = [item[l_type] for item in retrieved_items]
                
                # P@1
                p1 = compute_precision_at_k(retrieved_labels, q_label, k=1)
                results[rep_name][l_type]["p@1"] += p1
                
                # P@5
                p5 = compute_precision_at_k(retrieved_labels, q_label, k=5)
                results[rep_name][l_type]["p@5"] += p5

                # P@10
                p10 = compute_precision_at_k(retrieved_labels, q_label, k=10)
                results[rep_name][l_type]["p@10"] += p10

                # mAP@10
                ap = compute_ap(retrieved_labels, q_label, k=10)
                results[rep_name][l_type]["map@10"] += ap

                # MRR@10
                rr = compute_rr(retrieved_labels, q_label, k=10)
                results[rep_name][l_type]["mrr@10"] += rr

                detailed_rows.append({
                    "Query_Image": q_item["filename"],
                    "Representation": rep_name,
                    "Label_Type": l_type,
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
        for l_type in label_types:
            valid_queries = sum(1 for q in queries_meta if pd.notna(q[l_type]) and q[l_type] != '')
            if valid_queries > 0:
                results[rep_name][l_type]["p@1"] = (results[rep_name][l_type]["p@1"] / valid_queries) * 100.0
                results[rep_name][l_type]["p@5"] = (results[rep_name][l_type]["p@5"] / valid_queries) * 100.0
                results[rep_name][l_type]["p@10"] = (results[rep_name][l_type]["p@10"] / valid_queries) * 100.0
                results[rep_name][l_type]["map@10"] = (results[rep_name][l_type]["map@10"] / valid_queries) * 100.0
                results[rep_name][l_type]["mrr@10"] = (results[rep_name][l_type]["mrr@10"] / valid_queries) * 100.0

    # 5. Compile and Print Report
    report_lines = []
    report_lines.append("LUCAS 2018 Image Representation Semantic Retrieval Report")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"- Queries count: {len(queries_meta)} images")
    report_lines.append(f"- Database count: {len(database_meta)} images")
    report_lines.append("")

    print("\n" + "="*95)
    print("                    LUCAS 2018 SEMANTIC RETRIEVAL BENCHMARK REPORT")
    print("="*95)

    for l_type in label_types:
        print(f"\n{label_names[l_type]} Evaluation:")
        
        row_format = "{:<24} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}"
        header = row_format.format("Representation", "P@1 (%)", "P@5 (%)", "P@10 (%)", "mAP@10 (%)", "MRR@10 (%)")
        print("-" * 90)
        print(header)
        print("-" * 90)
        
        report_lines.append(f"--- {label_names[l_type]} Evaluation ---")
        report_lines.append(header)
        report_lines.append("-" * 90)

        for rep_name in results.keys():
            metrics = results[rep_name][l_type]
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
            
        print("-" * 90)
        report_lines.append("")

    print("="*95)

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
