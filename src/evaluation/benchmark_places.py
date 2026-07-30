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

def load_places365_labels(filepath):
    """Loads the valid Places365 categories from the Excel file and returns a mapping to macro and sub-categories."""
    if not os.path.exists(filepath):
        print(f"Warning: Labels file '{filepath}' not found. Continuing with directory names only.")
        return {}, [], []

    print(f"Loading labels from {filepath}...")
    try:
        df = pd.read_excel(filepath, header=[0, 1])
        mapping = {}
        all_sub_cats = set()
        all_types = []

        for col in df.columns:
            if col[0].startswith("Level 2"):
                all_sub_cats.add(col[1])

        for _, row in df.iterrows():
            raw_cat = row[('Unnamed: 0_level_0', 'category')]
            if not isinstance(raw_cat, str):
                continue

            clean_cat = raw_cat.strip("'").strip("/")
            if '/' in clean_cat:
                parts = clean_cat.split('/')
                if len(parts[0]) == 1:
                    clean_cat = "/".join(parts[1:])

            clean_cat = clean_cat.replace("/", "-")
            all_types.append(clean_cat)

            macro = "unknown"
            if row[('Level 1', 'indoor')] == 1:
                macro = "indoor"
            elif row[('Level 1', 'outdoor, natural')] == 1:
                macro = "outdoor natural"
            elif row[('Level 1', 'outdoor, man-made')] == 1:
                macro = "outdoor man-made"

            sub_cat = "unknown"
            for col in df.columns:
                if col[0].startswith("Level 2") and row[col] == 1:
                    sub_cat = col[1]
                    break

            mapping[clean_cat] = {"macro": macro, "sub_category": sub_cat}

        return mapping, sorted(list(all_sub_cats)), sorted(all_types)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}, [], []


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


def compute_ap(retrieved_labels, query_label, k=10):
    """Computes Average Precision at K for a query label and retrieved labels."""
    ap = 0.0
    correct_count = 0
    for i in range(min(len(retrieved_labels), k)):
        if retrieved_labels[i] == query_label:
            correct_count += 1
            precision_at_i = correct_count / (i + 1)
            ap += precision_at_i
    if correct_count > 0:
        ap /= correct_count
    return ap


def main():
    parser = argparse.ArgumentParser(description="Places365 Representation Semantic Retrieval Benchmarking Suite.")
    parser.add_argument("--labels", type=str, default="/user/aaniraj/home/Documents/Projects/data/places365/versions/1/Scene_hierarchy.xlsx",
                        help="Path to the Scene_hierarchy.xlsx file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing Places365 images.")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of query evaluations to run.")
    parser.add_argument("--num_database", type=int, default=500, help="Number of database images to search against (0 for all remaining).")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU batch size for feature extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tips_model_path", type=str, default=None,
                        help="Path to the official TIPSv2 model checkpoint (.npz). If None, uses Hugging Face 'google/tipsv2-b14'.")
    parser.add_argument("--tips_model_variant", type=str, default="B", choices=["S", "B", "L", "So400m", "g"],
                        help="Variant of the official TIPSv2 model.")
    parser.add_argument("--tips_low_res", action="store_true", help="Set image resolution to 224px instead of 448px.")
    parser.add_argument("--compare_clip", action="store_true", help="Include standard CLIP model in the comparison.")
    parser.add_argument("--compare_hf_tips", action="store_true", help="Include Hugging Face google/tipsv2-b14 model in the comparison.")
    parser.add_argument("--output_report", type=str, default="./benchmark_results/places_report.txt", help="Path to write the report summary.")
    parser.add_argument("--output_csv", type=str, default="./benchmark_results/places_results.csv", help="Path to write detailed query CSV results.")
    args = parser.parse_args()

    # Format output paths dynamically by appending seed and num_queries
    report_base, report_ext = os.path.splitext(args.output_report)
    args.output_report = f"{report_base}_s{args.seed}_q{args.num_queries}{report_ext}"
    
    csv_base, csv_ext = os.path.splitext(args.output_csv)
    args.output_csv = f"{csv_base}_s{args.seed}_q{args.num_queries}{csv_ext}"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Load labels mapping
    labels_mapping, sub_categories_list, type_of_places_list = load_places365_labels(args.labels)

    # Scan directory for images
    print(f"Scanning directory structure in {args.img_dir}...")
    images_by_class = {}
    for root, _, files in os.walk(args.img_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                class_folder_name = os.path.basename(root)
                ground_truth_label = class_folder_name
                
                # Normalize class name to match excel keys
                clean_class = ground_truth_label.replace("/", "-")
                
                macro = "unknown"
                sub_category = "unknown"
                if clean_class in labels_mapping:
                    macro = labels_mapping[clean_class]["macro"]
                    sub_category = labels_mapping[clean_class]["sub_category"]

                if ground_truth_label not in images_by_class:
                    images_by_class[ground_truth_label] = []
                images_by_class[ground_truth_label].append({
                    'path': image_path,
                    'filename': filename,
                    'place_label': clean_class,
                    'macro_label': macro,
                    'sub_label': sub_category
                })

    if not images_by_class:
        print("Error: No images found in the provided directory.")
        return

    # Sort classes and images within each class for determinism
    sorted_classes = sorted(images_by_class.keys())
    for cls in sorted_classes:
        images_by_class[cls].sort(key=lambda x: x['path'])
        random.shuffle(images_by_class[cls])

    # Interleave images to maximize class coverage
    selected_images = []
    max_images_in_any_class = max(len(images_by_class[cls]) for cls in sorted_classes)

    for i in range(max_images_in_any_class):
        for cls in sorted_classes:
            if i < len(images_by_class[cls]):
                selected_images.append(images_by_class[cls][i])

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
    
    # Load CLIP baseline if requested
    if args.compare_clip:
        print("Loading standard CLIP model...")
        from sentence_transformers import SentenceTransformer
        clip_model = SentenceTransformer("clip-ViT-B-32")
    
    # Load Hugging Face TIPSv2 model as baseline if requested (and if we are loading official checkpoint)
    hf_tipsv2 = None
    if args.compare_hf_tips and args.tips_model_path:
        print("Loading Hugging Face google/tipsv2-b14 model as baseline...")
        hf_tipsv2 = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).eval().to(device)

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
    representations = {}
    if args.compare_clip:
        representations["CLIP CLS"] = {"query": [], "db": []}
    if hf_tipsv2 is not None:
        representations["TIPSv2 HF CLS"] = {"query": [], "db": []}
        
    if args.tips_model_path:
        representations.update({
            "TIPSv2 1st CLS": {"query": [], "db": []},
            "TIPSv2 2nd CLS": {"query": [], "db": []},
            "TIPSv2 Average Patch": {"query": [], "db": []},
            "TIPSv2 Seg-Masked": {"query": [], "db": []}
        })
    else:
        representations.update({
            "TIPSv2 CLS": {"query": [], "db": []},
            "TIPSv2 Average Patch": {"query": [], "db": []},
            "TIPSv2 Seg-Masked": {"query": [], "db": []}
        })

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

            # B. CLIP feature extraction (optional)
            if args.compare_clip:
                with torch.no_grad():
                    clip_cls = clip_model.encode(batch_imgs, convert_to_numpy=True)
                representations["CLIP CLS"][split_key].extend(clip_cls)

            # C. Hugging Face TIPSv2 feature extraction (optional)
            if hf_tipsv2 is not None:
                hf_imgs = [img.resize((448, 448)) for img in batch_imgs] if image_size != 448 else batch_imgs
                img_tensors_hf = torch.stack([transform(img) for img in hf_imgs]).to(device)
                with torch.no_grad():
                    out_hf = hf_tipsv2.encode_image(img_tensors_hf)
                    cls_hf = out_hf.cls_token.cpu().numpy()
                    if cls_hf.ndim == 3:
                        cls_hf = cls_hf.squeeze(1)
                representations["TIPSv2 HF CLS"][split_key].extend(cls_hf)

            # D. TIPSv2 feature extraction
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
    label_types = ["place_label", "sub_label", "macro_label"]
    label_names = {"place_label": "Exact Place", "sub_label": "Sub-Category", "macro_label": "Macro Category"}
    
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
            
            # Compute similarities
            similarities = np.dot(db_vectors_norm, q_vec)
            sorted_db_indices = np.argsort(similarities)[::-1]

            for l_type in label_types:
                q_label = q_item[l_type]
                if pd.isna(q_label) or q_label == '' or q_label == 'unknown':
                    continue
                
                retrieved_items = [database_meta[idx] for idx in sorted_db_indices[:10]]
                retrieved_labels = [item[l_type] for item in retrieved_items]
                
                # P@1
                if retrieved_labels[0] == q_label:
                    results[rep_name][l_type]["p@1"] += 1.0
                
                # P@5
                matches_5 = sum(1.0 for l in retrieved_labels[:5] if l == q_label)
                results[rep_name][l_type]["p@5"] += matches_5 / 5.0

                # P@10
                matches_10 = sum(1.0 for l in retrieved_labels[:10] if l == q_label)
                results[rep_name][l_type]["p@10"] += matches_10 / 10.0

                # mAP@10
                ap = compute_ap(retrieved_labels, q_label, k=10)
                results[rep_name][l_type]["map@10"] += ap

                # MRR@10
                rr = 0.0
                for rank_idx, label in enumerate(retrieved_labels):
                    if label == q_label:
                        rr = 1.0 / (rank_idx + 1)
                        break
                results[rep_name][l_type]["mrr@10"] += rr

                detailed_rows.append({
                    "Query_Image": q_item["filename"],
                    "Representation": rep_name,
                    "Label_Type": l_type,
                    "Ground_Truth": q_label,
                    "Top_1_Retrieved": retrieved_items[0]["filename"],
                    "Top_1_Label": retrieved_labels[0],
                    "P@1": 1.0 if retrieved_labels[0] == q_label else 0.0,
                    "P@5": matches_5 / 5.0,
                    "P@10": matches_10 / 10.0,
                    "AP@10": ap,
                    "RR@10": rr
                })

        # Normalize metrics by valid query count
        for l_type in label_types:
            valid_queries = sum(1 for q in queries_meta if pd.notna(q[l_type]) and q[l_type] != '' and q[l_type] != 'unknown')
            if valid_queries > 0:
                results[rep_name][l_type]["p@1"] = (results[rep_name][l_type]["p@1"] / valid_queries) * 100.0
                results[rep_name][l_type]["p@5"] = (results[rep_name][l_type]["p@5"] / valid_queries) * 100.0
                results[rep_name][l_type]["p@10"] = (results[rep_name][l_type]["p@10"] / valid_queries) * 100.0
                results[rep_name][l_type]["map@10"] = (results[rep_name][l_type]["map@10"] / valid_queries) * 100.0
                results[rep_name][l_type]["mrr@10"] = (results[rep_name][l_type]["mrr@10"] / valid_queries) * 100.0

    # 5. Compile and Print Report
    report_lines = []
    report_lines.append("Places365 Image Representation Semantic Retrieval Report")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"- Queries count: {len(queries_meta)} images")
    report_lines.append(f"- Database count: {len(database_meta)} images")
    report_lines.append("")

    print("\n" + "="*95)
    print("                    PLACES365 SEMANTIC RETRIEVAL BENCHMARK REPORT")
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
