import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoModel,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# Configuration Defaults
DEFAULT_CSV_PATH = "./iwildcam_subset/iwildcam_subset_metadata.csv"
DEFAULT_IMAGES_DIR = "./iwildcam_subset/train"
DEFAULT_OUTPUT_DIR = "./iwildcam_exps"

# Discard classes (same as ADE20K configurations)
DISCARD_CLASSES = {
    2,
    12,
    20,
    43,
    80,
    83,
    102,
    127,
}  # sky, person, car, sign, bus, truck, van, bike


def main():
    parser = argparse.ArgumentParser(
        description="Run local iWildCam retrieval comparison and evaluate camera trap localization Precision@3."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to the iWildCam subset metadata CSV.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing iWildCam images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory to save results.",
    )
    parser.add_argument(
        "--query_idx",
        type=int,
        default=8,
        help="Row index of the query image in the subset.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading metadata from {args.csv_path}...")
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows.")

    if args.query_idx >= len(df) or args.query_idx < 0:
        print(f"Error: --query_idx must be between 0 and {len(df) - 1}")
        return

    # Lazy image loading helper
    images = {}

    def get_image(idx):
        if idx in images:
            return images[idx]
        if idx >= len(df) or idx < 0:
            return None
        row = df.iloc[idx]
        img_path = os.path.join(args.images_dir, row["file_name"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            images[idx] = img
            return img
        return None

    # Check query image
    if get_image(args.query_idx) is None:
        print(f"Error: Query image index {args.query_idx} failed to load.")
        return

    # Embeddings cache path
    cache_path = os.path.join(
        os.path.dirname(args.csv_path), "iwildcam_embeddings_cache.pkl"
    )
    import pickle

    cls_embeddings = {}
    bg_embeddings = {}
    simple_embeddings = {}
    concat_simple_embeddings = {}
    concat_bg_embeddings = {}
    diagnostic_images = {}
    valid_indices = []

    # Segformer references for dynamic mask generation
    seg_model = None
    seg_processor = None

    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
            cls_embeddings = cache_data["cls"]
            bg_embeddings = cache_data["bg"]
            simple_embeddings = cache_data["simple"]
            concat_simple_embeddings = cache_data["concat_simple"]
            concat_bg_embeddings = cache_data["concat_bg"]
            valid_indices = cache_data["valid_indices"]
        print(f"Loaded {len(valid_indices)} cached embeddings successfully.")
    else:
        # Load Models
        print("\nLoading Segformer and TIPSv2 on device:", device)
        seg_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        seg_model = (
            SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            .eval()
            .to(device)
        )
        tipsv2 = (
            AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
            .eval()
            .to(device)
        )

        # Compute Embeddings
        print(
            "\nExtracting [CLS], [Simple-Average], [Background-Average], [CLS+Simple-Avg Concat], and [CLS+BG-Avg Concat] embeddings..."
        )
        transform = transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        )

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
            img = get_image(idx)
            if img is None:
                continue
            img_resized = img.resize((448, 448))

            # 1. Segformer Masking
            inputs = seg_processor(images=img_resized, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = seg_model(**inputs)

            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=(448, 448), mode="bilinear", align_corners=False
            )
            pred_segmentation = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()

            patch_weights = np.ones((32, 32), dtype=np.float32)
            for r in range(32):
                for c in range(32):
                    patch_pixels = pred_segmentation[
                        r * 14 : (r + 1) * 14, c * 14 : (c + 1) * 14
                    ]
                    classes, counts = np.unique(patch_pixels, return_counts=True)
                    dominant_class = classes[np.argmax(counts)]
                    if dominant_class in DISCARD_CLASSES:
                        patch_weights[r, c] = 0.0

            patch_weights_flat = patch_weights.flatten()[:, np.newaxis]

            # 2. TIPSv2 Extraction
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = tipsv2.encode_image(img_tensor)
                cls_token = out.cls_token.squeeze().cpu().numpy()
                patch_tokens = out.patch_tokens.squeeze(0).cpu().numpy()

            total_weight = np.sum(patch_weights_flat)
            bg_avg = (
                np.sum(patch_tokens * patch_weights_flat, axis=0) / total_weight
                if total_weight > 0
                else np.mean(patch_tokens, axis=0)
            )

            simple_avg = np.mean(patch_tokens, axis=0)

            # Concat CLS and Simple-Average
            cls_norm = cls_token / (np.linalg.norm(cls_token) + 1e-9)
            simple_norm = simple_avg / (np.linalg.norm(simple_avg) + 1e-9)
            concat_simple = np.concatenate([cls_norm, simple_norm], axis=0)
            concat_simple /= np.linalg.norm(concat_simple) + 1e-9

            # Concat CLS and Background-Average
            bg_norm = bg_avg / (np.linalg.norm(bg_avg) + 1e-9)
            concat_bg = np.concatenate([cls_norm, bg_norm], axis=0)
            concat_bg /= np.linalg.norm(concat_bg) + 1e-9

            cls_token /= np.linalg.norm(cls_token) + 1e-9
            bg_avg /= np.linalg.norm(bg_avg) + 1e-9
            simple_avg /= np.linalg.norm(simple_avg) + 1e-9

            cls_embeddings[idx] = cls_token
            bg_embeddings[idx] = bg_avg
            simple_embeddings[idx] = simple_avg
            concat_simple_embeddings[idx] = concat_simple
            concat_bg_embeddings[idx] = concat_bg
            valid_indices.append(idx)

        # Save embeddings to cache
        print(f"Caching embeddings to {cache_path}...")
        cache_data = {
            "cls": cls_embeddings,
            "bg": bg_embeddings,
            "simple": simple_embeddings,
            "concat_simple": concat_simple_embeddings,
            "concat_bg": concat_bg_embeddings,
            "valid_indices": valid_indices,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print("Embeddings cached successfully.")

    # Helper function to generate diagnostic visualization mask on-demand
    def get_diagnostic_image(idx):
        nonlocal seg_model, seg_processor
        if idx in diagnostic_images:
            return diagnostic_images[idx]

        if seg_model is None:
            print(
                f"\n[On-Demand] Loading Segformer to generate visualization mask for image {idx}..."
            )
            seg_processor = SegformerImageProcessor.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            seg_model = (
                SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )
                .eval()
                .to(device)
            )

        img = get_image(idx)
        img_resized = img.resize((448, 448))
        inputs = seg_processor(images=img_resized, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = seg_model(**inputs)

        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(448, 448), mode="bilinear", align_corners=False
        )
        pred_segmentation = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        patch_weights = np.ones((32, 32), dtype=np.float32)
        for r in range(32):
            for c in range(32):
                patch_pixels = pred_segmentation[
                    r * 14 : (r + 1) * 14, c * 14 : (c + 1) * 14
                ]
                classes, counts = np.unique(patch_pixels, return_counts=True)
                dominant_class = classes[np.argmax(counts)]
                if dominant_class in DISCARD_CLASSES:
                    patch_weights[r, c] = 0.0

        mask_overlay = np.zeros((448, 448, 3), dtype=np.uint8)
        for r in range(32):
            for c in range(32):
                color = [0, 255, 0] if patch_weights[r, c] > 0 else [255, 0, 0]
                mask_overlay[r * 14 : (r + 1) * 14, c * 14 : (c + 1) * 14] = color

        binary_mask_visual = (np.array(img_resized) * 0.6 + mask_overlay * 0.4).astype(
            np.uint8
        )

        diag_img = Image.new("RGB", (896, 448))
        diag_img.paste(img_resized, (0, 0))
        diag_img.paste(Image.fromarray(binary_mask_visual), (448, 0))
        diagnostic_images[idx] = diag_img
        return diag_img

    # Perform retrieval matching
    query_row = df.iloc[args.query_idx]
    query_location = query_row["location"]

    print("\n--- QUERY IMAGE DETAILS ---")
    print(f"Index: {args.query_idx}")
    print(f"File Name: {query_row['file_name']}")
    print(f"Camera Location: {query_location}")
    print(f"Datetime: {query_row['Captured_At']}")
    print(f"Category: {query_row['category_name']}")

    cls_similarities = {}
    bg_similarities = {}
    simple_similarities = {}
    concat_simple_similarities = {}
    concat_bg_similarities = {}
    q_cls = cls_embeddings[args.query_idx]
    q_bg = bg_embeddings[args.query_idx]
    q_simple = simple_embeddings[args.query_idx]
    q_concat_simple = concat_simple_embeddings[args.query_idx]
    q_concat_bg = concat_bg_embeddings[args.query_idx]

    for idx in valid_indices:
        if idx == args.query_idx:
            continue
        cls_similarities[idx] = np.dot(cls_embeddings[idx], q_cls)
        bg_similarities[idx] = np.dot(bg_embeddings[idx], q_bg)
        simple_similarities[idx] = np.dot(simple_embeddings[idx], q_simple)
        concat_simple_similarities[idx] = np.dot(
            concat_simple_embeddings[idx], q_concat_simple
        )
        concat_bg_similarities[idx] = np.dot(concat_bg_embeddings[idx], q_concat_bg)

    # Sort results (Top 3)
    top_cls = sorted(cls_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    top_simple = sorted(simple_similarities.items(), key=lambda x: x[1], reverse=True)[
        :3
    ]
    top_bg = sorted(bg_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    top_concat_simple = sorted(
        concat_simple_similarities.items(), key=lambda x: x[1], reverse=True
    )[:3]
    top_concat_bg = sorted(
        concat_bg_similarities.items(), key=lambda x: x[1], reverse=True
    )[:3]

    # Create organized directories
    exp_dir = os.path.join(args.output_dir, f"exp_query_{args.query_idx}")
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

    # Save query image and mask
    query_path = os.path.join(query_dir, "query_image.png")
    get_image(args.query_idx).save(query_path)
    query_diag_path = os.path.join(query_dir, "query_segmentation.png")
    get_diagnostic_image(args.query_idx).save(query_diag_path)

    def print_matches_and_eval(name, top_matches, save_dir, save_masks=False):
        print(f"\n🏆 === TOP 3 LOCAL MATCHES USING {name} === 🏆")
        correct_count = 0
        for rank, (idx, sim) in enumerate(top_matches, 1):
            row = df.iloc[idx]
            is_correct = (
                "CORRECT (Same Camera)"
                if row["location"] == query_location
                else "INCORRECT (Diff Camera)"
            )
            if row["location"] == query_location:
                correct_count += 1
            print(
                f" {rank}. Similarity: {sim:.4f} | Location: {row['location']} ({is_correct}) | Date: {row['Captured_At']} | Category: {row['category_name']}"
            )

            # Save raw image
            match_path = os.path.join(save_dir, f"match_{rank}_sim_{sim:.4f}.png")
            get_image(idx).save(match_path)

            # Optionally save mask
            if save_masks:
                match_diag_path = os.path.join(
                    save_dir, f"match_{rank}_sim_{sim:.4f}_segmentation.png"
                )
                get_diagnostic_image(idx).save(match_diag_path)

        p3 = correct_count / 3.0
        print(f"📊 Localization Precision@3: {p3:.2%}")
        return p3

    # Print and Evaluate
    p3_cls = print_matches_and_eval(
        "STANDARD [CLS]", top_cls, cls_dir, save_masks=False
    )
    p3_simple = print_matches_and_eval(
        "[SIMPLE-AVERAGE]", top_simple, simple_dir, save_masks=False
    )
    p3_bg = print_matches_and_eval(
        "[BACKGROUND-AVERAGE]", top_bg, bg_dir, save_masks=True
    )
    p3_concat_simple = print_matches_and_eval(
        "[CLS + SIMPLE-AVG CONCAT]",
        top_concat_simple,
        concat_simple_dir,
        save_masks=False,
    )
    p3_concat_bg = print_matches_and_eval(
        "[CLS + BG-AVG CONCAT]", top_concat_bg, concat_bg_dir, save_masks=False
    )

    print("\n" + "=" * 50)
    print("📈 FINAL SUMMARY OF CAMERA TRAP LOCALIZATION ACCURACY (PRECISION@3):")
    print(f" - Standard CLS:                     {p3_cls:.1%}")
    print(f" - Simple Average:                   {p3_simple:.1%}")
    print(f" - Background Average (Masked):      {p3_bg:.1%}")
    print(f" - CLS + Simple-Avg Concat:          {p3_concat_simple:.1%}")
    print(f" - CLS + Background-Avg Concat:      {p3_concat_bg:.1%}")
    print("=" * 50)

    print(f"\nExperiment output saved successfully to: {exp_dir}")


if __name__ == "__main__":
    main()
