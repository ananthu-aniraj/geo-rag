import argparse
import math
import os
import random
import re
import sys
import time

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from src.evaluation.metrics import compute_ap, compute_precision_at_k, compute_rr
from src.models import tips_image_encoder as image_encoder
from src.models.vision_model_inference import (
    extract_benchmark_features_single_pass,
    load_vision_model,
)
from src.utils import lucas_class_mapping
from src.utils.spatial_overlays import (
    get_crs_transformer,
    get_environmental_zone_label,
    get_eunis_label,
    lookup_raster_pixel,
)

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


def load_lucas_metadata(csv_path):
    """Loads LUCAS metadata and maps the 8-digit point ID to its land cover, land use, EUNIS class, and coordinates."""
    if not os.path.exists(csv_path):
        print(f"Error: Metadata file '{csv_path}' not found.")
        sys.exit(1)

    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    metadata = {}

    for _, row in df.iterrows():
        im_id = str(row["im_id"]).strip()
        match = re.search(r"(\d{8})", im_id)
        if match:
            clean_id = match.group(1)
        else:
            clean_id = im_id

        metadata[clean_id] = {
            "im_id": im_id,
            "lc_label": row.get("lc_label", ""),
            "lu_label": row.get("lu_label", ""),
            "eunis_class": row.get("eunis_class", ""),
            "lat": float(row.get("lat", 0.0)) if pd.notna(row.get("lat")) else 0.0,
            "lon": float(row.get("lon", 0.0)) if pd.notna(row.get("lon")) else 0.0,
        }

    return metadata, df


def map_lucas_coordinates_to_rasters(
    metadata_dict, eunis_raster_path=None, env_zones_raster_path=None
):
    """
    Overlays LUCAS point coordinates onto EUNIS and Environmental Zones GeoTIFF rasters
    and appends the resolved class names to each metadata record.
    """

    # 1. Handle EUNIS Raster mapping
    if eunis_raster_path:
        if os.path.exists(eunis_raster_path):
            print(
                f"Mapping LUCAS coordinates to EUNIS Ecosystem classes from: {eunis_raster_path}..."
            )
            try:
                with rasterio.open(eunis_raster_path) as r_ds:
                    transformer, has_axis_order = get_crs_transformer(r_ds.crs)

                    # Check for dynamic DBF mapping
                    dbf_path = os.path.splitext(eunis_raster_path)[0] + ".vat.dbf"
                    dynamic_mapping = {}
                    if os.path.exists(dbf_path):
                        try:
                            gdf = gpd.read_file(dbf_path)
                            val_col = next(
                                (c for c in gdf.columns if c.lower() == "value"), None
                            )
                            label_col = next(
                                (
                                    c
                                    for c in gdf.columns
                                    if c.lower()
                                    in ["maes_l2", "maes_level2", "class_name"]
                                ),
                                None,
                            )
                            if val_col and label_col:
                                for _, row in gdf.iterrows():
                                    val = int(float(str(row[val_col])))
                                    label = str(row[label_col]).strip()
                                    if label and label.lower() != "none":
                                        dynamic_mapping[val] = label
                        except Exception as ex:
                            print(
                                f"Warning: Failed to load DBF attribute mapping: {ex}"
                            )

                    for clean_id, item in metadata_dict.items():
                        lat, lon = item.get("lat", 0.0), item.get("lon", 0.0)
                        pixel_val = lookup_raster_pixel(
                            lat, lon, r_ds, transformer, has_axis_order
                        )
                        if (
                            pixel_val is None
                            or pixel_val == r_ds.nodata
                            or pixel_val <= 0
                        ):
                            item["eunis_raster_class"] = ""
                        else:
                            label = get_eunis_label(pixel_val, dynamic_mapping)
                            item["eunis_raster_class"] = (
                                label if label != "Unknown" else ""
                            )
            except Exception as e:
                print(f"Error mapping coordinates to EUNIS: {e}")
                for item in metadata_dict.values():
                    item["eunis_raster_class"] = ""
        else:
            print(
                f"Warning: EUNIS raster path '{eunis_raster_path}' does not exist. Skipping EUNIS raster evaluation."
            )
            for item in metadata_dict.values():
                item["eunis_raster_class"] = ""
    else:
        for item in metadata_dict.values():
            item["eunis_raster_class"] = ""

    # 2. Handle Environmental Zones mapping
    if env_zones_raster_path:
        if os.path.exists(env_zones_raster_path):
            print(
                f"Mapping LUCAS coordinates to Environmental Zones from: {env_zones_raster_path}..."
            )
            try:
                with rasterio.open(env_zones_raster_path) as r_ds:
                    transformer, has_axis_order = get_crs_transformer(r_ds.crs)

                    for clean_id, item in metadata_dict.items():
                        lat, lon = item.get("lat", 0.0), item.get("lon", 0.0)
                        pixel_val = lookup_raster_pixel(
                            lat, lon, r_ds, transformer, has_axis_order
                        )
                        if (
                            pixel_val is None
                            or pixel_val == r_ds.nodata
                            or pixel_val <= 0
                        ):
                            item["env_zone_class"] = ""
                        else:
                            label = get_environmental_zone_label(pixel_val)
                            item["env_zone_class"] = label if label != "Unknown" else ""
            except Exception as e:
                print(f"Error mapping coordinates to Environmental Zones: {e}")
                for item in metadata_dict.values():
                    item["env_zone_class"] = ""
        else:
            print(
                f"Warning: Environmental Zones raster path '{env_zones_raster_path}' does not exist. Skipping Environmental Zones evaluation."
            )
            for item in metadata_dict.values():
                item["env_zone_class"] = ""
    else:
        for item in metadata_dict.values():
            item["env_zone_class"] = ""


def get_class_lists(df):
    """Retrieves class lists from lucas_class_mapping if available, falling back to unique values in the metadata CSV."""
    lc_list, lu_list, eunis_list = [], [], []
    try:
        lc_list = list(lucas_class_mapping.lc1_class_mapping.values())
        lu_list = list(lucas_class_mapping.lu1_class_mapping.values())
        eunis_list = list(lucas_class_mapping.eunis_mapping.values())
        print("Successfully loaded class lists from lucas_class_mapping.")
    except ImportError:
        print(
            "Warning: lucas_class_mapping not found or could not be imported. Extracting unique classes from metadata CSV..."
        )
        if "lc_label" in df.columns:
            lc_list = sorted(df["lc_label"].dropna().unique().tolist())
        if "lu_label" in df.columns:
            lu_list = sorted(df["lu_label"].dropna().unique().tolist())
        if "eunis_class" in df.columns:
            eunis_list = sorted(df["eunis_class"].dropna().unique().tolist())

    return lc_list, lu_list, eunis_list


def encode_image_value_attention(model_image, img):
    """Extracts spatial value features from model vision encoder using the MaskCLIP values trick."""
    B, _, H, W = img.shape
    P = model_image.patch_size if hasattr(model_image, "patch_size") else 14
    new_H = math.ceil(H / P) * P
    new_W = math.ceil(W / P) * P

    if (H, W) != (new_H, new_W):
        img = F.interpolate(
            img, size=(new_H, new_W), mode="bicubic", align_corners=False
        )

    B, _, h_i, w_i = img.shape
    x = model_image.prepare_tokens_with_masks(img)

    num_register = getattr(model_image, "num_register_tokens", 1)
    all_blocks = list(model_image.blocks)
    for i, blk in enumerate(all_blocks):
        if i < len(all_blocks) - 1:
            x = blk(x)
        else:
            x_normed = blk.norm1(x)
            b_dim, n_dim, c_dim = x_normed.shape
            qkv = (
                blk.attn.qkv(x_normed)
                .reshape(
                    b_dim, n_dim, 3, blk.attn.num_heads, c_dim // blk.attn.num_heads
                )
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
    patch_tokens = x_val[:, 1 + num_register :, :]
    blocks_patches = patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
    return blocks_patches


def main():
    parser = argparse.ArgumentParser(
        description="LUCAS Representation Semantic Retrieval Benchmarking Suite."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/user/aaniraj/home/Documents/Projects/data/LUCAS2018/Sen4Map_Metadata_test.csv",
        help="Path to the Sen4Map_Metadata_test.csv file.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Path to the directory containing LUCAS images.",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=100,
        help="Number of query evaluations to run.",
    )
    parser.add_argument(
        "--num_database",
        type=int,
        default=500,
        help="Number of database images to search against (0 for all remaining).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="GPU batch size for feature extraction.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--tips_model_path",
        type=str,
        default=None,
        help="Path to the official TIPSv2 model checkpoint (.npz). If None, uses --model_name.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/tipsv2-b14",
        help="Hugging Face model identifier or timm model name.",
    )
    parser.add_argument(
        "--tips_model_variant",
        type=str,
        default="B",
        choices=["S", "B", "L", "So400m", "g"],
        help="Variant of the official TIPSv2 model.",
    )
    parser.add_argument(
        "--tips_low_res",
        action="store_true",
        help="Set image resolution to 224px instead of 448px.",
    )
    parser.add_argument(
        "--no_segformer",
        action="store_true",
        help="Skip SegFormer background segmentation.",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="./benchmark_results/lucas_report.txt",
        help="Path to write the report summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./benchmark_results/lucas_results.csv",
        help="Path to write detailed query CSV results.",
    )
    parser.add_argument(
        "--eunis_raster",
        type=str,
        default=None,
        help="Path to the EUNIS Ecosystem GeoTIFF raster file for spatial overlay.",
    )
    parser.add_argument(
        "--env_zones_raster",
        type=str,
        default=None,
        help="Path to the Metzger Environmental Zones GeoTIFF raster file for spatial overlay.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Load metadata and scan for local images
    metadata_dict, df = load_lucas_metadata(args.csv)
    lc_list, lu_list, eunis_list = get_class_lists(df)

    # Map coordinates to rasters if paths are provided
    map_lucas_coordinates_to_rasters(
        metadata_dict, args.eunis_raster, args.env_zones_raster
    )

    print(f"Scanning for images in {args.img_dir}...")
    matched_images = []
    for root, _, files in os.walk(args.img_dir):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                basename = os.path.splitext(filename)[0]
                match = re.search(r"(\d{8})", basename)
                if match:
                    point_num = match.group(1)
                    dir_match = re.search(
                        rf"{point_num}([WENS])", basename, re.IGNORECASE
                    )
                    direction = dir_match.group(1).upper() if dir_match else "Unknown"

                    if point_num in metadata_dict:
                        matched_images.append(
                            {
                                "path": os.path.join(root, filename),
                                "filename": filename,
                                "point_id": point_num,
                                "direction": direction,
                                "lc_label": metadata_dict[point_num]["lc_label"],
                                "lu_label": metadata_dict[point_num]["lu_label"],
                                "eunis_class": metadata_dict[point_num]["eunis_class"],
                                "eunis_raster_class": metadata_dict[point_num].get(
                                    "eunis_raster_class", ""
                                ),
                                "env_zone_class": metadata_dict[point_num].get(
                                    "env_zone_class", ""
                                ),
                                "lat": metadata_dict[point_num].get("lat", 0.0),
                                "lon": metadata_dict[point_num].get("lon", 0.0),
                            }
                        )

    if not matched_images:
        print("Error: No matching images found in the provided directory.")
        return

    print(f"Found {len(matched_images):,} matched image files in directory.")
    print(
        "Partitioning queries and database geographically to prevent spatial autocorrelation/point-level leakage..."
    )

    # 1. Map each image to an H3 Resolution 4 parent block (~11,000 km2)
    for item in matched_images:
        try:
            h3_res4 = h3.latlng_to_cell(float(item["lat"]), float(item["lon"]), 4)
        except Exception:
            h3_res4 = "unknown"
        item["parent_block"] = h3_res4

    # 2. Perform Greedy Block Stratification to ensure full class coverage (stratified by lc_label)
    class_to_blocks = {}
    for item in matched_images:
        blk = item["parent_block"]
        if blk == "unknown":
            continue
        cls = item["lc_label"]
        if cls not in class_to_blocks:
            class_to_blocks[cls] = set()
        class_to_blocks[cls].add(blk)

    sorted_classes = sorted(
        class_to_blocks.keys(), key=lambda c: len(class_to_blocks[c])
    )
    query_blocks = set()
    database_blocks = set()
    random.seed(args.seed)

    for cls in sorted_classes:
        blocks = list(class_to_blocks[cls])
        random.shuffle(blocks)

        assigned_q = sum(1 for b in blocks if b in query_blocks)
        assigned_db = sum(1 for b in blocks if b in database_blocks)

        total_class_blocks = len(blocks)
        target_q = (
            max(1, int(total_class_blocks * 0.20)) if total_class_blocks > 1 else 0
        )

        unassigned_blocks = [
            b for b in blocks if b not in query_blocks and b not in database_blocks
        ]

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
    all_blocks = set(
        item["parent_block"]
        for item in matched_images
        if item["parent_block"] != "unknown"
    )
    unassigned_all = all_blocks - query_blocks - database_blocks
    for b in unassigned_all:
        if random.random() < 0.20:
            query_blocks.add(b)
        else:
            database_blocks.add(b)

    query_candidates = [
        item for item in matched_images if item["parent_block"] in query_blocks
    ]
    database_candidates = [
        item for item in matched_images if item["parent_block"] in database_blocks
    ]

    unique_blocks = sorted(list(all_blocks))

    print(f" -> Found {len(unique_blocks)} unique H3 blocks.")
    print(
        f" -> Query block pool: {len(query_blocks)} blocks ({len(query_candidates):,} images)"
    )
    print(
        f" -> Database block pool: {len(database_blocks)} blocks ({len(database_candidates):,} images)"
    )

    # Group query candidates by point
    query_by_point = {}
    for item in query_candidates:
        pt_id = item["point_id"]
        if pt_id not in query_by_point:
            query_by_point[pt_id] = []
        query_by_point[pt_id].append(item)

    sorted_q_points = sorted(query_by_point.keys())
    random.shuffle(sorted_q_points)

    selected_queries = []
    max_q_dirs = (
        max(len(query_by_point[pt]) for pt in sorted_q_points) if sorted_q_points else 0
    )
    for dir_idx in range(max_q_dirs):
        for pt in sorted_q_points:
            if dir_idx < len(query_by_point[pt]):
                selected_queries.append(query_by_point[pt][dir_idx])

    # Group database candidates by point
    db_by_point = {}
    for item in database_candidates:
        pt_id = item["point_id"]
        if pt_id not in db_by_point:
            db_by_point[pt_id] = []
        db_by_point[pt_id].append(item)

    sorted_db_points = sorted(db_by_point.keys())
    random.shuffle(sorted_db_points)

    selected_db = []
    max_db_dirs = (
        max(len(db_by_point[pt]) for pt in sorted_db_points) if sorted_db_points else 0
    )
    for dir_idx in range(max_db_dirs):
        for pt in sorted_db_points:
            if dir_idx < len(db_by_point[pt]):
                selected_db.append(db_by_point[pt][dir_idx])

    # Determine subset counts
    if len(selected_queries) < args.num_queries:
        print(
            f"Warning: Only {len(selected_queries)} query candidates available in query blocks. Adjusting --num_queries."
        )
        args.num_queries = len(selected_queries)

    queries_meta = selected_queries[: args.num_queries]
    database_meta = (
        selected_db[: args.num_database] if args.num_database > 0 else selected_db
    )

    print(
        f"Split data into: {len(queries_meta)} Queries and {len(database_meta)} Database images."
    )
    if len(queries_meta) == 0 or len(database_meta) == 0:
        print("Error: Empty query or database split. Adjust your parameters.")
        return

    # 2. Initialize Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing models on {device}...")

    # Load TIPSv2 Model (either official checkpoint or Hugging Face)
    if args.tips_model_path:
        print(f"Loading official TIPSv2 checkpoint from: {args.tips_model_path}...")
        model_def = {
            "S": image_encoder.vit_small,
            "B": image_encoder.vit_base,
            "L": image_encoder.vit_large,
            "So400m": image_encoder.vit_so400m,
            "g": image_encoder.vit_giant2,
        }[args.tips_model_variant]

        ffn_layer = "swiglu" if args.tips_model_variant == "g" else "mlp"

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
        # Load from HF or timm via unified helper
        tipsv2, transform, image_size = load_vision_model(args.model_name, device)

    seg_processor = None
    seg_model = None
    if not args.no_segformer:
        print("Loading SegFormer model...")
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

    # 3. Setup representations dynamically
    model_label = (
        "TIPSv2"
        if (args.tips_model_path or "tipsv2" in args.model_name.lower())
        else os.path.basename(args.model_name)
    )
    if args.tips_model_path:
        representations = {
            f"{model_label} 1st CLS": {"query": [], "db": []},
            f"{model_label} 2nd CLS": {"query": [], "db": []},
            f"{model_label} Average Patch": {"query": [], "db": []},
            f"{model_label} 1st CLS + Avg Patch": {"query": [], "db": []},
        }
        if not args.no_segformer:
            representations[f"{model_label} Seg-Masked"] = {"query": [], "db": []}
    else:
        representations = {
            f"{model_label} CLS": {"query": [], "db": []},
            f"{model_label} Average Patch": {"query": [], "db": []},
            f"{model_label} CLS + Avg Patch": {"query": [], "db": []},
        }
        if not args.no_segformer:
            representations[f"{model_label} Seg-Masked"] = {"query": [], "db": []}

    transform = transform
    grid_size = 16 if (args.tips_model_path and args.tips_low_res) else 32
    num_patches = grid_size * grid_size

    def extract_features_batch(metadata_list, split_key):
        print(
            f"Extracting features for {len(metadata_list)} images ({split_key} split)..."
        )
        for i in tqdm(
            range(0, len(metadata_list), args.batch_size),
            desc=f"Extraction ({split_key})",
        ):
            batch_meta = metadata_list[i : i + args.batch_size]
            batch_imgs = []

            for item in batch_meta:
                try:
                    with Image.open(item["path"]) as img:
                        img_resized = img.resize((image_size, image_size)).convert(
                            "RGB"
                        )
                        batch_imgs.append(img_resized)
                except Exception as e:
                    print(f"Warning: Failed to load image '{item['path']}': {e}")

            if not batch_imgs:
                continue

            # A. SegFormer segmentation masks
            pred_masks = None
            if not args.no_segformer:
                inputs = seg_processor(images=batch_imgs, return_tensors="pt").to(
                    device
                )
                with torch.no_grad():
                    outputs = seg_model(**inputs)
                logits = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                pred_masks = logits.argmax(dim=1).cpu().numpy()

                # Free SegFormer GPU memory before running TIPSv2
                del inputs, outputs, logits

            # B. Feature extraction
            img_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
            with torch.no_grad():
                is_local = bool(args.tips_model_path)

                cls_out, patch_tokens_vals = extract_benchmark_features_single_pass(
                    tipsv2, img_tensors, is_local=is_local
                )
                patch_tokens_vals = patch_tokens_vals.reshape(
                    len(batch_imgs), -1, patch_tokens_vals.shape[-1]
                )
                curr_num_patches = patch_tokens_vals.shape[1]
                curr_grid_size = int(math.sqrt(curr_num_patches))
                curr_patch_size = image_size // curr_grid_size

                if is_local:
                    first_cls, second_cls = cls_out
                    representations[f"{model_label} 1st CLS"][split_key].extend(
                        first_cls
                    )
                    representations[f"{model_label} 2nd CLS"][split_key].extend(
                        second_cls
                    )
                else:
                    cls_tokens = cls_out
                    representations[f"{model_label} CLS"][split_key].extend(cls_tokens)

            # Free image tensors from GPU memory
            del img_tensors

            # Process patch poolings for each image in batch
            for idx in range(len(batch_imgs)):
                patch_tokens = patch_tokens_vals[idx]  # (num_patches, D)

                # Average Patch
                avg_patch = np.mean(patch_tokens, axis=0)
                representations[f"{model_label} Average Patch"][split_key].append(
                    avg_patch
                )

                # CLS + Average Patch Concatenation
                if args.tips_model_path:
                    cls_val = first_cls[idx]
                    representations[f"{model_label} 1st CLS + Avg Patch"][
                        split_key
                    ].append(np.concatenate([cls_val, avg_patch]))
                else:
                    cls_val = cls_tokens[idx]
                    representations[f"{model_label} CLS + Avg Patch"][split_key].append(
                        np.concatenate([cls_val, avg_patch])
                    )

                # Seg-Masked
                if not args.no_segformer:
                    pred_mask = pred_masks[idx]
                    keep_mask = np.ones_like(pred_mask, dtype=float)
                    for c in DISCARD_CLASSES:
                        keep_mask[pred_mask == c] = 0.0

                    # Downsample keep mask to grid_size x grid_size patch resolution
                    patch_weights = np.zeros((curr_grid_size, curr_grid_size))
                    for r in range(curr_grid_size):
                        for c in range(curr_grid_size):
                            patch_weights[r, c] = np.mean(
                                keep_mask[
                                    r * curr_patch_size : (r + 1) * curr_patch_size,
                                    c * curr_patch_size : (c + 1) * curr_patch_size,
                                ]
                            )
                    patch_weights_flat = patch_weights.flatten()[
                        :, np.newaxis
                    ]  # (num_patches, 1)

                    masked_patch_sum = np.sum(patch_tokens * patch_weights_flat, axis=0)
                    masked_patch_weight_sum = np.sum(patch_weights_flat)
                    if masked_patch_weight_sum > 0:
                        masked_avg_embed = masked_patch_sum / (
                            masked_patch_weight_sum + 1e-9
                        )
                    else:
                        masked_avg_embed = avg_patch

                    representations[f"{model_label} Seg-Masked"][split_key].append(
                        masked_avg_embed
                    )

            # Explicitly release image and GPU memory to keep RAM/VRAM flat
            for img in batch_imgs:
                img.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    extract_features_batch(queries_meta, "query")
    extract_features_batch(database_meta, "db")

    # 4. Retrieval & Similarity Benchmarking
    expanded_representations = {}
    for rep_name, splits in list(representations.items()):
        expanded_representations[f"{rep_name} (FP32)"] = splits

        q_fp16 = [v.astype(np.float16).astype(np.float32) for v in splits["query"]]
        db_fp16 = [v.astype(np.float16).astype(np.float32) for v in splits["db"]]
        expanded_representations[f"{rep_name} (FP16)"] = {
            "query": q_fp16,
            "db": db_fp16,
        }
    representations = expanded_representations

    label_types = ["lc_label", "lu_label", "eunis_class"]
    label_names = {
        "lc_label": "Land Cover",
        "lu_label": "Land Use",
        "eunis_class": "EUNIS Class (CSV)",
    }

    # Check if raster values were loaded and append them
    has_eunis_raster = any(q.get("eunis_raster_class", "") != "" for q in queries_meta)
    if has_eunis_raster:
        label_types.append("eunis_raster_class")
        label_names["eunis_raster_class"] = "EUNIS Ecosystem (Raster)"

    has_env_zones = any(q.get("env_zone_class", "") != "" for q in queries_meta)
    if has_env_zones:
        label_types.append("env_zone_class")
        label_names["env_zone_class"] = "Environmental Zones (Raster)"

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
                "mrr@10": 0.0,
            }

        # Compute all similarities in a single batched operation: (Q, D_dim) x (D_dim, DB) -> (Q, DB)
        sim_matrix = np.dot(q_vectors_norm, db_vectors_norm.T)
        # Retrieve top 10 database indices for all queries at once
        top_indices = np.argsort(-sim_matrix, axis=1)[:, :10]

        # Query loop
        for q_idx in range(len(queries_meta)):
            q_item = queries_meta[q_idx]
            sorted_db_indices = top_indices[q_idx]

            for l_type in label_types:
                q_label = q_item[l_type]
                if pd.isna(q_label) or q_label == "":
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

                detailed_rows.append(
                    {
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
                        "RR@10": rr,
                    }
                )

        # Normalize metrics by query count
        for l_type in label_types:
            valid_queries = sum(
                1 for q in queries_meta if pd.notna(q[l_type]) and q[l_type] != ""
            )
            if valid_queries > 0:
                results[rep_name][l_type]["p@1"] = (
                    results[rep_name][l_type]["p@1"] / valid_queries
                ) * 100.0
                results[rep_name][l_type]["p@5"] = (
                    results[rep_name][l_type]["p@5"] / valid_queries
                ) * 100.0
                results[rep_name][l_type]["p@10"] = (
                    results[rep_name][l_type]["p@10"] / valid_queries
                ) * 100.0
                results[rep_name][l_type]["map@10"] = (
                    results[rep_name][l_type]["map@10"] / valid_queries
                ) * 100.0
                results[rep_name][l_type]["mrr@10"] = (
                    results[rep_name][l_type]["mrr@10"] / valid_queries
                ) * 100.0

    # 5. Compile and Print Report
    report_lines = []
    report_lines.append("LUCAS 2018 Image Representation Semantic Retrieval Report")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"- Queries count: {len(queries_meta)} images")
    report_lines.append(f"- Database count: {len(database_meta)} images")
    report_lines.append("")

    print("\n" + "=" * 95)
    print("                    LUCAS 2018 SEMANTIC RETRIEVAL BENCHMARK REPORT")
    print("=" * 95)

    for l_type in label_types:
        print(f"\n{label_names[l_type]} Evaluation:")

        row_format = "{:<24} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}"
        header = row_format.format(
            "Representation",
            "P@1 (%)",
            "P@5 (%)",
            "P@10 (%)",
            "mAP@10 (%)",
            "MRR@10 (%)",
        )
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
                f"{metrics['mrr@10']:.1f}%",
            )
            print(row_str)
            report_lines.append(row_str)

        print("-" * 90)
        report_lines.append("")

    print("=" * 95)

    # Save TXT report summary
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(
        f"\nBenchmark report saved successfully to: {os.path.abspath(args.output_report)}"
    )

    # Save detailed CSV results
    if detailed_rows:
        df_detailed = pd.DataFrame(detailed_rows)
        df_detailed.to_csv(args.output_csv, index=False)
        print(f"Detailed query results saved to: {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
