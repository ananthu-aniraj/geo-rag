import argparse
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from src.models.vision_model_inference import (
    extract_benchmark_features_single_pass,
    load_vision_model,
)
from src.utils.io import download_image, load_dataframe

# Try to load .env variables if not already set
if not os.environ.get("MAPILLARY_TOKEN") and os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "MAPILLARY_TOKEN":
                        os.environ["MAPILLARY_TOKEN"] = v.strip().strip('"').strip("'")
                        break
    except Exception:
        pass

MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "")
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

DEFAULT_CONFIG_PATH = "config/evaluation/benchmark_representations.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads YAML configuration with sensible defaults."""
    cfg: Dict[str, Any] = {
        "model": {
            "name": "google/tipsv2-b14",
            "checkpoint_path": None,
        },
        "dataset": {
            "input_path": "full_pipeline_output/geo_space_deduplicated.parquet",
            "offline_dataset_dirs": "",
            "num_images": 300,
            "num_queries": 50,
            "seed": 42,
            "batch_size": 16,
        },
        "options": {
            "enable_pca": False,
            "enable_fp16": True,
            "use_segformer": True,
            "fg_attn_threshold": 2.0,
            "max_fg_ratio": 0.05,
        },
        "output": {
            "output_dir": "./benchmark_results",
            "output_report": "representations_report.txt",
            "output_csv": "representations_results.csv",
            "save_plots": True,
        },
        "representations": [
            "cls",
            "unmasked_avg",
            "cls_attn_fg_removed",
            "cls_plus_fg_removed_concat",
            "segformer_masked",
            "tips_ade_masked",
            "cls_tips_ade_concat",
            "cls_unmasked_concat",
        ],
    }

    path_to_try = config_path or DEFAULT_CONFIG_PATH
    if path_to_try and os.path.exists(path_to_try):
        try:
            with open(path_to_try, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    for section, values in loaded.items():
                        if (
                            isinstance(values, dict)
                            and section in cfg
                            and isinstance(cfg[section], dict)
                        ):
                            cfg[section].update(values)
                        else:
                            cfg[section] = values
            print(f"Loaded configuration from: {path_to_try}")
        except Exception as e:
            print(
                f"Warning: Failed to parse configuration file '{path_to_try}': {e}. Using defaults."
            )
    return cfg


def resolve_dataset_path(path: str) -> str:
    """Finds an existing dataset path checking common variations (.parquet, .csv)."""
    if os.path.exists(path):
        return path

    candidates = [
        path.replace(".parquet", ".csv")
        if path.endswith(".parquet")
        else path.replace(".csv", ".parquet"),
        "full_pipeline_output/geo_space_deduplicated.parquet",
        "full_pipeline_output/geo_space_deduplicated.csv",
        "full_pipeline_output/geo_space_cleaned.parquet",
        "full_pipeline_output/geo_space_cleaned.csv",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return path


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names flexibly across legacy and new dataset formats."""
    cols_lower = {c.lower(): c for c in df.columns}

    lat_col = cols_lower.get("latitude") or cols_lower.get("lat")
    lon_col = cols_lower.get("longitude") or cols_lower.get("lon")
    url_col = cols_lower.get("image_url") or cols_lower.get("url")
    id_col = (
        cols_lower.get("photo_id")
        or cols_lower.get("photo_key")
        or cols_lower.get("id")
    )
    platform_col = cols_lower.get("platform")
    country_col = cols_lower.get("country")

    if not lat_col or not lon_col:
        raise ValueError("Database must contain latitude and longitude columns.")

    standard_df = pd.DataFrame()
    standard_df["Latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    standard_df["Longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    standard_df["Image_URL"] = df[url_col].astype(str) if url_col else ""
    standard_df["Photo_ID"] = df[id_col].astype(str) if id_col else ""
    standard_df["Platform"] = (
        df[platform_col].astype(str) if platform_col else "unknown"
    )
    standard_df["country"] = df[country_col].astype(str) if country_col else "Unknown"

    standard_df = standard_df.dropna(subset=["Latitude", "Longitude"]).reset_index(
        drop=True
    )
    return standard_df


def extract_cls_attention_maps(model, batch_tensors, is_local=False):
    """
    Dedicated extraction of [CLS]-to-patch attention weights from the final transformer layer.
    Only executed when foreground-filtering representations are explicitly requested.
    """
    try:
        # Case 1: TIPSv2 (HF wrapper or local checkpoint)
        if is_local or hasattr(model, "vision_encoder"):
            vision_encoder = (
                model
                if is_local
                else (
                    model.vision_encoder if hasattr(model, "vision_encoder") else model
                )
            )
            all_blocks = list(vision_encoder.blocks)
            x = vision_encoder.prepare_tokens_with_masks(batch_tensors)
            num_register = getattr(vision_encoder, "num_register_tokens", 1)
            for blk in all_blocks[:-1]:
                x = blk(x)
            last_blk = all_blocks[-1]
            x_normed = last_blk.norm1(x)
            b_dim, n_dim, c_dim = x_normed.shape
            num_heads = last_blk.attn.num_heads
            head_dim = c_dim // num_heads
            qkv = (
                last_blk.attn.qkv(x_normed)
                .reshape(b_dim, n_dim, 3, num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q_cls = qkv[0][:, :, 0:1, :]
            k = qkv[1]
            scale = 1.0 / math.sqrt(head_dim)
            attn = F.softmax(torch.matmul(q_cls, k.transpose(-2, -1)) * scale, dim=-1)
            # Average across heads, take attention from CLS to patch tokens (after CLS + registers)
            return attn[:, :, 0, 1 + num_register :].mean(dim=1).cpu().numpy()

        # Case 2: timm ViT models with blocks
        elif hasattr(model, "blocks") and len(model.blocks) > 0:
            captured_inputs = []
            last_attn = model.blocks[-1].attn
            hook = last_attn.register_forward_hook(
                lambda mod, inp, out: captured_inputs.append(inp[0])
            )
            try:
                with torch.no_grad():
                    model.forward_features(batch_tensors)
            finally:
                hook.remove()

            if captured_inputs:
                inp = captured_inputs[0]
                B, N, C = inp.shape
                qkv = (
                    last_attn.qkv(inp)
                    .reshape(B, N, 3, last_attn.num_heads, last_attn.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                q_cls = qkv[0][:, :, 0:1, :]
                k = qkv[1]
                scale = getattr(last_attn, "scale", 1.0 / math.sqrt(last_attn.head_dim))
                attn = F.softmax(
                    torch.matmul(q_cls, k.transpose(-2, -1)) * scale, dim=-1
                )
                num_prefix = getattr(model, "num_prefix_tokens", 1)
                return attn[:, :, 0, num_prefix:].mean(dim=1).cpu().numpy()
    except Exception as e:
        print(f"Warning: Failed to extract CLS attention weights: {e}")

    return None


def snap_mask_with_pca(seg_keep, pca_rgb, k_segments=4, use_gpu=True):
    """Refines/snaps a segment keep-mask to high-resolution visual boundaries from the upscaled PCA map."""
    try:
        import faiss
    except ImportError:
        return seg_keep

    h, w, c = pca_rgb.shape
    pixels = pca_rgb.reshape(-1, 3).astype(np.float32) / 255.0

    d = 3
    kmeans = faiss.Kmeans(d, k_segments, niter=10, verbose=False, gpu=use_gpu, seed=42)
    kmeans.train(pixels)
    _, labels_flat = kmeans.index.search(pixels, 1)
    labels = labels_flat.ravel().reshape(h, w)
    discard_mask = 1.0 - seg_keep
    snapped_keep = np.ones_like(seg_keep)

    for r in range(k_segments):
        segment_mask = labels == r
        segment_size = np.sum(segment_mask)
        if segment_size == 0:
            continue
        overlap = np.sum(segment_mask & (discard_mask == 1.0)) / segment_size
        if overlap >= 0.65:
            snapped_keep[segment_mask] = 0.0

    return snapped_keep


def haversine_distance(lat1, lon1, lat2, lon2):
    """Computes distance between coordinates in kilometers (vectorized)."""
    deg_to_rad = np.pi / 180.0
    phi1 = np.asarray(lat1, dtype=float) * deg_to_rad
    phi2 = np.asarray(lat2, dtype=float) * deg_to_rad
    dphi = (np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float)) * deg_to_rad
    dlambda = (
        np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float)
    ) * deg_to_rad

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(
        np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0))
    )
    return 6371.0 * c


def main():
    parser = argparse.ArgumentParser(
        description="Representation & Layout Retrieval Benchmarking Suite."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Hugging Face model identifier or timm model name. Overrides config.",
    )
    parser.add_argument(
        "--tips_model_path",
        type=str,
        default=None,
        help="Path to local TIPSv2 checkpoint (.npz or .pt). Overrides config.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to database file (CSV or Parquet). Overrides config.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Total database images to download/load. Overrides config.",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Number of query evaluations to run. Overrides config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="GPU batch size for feature extraction. Overrides config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Overrides config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save benchmark results. Overrides config.",
    )
    parser.add_argument(
        "--enable_pca",
        action="store_true",
        default=None,
        help="Enable PCA-based representations (AnyUp-PCA & Global PCA Layout). Overrides config.",
    )
    parser.add_argument(
        "--disable_pca",
        action="store_true",
        help="Disable PCA representations even if enabled in config.",
    )
    parser.add_argument(
        "--no_segformer",
        action="store_true",
        help="Skip SegFormer semantic segmentation masking.",
    )
    parser.add_argument(
        "--fg_attn_threshold",
        type=float,
        default=None,
        help="Foreground attention threshold multiplier (over uniform 1/N). Overrides config.",
    )
    parser.add_argument(
        "--max_fg_ratio",
        type=float,
        default=None,
        help="Maximum foreground ratio allowed to be removed (e.g. 0.05 for 1/20). Overrides config.",
    )
    parser.add_argument(
        "--offline_dataset_dirs",
        type=str,
        default=None,
        help="Space-separated paths to local image dirs (e.g. iWildCam).",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Resolve model parameters
    model_name = args.model_name or cfg.get("model", {}).get(
        "name", "google/tipsv2-b14"
    )
    tips_model_path = args.tips_model_path or cfg.get("model", {}).get(
        "checkpoint_path"
    )
    model_label = (
        "TIPSv2"
        if (tips_model_path or "tipsv2" in model_name.lower())
        else os.path.basename(model_name)
    )

    # CLI overrides
    input_path = args.csv_path or cfg["dataset"].get(
        "input_path", "full_pipeline_output/geo_space_deduplicated.parquet"
    )
    num_images = (
        args.num_images
        if args.num_images is not None
        else cfg["dataset"].get("num_images", 300)
    )
    num_queries = (
        args.num_queries
        if args.num_queries is not None
        else cfg["dataset"].get("num_queries", 50)
    )
    seed = args.seed if args.seed is not None else cfg["dataset"].get("seed", 42)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else cfg["dataset"].get("batch_size", 16)
    )
    offline_dirs_str = args.offline_dataset_dirs or cfg["dataset"].get(
        "offline_dataset_dirs", ""
    )

    enable_pca = cfg["options"].get("enable_pca", False)
    if args.enable_pca is True:
        enable_pca = True
    if args.disable_pca:
        enable_pca = False

    use_segformer = cfg["options"].get("use_segformer", True)
    if args.no_segformer:
        use_segformer = False

    enable_fp16 = cfg["options"].get("enable_fp16", True)
    fg_attn_threshold = (
        args.fg_attn_threshold
        if args.fg_attn_threshold is not None
        else cfg["options"].get("fg_attn_threshold", 2.0)
    )
    max_fg_ratio = (
        args.max_fg_ratio
        if args.max_fg_ratio is not None
        else cfg["options"].get("max_fg_ratio", 0.05)
    )

    output_dir = args.output_dir or cfg["output"].get(
        "output_dir", "./benchmark_results"
    )
    output_report_file = cfg["output"].get(
        "output_report", "representations_report.txt"
    )
    output_csv_file = cfg["output"].get("output_csv", "representations_results.csv")
    save_plots = cfg["output"].get("save_plots", True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("      Starting Representation & Layout Retrieval Benchmarking Suite")
    print("=" * 80)
    print(
        f"Target Model: {model_name} ({'Local Checkpoint: ' + tips_model_path if tips_model_path else 'Hub/timm'})"
    )
    print(f"Target Database: {input_path}")
    print(f"Images to Sample: {num_images} (Queries: {num_queries})")
    print(f"Batch Size: {batch_size}, Seed: {seed}")
    print(
        f"PCA Enabled: {enable_pca}, SegFormer Enabled: {use_segformer}, FP16: {enable_fp16}"
    )
    print(
        f"Foreground Filtering: Threshold Multiplier={fg_attn_threshold}x, Max Ratio={max_fg_ratio:.1%}"
    )

    # Load Vision Model via unified loader
    is_local = bool(tips_model_path)
    if is_local:
        print(f"Loading local checkpoint from: {tips_model_path}...")
        try:
            from src.indexing.label_clusters_mllm import ImageEncoder

            state_dict = np.load(tips_model_path, allow_pickle=True)
            checkpoint = {k: torch.tensor(v) for k, v in state_dict.items()}
            v_model = ImageEncoder(
                embed_dim=768, depth=12, num_heads=12, patch_size=14, in_chans=3
            )
            v_model.load_state_dict(checkpoint)
            vision_model = v_model.eval().to(device)
            image_size = 448
            from torchvision import transforms

            transform = transforms.Compose(
                [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
            )
        except Exception as e:
            print(
                f"Failed to load local checkpoint: {e}. Falling back to default loader..."
            )
            vision_model, transform, image_size = load_vision_model(model_name, device)
    else:
        vision_model, transform, image_size = load_vision_model(model_name, device)

    # Check model capabilities
    has_text_encoder = hasattr(vision_model, "encode_text")
    has_self_attn = (
        is_local
        or hasattr(vision_model, "vision_encoder")
        or (
            hasattr(vision_model, "blocks")
            and len(vision_model.blocks) > 0
            and hasattr(vision_model.blocks[-1], "attn")
        )
    )

    # Representation mapping
    rep_key_to_name = {
        "cls": f"{model_label} CLS",
        "unmasked_avg": f"{model_label} Unmasked Patch Average",
        "cls_attn_fg_removed": f"{model_label} CLS-Attn FG-Removed Average",
        "cls_plus_fg_removed_concat": f"{model_label} CLS + FG-Removed Average (Concat)",
        "segformer_masked": f"{model_label} Segformer-Masked Average",
        "tips_ade_masked": f"{model_label} Zero-Shot ADE150-Masked Average",
        "anyup_snapped": f"{model_label} AnyUp-PCA Snapped Mask Average",
        "cls_tips_ade_concat": f"{model_label} CLS + Zero-Shot ADE150-Masked (Concat)",
        "cls_unmasked_concat": f"{model_label} CLS + Unmasked Average (Concat)",
        "hybrid_land_use": f"{model_label} Hybrid Land Use Signature",
    }

    # Filter representations based on options and model capabilities
    configured_reps = cfg.get("representations", list(rep_key_to_name.keys()))
    selected_reps = []
    for r in configured_reps:
        r_clean = r.strip().lower()
        if not enable_pca and r_clean in ["anyup_snapped", "hybrid_land_use"]:
            print(f" -> Skipping representation '{r_clean}' because PCA is disabled.")
            continue
        if not use_segformer and r_clean in ["segformer_masked", "anyup_snapped"]:
            print(
                f" -> Skipping representation '{r_clean}' because SegFormer is disabled."
            )
            continue
        if not has_text_encoder and r_clean in [
            "tips_ade_masked",
            "cls_tips_ade_concat",
            "hybrid_land_use",
        ]:
            print(
                f" -> Skipping representation '{r_clean}' because model '{model_name}' has no text encoder (encode_text)."
            )
            continue
        if not has_self_attn and r_clean in [
            "cls_attn_fg_removed",
            "cls_plus_fg_removed_concat",
        ]:
            print(
                f" -> Skipping representation '{r_clean}' because model '{model_name}' has no transformer self-attention blocks."
            )
            continue
        if r_clean in rep_key_to_name:
            selected_reps.append(r_clean)
        else:
            print(f" -> Warning: Unknown representation key '{r}'. Ignoring.")

    if not selected_reps:
        print("Error: No valid representations selected for benchmarking.")
        sys.exit(1)

    print("Active Representations:")
    for r in selected_reps:
        print(f"  - {r}: {rep_key_to_name[r]}")
    print("=" * 80)

    # Check if dedicated CLS attention extraction is needed
    need_cls_attn = any(
        r in ["cls_attn_fg_removed", "cls_plus_fg_removed_concat"]
        for r in selected_reps
    )

    # 1. Load dataset with io utils
    resolved_path = resolve_dataset_path(input_path)
    if not os.path.exists(resolved_path):
        print(f"Error: Database file not found at: {resolved_path}")
        sys.exit(1)

    print(f"Loading metadata from {resolved_path} using src.utils.io.load_dataframe...")
    raw_df = load_dataframe(resolved_path)
    df = normalize_columns(raw_df)
    print(f"Successfully loaded and standardized {len(df)} database records.")

    # 2. Sample balanced subset of Flickr and Mapillary images if possible
    flickr_df = df[df["Platform"].str.lower() == "flickr"]
    mapillary_df = df[df["Platform"].str.lower() == "mapillary"]

    half_size = num_images // 2
    if len(flickr_df) >= half_size and len(mapillary_df) >= half_size:
        sampled_df = (
            pd.concat(
                [
                    flickr_df.sample(half_size, random_state=seed),
                    mapillary_df.sample(half_size, random_state=seed),
                ]
            )
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    else:
        print(
            "Warning: Insufficient platform records for balanced sampling. Using random sampling."
        )
        sampled_df = df.sample(min(num_images, len(df)), random_state=seed).reset_index(
            drop=True
        )

    # 3. Load or download images using src.utils.io.download_image
    offline_dirs = (
        [d.strip() for d in offline_dirs_str.split() if d.strip()]
        if offline_dirs_str
        else None
    )

    print(
        f"Downloading/loading {len(sampled_df)} images for benchmarking in parallel..."
    )
    images = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(
                download_image,
                url=row["Image_URL"],
                mapillary_token=MAPILLARY_TOKEN,
                photo_id=row.get("Photo_ID"),
                platform=row.get("Platform"),
                offline_dirs=offline_dirs,
                image_size=image_size,
            ): idx
            for idx, row in sampled_df.iterrows()
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading Images"
        ):
            idx = futures[future]
            img = future.result()
            if img:
                images[idx] = img

    print(f"Successfully loaded {len(images)} images.")
    if len(images) < 10:
        print("Error: Too few images successfully loaded to run benchmark.")
        return

    # Keep only successfully loaded items
    active_indices = sorted(list(images.keys()))
    sampled_df = sampled_df.iloc[active_indices].reset_index(drop=True)
    images = {i: images[idx] for i, idx in enumerate(active_indices)}

    # Load SegFormer only if needed
    need_segformer = use_segformer and any(
        r in ["segformer_masked", "anyup_snapped"] for r in selected_reps
    )
    seg_model = None
    seg_processor = None
    if need_segformer:
        print("Loading SegFormer model (nvidia/segformer-b0-finetuned-ade-512-512)...")
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

    # Load AnyUp only if PCA is enabled and anyup_snapped is selected
    anyup_model = None
    if enable_pca and "anyup_snapped" in selected_reps:
        print("Loading AnyUp model for feature upsampling...")
        try:
            from anyup import anyup_multi_backbone

            anyup_model = anyup_multi_backbone(
                use_natten=False, pretrained=True, device=device
            ).eval()
            print(" -> AnyUp loaded successfully.")
        except Exception as e:
            print(
                f" -> [WARNING] Failed to load AnyUp model locally: {e}. Trying torch.hub..."
            )
            try:
                anyup_model = torch.hub.load(
                    "wimmerth/anyup",
                    "anyup_multi_backbone",
                    use_natten=False,
                    pretrained=True,
                    device=device,
                ).eval()
                print(" -> AnyUp loaded successfully via torch.hub.")
            except Exception as e_hub:
                print(
                    f" -> [WARNING] Torch Hub AnyUp load failed: {e_hub}. Falling back to standard resizing."
                )
                anyup_model = None

    # Zero-shot ADE150 embeddings if model supports encode_text
    need_ade_text = has_text_encoder and any(
        r in ["tips_ade_masked", "cls_tips_ade_concat", "hybrid_land_use"]
        for r in selected_reps
    )
    ade_text_embeds_norm = None
    if need_ade_text:
        print("Pre-encoding 150 ADE20K semantic labels for zero-shot masking...")
        ade_labels = [
            f"ade20k class {i}"
            if seg_model is None
            else seg_model.config.id2label.get(i, f"class {i}")
            for i in range(150)
        ]
        with torch.no_grad():
            ade_text_embeds = vision_model.encode_text(ade_labels)
            ade_text_embeds_np = ade_text_embeds.cpu().numpy()
            ade_text_embeds_norm = ade_text_embeds_np / (
                np.linalg.norm(ade_text_embeds_np, axis=1, keepdims=True) + 1e-9
            )

    # Storage for extracted features
    features_dict = {r: [] for r in selected_reps}
    raw_patch_tokens_list = []
    tips_ade_keep_masks_list = []
    seg_viz_samples = []

    print("Extracting spatial representations in batches...")
    for i in tqdm(range(0, len(images), batch_size), desc="Feature Extraction"):
        batch_keys = list(range(i, min(i + batch_size, len(images))))
        batch_imgs = [images[k] for k in batch_keys]

        # Segformer pass if required
        pred_masks = None
        if seg_model and seg_processor:
            inputs = seg_processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = seg_model(**inputs)
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = logits.argmax(dim=1).cpu().numpy()

        # Model feature extractions
        img_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            cls_out, patch_tokens_vals = extract_benchmark_features_single_pass(
                vision_model, img_tensors, is_local=is_local
            )
            if is_local:
                cls_tokens = cls_out[0]
            else:
                cls_tokens = cls_out
            if cls_tokens.ndim == 3:
                cls_tokens = cls_tokens.squeeze(1)

            patch_tokens_vals = patch_tokens_vals.reshape(
                len(batch_keys), -1, patch_tokens_vals.shape[-1]
            )
            curr_num_patches = patch_tokens_vals.shape[1]
            curr_grid_size = (
                int(math.sqrt(curr_num_patches)) if curr_num_patches > 0 else 1
            )
            curr_patch_size = max(1, image_size // curr_grid_size)

        # Dedicated CLS attention extraction (only if requested)
        cls_attn_batch = None
        if need_cls_attn:
            cls_attn_batch = extract_cls_attention_maps(
                vision_model, img_tensors, is_local=is_local
            )

        # Process each image in batch
        for batch_i in range(len(batch_keys)):
            cls_token = cls_tokens[batch_i]
            patch_tokens = patch_tokens_vals[batch_i]
            if enable_pca:
                raw_patch_tokens_list.append(patch_tokens)

            cls_norm = cls_token / (np.linalg.norm(cls_token) + 1e-9)
            simple_avg = np.mean(patch_tokens, axis=0)
            simple_norm = simple_avg / (np.linalg.norm(simple_avg) + 1e-9)

            if "cls" in selected_reps:
                features_dict["cls"].append(cls_norm)
            if "unmasked_avg" in selected_reps:
                features_dict["unmasked_avg"].append(simple_norm)

            # CLS Attention Foreground Removal (Size-Gated)
            fg_mask_grid = None
            if cls_attn_batch is not None:
                patch_attn = cls_attn_batch[batch_i]
                norm_attn = patch_attn / (np.sum(patch_attn) + 1e-9)
                uniform_attn = 1.0 / curr_num_patches
                fg_thresh = fg_attn_threshold * uniform_attn
                fg_indices = np.where(norm_attn > fg_thresh)[0]
                num_fg = len(fg_indices)

                max_fg_patches = int(math.floor(max_fg_ratio * curr_num_patches))

                if 1 <= num_fg <= max_fg_patches:
                    fg_bool = np.zeros(curr_num_patches, dtype=bool)
                    fg_bool[fg_indices] = True
                    bg_indices = np.where(~fg_bool)[0]
                    bg_patch_avg = np.mean(patch_tokens[bg_indices], axis=0)
                    fg_mask_grid = (~fg_bool).astype(float)
                else:
                    bg_patch_avg = simple_avg
                    fg_mask_grid = np.ones(curr_num_patches, dtype=float)

                bg_patch_norm = bg_patch_avg / (np.linalg.norm(bg_patch_avg) + 1e-9)
                if "cls_attn_fg_removed" in selected_reps:
                    features_dict["cls_attn_fg_removed"].append(bg_patch_norm)

            # Segformer masking
            keep_mask = None
            patch_weights = None
            if pred_masks is not None:
                pred_mask = pred_masks[batch_i]
                keep_mask = np.ones_like(pred_mask, dtype=float)
                for c in DISCARD_CLASSES:
                    keep_mask[pred_mask == c] = 0.0

                patch_weights = np.zeros((curr_grid_size, curr_grid_size))
                for r_idx in range(curr_grid_size):
                    for c_idx in range(curr_grid_size):
                        patch_weights[r_idx, c_idx] = np.mean(
                            keep_mask[
                                r_idx * curr_patch_size : (r_idx + 1) * curr_patch_size,
                                c_idx * curr_patch_size : (c_idx + 1) * curr_patch_size,
                            ]
                        )
                patch_weights_flat = patch_weights.flatten()[:, np.newaxis]
                total_seg_weight = np.sum(patch_weights_flat)
                seg_avg = (
                    np.sum(patch_tokens * patch_weights_flat, axis=0) / total_seg_weight
                    if total_seg_weight > 0
                    else simple_avg
                )
                seg_norm = seg_avg / (np.linalg.norm(seg_avg) + 1e-9)
                if "segformer_masked" in selected_reps:
                    features_dict["segformer_masked"].append(seg_norm)

            # Zero-shot text ADE20K masking (if model has text encoder)
            tips_ade_keep_mask = None
            if ade_text_embeds_norm is not None:
                norm_patches = patch_tokens / (
                    np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-9
                )
                patch_ade_sim = np.dot(norm_patches, ade_text_embeds_norm.T)
                best_ade_idx = np.argmax(patch_ade_sim, axis=1)

                tips_ade_keep_mask = np.ones(curr_num_patches, dtype=float)
                for c in DISCARD_CLASSES:
                    tips_ade_keep_mask[best_ade_idx == c] = 0.0
                tips_ade_keep_masks_list.append(tips_ade_keep_mask)

                tips_ade_keep_mask_flat = tips_ade_keep_mask[:, np.newaxis]
                total_tips_ade_weight = np.sum(tips_ade_keep_mask_flat)
                tips_ade_avg = (
                    np.sum(patch_tokens * tips_ade_keep_mask_flat, axis=0)
                    / total_tips_ade_weight
                    if total_tips_ade_weight > 0
                    else simple_avg
                )
                tips_ade_norm = tips_ade_avg / (np.linalg.norm(tips_ade_avg) + 1e-9)
                if "tips_ade_masked" in selected_reps:
                    features_dict["tips_ade_masked"].append(tips_ade_norm)

            # Local PCA + AnyUp snapping (only if PCA enabled and anyup_snapped requested)
            snapped_keep = None
            pca_rgb = None
            if (
                enable_pca
                and "anyup_snapped" in selected_reps
                and keep_mask is not None
            ):
                with torch.no_grad():
                    patches_t = torch.tensor(patch_tokens, device=device)
                    patches_centered = patches_t - patches_t.mean(dim=0)
                    _, _, V_local = torch.pca_lowrank(
                        patches_centered, q=3, center=False
                    )
                    proj_local = torch.matmul(patches_centered, V_local)
                    p_min = proj_local.min(dim=0, keepdim=True).values
                    p_max = proj_local.max(dim=0, keepdim=True).values
                    p_norm = (proj_local - p_min) / (p_max - p_min + 1e-9)
                    pca_3d = p_norm.reshape(
                        1, curr_grid_size, curr_grid_size, 3
                    ).permute(0, 3, 1, 2)

                    if anyup_model is not None:
                        try:
                            hr_img_t = img_tensors[batch_i : batch_i + 1]
                            mean = torch.tensor(
                                [0.485, 0.456, 0.406], device=device
                            ).view(1, 3, 1, 1)
                            std = torch.tensor(
                                [0.229, 0.224, 0.225], device=device
                            ).view(1, 3, 1, 1)
                            hr_img_norm = (hr_img_t - mean) / std
                            hr_pca_t = (
                                anyup_model(hr_img_norm, pca_3d)
                                .squeeze(0)
                                .permute(1, 2, 0)
                            )
                            h_min = hr_pca_t.min()
                            h_max = hr_pca_t.max()
                            hr_pca_t = (hr_pca_t - h_min) / (h_max - h_min + 1e-9)
                            pca_rgb = (hr_pca_t.cpu().numpy() * 255).astype(np.uint8)
                            snapped_keep = snap_mask_with_pca(
                                keep_mask, pca_rgb, k_segments=4
                            )
                        except Exception:
                            pca_rgb = (
                                (p_norm.cpu().numpy() * 255)
                                .astype(np.uint8)
                                .reshape(curr_grid_size, curr_grid_size, 3)
                            )
                            snapped_keep = keep_mask
                    else:
                        pca_rgb = (
                            (p_norm.cpu().numpy() * 255)
                            .astype(np.uint8)
                            .reshape(curr_grid_size, curr_grid_size, 3)
                        )
                        snapped_keep = keep_mask

                if pca_rgb.shape[:2] == (image_size, image_size):
                    try:
                        snapped_pil = Image.fromarray(
                            (snapped_keep * 255).astype(np.uint8)
                        ).resize(
                            (curr_grid_size, curr_grid_size), resample=Image.BILINEAR
                        )
                        snapped_grid = (np.array(snapped_pil) > 127).astype(float)
                    except Exception:
                        snapped_grid = (
                            patch_weights
                            if patch_weights is not None
                            else np.ones((curr_grid_size, curr_grid_size))
                        )
                else:
                    snapped_grid = (
                        patch_weights
                        if patch_weights is not None
                        else np.ones((curr_grid_size, curr_grid_size))
                    )

                snapped_keep_flat = snapped_grid.reshape(-1, 1)
                total_snapped_weight = np.sum(snapped_keep_flat)
                snapped_avg = (
                    np.sum(patch_tokens * snapped_keep_flat, axis=0)
                    / total_snapped_weight
                    if total_snapped_weight > 0
                    else simple_avg
                )
                snapped_norm = snapped_avg / (np.linalg.norm(snapped_avg) + 1e-9)
                features_dict["anyup_snapped"].append(snapped_norm)

            # Store diagnostic visualization samples (up to 3)
            if save_plots and len(seg_viz_samples) < 3:
                seg_viz_samples.append(
                    {
                        "img": images[batch_keys[batch_i]],
                        "seg_keep": keep_mask,
                        "fg_keep": fg_mask_grid.reshape(curr_grid_size, curr_grid_size)
                        if fg_mask_grid is not None
                        else None,
                        "tips_ade_keep": tips_ade_keep_mask.reshape(
                            curr_grid_size, curr_grid_size
                        )
                        if tips_ade_keep_mask is not None
                        else None,
                        "pca_rgb": pca_rgb,
                        "snapped_keep": snapped_keep,
                    }
                )

    # 4. Build active representation matrices
    representations = {}
    for r in selected_reps:
        if r in [
            "cls",
            "unmasked_avg",
            "cls_attn_fg_removed",
            "segformer_masked",
            "tips_ade_masked",
            "anyup_snapped",
        ]:
            if features_dict[r]:
                representations[rep_key_to_name[r]] = np.vstack(features_dict[r])

    # Synthesize concatenated representations if selected
    if (
        "cls_plus_fg_removed_concat" in selected_reps
        and "cls" in features_dict
        and "cls_attn_fg_removed" in features_dict
    ):
        cls_mat = np.vstack(features_dict["cls"])
        fg_mat = np.vstack(features_dict["cls_attn_fg_removed"])
        concat = np.concatenate([cls_mat, fg_mat], axis=1)
        concat = concat / (np.linalg.norm(concat, axis=1, keepdims=True) + 1e-9)
        representations[rep_key_to_name["cls_plus_fg_removed_concat"]] = concat

    if (
        "cls_tips_ade_concat" in selected_reps
        and "cls" in features_dict
        and "tips_ade_masked" in features_dict
    ):
        cls_mat = np.vstack(features_dict["cls"])
        ade_mat = np.vstack(features_dict["tips_ade_masked"])
        concat = np.concatenate([cls_mat, ade_mat], axis=1)
        concat = concat / (np.linalg.norm(concat, axis=1, keepdims=True) + 1e-9)
        representations[rep_key_to_name["cls_tips_ade_concat"]] = concat

    if (
        "cls_unmasked_concat" in selected_reps
        and "cls" in features_dict
        and "unmasked_avg" in features_dict
    ):
        cls_mat = np.vstack(features_dict["cls"])
        unm_mat = np.vstack(features_dict["unmasked_avg"])
        concat = np.concatenate([cls_mat, unm_mat], axis=1)
        concat = concat / (np.linalg.norm(concat, axis=1, keepdims=True) + 1e-9)
        representations[rep_key_to_name["cls_unmasked_concat"]] = concat

    # 5. Global PCA layout histogram (only if PCA enabled and hybrid_land_use requested)
    if (
        enable_pca
        and "hybrid_land_use" in selected_reps
        and raw_patch_tokens_list
        and "tips_ade_masked" in features_dict
    ):
        print("\nFitting Global PCA model on patch token sample using GPU...")
        all_patches_tensor = torch.tensor(
            np.stack(raw_patch_tokens_list), device=device
        )
        D_dim = all_patches_tensor.shape[-1]
        all_patches_flat = all_patches_tensor.reshape(-1, D_dim)

        sample_indices = torch.randperm(all_patches_flat.shape[0], device=device)[
            : min(50000, all_patches_flat.shape[0])
        ]
        sampled_patches = all_patches_flat[sample_indices]

        with torch.no_grad():
            mean_vector = sampled_patches.mean(dim=0)
            _, _, V = torch.pca_lowrank(sampled_patches, q=16, center=True, niter=3)

        print("Generating Global PCA spatial layout histograms on GPU...")
        masked_pca_histograms = []
        with torch.no_grad():
            for idx_img, p_toks in enumerate(all_patches_tensor):
                centered = p_toks - mean_vector
                projected = torch.matmul(centered, V)
                dominant_components = torch.argmax(projected, dim=1)

                keep_mask_np = (
                    tips_ade_keep_masks_list[idx_img]
                    if idx_img < len(tips_ade_keep_masks_list)
                    else np.ones(curr_num_patches)
                )
                keep_mask_t = torch.tensor(keep_mask_np, device=device)
                masked_dominant = dominant_components[keep_mask_t == 1.0]
                if len(masked_dominant) > 0:
                    m_hist = torch.bincount(masked_dominant, minlength=16).float()
                    hist_masked = m_hist / m_hist.sum()
                else:
                    hist_masked = torch.zeros(16, device=device)
                masked_pca_histograms.append(hist_masked.cpu().numpy())

        masked_pca_mat = np.vstack(masked_pca_histograms)
        ade_mat = np.vstack(features_dict["tips_ade_masked"])
        hybrid_mat = np.concatenate([ade_mat, masked_pca_mat], axis=1)
        representations[rep_key_to_name["hybrid_land_use"]] = hybrid_mat

    # Expand representations to include FP16 if enabled
    if enable_fp16:
        expanded_representations = {}
        for rep_name, matrix in representations.items():
            expanded_representations[f"{rep_name} (FP32)"] = matrix
            expanded_representations[f"{rep_name} (FP16)"] = matrix.astype(
                np.float16
            ).astype(np.float32)
        representations = expanded_representations

    # 6. Run Benchmarking Suite
    print(f"\nRunning comparative benchmarks over {num_queries} queries...")
    query_indices = np.random.choice(
        len(sampled_df), min(num_queries, len(sampled_df)), replace=False
    )

    results = {
        rep: {
            "distances": [],
            "top_10_distances": [],
            "r_1_5km": 0,
            "r_5_5km": 0,
            "r_1_50km": 0,
            "r_5_50km": 0,
            "country_match": 0,
            "cross_platform_success": 0,
            "cross_platform_total": 0,
        }
        for rep in representations
    }

    for q_idx in query_indices:
        q_row = sampled_df.iloc[q_idx]
        q_lat, q_lon = q_row["Latitude"], q_row["Longitude"]
        q_plat = str(q_row["Platform"]).lower()
        q_country = q_row.get("country", "Unknown")

        db_mask = np.ones(len(sampled_df), dtype=bool)
        db_mask[q_idx] = False
        db_df = sampled_df[db_mask]

        db_distances = haversine_distance(
            q_lat, q_lon, db_df["Latitude"].values, db_df["Longitude"].values
        )

        for rep_name, representation_matrix in representations.items():
            q_vector = representation_matrix[q_idx]
            db_vectors = representation_matrix[db_mask]

            if "Hybrid" in rep_name:
                q_sem = q_vector[:768]
                q_hist = q_vector[768:]
                db_sem = db_vectors[:, :768]
                db_hist = db_vectors[:, 768:]
                cos_sim = np.dot(db_sem, q_sem)
                d_sem = 1.0 - cos_sim
                d_hist = np.sum(np.abs(db_hist - q_hist), axis=1) / 2.0
                blended_dist = 0.7 * d_sem + 0.3 * d_hist
                sorted_indices = np.argsort(blended_dist)
            else:
                similarities = np.dot(db_vectors, q_vector)
                sorted_indices = np.argsort(similarities)[::-1]

            top_1_dist = db_distances[sorted_indices[0]]
            top_5_dists = db_distances[sorted_indices[:5]]
            top_10_dists = db_distances[sorted_indices[:10]]
            top_1_row = db_df.iloc[sorted_indices[0]]

            results[rep_name]["distances"].append(top_1_dist)
            results[rep_name]["top_10_distances"].append(top_10_dists)

            if top_1_dist <= 5.0:
                results[rep_name]["r_1_5km"] += 1
            if np.any(top_5_dists <= 5.0):
                results[rep_name]["r_5_5km"] += 1

            if top_1_dist <= 50.0:
                results[rep_name]["r_1_50km"] += 1
            if np.any(top_5_dists <= 50.0):
                results[rep_name]["r_5_50km"] += 1

            if (
                q_country != "Unknown"
                and top_1_row.get("country", "Unknown") == q_country
            ):
                results[rep_name]["country_match"] += 1

            # Cross-Platform retrieval
            cross_db_indices = np.where(db_df["Platform"].str.lower() != q_plat)[0]
            if len(cross_db_indices) > 0:
                cross_db_dists = db_distances[cross_db_indices]
                if np.any(cross_db_dists <= 50.0):
                    results[rep_name]["cross_platform_total"] += 1
                    sorted_db_plats = (
                        db_df["Platform"].iloc[sorted_indices].str.lower().values
                    )
                    first_cross_idx = np.where(sorted_db_plats != q_plat)[0][0]
                    if db_distances[sorted_indices[first_cross_idx]] <= 50.0:
                        results[rep_name]["cross_platform_success"] += 1

    # 7. Print and Save Benchmark Report
    print("\n" + "=" * 80)
    print("                   GEOGRAPHIC RETRIEVAL BENCHMARK REPORT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Database Size: {len(sampled_df)} images")
    print(f"Evaluation Queries: {len(query_indices)} diverse samples")
    print("-" * 80)

    row_format = "{:<44} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}"
    header_str = row_format.format(
        "Representation",
        "Median Err",
        "R@1 (5km)",
        "R@5 (5km)",
        "R@1 (50km)",
        "Cross-Plat",
    )
    unit_str = row_format.format("", "(km)", "(%)", "(%)", "(%)", "Recall (%)")
    print(header_str)
    print(unit_str)
    print("-" * 80)

    report_lines = [
        "Geographic Representation & Layout Retrieval Benchmark Report",
        f"Model: {model_name}",
        f"Database: {resolved_path}",
        f"Database Size: {len(sampled_df)} images",
        f"Evaluation Queries: {len(query_indices)} samples",
        "-" * 80,
        header_str,
        unit_str,
        "-" * 80,
    ]

    report_rows = []
    for rep_name, metrics in results.items():
        median_err = (
            float(np.median(metrics["distances"])) if metrics["distances"] else 0.0
        )
        r_1_5 = (metrics["r_1_5km"] / len(query_indices)) * 100
        r_5_5 = (metrics["r_5_5km"] / len(query_indices)) * 100
        r_1_50 = (metrics["r_1_50km"] / len(query_indices)) * 100

        cross_plat_recall = 0.0
        if metrics["cross_platform_total"] > 0:
            cross_plat_recall = (
                metrics["cross_platform_success"] / metrics["cross_platform_total"]
            ) * 100

        row_text = row_format.format(
            rep_name,
            f"{median_err:.2f}",
            f"{r_1_5:.1f}%",
            f"{r_5_5:.1f}%",
            f"{r_1_50:.1f}%",
            f"{cross_plat_recall:.1f}%"
            if metrics["cross_platform_total"] > 0
            else "N/A",
        )
        print(row_text)
        report_lines.append(row_text)

        report_rows.append(
            {
                "Model": model_name,
                "Representation": rep_name,
                "Median_Error_km": round(median_err, 2),
                "R@1_5km": round(r_1_5, 2),
                "R@5_5km": round(r_5_5, 2),
                "R@1_50km": round(r_1_50, 2),
                "Cross_Platform_Recall": round(cross_plat_recall, 2)
                if metrics["cross_platform_total"] > 0
                else np.nan,
            }
        )

    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)

    # Save TXT report
    model_clean = model_name.replace("/", "_")
    report_fname = output_report_file
    if report_fname == "representations_report.txt":
        report_fname = f"representations_report_{model_clean}.txt"

    report_path = os.path.join(output_dir, report_fname)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nSaved benchmark text report to: {os.path.abspath(report_path)}")

    # Save CSV results
    csv_fname = output_csv_file
    if csv_fname == "representations_results.csv":
        csv_fname = f"representations_results_{model_clean}.csv"

    csv_out_path = os.path.join(output_dir, csv_fname)
    df_results = pd.DataFrame(report_rows)
    df_results.to_csv(csv_out_path, index=False)
    print(f"Saved benchmark results CSV to: {os.path.abspath(csv_out_path)}")

    # 8. Generate Visualization Plots
    if save_plots and len(results) > 0:
        try:
            import matplotlib.pyplot as plt

            # Plot 1: Recall Curve and Error CDF
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            for rep_name, metrics in results.items():
                k_values = list(range(1, 11))
                recalls = []
                top_10_dist_array = np.vstack(metrics["top_10_distances"])
                for k in k_values:
                    success = np.any(top_10_dist_array[:, :k] <= 50.0, axis=1)
                    recalls.append(np.mean(success) * 100)
                axes[0].plot(k_values, recalls, marker="o", label=rep_name)
            axes[0].set_title("Recall@K Curve (within 50 km)")
            axes[0].set_xlabel("K (Number of retrieved images)")
            axes[0].set_ylabel("Recall (%)")
            axes[0].set_xticks(range(1, 11))
            axes[0].grid(True, linestyle="--", alpha=0.6)
            axes[0].legend(loc="lower right", fontsize=7)

            for rep_name, metrics in results.items():
                sorted_errors = np.sort(metrics["distances"])
                cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
                axes[1].plot(sorted_errors, cdf, label=rep_name)
            axes[1].set_title("Error Cumulative Distribution Function (CDF)")
            axes[1].set_xlabel("Top-1 Geodesic Error (km)")
            axes[1].set_ylabel("Queries Resolved (%)")
            axes[1].set_xlim(0, 100)
            axes[1].grid(True, linestyle="--", alpha=0.6)
            axes[1].legend(loc="lower right", fontsize=7)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"benchmark_metrics_{model_clean}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved metrics comparison plot to: {os.path.abspath(plot_path)}")

            # Plot 2: Qualitative Query Sample Comparison Grid
            if len(query_indices) > 0:
                q_idx = query_indices[0]
                q_row = sampled_df.iloc[q_idx]
                q_img = images[q_idx]
                reps_to_visualize = list(representations.keys())

                fig2, axes2 = plt.subplots(
                    len(reps_to_visualize), 4, figsize=(16, 3 * len(reps_to_visualize))
                )
                if len(reps_to_visualize) == 1:
                    axes2 = np.expand_dims(axes2, axis=0)

                db_mask = np.ones(len(sampled_df), dtype=bool)
                db_mask[q_idx] = False
                db_df = sampled_df[db_mask]
                db_distances = haversine_distance(
                    q_row["Latitude"],
                    q_row["Longitude"],
                    db_df["Latitude"].values,
                    db_df["Longitude"].values,
                )

                for row_i, rep_name in enumerate(reps_to_visualize):
                    rep_mat = representations[rep_name]
                    q_vec = rep_mat[q_idx]
                    db_vecs = rep_mat[db_mask]

                    if "Hybrid" in rep_name:
                        distances = np.sum(np.abs(db_vecs - q_vec), axis=1)
                        sorted_idx = np.argsort(distances)
                    else:
                        sims = np.dot(db_vecs, q_vec)
                        sorted_idx = np.argsort(sims)[::-1]

                    axes2[row_i, 0].imshow(q_img)
                    axes2[row_i, 0].set_title(
                        f"Query ({q_row['Platform']})", fontsize=10
                    )
                    axes2[row_i, 0].axis("off")

                    for col_j in range(1, 4):
                        retrieved_db_idx = sorted_idx[col_j - 1]
                        retrieved_img_idx = db_df.index[retrieved_db_idx]
                        retrieved_img = images[retrieved_img_idx]
                        retrieved_row = db_df.iloc[retrieved_db_idx]
                        dist_err = db_distances[retrieved_db_idx]

                        clean_title = rep_name.replace(" (L1 Dist)", "").replace(
                            " (Concat)", "\n(Concat)"
                        )
                        axes2[row_i, col_j].imshow(retrieved_img)
                        axes2[row_i, col_j].set_title(
                            f"{clean_title} Top-{col_j}\n({retrieved_row['Platform']}) {dist_err:.1f}km",
                            fontsize=8,
                        )
                        axes2[row_i, col_j].axis("off")

                plt.tight_layout()
                grid_plot_path = os.path.join(
                    output_dir, f"benchmark_retrieval_{model_clean}.png"
                )
                plt.savefig(grid_plot_path, dpi=150)
                plt.close()
                print(
                    f"Saved qualitative retrieval example grid to: {os.path.abspath(grid_plot_path)}"
                )

            # Plot 3: Segmentation & Attention Mask Diagnostics
            if len(seg_viz_samples) > 0:
                has_seg_keep = any(
                    s.get("seg_keep") is not None for s in seg_viz_samples
                )
                has_fg_keep = any(s.get("fg_keep") is not None for s in seg_viz_samples)
                has_ade_keep = any(
                    s.get("tips_ade_keep") is not None for s in seg_viz_samples
                )

                num_cols = 1
                if has_seg_keep:
                    num_cols += 1
                if has_fg_keep:
                    num_cols += 1
                if has_ade_keep:
                    num_cols += 1
                if enable_pca:
                    num_cols += 2

                if num_cols > 1:
                    fig3, axes3 = plt.subplots(
                        len(seg_viz_samples),
                        num_cols,
                        figsize=(4 * num_cols, 4 * len(seg_viz_samples)),
                    )
                    if len(seg_viz_samples) == 1:
                        axes3 = np.expand_dims(axes3, axis=0)

                    for idx_s, sample in enumerate(seg_viz_samples):
                        col_curr = 0
                        axes3[idx_s, col_curr].imshow(sample["img"])
                        axes3[idx_s, col_curr].set_title(
                            f"Image {idx_s + 1}", fontsize=10
                        )
                        axes3[idx_s, col_curr].axis("off")
                        col_curr += 1

                        if has_seg_keep:
                            if sample.get("seg_keep") is not None:
                                seg_keep_rgb = np.zeros(
                                    (image_size, image_size, 3), dtype=np.uint8
                                )
                                seg_keep_rgb[sample["seg_keep"] == 1.0] = [34, 139, 34]
                                seg_keep_rgb[sample["seg_keep"] == 0.0] = [178, 34, 34]
                                axes3[idx_s, col_curr].imshow(seg_keep_rgb)
                                axes3[idx_s, col_curr].set_title(
                                    "Segformer ADE150 Mask", fontsize=9
                                )
                            axes3[idx_s, col_curr].axis("off")
                            col_curr += 1

                        if has_fg_keep:
                            if sample.get("fg_keep") is not None:
                                fg_mask = sample["fg_keep"]
                                fg_rgb = np.zeros((*fg_mask.shape, 3), dtype=np.uint8)
                                fg_rgb[fg_mask == 1.0] = [
                                    34,
                                    139,
                                    34,
                                ]  # Background: Green
                                fg_rgb[fg_mask == 0.0] = [
                                    178,
                                    34,
                                    34,
                                ]  # Removed Foreground: Red
                                fg_upsampled = Image.fromarray(fg_rgb).resize(
                                    (image_size, image_size), resample=Image.NEAREST
                                )
                                axes3[idx_s, col_curr].imshow(fg_upsampled)
                                axes3[idx_s, col_curr].set_title(
                                    "CLS-Attn FG Mask (<=1/20)", fontsize=9
                                )
                            axes3[idx_s, col_curr].axis("off")
                            col_curr += 1

                        if has_ade_keep:
                            if sample.get("tips_ade_keep") is not None:
                                ade_mask = sample["tips_ade_keep"]
                                ade_keep_rgb = np.zeros(
                                    (*ade_mask.shape, 3), dtype=np.uint8
                                )
                                ade_keep_rgb[ade_mask == 1.0] = [34, 139, 34]
                                ade_keep_rgb[ade_mask == 0.0] = [178, 34, 34]
                                ade_keep_upsampled = Image.fromarray(
                                    ade_keep_rgb
                                ).resize(
                                    (image_size, image_size), resample=Image.NEAREST
                                )
                                axes3[idx_s, col_curr].imshow(ade_keep_upsampled)
                                axes3[idx_s, col_curr].set_title(
                                    f"{model_label} Zero-Shot Mask", fontsize=9
                                )
                            axes3[idx_s, col_curr].axis("off")
                            col_curr += 1

                        if (
                            enable_pca
                            and sample.get("pca_rgb") is not None
                            and sample.get("snapped_keep") is not None
                        ):
                            pca_rgb_np = sample["pca_rgb"]
                            if pca_rgb_np.shape[:2] != (image_size, image_size):
                                pca_rgb_np = np.array(
                                    Image.fromarray(pca_rgb_np).resize(
                                        (image_size, image_size), resample=Image.NEAREST
                                    )
                                )
                            axes3[idx_s, col_curr].imshow(pca_rgb_np)
                            axes3[idx_s, col_curr].set_title(
                                "Local PCA Projection", fontsize=9
                            )
                            axes3[idx_s, col_curr].axis("off")
                            col_curr += 1

                            snapped_keep_rgb = np.zeros(
                                (image_size, image_size, 3), dtype=np.uint8
                            )
                            snapped_keep_rgb[sample["snapped_keep"] == 1.0] = [
                                34,
                                139,
                                34,
                            ]
                            snapped_keep_rgb[sample["snapped_keep"] == 0.0] = [
                                178,
                                34,
                                34,
                            ]
                            axes3[idx_s, col_curr].imshow(snapped_keep_rgb)
                            axes3[idx_s, col_curr].set_title(
                                "AnyUp Snapped Mask", fontsize=9
                            )
                            axes3[idx_s, col_curr].axis("off")

                    plt.tight_layout()
                    seg_plot_path = os.path.join(
                        output_dir, f"benchmark_segmentation_{model_clean}.png"
                    )
                    plt.savefig(seg_plot_path, dpi=150)
                    plt.close()
                    print(
                        f"Saved segmentation visualization masks to: {os.path.abspath(seg_plot_path)}"
                    )

        except Exception as e:
            print(f"Warning: Failed to render plots: {e}")


if __name__ == "__main__":
    main()
