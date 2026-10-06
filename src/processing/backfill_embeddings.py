import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

import src.models.tips_image_encoder as image_encoder
from src.models.vision_model_inference import (
    extract_regular_embeddings,
    load_vision_model,
    model_has_cls_token,
)
from src.utils.credentials import get_mapillary_token
from src.utils.io import (
    download_image,
    load_dataframe,
    save_dataframe,
)


def find_companion_files(target_path: str, model_name: str, representation_type: str):
    """
    Find existing companion .npy and .keys.parquet for a target dataset.
    Searches exact names and canonical base names in the target's directory
    and parent directory.
    """
    if not target_path:
        return None, None

    db_dir = os.path.dirname(os.path.abspath(target_path))
    if not os.path.exists(db_dir):
        return None, None

    candidate_dirs = [db_dir]
    parent_dir = os.path.dirname(db_dir)
    if parent_dir and os.path.exists(parent_dir) and parent_dir != db_dir:
        candidate_dirs.append(parent_dir)

    raw_base = os.path.splitext(os.path.basename(target_path))[0]
    candidate_bases = [raw_base]
    if raw_base.endswith("_cleaned"):
        candidate_bases.append(raw_base[:-8])
    if raw_base.endswith("_offline"):
        candidate_bases.append(raw_base[:-8])
    if raw_base.endswith("_cleaned_offline"):
        candidate_bases.append(raw_base[:-16])

    model_suffix = ""
    if model_name and model_name != "google/tipsv2-b14":
        model_suffix = "_" + model_name.replace("/", "_")

    for d in candidate_dirs:
        for b in candidate_bases:
            npy_path = os.path.join(
                d, f"{b}{model_suffix}_{representation_type}_embeddings.npy"
            )
            keys_path = npy_path.replace(".npy", ".keys.parquet")
            if os.path.exists(npy_path) and os.path.exists(keys_path):
                return npy_path, keys_path

    return None, None


def resolve_companion_with_derivation(
    target_path: str, model_name: str, representation_type: str
):
    """
    Finds exact companion files or derivable alternatives (e.g. cls_avg_patch for ViT,
    or Global Average Pooling aliases for CNN).

    Returns:
        tuple: (npy_path, keys_path, derivation_mode) or (None, None, None)
        derivation_mode can be:
          - 'exact': exact representation match
          - 'slice_cls': slice first half [:, :D] from cls_avg_patch
          - 'slice_avg_patch': slice second half [:, D:] from cls_avg_patch
          - 'cnn_alias': reuse Global Average Pooling representation between cls and avg_patch
          - 'cnn_duplicate': concatenate GAP representation with itself for cls_avg_patch
    """
    if not target_path:
        return None, None, None

    # 1. Exact match
    npy_p, keys_p = find_companion_files(target_path, model_name, representation_type)
    if npy_p and keys_p:
        return npy_p, keys_p, "exact"

    # 2. Derive cls or avg_patch from cls_avg_patch (ViT or any model with cls_avg_patch)
    if representation_type in ["cls", "avg_patch"]:
        sup_npy, sup_keys = find_companion_files(
            target_path, model_name, "cls_avg_patch"
        )
        if sup_npy and sup_keys:
            mode = "slice_cls" if representation_type == "cls" else "slice_avg_patch"
            return sup_npy, sup_keys, mode

    # 3. Architectures without discrete CLS token: interchangeable cls <-> avg_patch (both are spatial pooling)
    if not model_has_cls_token(model_name):
        if representation_type in ["cls", "avg_patch"]:
            other_rep = "avg_patch" if representation_type == "cls" else "cls"
            alt_npy, alt_keys = find_companion_files(target_path, model_name, other_rep)
            if alt_npy and alt_keys:
                return alt_npy, alt_keys, "cnn_alias"
        elif representation_type == "cls_avg_patch":
            for alt_rep in ["cls", "avg_patch"]:
                alt_npy, alt_keys = find_companion_files(
                    target_path, model_name, alt_rep
                )
                if alt_npy and alt_keys:
                    return alt_npy, alt_keys, "cnn_duplicate"

    return None, None, None


def resolve_checkpoint_paths(
    target_path: str, model_name: str, representation_type: str
):
    """
    Resolves the expected paths for checkpoint companion files.
    """
    db_dir = os.path.dirname(os.path.abspath(target_path))
    raw_base = os.path.splitext(os.path.basename(target_path))[0]
    if raw_base.endswith("_cleaned"):
        raw_base = raw_base[:-8]

    model_suffix = ""
    if model_name and model_name != "google/tipsv2-b14":
        model_suffix = "_" + model_name.replace("/", "_")

    ckpt_npy_name = (
        f"{raw_base}{model_suffix}_{representation_type}_embeddings_checkpoint.npy"
    )
    ckpt_npy = os.path.join(db_dir, ckpt_npy_name)
    ckpt_keys = ckpt_npy.replace(".npy", ".keys.parquet")
    return ckpt_npy, ckpt_keys


def save_checkpoint_companion(ckpt_npy, ckpt_keys, keys_list, embs_array, dtype):
    """
    Atomically writes intermediate embeddings and keys to checkpoint companion files.
    """
    tmp_npy = ckpt_npy.replace(".npy", ".tmp.npy")
    tmp_keys = ckpt_keys.replace(".parquet", ".tmp.parquet")
    np.save(tmp_npy, embs_array.astype(dtype))
    pd.DataFrame({"photo_key": keys_list}).to_parquet(tmp_keys, compression="zstd")
    os.replace(tmp_npy, ckpt_npy)
    os.replace(tmp_keys, ckpt_keys)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone utility to compute/backfill TIPSv2 embeddings (CLS, average patch, or concatenated) for an existing dataset."
    )
    parser.add_argument(
        "--input",
        "--in",
        dest="input",
        type=str,
        required=True,
        help="Path to the input metadata file (.parquet, .csv, or .pkl).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/tipsv2-b14",
        help="Hugging Face identifier or timm model name of the vision encoder to load.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=str,
        default=None,
        help="Path to save the output metadata (defaults to overwriting input in-place).",
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        default="cls",
        choices=["cls", "avg_patch", "cls_avg_patch"],
        help="Type of representation embedding to extract (cls, avg_patch, or cls_avg_patch).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Stored precision of companion binary file (float32 or float16).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for GPU forward passes."
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        nargs="+",
        default=None,
        help="Optional root directories containing local images (for offline datasets).",
    )
    parser.add_argument(
        "--tips_model_path",
        type=str,
        default=None,
        help="Optional path to local TIPSv2 checkpoint .npy weight file.",
    )
    parser.add_argument(
        "--tips_model_variant",
        type=str,
        default="b",
        choices=["s", "b", "l", "g"],
        help="TIPSv2 model variant to load if local checkpoint is specified (s, b, l, g).",
    )
    parser.add_argument(
        "--tips_low_res",
        action="store_true",
        help="Use 224x224 input resolution for local checkpoints instead of 448x448.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Number of images to process in parallel download chunks.",
    )
    parser.add_argument(
        "--mapillary_token",
        type=str,
        default=None,
        help="Mapillary API token for downloading images.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume computation by reusing existing embeddings for already-processed images (enabled by default).",
    )
    parser.add_argument(
        "--no_resume",
        "--force",
        dest="resume",
        action="store_false",
        help="Force recomputation of all embeddings from scratch.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=300,
        help="Interval in seconds for saving progress checkpoints (default: 300, set to 0 to disable).",
    )
    args = parser.parse_args()

    if not args.mapillary_token:
        args.mapillary_token = get_mapillary_token()

    out_path = args.output if args.output else args.input
    if out_path.endswith(".csv"):
        out_path = out_path.replace(".csv", ".parquet")
        print(f" -> Enforcing decoupled Parquet output format: {out_path}")
    elif out_path.endswith(".pkl") or out_path.endswith(".pickle"):
        out_path = os.path.splitext(out_path)[0] + ".parquet"
        print(f" -> Enforcing decoupled Parquet output format: {out_path}")

    # 1. Load dataset metadata
    print(f"Loading dataset from {args.input}...")
    is_pkl = args.input.endswith(".pkl")
    if is_pkl:
        with open(args.input, "rb") as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
    else:
        df = load_dataframe(args.input)

    if len(df) == 0:
        print("Error: Input dataset is empty.")
        return

    # Normalize schema using common column mappings (aligned with process_scraped_data.py)
    col_map = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "photo_id": "Photo_ID",
        "ID": "Photo_ID",
        "captured_at": "Captured_At",
        "Date_Observed": "Captured_At",
        "observed_on_string": "Captured_At",
        "license": "License",
        "platform": "Platform",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Map Image_URL fallback column
    if "Image_URL" not in df.columns:
        for fallback in ["local_path", "Image_Location", "file_name", "path"]:
            if fallback in df.columns:
                df = df.rename(columns={fallback: "Image_URL"})
                print(
                    f" -> Mapping missing column 'Image_URL' to existing column '{fallback}'."
                )
                break

    # Default missing Platform to 'Offline'
    if "Platform" not in df.columns:
        df["Platform"] = "Offline"
        print(" -> Platform column missing. Defaulting to 'Offline'.")

    # Verify final key columns exist
    for col in ["Platform", "Photo_ID", "Image_URL"]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' is missing from the dataset schema (even after mapping column aliases)."
            )

    # Standardize photo_key on df
    if "photo_key" not in df.columns:
        df["photo_key"] = (
            df["Platform"].astype(str).str.lower() + "_" + df["Photo_ID"].astype(str)
        )

    # Determine model suffix for npy file name
    is_local = bool(args.tips_model_path)
    model_name_for_save = (
        f"local_tipsv2_{args.tips_model_variant}" if is_local else args.model_name
    )
    target_dtype = np.float32 if args.precision == "float32" else np.float16

    # 2. Check for existing companion embeddings and checkpoint files (Resume & Derivation support)
    known_keys = []
    known_embs = []

    def load_companion_pair(npy_p, keys_p, label="", derivation_mode="exact"):
        if not (npy_p and keys_p and os.path.exists(npy_p) and os.path.exists(keys_p)):
            return False
        try:
            keys_df = pd.read_parquet(keys_p, columns=["photo_key"])
            k_arr = keys_df["photo_key"].astype(str).str.lower().values
            e_arr = np.load(npy_p, mmap_mode="r")
            if len(k_arr) != len(e_arr) or len(k_arr) == 0:
                print(
                    f" -> [WARNING] Length mismatch between keys ({len(k_arr)}) and embeddings ({len(e_arr)}) in {npy_p}. Skipping."
                )
                return False

            if derivation_mode == "slice_cls":
                if e_arr.ndim == 2 and e_arr.shape[1] % 2 == 0:
                    d = e_arr.shape[1] // 2
                    e_arr = e_arr[:, :d]
                    print(
                        f" -> Automatically derived 'cls' (dim: {d}) by slicing first half of {label or npy_p} ({len(k_arr):,} vectors)"
                    )
                else:
                    print(
                        f" -> [WARNING] Cannot derive 'cls' from {npy_p}: embedding dim {e_arr.shape[1]} not even. Skipping."
                    )
                    return False
            elif derivation_mode == "slice_avg_patch":
                if e_arr.ndim == 2 and e_arr.shape[1] % 2 == 0:
                    d = e_arr.shape[1] // 2
                    e_arr = e_arr[:, d:]
                    print(
                        f" -> Automatically derived 'avg_patch' (dim: {d}) by slicing second half of {label or npy_p} ({len(k_arr):,} vectors)"
                    )
                else:
                    print(
                        f" -> [WARNING] Cannot derive 'avg_patch' from {npy_p}: embedding dim {e_arr.shape[1]} not even. Skipping."
                    )
                    return False
            elif derivation_mode == "cnn_alias":
                print(
                    f" -> Model '{model_name_for_save}' has no discrete CLS token; reusing spatial pooled representation from {label or npy_p} as '{args.representation_type}' ({len(k_arr):,} vectors)"
                )
            elif derivation_mode == "cnn_duplicate":
                d = e_arr.shape[1]
                e_arr = np.concatenate([e_arr, e_arr], axis=-1)
                print(
                    f" -> Model '{model_name_for_save}' has no discrete CLS token; forming 'cls_avg_patch' (dim: {2*d}) from spatial pooling in {label or npy_p} ({len(k_arr):,} vectors)"
                )
            else:
                print(
                    f" -> Found existing companion embeddings ({len(k_arr):,} vectors): {label or npy_p}"
                )

            known_keys.append(k_arr)
            known_embs.append(e_arr)
            return True
        except Exception as e:
            print(
                f" -> [WARNING] Failed loading existing companion files ({npy_p}): {e}"
            )
            return False

    def load_concatenated_companion_pair(
        cls_npy, cls_keys, patch_npy, patch_keys, label=""
    ):
        if not (cls_npy and cls_keys and patch_npy and patch_keys):
            return False
        try:
            k1_df = pd.read_parquet(cls_keys, columns=["photo_key"])
            k2_df = pd.read_parquet(patch_keys, columns=["photo_key"])
            k1 = k1_df["photo_key"].astype(str).str.lower().values
            k2 = k2_df["photo_key"].astype(str).str.lower().values
            e1 = np.load(cls_npy, mmap_mode="r")
            e2 = np.load(patch_npy, mmap_mode="r")

            if len(k1) != len(e1) or len(k2) != len(e2) or len(k1) == 0:
                return False

            if not np.array_equal(k1, k2):
                idx1 = pd.Index(k1)
                idx2 = pd.Index(k2)
                common = idx1.intersection(idx2)
                if len(common) == 0:
                    return False
                m1 = idx1.get_indexer(common)
                m2 = idx2.get_indexer(common)
                e1 = e1[m1]
                e2 = e2[m2]
                k_arr = common.values
            else:
                k_arr = k1

            e_arr = np.concatenate([e1, e2], axis=-1)
            print(
                f" -> Automatically assembled 'cls_avg_patch' (dim: {e_arr.shape[1]}) by concatenating separate 'cls' and 'avg_patch' companions ({len(k_arr):,} vectors): {label}"
            )
            known_keys.append(k_arr)
            known_embs.append(e_arr)
            return True
        except Exception as e:
            print(f" -> [WARNING] Failed assembling concatenated companion: {e}")
            return False

    def resolve_and_load_companion(path_target, label_prefix):
        if not path_target:
            return False

        # 1. Exact or single-file derived companion (e.g. slice from cls_avg_patch or CNN GAP alias)
        npy_p, keys_p, mode = resolve_companion_with_derivation(
            path_target, model_name_for_save, args.representation_type
        )
        if npy_p and keys_p:
            if load_companion_pair(
                npy_p, keys_p, f"{label_prefix} ({npy_p})", derivation_mode=mode
            ):
                return True

        # 2. ViT architecture with CLS: assemble cls_avg_patch from separate cls and avg_patch companions
        if args.representation_type == "cls_avg_patch" and model_has_cls_token(
            model_name_for_save
        ):
            c_npy, c_keys = find_companion_files(
                path_target, model_name_for_save, "cls"
            )
            p_npy, p_keys = find_companion_files(
                path_target, model_name_for_save, "avg_patch"
            )
            if c_npy and c_keys and p_npy and p_keys:
                if load_concatenated_companion_pair(
                    c_npy, c_keys, p_npy, p_keys, f"{label_prefix} [cls + avg_patch]"
                ):
                    return True

        return False

    ckpt_npy, ckpt_keys = resolve_checkpoint_paths(
        out_path, model_name_for_save, args.representation_type
    )

    if args.resume:
        # A. Check output path companion files (exact or derived)
        resolve_and_load_companion(out_path, "output companion")

        # B. Check input path companion files (if input is different from output)
        if os.path.abspath(args.input) != os.path.abspath(out_path):
            resolve_and_load_companion(args.input, "input companion")

        # C. Check intermediate checkpoint files for output (exact only)
        if os.path.exists(ckpt_npy) and os.path.exists(ckpt_keys):
            load_companion_pair(
                ckpt_npy,
                ckpt_keys,
                f"checkpoint companion ({ckpt_npy})",
                derivation_mode="exact",
            )

    unique_keys_index = pd.Index([])
    unique_embs = None

    if known_keys and known_embs:
        cat_keys = np.concatenate(known_keys)
        cat_embs = np.concatenate(known_embs, axis=0)

        series_keys = pd.Series(cat_keys)
        unique_mask = ~series_keys.duplicated(keep="last").values
        unique_keys_index = pd.Index(cat_keys[unique_mask])
        unique_embs = cat_embs[unique_mask].astype(target_dtype)
        print(f" -> Indexed {len(unique_keys_index):,} unique pre-existing embeddings.")

    target_keys = df["photo_key"].astype(str).str.lower().values
    known_indexer = unique_keys_index.get_indexer(target_keys)
    already_computed_mask = known_indexer >= 0
    missing_indices = np.where(~already_computed_mask)[0]

    num_already = int(already_computed_mask.sum())
    num_missing = len(missing_indices)
    print(
        f" -> Pre-flight status: {num_already:,}/{len(df):,} images already have valid embeddings."
    )
    print(f" -> Images requiring feature extraction: {num_missing:,}")

    # If all images already have embeddings, finish immediately without loading the vision model!
    if num_missing == 0:
        print(
            "\n✅ All images in the dataset already have embeddings! No feature extraction required."
        )
        embeddings_matrix = unique_embs[known_indexer]
        df["embedding"] = list(embeddings_matrix)
        print(f"Saving decoupled dataset to: {out_path}")
        save_dataframe(
            df,
            out_path,
            representation_type=args.representation_type,
            precision=args.precision,
            model_name=model_name_for_save,
        )
        for p in [ckpt_npy, ckpt_keys]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        print("✅ Embeddings backfilling and metadata alignment complete!")
        return

    # 3. Setup Device & Initialize Model (only when missing embeddings need computation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing model on {device}...")

    # Load model transforms and initialize model
    if is_local:
        print(f"Loading local checkpoint from {args.tips_model_path}...")
        model_def = {
            "s": image_encoder.vit_small14,
            "b": image_encoder.vit_base14,
            "l": image_encoder.vit_large14,
            "g": image_encoder.vit_giant2,
        }[args.tips_model_variant]

        image_size = 224 if args.tips_low_res else 448
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        ffn_layer = "swiglu" if args.tips_model_variant == "g" else "mlp"
        checkpoint = dict(np.load(args.tips_model_path, allow_pickle=False))
        for key in checkpoint:
            checkpoint[key] = torch.tensor(checkpoint[key])

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
        model = model.eval().to(device)
    else:
        # Load any timm or Hugging Face model dynamically
        model, transform, image_size = load_vision_model(args.model_name, device)

    # 4. Compute Embeddings in Parallel Chunks with Checkpointing
    print(
        f"Computing '{args.representation_type}' embeddings with {args.precision} precision..."
    )

    def download_thread_fn(
        global_idx, url, photo_id, platform, offline_dirs, image_size, mapillary_token
    ):
        try:
            img = download_image(
                url,
                photo_id=photo_id,
                platform=platform,
                offline_dirs=offline_dirs,
                image_size=image_size,
                mapillary_token=mapillary_token,
            )
            return global_idx, img
        except Exception:
            return global_idx, None

    computed_new_dict = {}
    orig_photo_keys = df["photo_key"].astype(str).str.lower().to_dict()
    last_checkpoint_time = time.time()

    def flush_checkpoint():
        if not computed_new_dict:
            return
        print(
            f"\n[Checkpoint] Saving intermediate checkpoint ({len(computed_new_dict):,} new embeddings)..."
        )
        new_k = np.array([orig_photo_keys[idx] for idx in computed_new_dict.keys()])
        new_v = np.stack(list(computed_new_dict.values()))
        if unique_embs is not None and len(unique_keys_index) > 0:
            cat_k = np.concatenate([unique_keys_index.values, new_k])
            cat_v = np.concatenate([unique_embs, new_v], axis=0)
            u_m = ~pd.Series(cat_k).duplicated(keep="last").values
            final_ckpt_k = cat_k[u_m]
            final_ckpt_v = cat_v[u_m]
        else:
            final_ckpt_k = new_k
            final_ckpt_v = new_v

        save_checkpoint_companion(
            ckpt_npy, ckpt_keys, final_ckpt_k, final_ckpt_v, target_dtype
        )
        print(f" -> Intermediate checkpoint saved to {ckpt_npy}")

    try:
        t0 = time.time()
        valid_count = 0
        chunk_size = args.chunk_size
        print(
            f"Processing {num_missing:,} missing images in chunks of {chunk_size} with parallel downloads..."
        )

        for chunk_start in range(0, num_missing, chunk_size):
            chunk_batch_indices = missing_indices[
                chunk_start : chunk_start + chunk_size
            ]
            chunk_df = df.iloc[chunk_batch_indices]
            print(
                f"\n--- Processing chunk {chunk_start // chunk_size + 1}/{(num_missing + chunk_size - 1) // chunk_size} ({chunk_start} to {chunk_start + len(chunk_df)}) ---"
            )

            # Parallel downloads for the chunk
            db_dict = {}
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = {
                    executor.submit(
                        download_thread_fn,
                        global_idx,
                        row["Image_URL"],
                        row["Photo_ID"],
                        row["Platform"],
                        args.image_root_dir,
                        image_size,
                        args.mapillary_token,
                    ): global_idx
                    for global_idx, row in chunk_df.iterrows()
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Downloading Chunk"
                ):
                    global_idx, img = future.result()
                    if img is not None:
                        db_dict[global_idx] = img

            # Run model inference in batches of args.batch_size on successfully downloaded images
            active_indices = sorted(list(db_dict.keys()))
            valid_imgs = [db_dict[idx] for idx in active_indices]

            if len(valid_imgs) > 0:
                for b_start in tqdm(
                    range(0, len(valid_imgs), args.batch_size),
                    desc="Processing Batches",
                ):
                    batch_imgs = valid_imgs[b_start : b_start + args.batch_size]
                    batch_indices = active_indices[b_start : b_start + args.batch_size]

                    try:
                        batch_tensors = torch.stack(
                            [transform(img) for img in batch_imgs]
                        ).to(device)
                        features = extract_regular_embeddings(
                            model,
                            batch_tensors,
                            representation_type=args.representation_type,
                            is_local=is_local,
                        )

                        for b_i, global_idx in enumerate(batch_indices):
                            computed_new_dict[global_idx] = features[b_i].astype(
                                target_dtype
                            )
                            valid_count += 1
                    except Exception as e:
                        print(
                            f"Warning: Failed processing model batch starting at index {batch_indices[0]}: {e}"
                        )

                # Immediately close PIL images to free RAM
                for img in valid_imgs:
                    if hasattr(img, "close"):
                        img.close()

            # Periodic checkpoint flushing
            if args.checkpoint_interval > 0 and (
                time.time() - last_checkpoint_time > args.checkpoint_interval
            ):
                flush_checkpoint()
                last_checkpoint_time = time.time()

        elapsed = time.time() - t0
        print(
            f" -> Feature extraction complete. Processed {valid_count}/{num_missing} images successfully in {elapsed:.2f}s."
        )

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Process interrupted by user! Saving checkpoint...")
        flush_checkpoint()
        print(
            f"Safe checkpoint saved to {ckpt_npy}. You can resume later with --resume."
        )
        sys.exit(0)

    # 5. Assemble Final Outputs
    all_final_keys = []
    all_final_vectors = []
    if unique_embs is not None and len(unique_keys_index) > 0:
        all_final_keys.append(unique_keys_index.values)
        all_final_vectors.append(unique_embs)
    if computed_new_dict:
        new_keys_arr = np.array(
            [orig_photo_keys[idx] for idx in computed_new_dict.keys()]
        )
        new_vecs_arr = np.stack(list(computed_new_dict.values()))
        all_final_keys.append(new_keys_arr)
        all_final_vectors.append(new_vecs_arr)

    if not all_final_vectors:
        print("\n❌ Error: No images were successfully processed or loaded.")
        print(
            "Please verify that your image file paths exist or that you passed the correct directory to --image_root_dir."
        )
        sys.exit(1)

    cat_keys = np.concatenate(all_final_keys)
    cat_vecs = np.concatenate(all_final_vectors, axis=0)

    series_keys = pd.Series(cat_keys)
    u_m = ~series_keys.duplicated(keep="last").values
    master_keys_index = pd.Index(cat_keys[u_m])
    master_vectors = cat_vecs[u_m].astype(target_dtype)

    # Filter df to records that have embeddings
    current_keys = df["photo_key"].astype(str).str.lower().values
    indexer = master_keys_index.get_indexer(current_keys)
    valid_mask = indexer >= 0

    if not valid_mask.any():
        print("\n❌ Error: None of the dataset images have valid embeddings.")
        sys.exit(1)

    if not valid_mask.all():
        failed_count = len(df) - valid_mask.sum()
        print(f"Filtering out {failed_count:,} rows that failed to load or download...")
        df = df[valid_mask].reset_index(drop=True)
        current_keys = df["photo_key"].astype(str).str.lower().values
        indexer = master_keys_index.get_indexer(current_keys)

    embeddings_matrix = master_vectors[indexer]
    df["embedding"] = list(embeddings_matrix)

    print(f"Saving decoupled dataset to: {out_path}")
    save_dataframe(
        df,
        out_path,
        representation_type=args.representation_type,
        precision=args.precision,
        model_name=model_name_for_save,
    )

    # Clean up checkpoint files on successful completion
    for p in [ckpt_npy, ckpt_keys]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    print("✅ Embeddings backfilling and metadata alignment complete!")


if __name__ == "__main__":
    main()
