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
from transformers import AutoModel

import src.models.tips_image_encoder as image_encoder
from src.models.vision_model_inference import extract_model_embeddings
from src.utils.io import (
    download_image,
    load_dataframe,
    save_dataframe,
)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone utility to compute/backfill TIPSv2 embeddings (CLS, average patch, or concatenated) for an existing dataset."
    )
    parser.add_argument("--input", "--in", dest="input", type=str, required=True,
                        help="Path to the input metadata file (.parquet, .csv, or .pkl).")
    parser.add_argument("--output", "--out", dest="output", type=str, default=None,
                        help="Path to save the output metadata (defaults to overwriting input in-place).")
    parser.add_argument("--representation_type", type=str, default="cls", choices=["cls", "avg_patch", "cls_avg_patch"],
                        help="Type of representation embedding to extract (cls, avg_patch, or cls_avg_patch).")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16"],
                        help="Stored precision of companion binary file (float32 or float16).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for GPU forward passes.")
    parser.add_argument("--image_root_dir", type=str, nargs="+", default=None,
                        help="Optional root directories containing local images (for offline datasets).")
    parser.add_argument("--tips_model_path", type=str, default=None,
                        help="Optional path to local TIPSv2 checkpoint .npy weight file.")
    parser.add_argument("--tips_model_variant", type=str, default="b", choices=["s", "b", "l", "g"],
                        help="TIPSv2 model variant to load if local checkpoint is specified (s, b, l, g).")
    parser.add_argument("--tips_low_res", action="store_true",
                        help="Use 224x224 input resolution for local checkpoints instead of 448x448.")
    parser.add_argument("--chunk_size", type=int, default=512, help="Number of images to process in parallel download chunks.")
    args = parser.parse_args()

    out_path = args.output if args.output else args.input
    if out_path.endswith('.csv'):
        out_path = out_path.replace('.csv', '.parquet')
        print(f" -> Enforcing decoupled Parquet output format: {out_path}")
    elif out_path.endswith('.pkl') or out_path.endswith('.pickle'):
        out_path = os.path.splitext(out_path)[0] + '.parquet'
        print(f" -> Enforcing decoupled Parquet output format: {out_path}")

    # 1. Load dataset metadata
    print(f"Loading dataset from {args.input}...")
    is_pkl = args.input.endswith('.pkl')
    is_csv = args.input.endswith('.csv')
    if is_pkl:
        with open(args.input, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
    else:
        df = load_dataframe(args.input)

    if len(df) == 0:
        print("Error: Input dataset is empty.")
        return

    # Normalize schema using common column mappings (aligned with process_scraped_data.py)
    # 1. Map other standard fields (excluding URL fallbacks to prevent duplicates)
    col_map = {
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'photo_id': 'Photo_ID',
        'ID': 'Photo_ID',
        'captured_at': 'Captured_At',
        'Date_Observed': 'Captured_At',
        'observed_on_string': 'Captured_At',
        'license': 'License',
        'platform': 'Platform'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 2. Map Image_URL fallback column (only map the first available one to avoid duplicates)
    if 'Image_URL' not in df.columns:
        for fallback in ['local_path', 'Image_Location', 'file_name', 'path']:
            if fallback in df.columns:
                df = df.rename(columns={fallback: 'Image_URL'})
                print(f" -> Mapping missing column 'Image_URL' to existing column '{fallback}'.")
                break

    # Default missing Platform to 'Offline'
    if 'Platform' not in df.columns:
        df['Platform'] = 'Offline'
        print(" -> Platform column missing. Defaulting to 'Offline'.")

    # Verify final key columns exist
    for col in ['Platform', 'Photo_ID', 'Image_URL']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset schema (even after mapping column aliases).")

    # 2. Setup Device & Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing model on {device}...")

    # Load model transforms
    image_size = 224 if (args.tips_model_path and args.tips_low_res) else 448
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.tips_model_path:
        print(f"Loading local checkpoint from {args.tips_model_path}...")
        model_def = {
            's': image_encoder.vit_small14,
            'b': image_encoder.vit_base14,
            'l': image_encoder.vit_large14,
            'g': image_encoder.vit_giant2,
        }[args.tips_model_variant]

        ffn_layer = 'swiglu' if args.tips_model_variant == 'g' else 'mlp'
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
        tipsv2 = model.eval().to(device)
    else:
        print("Loading Hugging Face google/tipsv2-b14 model...")
        tipsv2 = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).eval().to(device)

    # 3. Compute Embeddings in Parallel Chunks
    print(f"Computing '{args.representation_type}' embeddings with {args.precision} precision...")
    num_rows = len(df)

    embeddings_matrix = None
    successful_indices = []

    def download_thread_fn(global_idx, url, photo_id, platform, offline_dirs, image_size):
        try:
            img = download_image(url, photo_id=photo_id, platform=platform, offline_dirs=offline_dirs, image_size=image_size)
            return global_idx, img
        except Exception:
            return global_idx, None

    t0 = time.time()
    valid_count = 0

    # Process dataset in parallel download chunks (e.g. 512 images at a time)
    chunk_size = args.chunk_size
    print(f"Processing dataset of {num_rows} images in chunks of {chunk_size} with parallel downloads...")

    for chunk_start in range(0, num_rows, chunk_size):
        chunk_df = df.iloc[chunk_start: chunk_start + chunk_size]
        print(
            f"\n--- Processing chunk {chunk_start // chunk_size + 1} ({chunk_start} to {chunk_start + len(chunk_df)}) ---")

        # Parallel downloads for the chunk
        db_dict = {}
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(
                    download_thread_fn,
                    global_idx,
                    row['Image_URL'],
                    row['Photo_ID'],
                    row['Platform'],
                    args.image_root_dir,
                    image_size
                ): global_idx
                for global_idx, row in chunk_df.iterrows()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Chunk"):
                global_idx, img = future.result()
                if img is not None:
                    db_dict[global_idx] = img

        # Run model inference in batches of args.batch_size on successfully downloaded images
        active_indices = sorted(list(db_dict.keys()))
        valid_imgs = [db_dict[idx] for idx in active_indices]

        if len(valid_imgs) > 0:
            for b_start in range(0, len(valid_imgs), args.batch_size):
                batch_imgs = valid_imgs[b_start: b_start + args.batch_size]
                batch_indices = active_indices[b_start: b_start + args.batch_size]

                try:
                    batch_tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
                    with torch.no_grad():
                        features = extract_model_embeddings(tipsv2, batch_tensors,
                                                            representation_type=args.representation_type)

                    if embeddings_matrix is None:
                        dim = features.shape[1]
                        dtype = np.float32 if args.precision == 'float32' else np.float16
                        embeddings_matrix = np.zeros((num_rows, dim), dtype=dtype)
                        print(f"Detected representation feature dimension: {dim}")

                    for b_i, global_idx in enumerate(batch_indices):
                        embeddings_matrix[global_idx] = features[b_i].astype(embeddings_matrix.dtype)
                        successful_indices.append(global_idx)
                        valid_count += 1
                except Exception as e:
                    print(f"Warning: Failed processing model batch starting at index {batch_indices[0]}: {e}")

            # Immediately close PIL images to free RAM
            for img in valid_imgs:
                if hasattr(img, 'close'):
                    img.close()

    elapsed = time.time() - t0
    print(
        f" -> Processed {valid_count}/{num_rows} images successfully in {elapsed:.2f}s ({num_rows / elapsed:.2f} img/s).")

    # Filter out records where the download/load failed (expired links, etc.)
    if embeddings_matrix is None:
        print("\n❌ Error: No images were successfully processed or loaded.")
        print("Please verify that your image file paths exist or that you passed the correct directory to --image_root_dir.")
        sys.exit(1)

    if len(successful_indices) < num_rows:
        print(f"Filtering out {num_rows - len(successful_indices)} rows that failed to load or download...")
        df = df.iloc[successful_indices].reset_index(drop=True)
        embeddings_matrix = embeddings_matrix[successful_indices]

    # 4. Save Outputs
    # Re-integrate embeddings into the dataframe so that save_dataframe handles decoupling and key mapping automatically
    df['embedding'] = list(embeddings_matrix)

    print(f"Saving decoupled dataset to: {out_path}")
    save_dataframe(df, out_path, representation_type=args.representation_type, precision=args.precision)

    print("✅ Embeddings backfilling and metadata alignment complete!")


if __name__ == "__main__":
    main()
