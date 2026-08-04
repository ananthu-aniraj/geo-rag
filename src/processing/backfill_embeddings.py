import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel

from src.models.vision_model_inference import extract_model_embeddings
from src.utils.io import load_dataframe, save_dataframe


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
    args = parser.parse_args()

    out_path = args.output if args.output else args.input

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

    # Verify key columns exist
    for col in ['Platform', 'Photo_ID', 'Image_URL']:
        if col not in df.columns:
            if col == 'Image_URL' and 'local_path' in df.columns:
                df['Image_URL'] = df['local_path']
            else:
                raise ValueError(f"Required column '{col}' is missing from the dataset schema.")

    # 2. Setup Device & Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing model on {device}...")

    # Load model transforms
    from torchvision import transforms
    image_size = 224 if (args.tips_model_path and args.tips_low_res) else 448
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.tips_model_path:
        print(f"Loading local checkpoint from {args.tips_model_path}...")
        from src.models import image_encoder
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

    # Import download_image from process_scraped_data
    from src.processing.process_scraped_data import download_image
    
    # 3. Compute Embeddings Batch-by-Batch
    print(f"Computing '{args.representation_type}' embeddings with {args.precision} precision...")
    num_rows = len(df)
    
    embeddings_matrix = None
    
    batch_imgs = []
    batch_indices = []
    
    t0 = time.time()
    valid_count = 0
    
    for idx in tqdm(range(num_rows), desc="Processing images"):
        row = df.iloc[idx]
        url = row['Image_URL']
        photo_id = row['Photo_ID']
        platform = row['Platform']
        
        # Download / Load image
        img = download_image(url, photo_id, platform, args.image_root_dir)
        if img is None:
            if platform.lower() == 'mapillary' and not url.startswith('mapillary://'):
                url = f"mapillary://{photo_id}"
            elif platform.lower() == 'kartaview' and not url.startswith('kartaview://'):
                url = f"kartaview://{photo_id}"
            img = download_image(url, photo_id, platform, args.image_root_dir)

        if img is not None:
            try:
                img_tensor = transform(img)
                batch_imgs.append(img_tensor)
                batch_indices.append(idx)
                valid_count += 1
            except Exception as e:
                print(f"Warning: Failed to preprocess image {photo_id}: {e}")
            finally:
                if hasattr(img, 'close'):
                    img.close()
                    
        # When batch size is reached or at the very end
        if len(batch_imgs) >= args.batch_size or (idx == num_rows - 1 and batch_imgs):
            batch_tensors = torch.stack(batch_imgs).to(device)
            
            with torch.no_grad():
                features = extract_model_embeddings(tipsv2, batch_tensors, representation_type=args.representation_type)
            
            # Check dimension and initialize output matrix if not done yet
            if embeddings_matrix is None:
                dim = features.shape[1]
                dtype = np.float32 if args.precision == 'float32' else np.float16
                embeddings_matrix = np.zeros((num_rows, dim), dtype=dtype)
                print(f"Detected representation feature dimension: {dim}")
                
            # Place into the pre-allocated matrix
            for b_i, global_idx in enumerate(batch_indices):
                embeddings_matrix[global_idx] = features[b_i].astype(embeddings_matrix.dtype)
                
            # Clear batch lists
            batch_imgs = []
            batch_indices = []

    elapsed = time.time() - t0
    print(f" -> Processed {valid_count}/{num_rows} images successfully in {elapsed:.2f}s ({num_rows/elapsed:.2f} img/s).")

    # 4. Save Outputs
    db_dir = os.path.dirname(os.path.abspath(out_path))
    base_name = os.path.splitext(os.path.basename(out_path))[0]
    
    from src.utils.io import get_core_base_name
    core_name = get_core_base_name(base_name)
    
    suffix = args.representation_type
    npy_name = f"{core_name}_{suffix}_embeddings.npy"
    npy_path = os.path.join(db_dir, npy_name)
    
    print(f"Saving companion embeddings matrix to: {npy_path}")
    np.save(npy_path, embeddings_matrix)
    
    # Update dataframe mapping columns
    df['embedding_idx'] = np.arange(len(df), dtype=np.int32)
    
    print(f"Saving metadata dataset to: {out_path}")
    # Drop existing embedded 'embedding' column if it exists to strictly enforce decoupled format
    meta_cols = [c for c in df.columns if c != 'embedding']
    df_meta = df[meta_cols].copy()
    
    save_dataframe(df_meta, out_path, representation_type=args.representation_type, precision=args.precision)
    
    print("✅ Embeddings backfilling and metadata alignment complete!")

if __name__ == "__main__":
    main()
