import argparse
import os
import sys
import time

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import load_dataframe, load_embeddings

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def find_elbow_point(k_values, losses):
    """Finds the elbow point on a curve using the distance-to-diagonal method."""
    if len(k_values) < 3:
        return k_values[-1]
    coords = np.column_stack((k_values, losses))
    p1 = coords[0]
    p2 = coords[-1]
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for p in coords:
        p_vec = p - p1
        proj = np.dot(p_vec, line_vec_norm) * line_vec_norm
        perp = p_vec - proj
        distances.append(np.linalg.norm(perp))
    return k_values[np.argmax(distances)]


def main():
    parser = argparse.ArgumentParser(
        description="Find the optimal number of clusters (k) using Spatial Block Cross-Validation with FAISS GPU."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="full_pipeline_output/geo_space_deduplicated.parquet",
        help="Path to the deduplicated Parquet dataset containing embeddings.",
    )
    parser.add_argument(
        "--k_min", type=int, default=10000, help="Minimum cluster count to evaluate."
    )
    parser.add_argument(
        "--k_max", type=int, default=50000, help="Maximum cluster count to evaluate."
    )
    parser.add_argument(
        "--k_step",
        type=int,
        default=10000,
        help="Step size for cluster count evaluation.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of coarse H3 blocks to hold out for validation (default: 0.1).",
    )
    parser.add_argument(
        "--block_res",
        type=int,
        default=4,
        help="H3 resolution for spatial block partitioning (default: 4, ~11k sq km).",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=0,
        help="Maximum training samples to use for evaluation (default: 0, which means use the entire dataset).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Enable FAISS GPU acceleration (default: True).",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="cluster_count_validation.png",
        help="Path to save the validation loss elbow plot.",
    )
    parser.add_argument(
        "--update_params",
        action="store_true",
        help="If set, mathematically calculates the elbow point and updates 'k_clusters' in params.yaml.",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default="params.yaml",
        help="Path to the parameters YAML file.",
    )
    args = parser.parse_args()

    print(
        "================================================================================"
    )
    print("🌍 GEOSPATIAL CLUSTER COUNT VALIDATION (FAISS GPU + SPATIAL BLOCK)")
    print(
        "================================================================================"
    )

    if not FAISS_AVAILABLE:
        print("Error: faiss-gpu (or faiss-cpu) library is required to run this script.")
        print("Please install it (e.g. 'pip install faiss-gpu') before executing.")
        sys.exit(1)

    # 1. Load Parquet (only select H3_Cell and embedding to save memory)
    if not os.path.exists(args.input):
        print(f"Error: Input dataset not found at '{args.input}'")
        sys.exit(1)

    print(f"Loading H3 cells from '{args.input}'...")
    t0 = time.time()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(args.input)
    cols_to_load = ["H3_Cell"]
    if "embedding_idx" in pf.schema_arrow.names:
        cols_to_load.append("embedding_idx")
    df = load_dataframe(args.input, columns=cols_to_load)
    print(f" -> Loaded {len(df):,} H3 cell records in {time.time() - t0:.2f}s.")

    # Load embeddings matrix
    print("Loading raw embedding matrix...")
    t0_emb = time.time()
    embeddings_matrix = load_embeddings(args.input)

    print(
        f" -> Successfully loaded raw embedding matrix in {time.time() - t0_emb:.2f}s."
    )

    # 2. Downscale H3 cells to coarse block resolution (Res 4)
    print(f"Downscaling H3 cells to resolution {args.block_res} parent blocks...")
    t0 = time.time()
    unique_cells = df["H3_Cell"].unique()
    res11_to_parent = {
        c: h3.cell_to_parent(c, args.block_res)
        if h3.get_resolution(c) != args.block_res
        else c
        for c in unique_cells
    }
    df["block_h3"] = df["H3_Cell"].map(res11_to_parent)
    print(f" -> Downscaled {len(unique_cells):,} cells in {time.time() - t0:.2f}s.")

    # 3. Spatial Block Split
    print(f"Partitioning dataset using spatial blocks (Res {args.block_res})...")
    unique_blocks = df["block_h3"].dropna().unique()
    val_size = int(len(unique_blocks) * args.val_ratio)
    np.random.seed(42)
    val_blocks = np.random.choice(unique_blocks, size=val_size, replace=False)

    train_df = df[~df["block_h3"].isin(val_blocks)]
    val_df = df[df["block_h3"].isin(val_blocks)]

    print(f" -> Total spatial blocks: {len(unique_blocks)}")
    print(
        f" -> Training blocks: {len(unique_blocks) - val_size} ({len(train_df):,} images)"
    )
    print(f" -> Validation blocks: {val_size} ({len(val_df):,} images)")

    # 4. Stratified/Downsample for Speed (Only active if sample_limit > 0)
    if args.sample_limit > 0:
        if len(train_df) > args.sample_limit:
            print(
                f"Downsampling training set to {args.sample_limit:,} images for evaluation speed..."
            )
            train_df = train_df.sample(n=args.sample_limit, random_state=42)

        val_limit = int(args.sample_limit * args.val_ratio)
        if len(val_df) > val_limit:
            print(f"Downsampling validation set to {val_limit:,} images...")
            val_df = val_df.sample(n=val_limit, random_state=42)

    print("Slicing train/val embedding matrices...")
    train_emb = embeddings_matrix[train_df.index.values]
    val_emb = embeddings_matrix[val_df.index.values]
    del embeddings_matrix  # Free memory immediately
    print(f" -> Train embeddings shape: {train_emb.shape}")
    print(f" -> Val embeddings shape: {val_emb.shape}")

    # 6. Evaluate K-Means reconstruction loss across different k values using FAISS
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    results = []
    d = train_emb.shape[1]

    print(f"\nStarting FAISS GPU evaluation loop (GPU Enabled: {args.gpu})...")
    for k in k_values:
        print(f"\nEvaluating k = {k}...")
        t_start = time.time()

        # Initialize FAISS Kmeans
        kmeans = faiss.Kmeans(d, k, niter=20, verbose=True, gpu=args.gpu, seed=42)
        kmeans.train(train_emb)
        train_fit_time = time.time() - t_start

        # Calculate Training Loss (Average squared distance to closest centroid)
        # kmeans.obj[-1] is the final sum of squared distances for the training points
        train_loss = kmeans.obj[-1] / len(train_emb)

        # Calculate Validation Loss using FAISS Index search on the GPU
        t_val = time.time()
        # D is the squared L2 distances, I is the indices of the closest centroids
        D, I = kmeans.index.search(val_emb, 1)
        val_loss = np.mean(D)
        val_eval_time = time.time() - t_val

        results.append(
            {
                "k": k,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "fit_time_sec": train_fit_time,
                "val_time_sec": val_eval_time,
            }
        )

        print(f" -> Train Reconstruction Loss: {train_loss:.6f}")
        print(f" -> Val Reconstruction Loss  : {val_loss:.6f}")
        print(f" -> Completed in {time.time() - t_start:.1f}s.")

    # 7. Print Summary Table
    df_results = pd.DataFrame(results)
    print(
        "\n================================================================================"
    )
    print("📊 EVALUATION RESULTS SUMMARY")
    print(
        "================================================================================"
    )
    print(df_results.to_string(index=False))
    print(
        "================================================================================"
    )

    # Find the optimal k using the elbow method on validation loss
    optimal_k = find_elbow_point(
        df_results["k"].tolist(), df_results["val_loss"].tolist()
    )
    print(f"\n💡 Mathematical Elbow Analysis suggests optimal k = {optimal_k}")

    if args.update_params:
        params_path = args.params_path
        if os.path.exists(params_path):
            try:
                import re

                with open(params_path, "r") as f:
                    content = f.read()
                new_content = re.sub(
                    r"(k_clusters:\s*)\d+", f"\\g<1>{optimal_k}", content
                )
                with open(params_path, "w") as f:
                    f.write(new_content)
                print(
                    f"✅ Successfully updated 'k_clusters' to {optimal_k} in {params_path}!"
                )
            except Exception as e:
                print(f"Warning: Failed to update params.yaml: {e}")
        else:
            print(
                "Warning: params.yaml not found at current directory. Skipping auto-update."
            )

    # 8. Generate & Save Plot
    print(f"Generating elbow plot and saving to '{args.output_plot}'...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df_results["k"],
        df_results["train_loss"],
        "o-",
        color="teal",
        label="Train Loss",
    )
    ax.plot(
        df_results["k"],
        df_results["val_loss"],
        "s-",
        color="coral",
        label="Validation Loss",
    )
    ax.set_title(
        "Cluster Count (k) vs. Spatial Reconstruction Loss",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax.set_ylabel("Average Reconstruction Loss (MSE)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    plt.close()
    print("Done! Evaluation completed successfully.")


if __name__ == "__main__":
    main()
