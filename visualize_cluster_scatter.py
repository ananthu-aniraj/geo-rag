import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import argparse
import umap


def create_scatter_plot(pkl_path, output_png, max_points=10000):
    print(f"Loading clustered data from {pkl_path}...")
    if pkl_path.endswith('.pkl'):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # Assume Parquet
        import pyarrow.parquet as pq
        import time
        try:
            parquet_file = pq.ParquetFile(pkl_path)
            available_cols = parquet_file.schema_arrow.names
            target_cols = ['Platform', 'cluster_id', 'cluster_label', 'parent_cluster_label']
            load_cols = [c for c in target_cols if c in available_cols]
            df = pd.read_parquet(pkl_path, columns=load_cols)
        except Exception:
            df = pd.read_parquet(pkl_path)

    if len(df) == 0:
        print("Error: Dataset is empty.")
        return

    # Sample data if too large for plotting (and for UMAP speed)
    if len(df) > max_points:
        print(f"Sampling {max_points} points from {len(df)} for UMAP visualization...")
        df_sampled = df.sample(n=max_points, random_state=42)
    else:
        df_sampled = df.copy()

    print("Loading raw embedding matrix using PyArrow...")
    t0 = time.time()
    table = pq.read_table(pkl_path, columns=["embedding"])
    
    num_rows = len(table)
    chunked_arr = table['embedding']
    dim = len(chunked_arr.chunk(0)[0].as_py())
    
    embeddings = np.empty((num_rows, dim), dtype=np.float32)
    current_row = 0
    for chunk in chunked_arr.chunks:
        chunk_len = len(chunk)
        flat_chunk = chunk.flatten().to_numpy()
        embeddings[current_row:current_row + chunk_len] = flat_chunk.reshape(chunk_len, dim)
        current_row += chunk_len
        
    del table
    print(f" -> Successfully loaded raw embedding matrix in {time.time() - t0:.2f}s.")

    # Slice embeddings for the sampled indices
    embeddings_sampled = embeddings[df_sampled.index.values]
    del embeddings  # Free memory of full matrix immediately

    print("Extracting and normalizing embeddings for 2D projection...")
    if embeddings_sampled.ndim == 1:
        embeddings_sampled = embeddings_sampled.reshape(1, -1)
    embeddings_norm = normalize(embeddings_sampled)

    print("Computing 2D UMAP projection...")
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
    umap_coords = reducer.fit_transform(embeddings_norm)

    # Prepare DataFrame for plotting
    plot_list = []
    has_parents = 'parent_cluster_label' in df_sampled.columns

    for i, (_, item) in enumerate(df_sampled.iterrows()):
        if has_parents:
            hue_val = item.get('parent_cluster_label', 'Unknown Parent')
        else:
            label_suffix = f" ({item['cluster_label']})" if 'cluster_label' in item and item['cluster_label'] else ""
            hue_val = f"Cluster {item['cluster_id']}{label_suffix}"

        plot_list.append({
            'x': umap_coords[i][0],
            'y': umap_coords[i][1],
            'color_group': hue_val,
            'platform': item['Platform']
        })
    df_plot = pd.DataFrame(plot_list)

    # Create Plot
    plt.figure(figsize=(13, 9))
    sns.set_style("whitegrid")

    # Use tab20/tab20b/tab20c for categorical coloring if parents exist
    color_palette = 'tab20' if has_parents else 'viridis'
    
    scatter = sns.scatterplot(
        data=df_plot,
        x='x', y='y',
        hue='color_group',
        style='platform',
        palette=color_palette,
        alpha=0.7,
        s=45,
        edgecolor='none'
    )

    plt.title("Geo-RAG: Semantic Visual Clusters (UMAP 2D Projection)", fontsize=15)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout()

    print(f"Saving scatter plot to {output_png}...")
    plt.savefig(output_png, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize clusters in a 2D semantic scatter plot using UMAP.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the clustered .pkl file.")
    parser.add_argument("--out", type=str, default="cluster_scatter.png", help="Output PNG file name.")
    args = parser.parse_args()

    create_scatter_plot(args.pkl, args.out)
