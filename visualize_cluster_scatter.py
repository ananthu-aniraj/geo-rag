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
        df = pd.read_parquet(pkl_path)
        data = df.to_dict('records')

    if not data or 'cluster_id' not in data[0]:
        print("Error: Data must be clustered first.")
        return

    # Sample data if too large for plotting (and for UMAP speed)
    if len(data) > max_points:
        print(f"Sampling {max_points} points from {len(data)} for UMAP visualization...")
        import random
        random.seed(42)
        data = random.sample(data, max_points)

    print("Extracting and normalizing embeddings for 2D projection...")
    embeddings = np.array([item['embedding'] for item in data]).squeeze()
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    embeddings_norm = normalize(embeddings)

    print("Computing 2D UMAP projection...")
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
    umap_coords = reducer.fit_transform(embeddings_norm)

    # Prepare DataFrame for plotting
    plot_list = []
    for i, item in enumerate(data):
        plot_list.append({
            'x': umap_coords[i][0],
            'y': umap_coords[i][1],
            'cluster': f"Cluster {item['cluster_id']}",
            'platform': item['Platform']
        })
    df = pd.DataFrame(plot_list)

    # Create Plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    scatter = sns.scatterplot(
        data=df,
        x='x', y='y',
        hue='cluster',
        style='platform',
        palette='viridis',
        alpha=0.7,
        s=60,
        edgecolor='w'
    )

    plt.title("Geo-RAG: Semantic Visual Clusters (UMAP 2D Projection)", fontsize=15)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
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
