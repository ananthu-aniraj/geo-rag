import pickle
import numpy as np
import argparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
import os

def main():
    parser = argparse.ArgumentParser(description="Global Semantic Clustering of Geo-Images.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the .pkl file.")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters.")
    parser.add_argument("--use_umap", action="store_true", help="Use UMAP dimensionality reduction before clustering.")
    parser.add_argument("--reduce_dim", type=int, default=10, help="Dimensions to reduce to via UMAP if --use_umap is set.")
    parser.add_argument("--minibatch", action="store_true", help="Use MiniBatchKMeans for massive datasets (2M+ images).")
    parser.add_argument("--out", type=str, default="clustered_data.pkl", help="Output path.")
    args = parser.parse_args()

    print(f"Loading data from {args.pkl}...")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    if not data:
        print("No data found.")
        return

    print(f"Extracting embeddings for {len(data)} images...")
    embeddings = np.array([item['embedding'] for item in data]).squeeze()
    if embeddings.ndim == 1: 
        embeddings = embeddings.reshape(1, -1)
    
    # Pre-normalization (crucial for Cosine similarity in K-Means)
    print("Normalizing embeddings for cosine similarity...")
    embeddings_norm = normalize(embeddings)

    # Path A: UMAP Reduction
    if args.use_umap:
        import umap
        print(f"Reducing dimensions to {args.reduce_dim}D using UMAP...")
        reducer = umap.UMAP(n_components=args.reduce_dim, metric='cosine', random_state=42)
        cluster_input = reducer.fit_transform(embeddings_norm)
        clustering_mode = f"UMAP ({args.reduce_dim}D)"
    else:
        # Path B: Raw Embeddings
        cluster_input = embeddings_norm
        clustering_mode = "Raw 768D (Normalized)"

    # Clustering
    print(f"Running {'MiniBatch' if args.minibatch else ''}K-Means (k={args.k}) on {clustering_mode} space...")
    if args.minibatch:
        kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=42, n_init=3, batch_size=1024)
    else:
        kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    
    cluster_ids = kmeans.fit_predict(cluster_input)

    print("Updating metadata...")
    for i, item in enumerate(data):
        item['cluster_id'] = int(cluster_ids[i])

    print(f"Saving to {args.out}...")
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nClustering Complete using {clustering_mode}!")
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} images")

if __name__ == "__main__":
    main()
