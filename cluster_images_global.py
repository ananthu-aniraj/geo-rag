import pickle
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import os

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Global K-Means Clustering of Geo-Images.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the .pkl file containing embeddings.")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters.")
    parser.add_argument("--out", type=str, default="clustered_data.pkl", help="Output path for updated pkl.")
    args = parser.parse_args()

    print(f"Loading data from {args.pkl}...")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    if not data:
        print("No data found.")
        return

    print(f"Extracting embeddings for {len(data)} images...")
    # Squeeze to ensure embeddings are (N, D) even if saved as (N, 1, D)
    embeddings = np.array([item['embedding'] for item in data]).squeeze()
    
    # If only one image, squeeze() might make it 1D, so ensure it's 2D (N, D)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # Normalize embeddings for better K-Means performance (Cosine distance approximation)
    print("Normalizing embeddings...")
    embeddings_norm = normalize(embeddings)

    print(f"Running K-Means (k={args.k})...")
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings_norm)

    print("Updating metadata with cluster IDs...")
    for i, item in enumerate(data):
        item['cluster_id'] = int(cluster_ids[i])

    print(f"Saving clustered data to {args.out}...")
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)

    print("\nClustering Complete!")
    print(f"Cluster distribution:")
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} images")

if __name__ == "__main__":
    main()
