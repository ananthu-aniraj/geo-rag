import pickle
import numpy as np
import argparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
import os
import pandas as pd
import torch
from transformers import AutoModel
try:
    import faiss
except ImportError:
    faiss = None


# Curated Geo-LULC Vocabulary optimized for TIPSv2
GEO_LULC_VOCAB = {
    "High-density urban area (downtown, skyscrapers)": "An urban scene with tall buildings and high-density architecture.",
    "Residential neighborhood (houses, suburban streets)": "A residential area with houses, gardens, and quiet streets.",
    "Industrial zone or logistics park": "An industrial area with factories, warehouses, and heavy machinery.",
    "Agricultural land (crops, plantations, orchards)": "Farmland with cultivated crops, fields, or orchards.",
    "Transportation infrastructure (highways, railways, bridges)": "Major infrastructure like highways, overpasses, or railways.",
    "Dense forest or woodland": "A dense natural forest or wooded area with many trees.",
    "Sparse shrubland or scrub": "An open area with low-lying bushes, shrubs, and sparse vegetation.",
    "Open grassland or pasture": "Wide open fields of grass, meadows, or grazing land.",
    "Arid land or desert (sandy or rocky)": "A dry, sandy, or rocky desert landscape with very little vegetation.",
    "Snow-covered land, glacier, or ice cap": "A landscape dominated by snow, ice, glaciers, or polar conditions.",
    "Wetland or marshland": "A swampy or marshy area with water and specialized vegetation.",
    "Water body (river, lake, or sea)": "A view dominated by water, such as a river, lake, or the ocean.",
    "Busy street with heavy vehicle traffic": "A street scene filled with cars, buses, and traffic activity.",
    "Pedestrian-only zone or public plaza": "A walkable city area like a square, plaza, or pedestrian street.",
    "Active construction site": "An area with ongoing building work, cranes, and excavated earth.",
    "Outdoor recreation or park activity": "People enjoying a managed park, sports field, or playground."
}


def label_clusters(centroids, model, device, top_k=3):
    """Performs zero-shot labeling of cluster centroids using the Geo-LULC vocab (Top-K)."""
    print(f"Embedding Geo-LULC vocabulary and finding Top-{top_k} labels...")
    categories = list(GEO_LULC_VOCAB.keys())
    prompts = list(GEO_LULC_VOCAB.values())
    
    with torch.no_grad():
        text_features = model.encode_text(prompts).cpu().numpy()
    
    # Normalize for cosine similarity
    text_features = normalize(text_features)
    centroids_norm = normalize(centroids)
    
    # Matrix multiply to get similarities (K, 16)
    sims = np.dot(centroids_norm, text_features.T)
    
    results = []
    for i in range(len(centroids)):
        # Get indices of top_k similarities
        top_indices = np.argsort(sims[i])[::-1][:top_k]
        top_labels = [categories[idx] for idx in top_indices]
        results.append(", ".join(top_labels))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Global Semantic Clustering of Geo-Images.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the .pkl file.")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters.")
    parser.add_argument("--use_umap", action="store_true", help="Use UMAP dimensionality reduction before clustering.")
    parser.add_argument("--reduce_dim", type=int, default=10,
                        help="Dimensions to reduce to via UMAP if --use_umap is set.")
    parser.add_argument("--minibatch", action="store_true",
                        help="Use MiniBatchKMeans for massive datasets (2M+ images).")
    parser.add_argument("--gpu", action="store_true",
                        help="Use FAISS GPU for massive datasets.")
    parser.add_argument("--no_label", action="store_true", help="Disable automatic zero-shot cluster labeling.")
    parser.add_argument("--out", type=str, default="clustered_data.pkl", help="Output path.")
    args = parser.parse_args()

    print(f"Loading data from {args.pkl}...")
    if args.pkl.endswith('.pkl'):
        with open(args.pkl, 'rb') as f:
            data = pickle.load(f)
    else:
        # Assume Parquet
        df = pd.read_parquet(args.pkl)
        data = df.to_dict('records')

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
    if args.gpu:
        if faiss is None:
            raise ImportError("faiss is not installed. Please install it to use --gpu.")
        print(f"Running FAISS GPU K-Means (k={args.k}) on {clustering_mode} space...")
        # FAISS K-Means
        d = cluster_input.shape[1]
        kmeans_faiss = faiss.Kmeans(d, args.k, niter=20, verbose=True, gpu=True, seed=42)
        kmeans_faiss.train(cluster_input.astype('float32'))
        _, cluster_ids = kmeans_faiss.index.search(cluster_input.astype('float32'), 1)
        cluster_ids = cluster_ids.ravel()
        centroids = kmeans_faiss.centroids
    else:
        print(f"Running {'MiniBatch' if args.minibatch else ''}K-Means (k={args.k}) on {clustering_mode} space...")
        if args.minibatch:
            kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=42, n_init=3, batch_size=1024)
        else:
            kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(cluster_input)
        centroids = kmeans.cluster_centers_
    
    # Labeling
    cluster_labels = {}
    if not args.no_label:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TIPSv2 for labeling on {device}...")
        model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).to(device)
        model.eval()
        
        # If we used UMAP, we need centroids in the ORIGINAL 768D space for labeling
        if args.use_umap:
            # Manually compute centroids in raw space
            print("Computing centroids in original space for labeling...")
            d = embeddings_norm.shape[1]
            raw_centroids = np.zeros((args.k, d), dtype=embeddings_norm.dtype)
            np.add.at(raw_centroids, cluster_ids, embeddings_norm)
            counts = np.bincount(cluster_ids, minlength=args.k)
            valid = counts > 0
            raw_centroids[valid] /= counts[valid, None]
        else:
            raw_centroids = centroids
            
        labels = label_clusters(raw_centroids, model, device)
        for i, label in enumerate(labels):
            cluster_labels[i] = label
            print(f"  Cluster {i} Label: {label}")

    print("Updating metadata...")
    for i, item in enumerate(data):
        cid = int(cluster_ids[i])
        item['cluster_id'] = cid
        if cid in cluster_labels:
            item['cluster_label'] = cluster_labels[cid]

    print(f"Saving to {args.out}...")
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nClustering Complete using {clustering_mode}!")
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = f" ({cluster_labels[u]})" if u in cluster_labels else ""
        print(f"  Cluster {u}{lbl}: {c} images")


if __name__ == "__main__":
    main()
