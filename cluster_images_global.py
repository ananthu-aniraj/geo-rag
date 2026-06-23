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


# Exhaustive Natural LULC Vocabulary
NATURAL_LULC_VOCAB = {
    "Dense forest": "A dense natural forest or wooded area with many trees.",
    "Shrubland/scrub": "Low-lying bushes, shrubs, and sparse dry vegetation.",
    "Grassland/pasture": "Open grassy fields, meadows, grazing pastures, or plains.",
    "Arid desert/dunes": "Dry sandy dunes, rocky deserts, or barren arid landscapes.",
    "Snow/glacier": "Snow-covered mountains, glaciers, ice caps, or winter landscapes.",
    "Wetland/marsh": "Swampy, marshy, or boggy area with specialized vegetation.",
    "Water body": "Rivers, lakes, oceans, coastlines, or open water scenes.",
    "Barren rock/cliffs": "Exposed bedrock, mountains, cliffs, canyons, or rocky slopes."
}

# Exhaustive Man-made LULC Vocabulary
MAN_MADE_LULC_VOCAB = {
    "High-density urban": "Skyscrapers, city downtowns, high-rise buildings, and dense city center scenes.",
    "Suburban/residential": "Quiet streets with houses, gardens, villas, and residential yards.",
    "Commercial/retail": "Storefronts, shopping districts, malls, plazas, or commercial avenues.",
    "Industrial zone": "Factories, warehouses, refineries, power plants, and industrial complexes.",
    "Transportation infrastructure": "Highways, paved roads, railways, overpasses, bridges, or tunnels.",
    "Agricultural land": "Cultivated fields, crop rows, orchards, vineyards, or plantations.",
    "Active construction site": "Excavations, cranes, scaffolding, and buildings under construction.",
    "Managed park/recreation": "City parks, gardens, playgrounds, athletic fields, or golf courses."
}


def label_clusters(centroids, text_features, categories, top_k=3):
    """Performs zero-shot labeling of cluster centroids using given text features."""
    centroids_norm = normalize(centroids)
    sims = np.dot(centroids_norm, text_features.T)
    results = []
    for i in range(len(centroids)):
        top_indices = np.argsort(sims[i])[::-1][:top_k]
        top_labels = [categories[idx] for idx in top_indices]
        results.append(", ".join(top_labels))
    return results


def cluster_subset(subset_input, k_subset, gpu_enabled, minibatch_enabled, faiss_module):
    """Helper to run K-Means on a subset of embeddings."""
    if k_subset <= 0:
        return np.array([]), np.array([])
    if len(subset_input) < k_subset:
        k_subset = len(subset_input)
        
    if gpu_enabled:
        if faiss_module is None:
            raise ImportError("faiss is not installed. Please install it to use --gpu.")
        d = subset_input.shape[1]
        kmeans_faiss = faiss_module.Kmeans(d, k_subset, niter=20, verbose=True, gpu=True, seed=42)
        kmeans_faiss.train(subset_input.astype('float32'))
        _, cluster_ids = kmeans_faiss.index.search(subset_input.astype('float32'), 1)
        cluster_ids = cluster_ids.ravel()
        centroids = kmeans_faiss.centroids
    else:
        if minibatch_enabled:
            kmeans = MiniBatchKMeans(n_clusters=k_subset, random_state=42, n_init=3, batch_size=1024)
        else:
            kmeans = KMeans(n_clusters=k_subset, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(subset_input)
        centroids = kmeans.cluster_centers_
        
    return cluster_ids, centroids


def main():
    parser = argparse.ArgumentParser(description="Global Semantic Clustering of Geo-Images.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the .pkl or parquet file.")
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
        df = pd.read_parquet(args.pkl)
        data = df.to_dict('records')

    if not data:
        print("No data found.")
        return

    print(f"Extracting embeddings for {len(data)} images...")
    embeddings = np.array([item['embedding'] for item in data]).squeeze()
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # 1. Zero-shot screening and classification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TIPSv2 model on {device} to encode classification prompts...")
    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).to(device)
    model.eval()

    SCREENING_PROMPTS = [
        "An outdoor scene, landscape, street view, city street, building exterior, nature, or a road view looking out through a vehicle windshield or window.",
        "An indoor scene, room interior, office, household, inside of a building, or a vehicle cabin interior focusing on passengers, seats, or group selfies."
    ]
    
    SUBTYPE_PROMPTS = [
        "A natural landscape, wild nature, forest, grassland, desert, mountain, or water body.",
        "A man-made structure, city, road, building, industrial area, or transport infrastructure."
    ]

    natural_categories = list(NATURAL_LULC_VOCAB.keys())
    natural_prompts = list(NATURAL_LULC_VOCAB.values())
    man_made_categories = list(MAN_MADE_LULC_VOCAB.keys())
    man_made_prompts = list(MAN_MADE_LULC_VOCAB.values())

    # Encode prompts
    with torch.no_grad():
        screening_features = normalize(model.encode_text(SCREENING_PROMPTS).cpu().numpy())
        subtype_features = normalize(model.encode_text(SUBTYPE_PROMPTS).cpu().numpy())
        natural_features = normalize(model.encode_text(natural_prompts).cpu().numpy())
        man_made_features = normalize(model.encode_text(man_made_prompts).cpu().numpy())

    # Unload model to save memory
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print("Prompts embedded successfully. Model unloaded from memory.")

    print("Normalizing embeddings for cosine similarity classification...")
    embeddings_norm = normalize(embeddings)

    # Screening execution (Outdoor vs Indoor Noise)
    screening_sims = np.dot(embeddings_norm, screening_features.T)
    screening_preds = np.argmax(screening_sims, axis=1)

    indoor_indices = np.where(screening_preds == 1)[0]
    outdoor_indices = np.where(screening_preds == 0)[0]

    # Subtype execution for outdoor images (Natural vs Man-made)
    natural_indices = np.array([], dtype=int)
    man_made_indices = np.array([], dtype=int)
    k_nat, k_man = 0, 0

    if len(outdoor_indices) > 0:
        outdoor_embeddings = embeddings_norm[outdoor_indices]
        subtype_sims = np.dot(outdoor_embeddings, subtype_features.T)
        subtype_preds = np.argmax(subtype_sims, axis=1)
        
        natural_indices = outdoor_indices[subtype_preds == 0]
        man_made_indices = outdoor_indices[subtype_preds == 1]

    print(f"\nScreening Summary:")
    print(f"  Outdoor/Geo images: {len(outdoor_indices)}")
    print(f"  Indoor images (excluded): {len(indoor_indices)}")
    print(f"Outdoor Classification:")
    print(f"  Natural: {len(natural_indices)}")
    print(f"  Man-made: {len(man_made_indices)}")

    # Dynamic K allocation
    n_nat = len(natural_indices)
    n_man = len(man_made_indices)
    n_total = n_nat + n_man

    if n_total == 0:
        print("No outdoor images found to cluster.")
        return

    if args.k < 2:
        k_nat = 1 if n_nat >= n_man else 0
        k_man = 1 - k_nat
    else:
        k_nat = int(round(args.k * (n_nat / n_total)))
        k_nat = max(1, min(k_nat, args.k - 1))
        k_man = args.k - k_nat

    print(f"Allocated K: K_natural = {k_nat}, K_manmade = {k_man}")

    # Prepare inputs for clustering
    if args.use_umap:
        import umap
        print(f"\nReducing dimensions to {args.reduce_dim}D using UMAP...")
        reducer = umap.UMAP(n_components=args.reduce_dim, metric='cosine', random_state=42)
        cluster_input = reducer.fit_transform(embeddings_norm)
        clustering_mode = f"UMAP ({args.reduce_dim}D)"
    else:
        cluster_input = embeddings_norm
        clustering_mode = "Raw 768D (Normalized)"

    # Cluster subsets
    global_cluster_ids = np.zeros(len(data), dtype=int)
    global_cluster_ids[indoor_indices] = -1

    # Cluster Natural
    if k_nat > 0 and len(natural_indices) > 0:
        print(f"\nClustering Natural subset on {clustering_mode} space...")
        nat_input = cluster_input[natural_indices]
        nat_cluster_ids, _ = cluster_subset(nat_input, k_nat, args.gpu, args.minibatch, faiss)
        global_cluster_ids[natural_indices] = nat_cluster_ids
        
    # Cluster Man-made
    if k_man > 0 and len(man_made_indices) > 0:
        print(f"\nClustering Man-made subset on {clustering_mode} space...")
        man_input = cluster_input[man_made_indices]
        man_cluster_ids, _ = cluster_subset(man_input, k_man, args.gpu, args.minibatch, faiss)
        global_cluster_ids[man_made_indices] = man_cluster_ids + k_nat

    # Centroid derivation and labeling
    cluster_labels = {}
    cluster_labels[-1] = "Noise: Indoor"

    if not args.no_label:
        print("\nComputing centroids in original 768D space for labeling...")
        d = embeddings_norm.shape[1]
        raw_centroids = np.zeros((args.k, d), dtype=embeddings_norm.dtype)
        valid_mask = (global_cluster_ids >= 0)
        np.add.at(raw_centroids, global_cluster_ids[valid_mask], embeddings_norm[valid_mask])
        counts = np.bincount(global_cluster_ids[valid_mask], minlength=args.k)
        valid_counts = counts > 0
        raw_centroids[valid_counts] /= counts[valid_counts, None]

        if k_nat > 0:
            nat_centroids = raw_centroids[:k_nat]
            nat_labels = label_clusters(nat_centroids, natural_features, natural_categories)
            for i, label in enumerate(nat_labels):
                cluster_labels[i] = label
                print(f"  Cluster {i} (Natural) Label: {label}")

        if k_man > 0:
            man_centroids = raw_centroids[k_nat : k_nat + k_man]
            man_labels = label_clusters(man_centroids, man_made_features, man_made_categories)
            for i, label in enumerate(man_labels):
                cid = k_nat + i
                cluster_labels[cid] = label
                print(f"  Cluster {cid} (Man-made) Label: {label}")

    print("\nUpdating metadata...")
    for i, item in enumerate(data):
        cid = int(global_cluster_ids[i])
        item['cluster_id'] = cid
        if cid in cluster_labels:
            item['cluster_label'] = cluster_labels[cid]

    print(f"Saving to {args.out}...")
    if args.out.endswith('.pkl'):
        with open(args.out, 'wb') as f:
            pickle.dump(data, f)
    else:
        pd.DataFrame(data).to_parquet(args.out)

    print(f"\nClustering Complete!")
    unique, counts = np.unique(global_cluster_ids, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = f" ({cluster_labels[u]})" if u in cluster_labels else ""
        print(f"  Cluster {u}{lbl}: {c} images")


if __name__ == "__main__":
    main()
