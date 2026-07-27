import argparse
import os
import pickle
import sys
import time

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize


def cluster_data(input_embeddings, k, gpu_enabled=True, minibatch_enabled=False):
    """Performs K-Means clustering using FAISS (GPU/CPU) or scikit-learn."""
    if k <= 0:
        return np.array([]), np.array([])
    if len(input_embeddings) < k:
        k = len(input_embeddings)

    input_embeddings = input_embeddings.astype(np.float32)
    d = input_embeddings.shape[1]

    if gpu_enabled:
        if faiss is None:
            print("[WARNING] FAISS not available. Falling back to scikit-learn K-Means.")
            gpu_enabled = False
        else:
            print(f"Running FAISS GPU K-Means (k={k}, dim={d}, niter=20)...")
            t0 = time.time()
            kmeans_faiss = faiss.Kmeans(d, k, niter=20, verbose=True, gpu=True, seed=42)
            kmeans_faiss.train(input_embeddings)
            _, cluster_ids = kmeans_faiss.index.search(input_embeddings, 1)
            cluster_ids = cluster_ids.ravel()
            centroids = kmeans_faiss.centroids
            print(f" -> FAISS GPU K-Means completed in {time.time() - t0:.2f}s.")
            return cluster_ids, centroids

    if not gpu_enabled:
        t0 = time.time()
        if minibatch_enabled:
            print(f"Running MiniBatchKMeans (k={k})...")
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024)
        else:
            print(f"Running scikit-learn KMeans (k={k})...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(input_embeddings)
        centroids = kmeans.cluster_centers_
        print(f" -> K-Means completed in {time.time() - t0:.2f}s.")
        return cluster_ids, centroids


def main():
    parser = argparse.ArgumentParser(description="Global FAISS Semantic Clustering of Geo-Images (GPU-Accelerated).")
    parser.add_argument("--pkl", type=str, required=True, help="Path to input Parquet or .pkl file.")
    parser.add_argument("--k", type=int, default=40000, help="Number of fine-grained child clusters.")
    parser.add_argument("--k_parents", type=int, default=None, help="Number of parent clusters for hierarchical grouping (default: k // 80).")
    parser.add_argument("--minibatch", action="store_true", help="Use MiniBatchKMeans fallback if GPU is disabled.")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use FAISS GPU for clustering.")
    parser.add_argument("--no_gpu", action="store_false", dest="gpu", help="Disable FAISS GPU acceleration.")
    parser.add_argument("--out", type=str, default="clustered_data.parquet", help="Output Parquet path.")
    parser.add_argument("--clustering_mode", type=str, choices=["fit", "assign"], default="fit", help="Clustering mode: fit (re-cluster all) or assign (map to existing centroids).")
    parser.add_argument("--centroids_parquet", type=str, default=None, help="Path to pre-existing clustered Parquet database to load centroids and metadata from.")
    args = parser.parse_args()

    k_parents = args.k_parents
    if args.k_parents is None:
        k_parents = max(2, args.k // 80)

    print(f"Loading dataset from {args.pkl}...")
    if args.pkl.endswith('.pkl'):
        with open(args.pkl, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        del data
        embeddings = np.vstack(df['embedding'].values).astype(np.float32)
    else:
        try:
            parquet_file = pq.ParquetFile(args.pkl)
            metadata_cols = [c for c in parquet_file.schema_arrow.names if c != 'embedding']
            df = pd.read_parquet(args.pkl, columns=metadata_cols)
        except Exception as e:
            print(f"Schema inspection fallback: {e}")
            df = pd.read_parquet(args.pkl)

        print("Loading raw embedding matrix using PyArrow...")
        t0 = time.time()
        table = pq.read_table(args.pkl, columns=["embedding"])
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
        print(f" -> Successfully loaded {num_rows:,} embeddings in {time.time() - t0:.2f}s.")

    if 'Latitude' in df.columns:
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    if 'Longitude' in df.columns:
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    if len(df) == 0:
        print("No data found.")
        return

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    print("\nNormalizing embeddings for spherical cosine similarity clustering...")
    embeddings_norm = normalize(embeddings).astype(np.float32)

    if args.clustering_mode == "assign":
        if not args.centroids_parquet or not os.path.exists(args.centroids_parquet):
            print(f"Error: Centroids Parquet database '{args.centroids_parquet}' is required and must exist in assign mode.")
            sys.exit(1)
            
        print(f"Loading pre-existing clustered database from {args.centroids_parquet}...")
        start_assign = time.time()
        pf_old = pq.ParquetFile(args.centroids_parquet)
        
        # Accumulate centroids dynamically from the old clustered database using vectorized math
        raw_centroids = np.zeros((args.k, dim), dtype=np.float32)
        counts = np.zeros(args.k, dtype=np.int64)
        child_to_parent = np.zeros(args.k, dtype=np.int32)
        
        print("Computing centroids from pre-existing database...")
        for rg in range(pf_old.num_row_groups):
            table_old = pf_old.read_row_group(rg, columns=['cluster_id', 'parent_cluster_id', 'embedding'])
            df_rg_old = table_old.to_pandas()
            if len(df_rg_old) == 0:
                continue
                
            embs_old = np.vstack(df_rg_old['embedding'].values).astype(np.float32)
            c_ids_old = df_rg_old['cluster_id'].values
            p_ids_old = df_rg_old['parent_cluster_id'].values
            
            valid_mask = (c_ids_old >= 0) & (~pd.isna(c_ids_old)) & (c_ids_old < args.k)
            c_ids_valid = c_ids_old[valid_mask].astype(np.int32)
            
            # Vectorized addition and bincount accumulation
            np.add.at(raw_centroids, c_ids_valid, embs_old[valid_mask])
            counts += np.bincount(c_ids_valid, minlength=args.k)
            
            # Map child to parent vectorially
            child_to_parent[c_ids_valid] = np.where(pd.isna(p_ids_old[valid_mask]), c_ids_valid // 80, p_ids_old[valid_mask]).astype(np.int32)

        # Normalize centroids by their counts
        valid_counts = counts > 0
        raw_centroids[valid_counts] /= counts[valid_counts, None]
        
        # Keep only active cluster IDs
        unique_ids = np.where(valid_counts)[0]
        child_centroids = raw_centroids[unique_ids]
        child_to_parent = child_to_parent[unique_ids]
        
        print(f" -> Successfully loaded {len(unique_ids)} centroids from database.")
        
        print(f"\nAssigning {len(embeddings_norm):,} embeddings to pre-existing child centroids...")
        child_centroids = normalize(child_centroids).astype(np.float32)
        
        if args.gpu and faiss is not None:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.IndexFlatIP(dim)
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("Using FAISS GPU index for assignment.")
            except Exception as e:
                print(f"[WARNING] GPU FAISS error ({e}), falling back to CPU.")
                index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatIP(dim)
            
        index.add(child_centroids)
        _, child_indices = index.search(embeddings_norm, 1)
        child_indices = child_indices.ravel()
        
        # Convert indices back to correct cluster IDs
        child_cluster_ids = np.array([unique_ids[idx] for idx in child_indices])
        
        df['cluster_id'] = child_cluster_ids.astype(int)
        
        # Load unique cluster metadata labels from the old database
        print("Merging VLM labels and metadata from pre-existing database...")
        meta_cols = ['cluster_id']
        for col in ['cluster_label', 'parent_cluster_label', 'cluster_description', 'parent_cluster_id']:
            if col in pf_old.schema_arrow.names:
                meta_cols.append(col)
                
        # Read the unique metadata per cluster
        old_meta_df = pd.read_parquet(args.centroids_parquet, columns=meta_cols).drop_duplicates('cluster_id')
        df = df.merge(old_meta_df, on='cluster_id', how='left')
        
        # Fallback values
        if 'parent_cluster_id' not in df.columns:
            parent_map_dict = {unique_ids[i]: child_to_parent[i] for i in range(len(unique_ids))}
            df['parent_cluster_id'] = np.array([parent_map_dict.get(cid, cid // 80) for cid in df['cluster_id']])
            
        print(f" -> Mapping completed in {time.time() - start_assign:.2f}s.")
        
    else:
        # 1. Child Clustering
        print(f"\n--- [Stage 1/2] Fine-Grained Child Clustering (k={args.k}) ---")
        child_cluster_ids, child_centroids = cluster_data(embeddings_norm, args.k, gpu_enabled=args.gpu, minibatch_enabled=args.minibatch)

        # Re-compute exact mean centroids in original embedding space
        print("\nComputing exact child cluster centroids...")
        d = embeddings_norm.shape[1]
        raw_centroids = np.zeros((args.k, d), dtype=np.float32)
        valid_mask = (child_cluster_ids >= 0)
        np.add.at(raw_centroids, child_cluster_ids[valid_mask], embeddings_norm[valid_mask])
        counts = np.bincount(child_cluster_ids[valid_mask], minlength=args.k)
        valid_counts = counts > 0
        raw_centroids[valid_counts] /= counts[valid_counts, None]

        # 2. Hierarchical Parent Clustering using FAISS
        print(f"\n--- [Stage 2/2] Hierarchical Parent Clustering (k_parents={k_parents}) ---")
        centroids_norm_hac = normalize(raw_centroids).astype(np.float32)
        parent_ids, _ = cluster_data(centroids_norm_hac, k_parents, gpu_enabled=args.gpu, minibatch_enabled=args.minibatch)

        # Assign cluster IDs to DataFrame
        print("\nAssigning cluster IDs to metadata...")
        df['cluster_id'] = child_cluster_ids.astype(int)
        parent_id_map = {cid: int(parent_ids[cid]) for cid in range(args.k) if cid < len(parent_ids)}
        df['parent_cluster_id'] = df['cluster_id'].map(parent_id_map)

    # Save output
    print(f"\nSaving clustered dataset to {args.out}...")
    if args.out.endswith('.pkl'):
        data = df.to_dict('records')
        for i, item in enumerate(data):
            item['embedding'] = embeddings[i]
        with open(args.out, 'wb') as f:
            pickle.dump(data, f)
    else:
        df['embedding'] = list(embeddings)
        df.to_parquet(args.out, index=False)

    print(f"\n✅ Global FAISS Clustering Complete! Saved {len(df):,} clustered rows to {args.out}.")


if __name__ == "__main__":
    main()
