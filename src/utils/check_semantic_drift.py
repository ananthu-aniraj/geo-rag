import argparse
import sys
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize

try:
    import faiss
except ImportError:
    faiss = None

def main():
    parser = argparse.ArgumentParser(description="Check for semantic drift between new images and existing centroids.")
    parser.add_argument("--input", type=str, required=True, help="Path to new/combined dataset Parquet file.")
    parser.add_argument("--centroids_parquet", type=str, required=True, help="Path to pre-existing clustered Parquet database.")
    parser.add_argument("--k_clusters", type=int, default=40000, help="Number of clusters.")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of new embeddings to sample.")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU for FAISS search.")
    parser.add_argument("--no_gpu", action="store_false", dest="gpu")
    parser.add_argument("--outlier_threshold", type=float, default=0.70, help="Cosine similarity below which a vector is an outlier.")
    parser.add_argument("--drift_ratio_threshold", type=float, default=0.03, help="Outlier ratio threshold (e.g. 0.03 for 3%).")
    args = parser.parse_args()

    # Fallback to fit if the pre-existing file doesn't exist
    if not os.path.exists(args.centroids_parquet):
        print("fit")
        return

    # 1. Load a sample of the new embeddings
    try:
        pf_new = pq.ParquetFile(args.input)
        new_embeddings = []
        rows_read = 0
        for rg in range(pf_new.num_row_groups):
            table = pf_new.read_row_group(rg, columns=["embedding"])
            embs = np.vstack(table['embedding'].to_numpy()).astype(np.float32)
            new_embeddings.append(embs)
            rows_read += len(embs)
            if rows_read >= args.sample_size:
                break
        if len(new_embeddings) == 0:
            print("fit")
            return
        new_embs = np.vstack(new_embeddings)[:args.sample_size]
        new_embs = normalize(new_embs).astype(np.float32)
    except Exception:
        print("fit")
        return

    # 2. Load the old centroids from args.centroids_parquet
    try:
        pf_old = pq.ParquetFile(args.centroids_parquet)
        first_rg = pf_old.read_row_group(0, columns=["embedding"])
        first_embs = np.vstack(first_rg['embedding'].to_numpy())
        dim = first_embs.shape[1]

        # Accumulate centroids dynamically using vectorized addition
        raw_centroids = np.zeros((args.k_clusters, dim), dtype=np.float32)
        counts = np.zeros(args.k_clusters, dtype=np.int64)

        for rg in range(pf_old.num_row_groups):
            table_old = pf_old.read_row_group(rg, columns=['cluster_id', 'embedding'])
            df_rg_old = table_old.to_pandas()
            if len(df_rg_old) == 0:
                continue
            embs_old = np.vstack(df_rg_old['embedding'].values).astype(np.float32)
            c_ids_old = df_rg_old['cluster_id'].values
            
            valid_mask = (c_ids_old >= 0) & (~pd.isna(c_ids_old)) & (c_ids_old < args.k_clusters)
            c_ids_valid = c_ids_old[valid_mask].astype(np.int32)
            
            np.add.at(raw_centroids, c_ids_valid, embs_old[valid_mask])
            counts += np.bincount(c_ids_valid, minlength=args.k_clusters)

        valid_counts = counts > 0
        raw_centroids[valid_counts] /= counts[valid_counts, None]
        unique_ids = np.where(valid_counts)[0]
        child_centroids = raw_centroids[unique_ids]
        child_centroids = normalize(child_centroids).astype(np.float32)
    except Exception:
        print("fit")
        return

    # 3. Perform FAISS search
    try:
        gpu_enabled = args.gpu and (faiss is not None)
        if gpu_enabled:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.IndexFlatIP(dim)
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                gpu_enabled = False
                index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(child_centroids)
        similarities, _ = index.search(new_embs, 1)
        similarities = similarities.ravel()

        # 4. Calculate outlier ratio
        outliers = similarities < args.outlier_threshold
        outlier_ratio = np.mean(outliers)

        if outlier_ratio > args.drift_ratio_threshold:
            print("fit")
        else:
            print("assign")
    except Exception:
        print("fit")

if __name__ == "__main__":
    main()
