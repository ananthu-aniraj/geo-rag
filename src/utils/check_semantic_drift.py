import argparse
import os

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize


def main():
    parser = argparse.ArgumentParser(
        description="Check for semantic drift of new images against existing centroids."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to new/combined dataset Parquet file.",
    )
    parser.add_argument(
        "--centroids_parquet",
        type=str,
        required=True,
        help="Path to pre-existing clustered Parquet database.",
    )
    parser.add_argument(
        "--k_clusters", type=int, default=40000, help="Number of clusters."
    )
    parser.add_argument(
        "--gpu", action="store_true", default=True, help="Use GPU for FAISS search."
    )
    parser.add_argument("--no_gpu", action="store_false", dest="gpu")
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=0.70,
        help="Cosine similarity below which a vector is an outlier.",
    )
    parser.add_argument(
        "--drift_ratio_threshold",
        type=float,
        default=0.03,
        help="Outlier ratio threshold (e.g. 0.03 for 3%).",
    )
    args = parser.parse_args()

    # Fallback to fit if the pre-existing file doesn't exist
    if not os.path.exists(args.centroids_parquet):
        print("fit")
        return

    # 1. Load the unique identifiers (Photo_ID + Platform) of the old database
    try:
        pf_old = pq.ParquetFile(args.centroids_parquet)
        old_ids = []
        for rg in range(pf_old.num_row_groups):
            table_ids = pf_old.read_row_group(rg, columns=["Photo_ID", "Platform"])
            df_ids = table_ids.to_pandas()
            # Combine Photo_ID and Platform to create a unique key
            keys = df_ids["Platform"].astype(str) + "_" + df_ids["Photo_ID"].astype(str)
            old_ids.extend(keys.values)
        old_keys_set = set(old_ids)
        del old_ids
    except Exception:
        # Fallback to fit if schema doesn't match
        print("fit")
        return

    # 2. Sample the EMBEDDINGS of ONLY the newly added images
    try:
        from src.utils.io import load_embeddings

        pf_new = pq.ParquetFile(args.input)
        new_ids = []
        for rg in range(pf_new.num_row_groups):
            table_ids = pf_new.read_row_group(rg, columns=["Photo_ID", "Platform"])
            df_ids = table_ids.to_pandas()
            keys = df_ids["Platform"].astype(str) + "_" + df_ids["Photo_ID"].astype(str)
            new_ids.extend(keys.values)
        new_keys = np.array(new_ids)

        is_new_mask = ~np.isin(new_keys, list(old_keys_set))

        if not np.any(is_new_mask):
            print("assign")
            return

        embs_all_new = load_embeddings(args.input)
        new_embs = embs_all_new[is_new_mask].astype(np.float32)

        new_embs = normalize(new_embs).astype(np.float32)
        dim = new_embs.shape[1]
    except Exception:
        print("fit")
        return

    # 3. Load the old centroids dynamically using vectorized addition
    try:
        from src.utils.io import load_embeddings

        pf_old = pq.ParquetFile(args.centroids_parquet)
        c_ids_list = []
        for rg in range(pf_old.num_row_groups):
            c_ids_rg = pf_old.read_row_group(rg, columns=["cluster_id"])[
                "cluster_id"
            ].to_numpy()
            c_ids_list.append(c_ids_rg)
        c_ids_old = np.concatenate(c_ids_list)

        embs_old = load_embeddings(args.centroids_parquet)

        raw_centroids = np.zeros((args.k_clusters, dim), dtype=np.float32)
        counts = np.zeros(args.k_clusters, dtype=np.int64)

        valid_mask = (
            (c_ids_old >= 0) & (~pd.isna(c_ids_old)) & (c_ids_old < args.k_clusters)
        )
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

    # 4. Perform FAISS search on the new image embeddings
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

        # 5. Calculate outlier ratio
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
