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
        pf_new = pq.ParquetFile(args.input)
        new_embeddings = []
        rows_read = 0

        # Read the new combined database row-group by row-group
        for rg in range(pf_new.num_row_groups):
            table = pf_new.read_row_group(
                rg, columns=["Photo_ID", "Platform", "embedding"]
            )
            df_rg = table.to_pandas()
            if len(df_rg) == 0:
                continue

            # Create keys for this row group
            rg_keys = (
                df_rg["Platform"].astype(str) + "_" + df_rg["Photo_ID"].astype(str)
            )

            # Mask to filter out images that were already in the old run
            is_new_mask = ~rg_keys.isin(old_keys_set)

            if is_new_mask.any():
                new_embs_rg = np.vstack(
                    df_rg.loc[is_new_mask, "embedding"].values
                ).astype(np.float32)
                new_embeddings.append(new_embs_rg)
                rows_read += len(new_embs_rg)

        # If no new images were added at all, we can safely run assign mode (0 drift)
        if len(new_embeddings) == 0:
            print("assign")
            return

        new_embs = np.vstack(new_embeddings)
        new_embs = normalize(new_embs).astype(np.float32)
        dim = new_embs.shape[1]
    except Exception:
        print("fit")
        return

    # 3. Load the old centroids dynamically using vectorized addition
    try:
        raw_centroids = np.zeros((args.k_clusters, dim), dtype=np.float32)
        counts = np.zeros(args.k_clusters, dtype=np.int64)

        for rg in range(pf_old.num_row_groups):
            table_old = pf_old.read_row_group(rg, columns=["cluster_id", "embedding"])
            df_rg_old = table_old.to_pandas()
            if len(df_rg_old) == 0:
                continue
            embs_old = np.vstack(df_rg_old["embedding"].values).astype(np.float32)
            c_ids_old = df_rg_old["cluster_id"].values

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
