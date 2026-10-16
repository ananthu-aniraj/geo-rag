import argparse
import os
import stat
import sys

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize


def clean_photo_id(val) -> str:
    """Cleans photo ID by stripping whitespace and removing trailing .0."""
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    return val_str.removesuffix(".0")


def compute_photo_key(platform: str, photo_id: str) -> str:
    """Computes normalized photo key from platform and photo ID."""
    return f"{str(platform).strip().lower()}_{clean_photo_id(photo_id)}"


def extract_keys(parquet_path: str) -> tuple[np.ndarray, set[str]]:
    """Extracts normalized photo keys from a Parquet file without loading entire tables into memory."""
    pf = pq.ParquetFile(parquet_path)
    col_names = pf.schema_arrow.names

    keys_list = []
    if "photo_key" in col_names:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["photo_key"])
            keys = table["photo_key"].to_pandas().astype(str).str.lower().values
            keys_list.extend(keys)
    elif "Platform" in col_names and "Photo_ID" in col_names:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["Platform", "Photo_ID"])
            df = table.to_pandas()
            plats = df["Platform"].astype(str).str.strip().str.lower()
            pids = df["Photo_ID"].astype(str).str.strip().str.removesuffix(".0")
            keys = (plats + "_" + pids).values
            keys_list.extend(keys)
    elif "platform" in col_names and "photo_id" in col_names:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["platform", "photo_id"])
            df = table.to_pandas()
            plats = df["platform"].astype(str).str.strip().str.lower()
            pids = df["photo_id"].astype(str).str.strip().str.removesuffix(".0")
            keys = (plats + "_" + pids).values
            keys_list.extend(keys)
    else:
        raise ValueError(
            f"Parquet file '{parquet_path}' does not contain 'photo_key' or 'Platform'+'Photo_ID' columns."
        )

    keys_arr = np.array(keys_list)
    return keys_arr, set(keys_list)


def prune_clustered_parquet(
    centroids_parquet_path: str,
    keep_mask: np.ndarray,
) -> None:
    """Streams and prunes rows from centroids_parquet on disk matching keep_mask."""
    pf = pq.ParquetFile(centroids_parquet_path)
    schema_out = pf.schema_arrow
    tmp_file = centroids_parquet_path + ".tmp_prune.parquet"

    try:
        with pq.ParquetWriter(tmp_file, schema_out, compression="zstd") as writer:
            offset = 0
            for rg in range(pf.num_row_groups):
                rg_table = pf.read_row_group(rg)
                rg_len = len(rg_table)
                rg_mask = keep_mask[offset : offset + rg_len]
                offset += rg_len
                if np.any(rg_mask):
                    filtered_rg = rg_table.filter(pa.array(rg_mask))
                    writer.write_table(filtered_rg)

        # Close reference before replacing
        del pf

        # Preserve permissions (including read-only DVC files)
        orig_stat = os.stat(centroids_parquet_path)
        orig_mode = stat.S_IMODE(orig_stat.st_mode)

        if not (orig_mode & stat.S_IWUSR):
            try:
                os.chmod(centroids_parquet_path, orig_mode | stat.S_IWUSR)
            except Exception:
                pass

        os.replace(tmp_file, centroids_parquet_path)

        try:
            os.chmod(centroids_parquet_path, orig_mode)
        except Exception:
            pass

    except Exception as e:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass
        raise e


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
        help="Outlier ratio threshold (e.g. 0.03 for 3%%).",
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        default="cls",
        choices=["cls", "avg_patch", "cls_avg_patch"],
        help="Type of representation embedding to load (cls, avg_patch, or cls_avg_patch).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model name to disambiguate companion embedding files.",
    )
    args = parser.parse_args()

    # Fallback to fit if the pre-existing file doesn't exist
    if not os.path.exists(args.centroids_parquet):
        print("fit")
        return

    # 1. Load the unique identifiers of both databases
    try:
        old_keys_arr, old_keys_set = extract_keys(args.centroids_parquet)
    except Exception:
        sys.stderr.write(
            "Failed to extract keys from centroids_parquet. Falling back to fit.\n"
        )
        print("fit")
        return

    try:
        new_keys_arr, new_keys_set = extract_keys(args.input)
    except Exception:
        sys.stderr.write(
            "Failed to extract keys from input parquet. Falling back to fit.\n"
        )
        print("fit")
        return

    # 2. Synchronize: remove images from clustered database that were removed by cleaning
    try:
        keep_mask = np.isin(old_keys_arr, list(new_keys_set))
        num_removed = int(np.sum(~keep_mask))
        num_surviving = int(np.sum(keep_mask))

        if num_surviving == 0:
            sys.stderr.write(
                f"All {len(old_keys_arr):,} images in '{args.centroids_parquet}' were removed during cleaning. "
                "Triggering full reclustering (fit).\n"
            )
            print("fit")
            return

        if num_removed > 0:
            sys.stderr.write(
                f"Pruning {num_removed:,} cleaned-out images from clustered database '{args.centroids_parquet}' "
                f"({len(old_keys_arr):,} -> {num_surviving:,})...\n"
            )
            prune_clustered_parquet(args.centroids_parquet, keep_mask)
            old_keys_arr = old_keys_arr[keep_mask]
            old_keys_set = set(old_keys_arr)
    except Exception:
        sys.stderr.write("Failed during sidecar pruning. Falling back to fit.\n")
        print("fit")
        return

    # 3. Check for newly added images
    try:
        is_new_mask = ~np.isin(new_keys_arr, list(old_keys_set))

        if not np.any(is_new_mask):
            sys.stderr.write(
                "No newly added images found after synchronizing with cleaned parquet.\n"
            )
            print("assign")
            return

        from src.utils.io import load_embeddings

        embs_all_new = load_embeddings(
            args.input,
            representation_type=args.representation_type,
            model_name=args.model_name,
        )
        new_embs = embs_all_new[is_new_mask].astype(np.float32)

        new_embs = normalize(new_embs).astype(np.float32)
        dim = new_embs.shape[1]
    except Exception:
        sys.stderr.write(
            "Failed to load embeddings for new images. Falling back to fit.\n"
        )
        print("fit")
        return

    # 4. Load the centroids dynamically using vectorized addition
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

        try:
            embs_old = load_embeddings(
                args.centroids_parquet,
                representation_type=args.representation_type,
                model_name=args.model_name,
            )
        except Exception:
            new_keys_idx = pd.Index(new_keys_arr)
            indexer = new_keys_idx.get_indexer(old_keys_arr)
            valid_k = indexer >= 0
            if not valid_k.any():
                raise
            safe_indexer = np.clip(indexer, 0, len(embs_all_new) - 1)
            embs_old = embs_all_new[safe_indexer].astype(np.float32)
            embs_old[~valid_k] = 0.0

        max_c_id = (
            int(np.nanmax(c_ids_old[c_ids_old >= 0])) if np.any(c_ids_old >= 0) else 0
        )
        k_clusters_eff = max(args.k_clusters, max_c_id + 1)

        raw_centroids = np.zeros((k_clusters_eff, dim), dtype=np.float32)
        counts = np.zeros(k_clusters_eff, dtype=np.int64)

        valid_mask = (
            (c_ids_old >= 0) & (~pd.isna(c_ids_old)) & (c_ids_old < k_clusters_eff)
        )
        c_ids_valid = c_ids_old[valid_mask].astype(np.int32)

        np.add.at(raw_centroids, c_ids_valid, embs_old[valid_mask])
        counts += np.bincount(c_ids_valid, minlength=k_clusters_eff)

        valid_counts = counts > 0
        if not np.any(valid_counts):
            print("fit")
            return

        raw_centroids[valid_counts] /= counts[valid_counts, None]
        unique_ids = np.where(valid_counts)[0]
        child_centroids = raw_centroids[unique_ids]
        child_centroids = normalize(child_centroids).astype(np.float32)
    except Exception:
        sys.stderr.write(
            "Failed to compute centroids from old database. Falling back to fit.\n"
        )
        print("fit")
        return

    # 5. Perform FAISS search on the new image embeddings
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

        # 6. Calculate outlier ratio
        outliers = similarities < args.outlier_threshold
        outlier_ratio = np.mean(outliers)

        if outlier_ratio > args.drift_ratio_threshold:
            print("fit")
        else:
            print("assign")
    except Exception:
        sys.stderr.write("Failed during FAISS drift search. Falling back to fit.\n")
        print("fit")


if __name__ == "__main__":
    main()
