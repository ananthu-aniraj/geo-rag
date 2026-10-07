"""
Offline Dataset Synchronization & Multi-Model Companion Embeddings Pruning.

Synchronizes an offline Parquet dataset with a master online cleaned dataset:
1. Detects records in the offline dataset that were purged from the online dataset
   (e.g. from coordinate anomaly cleanup or Mapillary sequence purging).
2. Prunes metadata rows from the offline Parquet file in-place (atomically).
3. Automatically discovers companion embedding files (*_embeddings.npy and
   *_embeddings.keys.parquet) for ALL models (TIPSv2, DINOv2, SigLIP, etc.)
   and slices the .npy matrices and .keys.parquet to surviving keys (zero GPU inference).
4. (Optional) Deletes orphaned local image files on disk to reclaim storage space.
"""

import argparse
import glob
import os
import time
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def clean_photo_id(val) -> str:
    """Cleans photo ID by stripping whitespace and removing trailing .0."""
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    return val_str.removesuffix(".0")


def compute_photo_key(platform: str, photo_id: str) -> str:
    """Computes normalized photo key from platform and photo ID."""
    return f"{str(platform).strip().lower()}_{clean_photo_id(photo_id)}"


def extract_photo_keys(parquet_path: str) -> Tuple[Set[str], List[str]]:
    """Extracts photo_key set and list from a Parquet dataset via PyArrow.

    Returns:
        Tuple of (set_of_unique_keys, list_of_all_keys_in_order)
    """
    parquet_file = pq.ParquetFile(parquet_path)
    col_names = parquet_file.schema_arrow.names

    if "photo_key" in col_names:
        table = parquet_file.read(columns=["photo_key"])
        keys_list = [str(k) for k in table["photo_key"].to_pylist()]
    elif "Platform" in col_names and "Photo_ID" in col_names:
        table = parquet_file.read(columns=["Platform", "Photo_ID"])
        plats = table["Platform"].to_pylist()
        pids = table["Photo_ID"].to_pylist()
        keys_list = [compute_photo_key(p, i) for p, i in zip(plats, pids)]
    elif "photo_id" in col_names and "platform" in col_names:
        table = parquet_file.read(columns=["platform", "photo_id"])
        plats = table["platform"].to_pylist()
        pids = table["photo_id"].to_pylist()
        keys_list = [compute_photo_key(p, i) for p, i in zip(plats, pids)]
    else:
        raise ValueError(
            f"Dataset '{parquet_path}' does not contain 'photo_key' or 'Platform'+'Photo_ID' columns."
        )

    return set(keys_list), keys_list


def cleanup_stale_temp_files(
    search_dirs: List[str],
    dry_run: bool = False,
) -> int:
    """Removes leftover *.tmp_sync.* files from prior interrupted runs."""
    seen = set()
    cleaned_count = 0
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        patterns = [
            os.path.join(d, "*.tmp_sync.parquet"),
            os.path.join(d, "*.tmp_sync.npy"),
            os.path.join(d, "*.tmp_sync.keys.parquet"),
        ]
        for pat in patterns:
            for stale_f in glob.glob(pat):
                abs_f = os.path.abspath(stale_f)
                if abs_f not in seen and os.path.isfile(abs_f):
                    seen.add(abs_f)
                    if not dry_run:
                        try:
                            os.remove(abs_f)
                            cleaned_count += 1
                            print(
                                f" -> Cleaned up leftover temporary file: {os.path.basename(abs_f)}"
                            )
                        except OSError as e:
                            print(f" -> Warning: Could not remove {abs_f}: {e}")
                    else:
                        print(
                            f" -> [DRY RUN] Would clean up leftover temporary file: {os.path.basename(abs_f)}"
                        )
    return cleaned_count


def discover_companion_embedding_files(
    offline_parquet_path: str,
    extra_dirs: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Discovers all companion embeddings and keys files matching the offline dataset.

    Returns:
        List of tuples: [(npy_path, keys_parquet_path), ...]
    """
    directory = os.path.dirname(os.path.abspath(offline_parquet_path))
    base_name = os.path.splitext(os.path.basename(offline_parquet_path))[0]

    # Derive core prefix by stripping common offline/cleaning suffixes
    core_base = base_name
    for suffix in ["_cleaned", "_offline", "_filtered", "_deduplicated"]:
        core_base = core_base.replace(suffix, "")

    search_dirs = [directory]
    if extra_dirs:
        for d in extra_dirs:
            if d and os.path.isdir(d):
                abs_d = os.path.abspath(d)
                if abs_d not in search_dirs:
                    search_dirs.append(abs_d)

    companion_pairs = []
    seen = set()

    for d in search_dirs:
        patterns = [
            os.path.join(d, f"{base_name}*embeddings.keys.parquet"),
            os.path.join(d, f"{core_base}*embeddings.keys.parquet"),
            os.path.join(d, f"{base_name}*.keys.parquet"),
            os.path.join(d, f"{core_base}*.keys.parquet"),
        ]
        # In the offline directory itself, online and offline datasets are assumed
        # to be stored in separate locations, so any companion embeddings belong to the offline dataset.
        if d == directory:
            patterns.extend(
                [
                    os.path.join(d, "*embeddings.keys.parquet"),
                    os.path.join(d, "*.keys.parquet"),
                ]
            )

        for pat in patterns:
            for keys_f in glob.glob(pat):
                if ".tmp_sync." in keys_f:
                    continue
                abs_keys = os.path.abspath(keys_f)
                abs_npy = abs_keys.replace(".keys.parquet", ".npy")
                if (
                    abs_npy not in seen
                    and os.path.isfile(abs_npy)
                    and os.path.isfile(abs_keys)
                ):
                    seen.add(abs_npy)
                    companion_pairs.append((abs_npy, abs_keys))

    return sorted(companion_pairs, key=lambda x: x[0])


def discover_clustered_sidecar_files(
    offline_parquet_path: str,
    extra_dirs: Optional[List[str]] = None,
) -> List[str]:
    """Discovers all clustered sidecar Parquet files matching the offline dataset.

    Returns:
        List of absolute file paths to clustered sidecars.
    """
    directory = os.path.dirname(os.path.abspath(offline_parquet_path))
    base_name = os.path.splitext(os.path.basename(offline_parquet_path))[0]

    # Derive core prefix by stripping common offline suffixes
    core_base = base_name
    for suffix in ["_cleaned", "_offline", "_filtered", "_deduplicated"]:
        core_base = core_base.replace(suffix, "")

    search_dirs = [directory]
    if extra_dirs:
        for d in extra_dirs:
            if d and os.path.isdir(d):
                abs_d = os.path.abspath(d)
                if abs_d not in search_dirs:
                    search_dirs.append(abs_d)

    sidecar_files = []
    seen = set()

    for d in search_dirs:
        patterns = [
            os.path.join(d, f"{base_name}*clustered_k_*.parquet"),
            os.path.join(d, f"{core_base}*clustered_k_*.parquet"),
        ]
        if d == directory:
            patterns.append(os.path.join(d, "*clustered_k_*.parquet"))

        for pat in patterns:
            for f in glob.glob(pat):
                if (
                    f.endswith(".tmp_sync.parquet")
                    or f.endswith(".keys.parquet")
                    or ".tmp_sync." in f
                ):
                    continue
                abs_f = os.path.abspath(f)
                if abs_f not in seen and os.path.isfile(abs_f):
                    seen.add(abs_f)
                    sidecar_files.append(abs_f)

    return sorted(sidecar_files)


def prune_clustered_sidecar(
    sidecar_path: str,
    active_keys: Set[str],
    dry_run: bool = False,
) -> int:
    """Prunes rows from a clustered sidecar Parquet to only retain active keys.

    Returns:
        Number of rows purged.
    """
    sidecar_table = pq.read_table(sidecar_path)
    schema_names = sidecar_table.schema.names

    if "photo_key" in schema_names:
        keys_list = [str(k) for k in sidecar_table["photo_key"].to_pylist()]
    elif "Platform" in schema_names and "Photo_ID" in schema_names:
        plats = sidecar_table["Platform"].to_pylist()
        pids = sidecar_table["Photo_ID"].to_pylist()
        keys_list = [compute_photo_key(p, i) for p, i in zip(plats, pids)]
    elif "platform" in schema_names and "photo_id" in schema_names:
        plats = sidecar_table["platform"].to_pylist()
        pids = sidecar_table["photo_id"].to_pylist()
        keys_list = [compute_photo_key(p, i) for p, i in zip(plats, pids)]
    else:
        print(
            f" -> Skipping {os.path.basename(sidecar_path)}: missing 'photo_key' or 'Platform'+'Photo_ID'."
        )
        return 0

    keep_mask = [k in active_keys for k in keys_list]
    num_keep = sum(keep_mask)
    num_purged = len(keys_list) - num_keep

    if num_purged == 0:
        return 0

    if dry_run:
        print(
            f" -> [DRY RUN] Would prune {num_purged:,} rows from clustered sidecar: {os.path.basename(sidecar_path)} "
            f"({len(keys_list):,} -> {num_keep:,})."
        )
        return num_purged

    filtered_table = sidecar_table.filter(pa.array(keep_mask))
    tmp_sidecar = sidecar_path.replace(".parquet", ".tmp_sync.parquet")
    pq.write_table(filtered_table, tmp_sidecar, compression="zstd")
    os.replace(tmp_sidecar, sidecar_path)

    print(
        f" -> Pruned {num_purged:,} rows from clustered sidecar: {os.path.basename(sidecar_path)} "
        f"({len(keys_list):,} -> {num_keep:,})."
    )
    return num_purged


def prune_companion_embeddings(
    npy_path: str,
    keys_path: str,
    active_keys: Set[str],
    dry_run: bool = False,
) -> int:
    """Prunes companion embedding matrix and keys file to only retain active keys.

    Returns:
        Number of vectors purged.
    """
    keys_df = pd.read_parquet(keys_path)
    if "photo_key" in keys_df.columns:
        comp_keys = [str(k).strip() for k in keys_df["photo_key"]]
    elif "Platform" in keys_df.columns and "Photo_ID" in keys_df.columns:
        comp_keys = [
            compute_photo_key(p, i)
            for p, i in zip(keys_df["Platform"], keys_df["Photo_ID"])
        ]
    elif "platform" in keys_df.columns and "photo_id" in keys_df.columns:
        comp_keys = [
            compute_photo_key(p, i)
            for p, i in zip(keys_df["platform"], keys_df["photo_id"])
        ]
    else:
        print(f" -> Skipping {keys_path}: missing 'photo_key' column.")
        return 0

    # Memory-map slice the .npy matrix
    mmap_emb = np.load(npy_path, mmap_mode="r")
    if len(mmap_emb) != len(keys_df):
        print(
            f" ⚠️ WARNING: Dimension mismatch for {os.path.basename(npy_path)}: "
            f"matrix has {len(mmap_emb):,} rows but keys index has {len(keys_df):,} rows! Skipping."
        )
        return 0

    keep_indices = [i for i, k in enumerate(comp_keys) if k in active_keys]
    num_purged = len(comp_keys) - len(keep_indices)

    if num_purged == 0:
        return 0

    if dry_run:
        print(
            f" -> [DRY RUN] Would prune {num_purged:,} vectors from {os.path.basename(npy_path)} "
            f"({len(comp_keys):,} -> {len(keep_indices):,})."
        )
        return num_purged

    pruned_emb = mmap_emb[keep_indices]
    pruned_keys_df = keys_df.iloc[keep_indices].reset_index(drop=True)
    if "photo_key" not in pruned_keys_df.columns:
        pruned_keys_df["photo_key"] = [comp_keys[i] for i in keep_indices]

    # Atomic write to temporary files
    tmp_npy = npy_path.replace(".npy", ".tmp_sync.npy")
    tmp_keys = keys_path.replace(".keys.parquet", ".tmp_sync.keys.parquet")

    np.save(tmp_npy, pruned_emb)
    pruned_keys_df.to_parquet(tmp_keys, compression="zstd")

    os.replace(tmp_npy, npy_path)
    os.replace(tmp_keys, keys_path)

    print(
        f" -> Pruned {num_purged:,} vectors from {os.path.basename(npy_path)} "
        f"({len(comp_keys):,} -> {len(keep_indices):,})."
    )
    return num_purged


def sync_offline_dataset(
    online_path: str,
    offline_path: str,
    image_dir: Optional[str] = None,
    extra_dirs: Optional[List[str]] = None,
    delete_images: bool = False,
    dry_run: bool = False,
) -> dict:
    """Synchronizes offline dataset with online master dataset.

    Returns:
        Summary dict containing counts of purged rows, vectors, and deleted images.
    """
    print("=" * 80)
    print("🔄 OFFLINE DATASET & MULTI-MODEL EMBEDDINGS SYNCHRONIZATION")
    print("=" * 80)
    print(f"Master Online Dataset : {online_path}")
    print(f"Target Offline Dataset: {offline_path}")
    if image_dir:
        print(f"Image Directory       : {image_dir}")
    print(f"Delete Physical Images: {delete_images}")
    print(f"Dry Run Mode          : {dry_run}")
    print("-" * 80)

    if not os.path.exists(online_path):
        raise FileNotFoundError(f"Online master dataset not found: '{online_path}'")
    if not os.path.exists(offline_path):
        raise FileNotFoundError(f"Offline dataset not found: '{offline_path}'")

    # 1. Read keys
    t0 = time.time()
    print("Step 1: Reading photo keys from master online dataset...")
    online_keys, _ = extract_photo_keys(online_path)
    print(
        f" -> Found {len(online_keys):,} active keys in master online dataset ({time.time() - t0:.2f}s)."
    )

    t1 = time.time()
    print("Step 2: Inspecting target offline dataset...")
    offline_table = pq.read_table(offline_path)
    total_offline = len(offline_table)

    if "photo_key" in offline_table.schema.names:
        offline_keys_list = [str(k) for k in offline_table["photo_key"].to_pylist()]
    else:
        plats = offline_table["Platform"].to_pylist()
        pids = offline_table["Photo_ID"].to_pylist()
        offline_keys_list = [compute_photo_key(p, i) for p, i in zip(plats, pids)]

    # Compute surviving mask
    keep_mask = [k in online_keys for k in offline_keys_list]
    num_keep = sum(keep_mask)
    num_purged_rows = total_offline - num_keep

    print(f" -> Offline dataset records: {total_offline:,}")
    print(f" -> Matching active records: {num_keep:,}")
    print(f" -> Stale records to purge : {num_purged_rows:,}")

    summary = {
        "total_offline": total_offline,
        "surviving_rows": num_keep,
        "purged_rows": num_purged_rows,
        "companion_models_synced": 0,
        "purged_vectors_total": 0,
        "clustered_sidecars_synced": 0,
        "purged_sidecar_rows_total": 0,
        "deleted_images_count": 0,
        "reclaimed_bytes": 0,
    }

    # 2. Discover companion embeddings & clustered sidecars for ALL models
    print("\nStep 3: Discovering companion embeddings and clustered sidecars...")
    search_dirs = [os.path.dirname(os.path.abspath(offline_path))]
    if extra_dirs:
        for d in extra_dirs:
            if d and os.path.isdir(d):
                abs_d = os.path.abspath(d)
                if abs_d not in search_dirs:
                    search_dirs.append(abs_d)

    cleanup_stale_temp_files(search_dirs, dry_run=dry_run)

    companion_files = discover_companion_embedding_files(
        offline_path, extra_dirs=extra_dirs
    )
    if companion_files:
        print(f" -> Discovered {len(companion_files)} companion embedding file(s):")
        for npy_f, _ in companion_files:
            print(f"    * {os.path.basename(npy_f)}")
    else:
        print(" -> No companion embedding files (*_embeddings.keys.parquet) found.")

    sidecar_files = discover_clustered_sidecar_files(
        offline_path, extra_dirs=extra_dirs
    )
    if sidecar_files:
        print(f" -> Discovered {len(sidecar_files)} clustered sidecar file(s):")
        for sc_f in sidecar_files:
            print(f"    * {os.path.basename(sc_f)}")
    else:
        print(" -> No clustered sidecar files (*_clustered_k_*.parquet) found.")

    # 3. Prune offline metadata Parquet
    if num_purged_rows > 0:
        print(
            f"\nStep 4: Pruning {num_purged_rows:,} stale rows from offline Parquet metadata..."
        )
        purged_image_locations = []
        if "Image_Location" in offline_table.schema.names:
            locs = offline_table["Image_Location"].to_pylist()
            purged_image_locations = [
                loc for loc, keep in zip(locs, keep_mask) if not keep and loc
            ]
        elif "Image_URL" in offline_table.schema.names:
            urls = offline_table["Image_URL"].to_pylist()
            purged_image_locations = [
                u for u, keep in zip(urls, keep_mask) if not keep and u
            ]

        if not dry_run:
            filtered_table = offline_table.filter(pa.array(keep_mask))
            tmp_offline = offline_path.replace(".parquet", ".tmp_sync.parquet")
            pq.write_table(filtered_table, tmp_offline, compression="zstd")
            os.replace(tmp_offline, offline_path)
            print(f" -> Successfully updated offline Parquet in-place: {offline_path}")
        else:
            print(f" -> [DRY RUN] Would update offline Parquet: {offline_path}")
    else:
        print("\nStep 4: Offline Parquet metadata is already in sync (0 stale rows).")
        purged_image_locations = []

    # 4. Prune companion embeddings for all models
    if companion_files:
        print("\nStep 5: Pruning companion embeddings matrices for all models...")
        total_vectors_purged = 0
        models_synced = 0
        for npy_f, keys_f in companion_files:
            v_purged = prune_companion_embeddings(
                npy_f, keys_f, online_keys, dry_run=dry_run
            )
            total_vectors_purged += v_purged
            if v_purged > 0 or not dry_run:
                models_synced += 1

        summary["companion_models_synced"] = models_synced
        summary["purged_vectors_total"] = total_vectors_purged

    # 5. Prune clustered sidecars across all sweeps
    if sidecar_files:
        print("\nStep 6: Pruning companion clustered sidecars across all sweeps...")
        total_sidecar_rows_purged = 0
        sidecars_synced = 0
        for sc_f in sidecar_files:
            sc_purged = prune_clustered_sidecar(sc_f, online_keys, dry_run=dry_run)
            total_sidecar_rows_purged += sc_purged
            if sc_purged > 0 or not dry_run:
                sidecars_synced += 1

        summary["clustered_sidecars_synced"] = sidecars_synced
        summary["purged_sidecar_rows_total"] = total_sidecar_rows_purged

    if (
        num_purged_rows == 0
        and summary["purged_vectors_total"] == 0
        and summary["purged_sidecar_rows_total"] == 0
    ):
        print(
            "\n✅ Offline dataset, companion embeddings, and clustered sidecars are already 100% in sync! Nothing to prune."
        )
        return summary

    # 6. Delete orphaned physical image files on disk (if requested)
    if delete_images and purged_image_locations:
        print(
            f"\nStep 7: Cleaning up {len(purged_image_locations):,} orphaned image files on disk..."
        )
        deleted_count = 0
        bytes_saved = 0

        for loc in tqdm(purged_image_locations, desc="Deleting orphaned images"):
            candidate_paths = []
            if os.path.isabs(loc):
                candidate_paths.append(loc)
            elif image_dir:
                candidate_paths.append(os.path.join(image_dir, loc))
            else:
                candidate_paths.append(loc)
                candidate_paths.append(os.path.join(os.path.dirname(offline_path), loc))

            for path in candidate_paths:
                if os.path.isfile(path):
                    try:
                        f_size = os.path.getsize(path)
                        if not dry_run:
                            os.remove(path)
                        bytes_saved += f_size
                        deleted_count += 1
                        break
                    except Exception as e:
                        print(f"Warning: Failed to delete image '{path}': {e}")

        summary["deleted_images_count"] = deleted_count
        summary["reclaimed_bytes"] = bytes_saved
        mb_saved = bytes_saved / (1024 * 1024)
        print(
            f" -> {'[DRY RUN] Would delete' if dry_run else 'Deleted'} {deleted_count:,} image files "
            f"({mb_saved:.2f} MB disk space reclaimed)."
        )
    elif delete_images:
        print("\nStep 7: No image locations found to clean up on disk.")

    print("\n" + "=" * 80)
    print("🎉 SYNCHRONIZATION COMPLETE")
    print("=" * 80)
    print(f"Surviving Records: {num_keep:,} / {total_offline:,}")
    print(f"Purged Metadata  : {num_purged_rows:,}")
    if companion_files:
        print(
            f"Models Synced    : {summary['companion_models_synced']} companion file pairs"
        )
        print(f"Vectors Purged   : {summary['purged_vectors_total']:,}")
    if sidecar_files:
        print(
            f"Sidecars Synced  : {summary['clustered_sidecars_synced']} clustered sidecar(s)"
        )
        print(f"Sidecar Rows Out : {summary['purged_sidecar_rows_total']:,}")
    if delete_images:
        print(
            f"Images Deleted   : {summary['deleted_images_count']:,} ({summary['reclaimed_bytes'] / (1024 * 1024):.2f} MB)"
        )
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize offline dataset & multi-model companion embeddings with online cleaned master dataset."
    )
    parser.add_argument(
        "--online",
        "--master",
        type=str,
        required=True,
        help="Path to online master cleaned Parquet dataset (e.g. geo_space_cleaned.parquet).",
    )
    parser.add_argument(
        "--offline",
        type=str,
        required=True,
        help="Path to offline Parquet dataset (e.g. geo_space_cleaned_offline.parquet).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Root directory where offline images reside (used for resolving relative Image_Location).",
    )
    parser.add_argument(
        "--extra_dirs",
        nargs="*",
        default=[],
        help="Additional directories to search for companion clustered sidecars or embeddings (e.g. output directory).",
    )
    parser.add_argument(
        "--delete_images",
        action="store_true",
        help="Delete physical image files on disk for records purged from the dataset.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Simulate pruning without modifying files on disk.",
    )
    args = parser.parse_args()

    sync_offline_dataset(
        online_path=args.online,
        offline_path=args.offline,
        image_dir=args.image_dir,
        extra_dirs=args.extra_dirs,
        delete_images=args.delete_images,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
