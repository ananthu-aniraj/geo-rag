"""
Batch Mapillary Sequence Auditor and Purge Utility for Geo-RAG.

Audits Mapillary image sequences within a target geographic scope (e.g. continent),
detects invalid sequences caused by stationary cameras, multipath GPS drift, and
impossible speed spikes, and executes a streaming purge of the corrupted images.
"""

import argparse
import glob
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.credentials import get_mapillary_token
from src.utils.mapillary_trajectory_validator import MapillaryTrajectoryValidator
from src.utils.sync_offline_dataset import (
    cleanup_stale_temp_files,
    compute_photo_key,
    discover_clustered_sidecar_files,
    discover_companion_embedding_files,
    prune_clustered_sidecar,
    prune_companion_embeddings,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("audit_mapillary_sequences")


def get_parquet_writer(file_path: str, schema: pa.Schema) -> pq.ParquetWriter:
    """Returns a ParquetWriter with standard project compression settings."""
    return pq.ParquetWriter(
        file_path,
        schema,
        compression="zstd",
        compression_level=7,
        use_dictionary=True,
    )


def discover_companion_parquets(
    input_parquet: str,
    extra_dirs: Optional[List[str]] = None,
) -> List[str]:
    """Discovers sibling parquet datasets (such as *_cleaned.parquet) matching the prefix.

    Excludes companion keys (*.keys.parquet), sidecars (*clustered*.parquet),
    and semantic indices (*index*.parquet).
    """
    directory = os.path.dirname(os.path.abspath(input_parquet))
    base_name = os.path.splitext(os.path.basename(input_parquet))[0]
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

    discovered = []
    seen = {os.path.abspath(input_parquet)}

    for d in search_dirs:
        patterns = [
            os.path.join(d, f"{core_base}*.parquet"),
        ]
        for pat in patterns:
            for f in glob.glob(pat):
                abs_f = os.path.abspath(f)
                if abs_f in seen:
                    continue
                # Exclude non-main-dataset files
                if (
                    abs_f.endswith(".keys.parquet")
                    or "clustered" in abs_f
                    or "index" in abs_f
                    or ".tmp_" in abs_f
                ):
                    continue
                if os.path.isfile(abs_f):
                    seen.add(abs_f)
                    discovered.append(abs_f)

    return sorted(discovered)


def stream_purge_parquet(
    parquet_path: str,
    violating_photo_ids: Set[str],
    platform: str = "mapillary",
    output_path: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """
    Stream-purges rows with matching (platform, photo_id) from a Parquet dataset.
    Uses row-group streaming to keep memory overhead near zero.

    Args:
        parquet_path: Path to the target Parquet dataset.
        violating_photo_ids: Set of photo IDs to purge.
        platform: Target platform (e.g. 'mapillary').
        output_path: Optional output path. Defaults to in-place replacement of parquet_path.
        dry_run: If True, counts rows that would be purged without modifying disk.

    Returns:
        Tuple of (total_rows_in, total_rows_out, purged_count).
    """
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    num_row_groups = pf.num_row_groups
    total_in = pf.metadata.num_rows

    if total_in == 0 or not violating_photo_ids:
        if output_path and output_path != parquet_path and not dry_run:
            tbl = pf.read()
            with get_parquet_writer(output_path, schema) as writer:
                writer.write_table(tbl)
        return total_in, total_in, 0

    photo_id_col = (
        "Photo_ID"
        if "Photo_ID" in schema.names
        else ("photo_id" if "photo_id" in schema.names else None)
    )
    plat_col = (
        "Platform"
        if "Platform" in schema.names
        else ("platform" if "platform" in schema.names else None)
    )
    photo_key_col = "photo_key" if "photo_key" in schema.names else None

    if not photo_id_col and not photo_key_col:
        logger.warning(
            f"Dataset '{parquet_path}' has neither photo ID nor photo_key column; skipping purge."
        )
        return total_in, total_in, 0

    violating_keys = (
        {compute_photo_key(platform, pid) for pid in violating_photo_ids}
        if photo_key_col
        else set()
    )
    violating_pids_set = violating_photo_ids
    target_platform = platform.lower()

    target_output = output_path or parquet_path
    temp_output = target_output + f".tmp_audit_purge_{os.getpid()}"

    purged_count = 0
    total_out = 0

    if dry_run:
        cols_needed = (
            [photo_id_col, plat_col]
            if (photo_id_col and plat_col)
            else ([photo_key_col] if photo_key_col else [photo_id_col])
        )
        for rg in range(num_row_groups):
            tbl_rg = pf.read_row_group(rg, columns=cols_needed)
            if photo_id_col and plat_col:
                rg_pids = np.array(
                    [
                        str(p).strip().removesuffix(".0")
                        for p in tbl_rg[photo_id_col].to_pylist()
                    ]
                )
                rg_plat = np.char.lower(tbl_rg[plat_col].to_numpy().astype(str))
                is_violating = np.isin(rg_pids, list(violating_pids_set)) & (
                    rg_plat == target_platform
                )
            elif photo_key_col:
                rg_keys = tbl_rg[photo_key_col].to_numpy().astype(str)
                is_violating = np.isin(rg_keys, list(violating_keys))
            else:
                rg_pids = np.array(
                    [
                        str(p).strip().removesuffix(".0")
                        for p in tbl_rg[photo_id_col].to_pylist()
                    ]
                )
                is_violating = np.isin(rg_pids, list(violating_pids_set))

            purged_count += int(np.sum(is_violating))
        return total_in, total_in - purged_count, purged_count

    writer = get_parquet_writer(temp_output, schema)
    try:
        pbar = tqdm(
            range(num_row_groups),
            desc=f"Purging {os.path.basename(target_output)}",
            unit="rg",
            leave=False,
        )
        for rg in pbar:
            tbl_rg = pf.read_row_group(rg)
            if photo_id_col and plat_col:
                rg_pids = np.array(
                    [
                        str(p).strip().removesuffix(".0")
                        for p in tbl_rg[photo_id_col].to_pylist()
                    ]
                )
                rg_plat = np.char.lower(tbl_rg[plat_col].to_numpy().astype(str))
                is_violating = np.isin(rg_pids, list(violating_pids_set)) & (
                    rg_plat == target_platform
                )
            elif photo_key_col:
                rg_keys = tbl_rg[photo_key_col].to_numpy().astype(str)
                is_violating = np.isin(rg_keys, list(violating_keys))
            else:
                rg_pids = np.array(
                    [
                        str(p).strip().removesuffix(".0")
                        for p in tbl_rg[photo_id_col].to_pylist()
                    ]
                )
                is_violating = np.isin(rg_pids, list(violating_pids_set))

            keep_mask = ~is_violating
            filtered_tbl = tbl_rg.filter(pa.array(keep_mask))
            writer.write_table(filtered_tbl)

            rg_purged = int(np.sum(is_violating))
            purged_count += rg_purged
            total_out += len(filtered_tbl)
            pbar.set_postfix(purged=purged_count, refresh=False)
    except Exception:
        writer.close()
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except OSError:
                pass
        raise
    else:
        writer.close()
        os.replace(temp_output, target_output)

    return total_in, total_out, purged_count


def audit_and_purge_dataset(
    input_parquet: str,
    output_parquet: str,
    continent: Optional[str] = "Africa",
    platform: str = "mapillary",
    token: Optional[str] = None,
    db_path: str = "data/cache/mapillary_sequences.db",
    max_sequences: Optional[int] = None,
    dry_run: bool = False,
    report_json: Optional[str] = None,
    sync_embeddings: bool = True,
    embeddings_dir: Optional[str] = None,
    sync_sidecars: bool = True,
    extra_parquets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Audits Mapillary photos in the target continent, detects invalid sequences,
    and removes all associated photos. Also prunes matching rows from additional
    Parquet datasets (e.g. geo_space_cleaned.parquet), companion embedding matrices,
    and clustered sidecars matching deleted records.
    """
    t0 = time.time()
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(f"Input Parquet not found: {input_parquet}")

    token = token or get_mapillary_token() or ""
    if not token:
        raise ValueError(
            "Mapillary API token not found in environment, arguments, or .env file."
        )

    validator = MapillaryTrajectoryValidator(token=token, db_path=db_path)

    # Clean leftover stale temp files
    search_dirs = [os.path.dirname(os.path.abspath(input_parquet))]
    if embeddings_dir:
        search_dirs.append(os.path.abspath(embeddings_dir))
    cleanup_stale_temp_files(search_dirs, dry_run=dry_run)
    for d in search_dirs:
        for stale_f in glob.glob(os.path.join(d, "*.tmp_audit_purge*")):
            if not dry_run:
                try:
                    os.remove(stale_f)
                    print(
                        f" -> Cleaned up leftover temporary file: {os.path.basename(stale_f)}"
                    )
                except OSError:
                    pass

    # Resolve extra parquet datasets
    target_extra_parquets: List[str] = []
    if extra_parquets:
        for item in extra_parquets:
            if item.strip().lower() == "auto":
                discovered = discover_companion_parquets(
                    input_parquet,
                    extra_dirs=[embeddings_dir] if embeddings_dir else None,
                )
                for d_path in discovered:
                    if d_path not in target_extra_parquets:
                        target_extra_parquets.append(d_path)
            else:
                p_path = os.path.abspath(item.strip())
                if p_path not in target_extra_parquets and p_path != os.path.abspath(
                    input_parquet
                ):
                    target_extra_parquets.append(p_path)

    # Discover companion embeddings and clustered sidecars
    companion_pairs = (
        discover_companion_embedding_files(
            input_parquet, extra_dirs=[embeddings_dir] if embeddings_dir else None
        )
        if sync_embeddings
        else []
    )
    sidecar_files = (
        discover_clustered_sidecar_files(
            input_parquet, extra_dirs=[embeddings_dir] if embeddings_dir else None
        )
        if sync_sidecars
        else []
    )

    print(
        "================================================================================"
    )
    print("🚗 MAPILLARY SEQUENCE TRAJECTORY AUDITOR & PURGE UTILITY")
    print(
        "================================================================================"
    )
    print(f"Input Dataset  : {input_parquet}")
    print(f"Output Dataset : {output_parquet}")
    print(f"Scope Platform : {platform.lower()}")
    print(f"Scope Continent: {continent or 'Global (All)'}")
    print(f"Dry Run Mode   : {dry_run}")
    print(f"Cache Database : {db_path}")
    if target_extra_parquets:
        print(
            f"Extra Parquets : {len(target_extra_parquets)} additional dataset(s) scheduled:"
        )
        for ep in target_extra_parquets:
            print(f"  * {ep}")
    if sync_embeddings:
        print(f"Companion Embs : {len(companion_pairs)} discovered for pruning")
    if sync_sidecars:
        print(f"Sidecar Files  : {len(sidecar_files)} discovered for pruning")
    print(
        "--------------------------------------------------------------------------------"
    )

    # Step 1: Scan photo metadata in streaming row groups
    print("Step 1: Reading Photo IDs and scope columns from dataset...")
    pf = pq.ParquetFile(input_parquet)
    schema = pf.schema_arrow
    num_row_groups = pf.num_row_groups

    has_continent = "continent" in schema.names or "Continent" in schema.names
    cont_col = (
        "continent"
        if "continent" in schema.names
        else ("Continent" if "Continent" in schema.names else None)
    )
    photo_id_col = (
        "Photo_ID"
        if "Photo_ID" in schema.names
        else ("photo_id" if "photo_id" in schema.names else None)
    )

    if not photo_id_col:
        raise ValueError("Photo_ID column not found in Parquet schema.")

    cols_to_load = [photo_id_col, "Platform"]
    if has_continent and cont_col and continent:
        cols_to_load.append(cont_col)

    all_pids: List[str] = []
    all_plats: List[str] = []
    all_conts: List[str] = []

    for rg in tqdm(
        range(num_row_groups), desc="Scanning dataset row groups", unit="rg"
    ):
        tbl_rg = pf.read_row_group(rg, columns=cols_to_load)
        all_pids.extend(tbl_rg[photo_id_col].to_numpy().astype(str))
        all_plats.extend(tbl_rg["Platform"].to_numpy().astype(str))
        if has_continent and cont_col and continent:
            all_conts.extend(tbl_rg[cont_col].to_numpy().astype(str))

    df_scope = pd.DataFrame(
        {
            "Photo_ID": all_pids,
            "Platform": [p.lower() for p in all_plats],
        }
    )
    if has_continent and cont_col and continent:
        df_scope["continent"] = [c.lower() for c in all_conts]

    # Filter by platform
    mask = df_scope["Platform"] == platform.lower()
    if has_continent and cont_col and continent:
        mask = mask & (df_scope["continent"] == continent.lower())

    scoped_df = df_scope[mask]
    target_photo_ids = [
        str(p).strip().removesuffix(".0")
        for p in scoped_df["Photo_ID"].dropna().unique()
        if str(p).strip()
    ]

    print(f" -> Scanned {len(df_scope):,} total dataset rows.")
    print(
        f" -> Found {len(target_photo_ids):,} Mapillary photo(s) in scope ({continent or 'Global'})."
    )

    if not target_photo_ids:
        print("✅ No target Mapillary photos found in scope. Nothing to audit.")
        return {"status": "empty_scope", "purged_photos": 0}

    # Step 2: Resolve photo IDs to sequence IDs
    print("\nStep 2: Resolving sequence IDs for target Mapillary photos...")
    t_seq = time.time()
    photo_to_seq = validator.resolve_photo_sequences(target_photo_ids, batch_size=100)
    print(
        f" -> Resolved {len(photo_to_seq):,} photo(s) to parent sequence IDs in {time.time() - t_seq:.2f}s."
    )

    unique_seqs = list(dict.fromkeys(photo_to_seq.values()))
    print(
        f" -> Scoped photos belong to {len(unique_seqs):,} unique Mapillary sequence(s)."
    )

    if max_sequences and max_sequences < len(unique_seqs):
        print(f" -> Capping sequence audit to {max_sequences} sequences as requested.")
        unique_seqs = unique_seqs[:max_sequences]

    # Step 3: Audit each sequence's kinematic trajectory
    print("\nStep 3: Auditing kinematic trajectory validity per sequence...")
    invalid_sequences: Dict[str, Dict[str, Any]] = {}
    valid_sequences_count = 0

    seq_pbar = tqdm(unique_seqs, desc="Auditing sequences", unit="seq")
    for seq_id in seq_pbar:
        res = validator.validate_sequence(seq_id)
        if not res.is_valid:
            invalid_sequences[seq_id] = {
                "reason": res.reason,
                "metrics": res.metrics.to_dict()
                if hasattr(res.metrics, "to_dict")
                else (res.metrics.__dict__ if res.metrics else None),
            }
            seq_pbar.set_postfix(invalid=len(invalid_sequences), refresh=False)
        else:
            valid_sequences_count += 1

    print(
        "\n--------------------------------------------------------------------------------"
    )
    print("📊 AUDIT RESULTS SUMMARY")
    print(
        "--------------------------------------------------------------------------------"
    )
    print(f"Unique Sequences Evaluated : {len(unique_seqs):,}")
    print(f"Valid Moving Sequences     : {valid_sequences_count:,}")
    print(f"Invalid / Jitter Sequences : {len(invalid_sequences):,}")

    # Map invalid sequences to violating photo IDs
    invalid_seq_set = set(invalid_sequences.keys())
    violating_photo_ids: Set[str] = {
        pid for pid, sid in photo_to_seq.items() if sid in invalid_seq_set
    }
    print(f"Dataset Photos to Purge    : {len(violating_photo_ids):,}")
    print(
        "--------------------------------------------------------------------------------"
    )

    # Tally reasons
    reason_counts: Dict[str, int] = {}
    for info in invalid_sequences.values():
        r = info["reason"].split(":")[0]
        reason_counts[r] = reason_counts.get(r, 0) + 1

    print("Breakdown of Violations:")
    for reason_prefix, count in sorted(
        reason_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f" * {reason_prefix}: {count:,} sequence(s)")
    print(
        "================================================================================"
    )

    # Save report JSON if requested
    if report_json:
        os.makedirs(os.path.dirname(os.path.abspath(report_json)), exist_ok=True)
        report_data = {
            "timestamp": time.time(),
            "scope_continent": continent,
            "scope_platform": platform,
            "total_photos_in_scope": len(target_photo_ids),
            "unique_sequences": len(unique_seqs),
            "valid_sequences": valid_sequences_count,
            "invalid_sequences_count": len(invalid_sequences),
            "photos_to_purge_count": len(violating_photo_ids),
            "reason_breakdown": reason_counts,
            "invalid_sequences": invalid_sequences,
        }
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f" -> Saved detailed audit report to: {report_json}")

    violating_keys: Set[str] = {
        compute_photo_key(platform, pid) for pid in violating_photo_ids
    }

    if dry_run or not violating_photo_ids:
        if dry_run:
            print(
                "\n[DRY RUN] No files modified. Run without --dry_run to purge these records."
            )
            if target_extra_parquets and violating_photo_ids:
                print(
                    f"\n[DRY RUN] Inspecting {len(target_extra_parquets)} additional Parquet dataset(s)..."
                )
                for ep in target_extra_parquets:
                    if not os.path.isfile(ep):
                        continue
                    e_in, e_out, e_purged = stream_purge_parquet(
                        parquet_path=ep,
                        violating_photo_ids=violating_photo_ids,
                        platform=platform,
                        output_path=None,
                        dry_run=True,
                    )
                    print(
                        f" -> [DRY RUN] Would purge {e_purged:,} rows from extra dataset: {os.path.basename(ep)} ({e_in:,} -> {e_out:,})."
                    )

            if sync_embeddings and companion_pairs and violating_photo_ids:
                print(
                    f"\n[DRY RUN] Inspecting {len(companion_pairs)} companion embedding file(s)..."
                )
                for npy_path, keys_path in companion_pairs:
                    keys_df = pd.read_parquet(keys_path)
                    if "photo_key" in keys_df.columns:
                        comp_keys = [str(k).strip() for k in keys_df["photo_key"]]
                    elif (
                        "Platform" in keys_df.columns and "Photo_ID" in keys_df.columns
                    ):
                        comp_keys = [
                            compute_photo_key(p, i)
                            for p, i in zip(keys_df["Platform"], keys_df["Photo_ID"])
                        ]
                    else:
                        continue
                    active_keys = {k for k in comp_keys if k not in violating_keys}
                    prune_companion_embeddings(
                        npy_path, keys_path, active_keys, dry_run=True
                    )

            if sync_sidecars and sidecar_files and violating_photo_ids:
                print(
                    f"\n[DRY RUN] Inspecting {len(sidecar_files)} clustered sidecar file(s)..."
                )
                for sidecar_path in sidecar_files:
                    sidecar_tbl = pq.read_table(sidecar_path)
                    s_schema = sidecar_tbl.schema.names
                    if "photo_key" in s_schema:
                        s_keys = [str(k) for k in sidecar_tbl["photo_key"].to_pylist()]
                    elif "Platform" in s_schema and "Photo_ID" in s_schema:
                        s_keys = [
                            compute_photo_key(p, i)
                            for p, i in zip(
                                sidecar_tbl["Platform"].to_pylist(),
                                sidecar_tbl["Photo_ID"].to_pylist(),
                            )
                        ]
                    else:
                        continue
                    active_keys = {k for k in s_keys if k not in violating_keys}
                    prune_clustered_sidecar(sidecar_path, active_keys, dry_run=True)

        return {
            "status": "dry_run" if dry_run else "no_violations",
            "purged_photos": len(violating_photo_ids),
            "invalid_sequences": len(invalid_sequences),
        }

    # Step 4: Stream-purge violating photos from primary Parquet
    print(
        f"\nStep 4: Executing streaming purge of {len(violating_photo_ids):,} photos from primary Parquet..."
    )
    t_purge = time.time()
    total_in, total_out, purged_count = stream_purge_parquet(
        parquet_path=input_parquet,
        violating_photo_ids=violating_photo_ids,
        platform=platform,
        output_path=output_parquet,
        dry_run=False,
    )
    print(
        f" -> Purged {purged_count:,} rows. Dataset reduced from {total_in:,} to {total_out:,} rows in {time.time() - t_purge:.2f}s."
    )
    print(f" -> Updated clean Parquet dataset: {output_parquet}")

    # Step 4b: Stream-purge extra Parquet datasets
    extra_purged_summary: Dict[str, int] = {}
    if target_extra_parquets and violating_photo_ids:
        print(
            f"\nStep 4b: Stream-purging {len(violating_photo_ids):,} photos from {len(target_extra_parquets)} additional Parquet dataset(s)..."
        )
        for extra_pq in target_extra_parquets:
            if not os.path.isfile(extra_pq):
                print(f" -> Skipping non-existent extra parquet: {extra_pq}")
                continue
            e_in, e_out, e_purged = stream_purge_parquet(
                parquet_path=extra_pq,
                violating_photo_ids=violating_photo_ids,
                platform=platform,
                output_path=None,  # In-place purge
                dry_run=False,
            )
            extra_purged_summary[extra_pq] = e_purged
            print(
                f" -> Purged {e_purged:,} rows from extra dataset: {os.path.basename(extra_pq)} ({e_in:,} -> {e_out:,})."
            )

    # Step 5: Prune companion embedding matrices for all models
    total_vectors_purged = 0
    if sync_embeddings and companion_pairs and violating_photo_ids:
        print("\nStep 5: Pruning companion embeddings matrices for all models...")
        for npy_path, keys_path in companion_pairs:
            keys_df = pd.read_parquet(keys_path)
            if "photo_key" in keys_df.columns:
                comp_keys = [str(k).strip() for k in keys_df["photo_key"]]
            elif "Platform" in keys_df.columns and "Photo_ID" in keys_df.columns:
                comp_keys = [
                    compute_photo_key(p, i)
                    for p, i in zip(keys_df["Platform"], keys_df["Photo_ID"])
                ]
            else:
                continue
            active_keys = {k for k in comp_keys if k not in violating_keys}
            purged_vecs = prune_companion_embeddings(
                npy_path, keys_path, active_keys, dry_run=False
            )
            total_vectors_purged += purged_vecs

    # Step 6: Prune clustered sidecars across all sweeps
    total_sidecar_rows_purged = 0
    if sync_sidecars and sidecar_files and violating_photo_ids:
        print("\nStep 6: Pruning companion clustered sidecars across all sweeps...")
        for sidecar_path in sidecar_files:
            sidecar_tbl = pq.read_table(sidecar_path)
            s_schema = sidecar_tbl.schema.names
            if "photo_key" in s_schema:
                s_keys = [str(k) for k in sidecar_tbl["photo_key"].to_pylist()]
            elif "Platform" in s_schema and "Photo_ID" in s_schema:
                s_keys = [
                    compute_photo_key(p, i)
                    for p, i in zip(
                        sidecar_tbl["Platform"].to_pylist(),
                        sidecar_tbl["Photo_ID"].to_pylist(),
                    )
                ]
            else:
                continue
            active_keys = {k for k in s_keys if k not in violating_keys}
            purged_rows = prune_clustered_sidecar(
                sidecar_path, active_keys, dry_run=False
            )
            total_sidecar_rows_purged += purged_rows

    print(
        "\n================================================================================"
    )
    print("🎉 MAPILLARY AUDIT & PURGE COMPLETE")
    print(
        "================================================================================"
    )
    print(f"Total Time Taken        : {time.time() - t0:.2f}s")
    print(f"Dataset Photos Purged   : {purged_count:,}")
    if sync_embeddings and companion_pairs:
        print(
            f"Companion Vectors Purged: {total_vectors_purged:,} across {len(companion_pairs)} file(s)"
        )
    if sync_sidecars and sidecar_files:
        print(
            f"Clustered Rows Purged   : {total_sidecar_rows_purged:,} across {len(sidecar_files)} sidecar(s)"
        )
    if extra_purged_summary:
        print(
            f"Extra Parquet Rows Purged: {sum(extra_purged_summary.values()):,} across {len(extra_purged_summary)} file(s)"
        )
    print(
        "================================================================================"
    )

    # Update report JSON with actual purge counts if requested
    if report_json and os.path.exists(report_json):
        try:
            with open(report_json, "r", encoding="utf-8") as f:
                r_data = json.load(f)
            r_data["purged_photos"] = purged_count
            r_data["vectors_purged"] = total_vectors_purged
            r_data["sidecar_rows_purged"] = total_sidecar_rows_purged
            r_data["extra_parquets_purged"] = extra_purged_summary
            with open(report_json, "w", encoding="utf-8") as f:
                json.dump(r_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update report JSON: {e}")

    return {
        "status": "purged",
        "purged_photos": purged_count,
        "invalid_sequences": len(invalid_sequences),
        "total_records_remaining": total_out,
        "vectors_purged": total_vectors_purged,
        "sidecar_rows_purged": total_sidecar_rows_purged,
        "extra_parquets_purged": extra_purged_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit Mapillary sequence kinematics and purge invalid/jitter sequences."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="full_pipeline_output/geo_space_cleaned.parquet",
        help="Path to the input Parquet dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output Parquet dataset (defaults to input).",
    )
    parser.add_argument(
        "--extra_parquets",
        nargs="*",
        default=[],
        help=(
            "Optional additional Parquet datasets (e.g. geo_space_cleaned.parquet) "
            "to purge matching violating photos from simultaneously. Pass 'auto' to "
            "automatically discover sibling datasets matching the base prefix."
        ),
    )
    parser.add_argument(
        "--continent",
        type=str,
        default="Africa",
        help="Target continent to audit (default: 'Africa'). Pass 'all' or empty to audit globally.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="mapillary",
        help="Target platform to audit (default: 'mapillary').",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Optional limit on the number of unique sequences to audit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and report violations without modifying dataset files.",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="full_pipeline_output/mapillary_sequence_audit.json",
        help="Path to write JSON audit report.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/cache/mapillary_sequences.db",
        help="Path to SQLite validator cache.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Mapillary API token.",
    )
    parser.add_argument(
        "--sync_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically discover and prune matching companion embedding matrices (*_embeddings.npy and *.keys.parquet).",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Optional directory to search for companion embeddings and sidecars.",
    )
    parser.add_argument(
        "--sync_sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically discover and prune matching clustered sidecars (*_clustered_k_*.parquet).",
    )
    args = parser.parse_args()

    cont = None if args.continent.lower() in ("all", "none", "") else args.continent
    out_parquet = args.output or args.input

    audit_and_purge_dataset(
        input_parquet=args.input,
        output_parquet=out_parquet,
        continent=cont,
        platform=args.platform,
        token=args.token,
        db_path=args.db_path,
        max_sequences=args.max_sequences,
        dry_run=args.dry_run,
        report_json=args.report_json,
        sync_embeddings=args.sync_embeddings,
        embeddings_dir=args.embeddings_dir,
        sync_sidecars=args.sync_sidecars,
        extra_parquets=args.extra_parquets,
    )


if __name__ == "__main__":
    main()
