"""
Batch Mapillary Sequence Auditor and Purge Utility for Geo-RAG.

Audits Mapillary image sequences within a target geographic scope (e.g. continent),
detects invalid sequences caused by stationary cameras, multipath GPS drift, and
impossible speed spikes, and executes a streaming purge of the corrupted images.
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.credentials import get_mapillary_token
from src.utils.mapillary_trajectory_validator import MapillaryTrajectoryValidator

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
    input_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Audits Mapillary photos in the target continent, detects invalid sequences,
    and removes all associated photos.
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

    for rg in range(num_row_groups):
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

    for seq_id in tqdm(unique_seqs, desc="Auditing sequences"):
        res = validator.validate_sequence(seq_id)
        if not res.is_valid:
            invalid_sequences[seq_id] = {
                "reason": res.reason,
                "metrics": res.metrics.to_dict()
                if hasattr(res.metrics, "to_dict")
                else (res.metrics.__dict__ if res.metrics else None),
            }
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

    if dry_run or not violating_photo_ids:
        if dry_run:
            print(
                "\n[DRY RUN] No files modified. Run without --dry_run to purge these records."
            )
        return {
            "status": "dry_run" if dry_run else "no_violations",
            "purged_photos": len(violating_photo_ids),
            "invalid_sequences": len(invalid_sequences),
        }

    # Step 4: Stream-purge violating photos from Parquet
    print(
        f"\nStep 4: Executing streaming purge of {len(violating_photo_ids):,} photos from Parquet..."
    )
    t_purge = time.time()
    temp_output = output_parquet + ".tmp_audit_purge"

    total_in = 0
    total_out = 0
    purged_count = 0

    with get_parquet_writer(temp_output, schema) as writer:
        for rg in range(num_row_groups):
            tbl_rg = pf.read_row_group(rg)
            total_in += len(tbl_rg)

            rg_pids = tbl_rg[photo_id_col].to_numpy().astype(str)
            clean_pids = np.array([p.removesuffix(".0") for p in rg_pids])
            rg_plat = np.char.lower(tbl_rg["Platform"].to_numpy().astype(str))

            is_target_plat = rg_plat == platform.lower()
            is_violating = (
                np.isin(clean_pids, list(violating_photo_ids)) & is_target_plat
            )
            keep_mask = ~is_violating

            filtered_tbl = tbl_rg.filter(pa.array(keep_mask))
            writer.write_table(filtered_tbl)

            rg_purged = int(np.sum(is_violating))
            purged_count += rg_purged
            total_out += len(filtered_tbl)

    os.replace(temp_output, output_parquet)
    print(
        f" -> Purged {purged_count:,} rows. Dataset reduced from {total_in:,} to {total_out:,} rows in {time.time() - t_purge:.2f}s."
    )
    print(f" -> Updated clean Parquet dataset: {output_parquet}")

    # Step 5: Clean matching CSV if provided
    if input_csv and os.path.exists(input_csv):
        target_csv_out = output_csv or input_csv
        print(f"\nStep 5: Cleaning CSV metadata file: {input_csv}...")
        temp_csv = target_csv_out + ".tmp_audit_purge"
        first_chunk = True
        for chunk in pd.read_csv(
            input_csv,
            chunksize=100_000,
            dtype={photo_id_col: str, "Platform": str},
        ):
            c_pids = chunk[photo_id_col].fillna("").astype(str).str.removesuffix(".0")
            c_plat = chunk["Platform"].fillna("").astype(str).str.lower()
            violating_csv_mask = (c_plat == platform.lower()) & c_pids.isin(
                violating_photo_ids
            )
            clean_chunk = chunk[~violating_csv_mask]
            clean_chunk.to_csv(
                temp_csv,
                mode="w" if first_chunk else "a",
                index=False,
                header=first_chunk,
            )
            first_chunk = False
        os.replace(temp_csv, target_csv_out)
        print(f" -> Updated clean CSV file: {target_csv_out}")

    print(
        "\n================================================================================"
    )
    print("🎉 MAPILLARY AUDIT & PURGE COMPLETE")
    print(
        "================================================================================"
    )
    print(f"Total Time Taken: {time.time() - t0:.2f}s")
    print(
        "💡 Remember to synchronize companion embeddings and clustered sidecars using:"
    )
    print(
        f"   python3 src/utils/sync_offline_dataset.py --master_online {output_parquet} --target_offline full_pipeline_output/geo_space_cleaned_offline.parquet"
    )
    print(
        "================================================================================"
    )

    return {
        "status": "purged",
        "purged_photos": purged_count,
        "invalid_sequences": len(invalid_sequences),
        "total_records_remaining": total_out,
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
        "--csv",
        type=str,
        default=None,
        help="Optional path to matching CSV file to clean.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to output CSV file (defaults to input CSV).",
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
        input_csv=args.csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
