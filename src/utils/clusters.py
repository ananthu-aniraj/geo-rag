"""Cluster dataset loader for Parquet data and cluster sidecars."""

import os
import re

import pyarrow.parquet as pq

from src.utils.companion_paths import get_core_base_name
from src.utils.table_io import load_dataframe


def _load_dataset_with_clusters(
    parquet_path,
    k_clusters=50000,
    columns=None,
    representation_type=None,
    model_name=None,
    precision=None,
    _load_dataframe=None,
    _get_core_base_name=None,
    **kwargs,
):
    """
    Backward-compatible loader that returns metadata and cluster assignments.
    Merges sidecars automatically if the base parquet does not contain cluster columns.
    """
    if _load_dataframe is None:
        _load_dataframe = load_dataframe
    if _get_core_base_name is None:
        _get_core_base_name = get_core_base_name

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    # Auto-detect k_clusters if the input filename contains it
    match = re.search(r"_clustered_k_(\d+)", os.path.basename(parquet_path))
    if match:
        k_clusters = int(match.group(1))

    # Auto-extract k_clusters from filename if it has _k_X suffix
    match = re.search(r"_k_(\d+)", os.path.basename(parquet_path))
    if match:
        k_clusters = int(match.group(1))

    # Inspect schema names using PyArrow ParquetFile
    pf = pq.ParquetFile(parquet_path)
    schema_names = pf.schema_arrow.names

    known_sidecar_path = None
    # If it's a decoupled sidecar file (has cluster_id but lacks base columns like Latitude/Longitude),
    # resolve the path to the base metadata file instead.
    if "cluster_id" in schema_names and (
        "Latitude" not in schema_names or "Longitude" not in schema_names
    ):
        known_sidecar_path = parquet_path
        db_dir = os.path.dirname(os.path.abspath(parquet_path))
        parent_dir = os.path.dirname(db_dir)
        candidate_dirs = [db_dir]
        if parent_dir and parent_dir != db_dir:
            candidate_dirs.append(parent_dir)

        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        prefix = base_name.split("_clustered_k_")[0]

        if model_name is None and "model_name" in schema_names:
            try:
                m_val = pf.read(columns=["model_name"])["model_name"].to_pylist()[0]
                if m_val:
                    model_name = m_val
            except Exception:
                pass
        if (
            representation_type is None or representation_type == "cls"
        ) and "representation_type" in schema_names:
            try:
                r_val = pf.read(columns=["representation_type"])[
                    "representation_type"
                ].to_pylist()[0]
                if r_val:
                    representation_type = r_val
            except Exception:
                pass
        if precision is None and "precision" in schema_names:
            try:
                p_val = pf.read(columns=["precision"])["precision"].to_pylist()[0]
                if p_val:
                    precision = p_val
            except Exception:
                pass

        model_slug = model_name.replace("/", "_") if model_name else None
        stripped_prefix = prefix
        tags_to_strip = []
        if model_slug and representation_type and precision:
            tags_to_strip.append(f"_{model_slug}_{representation_type}_{precision}")
        if model_slug and representation_type:
            tags_to_strip.append(f"_{model_slug}_{representation_type}")
        if model_slug:
            tags_to_strip.append(f"_{model_slug}")
        tags_to_strip.extend(
            [
                "_float16",
                "_float32",
                "_cls_avg_patch",
                "_avg_patch",
                "_cls",
            ]
        )
        for tag in tags_to_strip:
            if tag in stripped_prefix:
                stripped_prefix = stripped_prefix.split(tag)[0]

        # Scan candidate directories for matching base datasets (longest prefix match first)
        candidate_base_files = []
        for d in candidate_dirs:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if (
                    f.endswith(".parquet")
                    and not f.endswith(".keys.parquet")
                    and "_clustered_k_" not in f
                    and "_h3_semantic_index" not in f
                ):
                    cand_stem = f[:-8]
                    if (
                        prefix.startswith(cand_stem)
                        or stripped_prefix.startswith(cand_stem)
                        or cand_stem.startswith(stripped_prefix)
                        or _get_core_base_name(stripped_prefix)
                        == _get_core_base_name(cand_stem)
                    ):
                        candidate_base_files.append(os.path.join(d, f))

        if not candidate_base_files:
            for d in candidate_dirs:
                if not os.path.exists(d):
                    continue
                for f in os.listdir(d):
                    if (
                        f.endswith(".parquet")
                        and not f.endswith(".keys.parquet")
                        and "_clustered_k_" not in f
                        and "_h3_semantic_index" not in f
                    ):
                        candidate_base_files.append(os.path.join(d, f))

        sidecar_num_rows = None
        if known_sidecar_path and os.path.exists(known_sidecar_path):
            try:
                sidecar_num_rows = pq.ParquetFile(known_sidecar_path).metadata.num_rows
            except Exception:
                pass

        def rank_candidate_base(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            # 1. Exact row count match with sidecar (highest priority)
            row_match = False
            if sidecar_num_rows is not None:
                try:
                    c_pf = pq.ParquetFile(p)
                    row_match = c_pf.metadata.num_rows == sidecar_num_rows
                except Exception:
                    pass

            # 2. Prioritize cleaned over deduplicated:
            # In the pipeline: raw -> deduplicated -> cleaned.
            # Clustered sidecars are produced on the cleaned dataset.
            is_cleaned = "cleaned" in stem
            is_dedup = "deduplicated" in stem
            clean_score = 2 if is_cleaned else (0 if is_dedup else 1)

            # 3. Exact stem match
            exact_stem = stem == stripped_prefix or stem == prefix

            return (row_match, clean_score, exact_stem, len(stem))

        candidate_base_files.sort(key=rank_candidate_base, reverse=True)

        found_base = False
        for fallback_path in candidate_base_files:
            if os.path.exists(fallback_path):
                parquet_path = fallback_path
                pf = pq.ParquetFile(parquet_path)
                schema_names = pf.schema_arrow.names
                found_base = True
                break

    cluster_cols = [
        "cluster_id",
        "cluster_label",
        "cluster_description",
        "parent_cluster_id",
        "parent_cluster_label",
        "parent_cluster_description",
        "visual_description",
        "parent_visual_description",
        "model_name",
        "representation_type",
        "precision",
    ]

    # Filter which columns belong to cluster variables vs base metadata
    if columns is None:
        requested_cols = list(schema_names) + cluster_cols
    else:
        requested_cols = columns

    req_cluster_cols = [c for c in requested_cols if c in cluster_cols]
    req_meta_cols = [
        c
        for c in requested_cols
        if c not in cluster_cols or c in ["Platform", "Photo_ID"]
    ]

    # Case A: Old format contains cluster columns directly (or we failed to fallback to base)
    if "cluster_id" in schema_names and "Latitude" in schema_names:
        return _load_dataframe(parquet_path, columns=columns, **kwargs)

    # Case B: Decoupled format
    df_meta = _load_dataframe(parquet_path, columns=req_meta_cols, **kwargs)

    if known_sidecar_path and os.path.exists(known_sidecar_path):
        sidecar_path = known_sidecar_path
    else:
        # Check for and load sidecar file
        db_dir = os.path.dirname(os.path.abspath(parquet_path))
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]

        # Trim '_clustered_k_X' suffix if present to find base name
        if "_clustered_k_" in base_name:
            base_name = base_name.split("_clustered_k_")[0]

        core_name = _get_core_base_name(base_name)
        model_slug = model_name.replace("/", "_") if model_name else None

        candidates = []
        if model_slug and representation_type and precision:
            candidates.extend(
                [
                    os.path.join(
                        db_dir,
                        f"{base_name}_{model_slug}_{representation_type}_{precision}_clustered_k_{k_clusters}.parquet",
                    ),
                    os.path.join(
                        db_dir,
                        f"{core_name}_{model_slug}_{representation_type}_{precision}_clustered_k_{k_clusters}.parquet",
                    ),
                ]
            )
        if model_slug and representation_type:
            candidates.extend(
                [
                    os.path.join(
                        db_dir,
                        f"{base_name}_{model_slug}_{representation_type}_clustered_k_{k_clusters}.parquet",
                    ),
                    os.path.join(
                        db_dir,
                        f"{core_name}_{model_slug}_{representation_type}_clustered_k_{k_clusters}.parquet",
                    ),
                ]
            )
        if model_slug:
            candidates.extend(
                [
                    os.path.join(
                        db_dir,
                        f"{base_name}_{model_slug}_clustered_k_{k_clusters}.parquet",
                    ),
                    os.path.join(
                        db_dir,
                        f"{core_name}_{model_slug}_clustered_k_{k_clusters}.parquet",
                    ),
                ]
            )
        candidates.extend(
            [
                os.path.join(db_dir, f"{base_name}_clustered_k_{k_clusters}.parquet"),
                os.path.join(db_dir, f"{core_name}_clustered_k_{k_clusters}.parquet"),
            ]
        )

        sidecar_path = None
        for p in candidates:
            if os.path.exists(p):
                sidecar_path = p
                break

    if sidecar_path and os.path.exists(sidecar_path) and req_cluster_cols:
        # We need join keys in both dataframes for merging
        pf_side = pq.ParquetFile(sidecar_path)
        side_names = pf_side.schema_arrow.names
        join_cols = ["Platform", "Photo_ID"]
        if "photo_key" in side_names and "photo_key" in df_meta.columns:
            join_cols = ["photo_key"]

        sidecar_cols = list(set(join_cols + req_cluster_cols))
        side_avail_cols = [c for c in sidecar_cols if c in side_names]

        df_sidecar = _load_dataframe(sidecar_path, columns=side_avail_cols, **kwargs)
        if known_sidecar_path:
            # Caller requested the clustered sidecar directly: preserve sidecar rows and sidecar ordering exactly
            meta_cols_to_merge = [
                c
                for c in df_meta.columns
                if c not in df_sidecar.columns or c in join_cols
            ]
            df_meta = df_sidecar.merge(
                df_meta[meta_cols_to_merge], on=join_cols, how="left"
            )
        else:
            df_meta = df_meta.merge(df_sidecar, on=join_cols, how="left")

    return df_meta


def load_dataset_with_clusters(
    parquet_path,
    k_clusters=50000,
    columns=None,
    representation_type=None,
    model_name=None,
    precision=None,
    **kwargs,
):
    """Load metadata and cluster assignments from a Parquet dataset."""
    return _load_dataset_with_clusters(
        parquet_path,
        k_clusters=k_clusters,
        columns=columns,
        representation_type=representation_type,
        model_name=model_name,
        precision=precision,
        _load_dataframe=load_dataframe,
        _get_core_base_name=get_core_base_name,
        **kwargs,
    )
