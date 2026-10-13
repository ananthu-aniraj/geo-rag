"""Embedding matrix loader for inline Parquet columns and companion .npy files."""

import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.companion_paths import get_core_base_name


def _load_embeddings(
    parquet_path,
    column="embedding",
    representation_type="cls",
    model_name=None,
    precision=None,
    core_name_resolver=None,
):
    """
    Backward-compatible loader that returns memory-mapped or raw embedding matrices.
    Supports dynamic mapping lookup via 'embedding_idx' to load from a shared base file.
    """
    if core_name_resolver is None:
        core_name_resolver = get_core_base_name

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    if column in pf.schema_arrow.names:
        # Case A: Combined format (read via pyarrow table and stack)
        return _read_inline_embeddings(pf, column)

    # Case B: Decoupled format (.npy)
    db_dir, candidate_dirs = _candidate_dirs(parquet_path)

    # Auto-detect sidecar metadata from Parquet schema if not explicitly provided
    model_name, representation_type, precision = _read_sidecar_metadata(
        pf, model_name, representation_type, precision
    )

    base_name = os.path.splitext(os.path.basename(parquet_path))[0]

    # Trim '_clustered_k_X' suffix if present to find base name
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]

    is_tipsv2 = (model_name is None) or ("tipsv2" in model_name.lower())
    model_slug = model_name.replace("/", "_") if model_name else None

    # Strip any known provenance tags that may have been baked into sidecar name
    # e.g., geo_space_offline_google_tipsv2-b14_cls_float16 -> geo_space_offline
    stripped_base = _strip_provenance_tags(
        base_name, model_slug, representation_type, precision
    )

    # Find candidate base dataset stem by scanning candidate dirs
    candidate_bases = _find_candidate_bases(
        candidate_dirs,
        base_name,
        stripped_base,
        core_name_resolver=core_name_resolver,
    )
    clean_base = candidate_bases[0] if candidate_bases else stripped_base
    core_name = core_name_resolver(clean_base)

    bases_to_check = _unique_bases(
        [
            clean_base,
            core_name,
            stripped_base,
            core_name_resolver(stripped_base),
            base_name,
            core_name_resolver(base_name),
        ]
    )

    npy_path = _resolve_npy_path(
        column=column,
        representation_type=representation_type,
        model_name=model_name,
        is_tipsv2=is_tipsv2,
        base_name=base_name,
        bases_to_check=bases_to_check,
        candidate_dirs=candidate_dirs,
        pf=pf,
        db_dir=db_dir,
    )
    if os.path.exists(npy_path):
        return _read_companion_embeddings(
            pf,
            npy_path,
            base_name,
            db_dir,
            core_name_resolver=core_name_resolver,
        )

    if not is_tipsv2:
        raise FileNotFoundError(
            f"Embeddings for model '{model_name}' (representation '{representation_type}') were not found on disk for '{parquet_path}' in '{db_dir}'. "
            f"(Note: Embedding files without a model name are assumed to be TIPSv2). "
            f"Please run 'python -m src.processing.backfill_embeddings --input {parquet_path} --model_name {model_name} --representation_type {representation_type}' "
            f"to compute and backfill embeddings for this model before proceeding."
        )
    else:
        model_display = model_name or "google/tipsv2-b14"
        raise FileNotFoundError(
            f"Embeddings for model '{model_display}' (representation '{representation_type}') were not found on disk for '{parquet_path}' in '{db_dir}'. "
            f"Please run 'python -m src.processing.backfill_embeddings --input {parquet_path} --model_name {model_display} --representation_type {representation_type}' "
            f"to compute and backfill embeddings for this representation before proceeding."
        )


def _read_inline_embeddings(pf, column):
    """Case A: stacks an embedding column stored directly in the Parquet file."""
    table = pf.read(columns=[column])
    chunked_arr = table[column]
    num_rows = len(table)
    dim = len(chunked_arr.chunk(0)[0].as_py())

    emb_matrix = np.empty((num_rows, dim), dtype=np.float32)
    current_row = 0
    for chunk in chunked_arr.chunks:
        chunk_len = len(chunk)
        flat_chunk = chunk.flatten().to_numpy()
        emb_matrix[current_row : current_row + chunk_len] = flat_chunk.reshape(
            chunk_len, dim
        )
        current_row += chunk_len
    return emb_matrix


def _candidate_dirs(parquet_path):
    """Returns the directory of the Parquet file and its parent, if different."""
    db_dir = os.path.dirname(os.path.abspath(parquet_path))
    parent_dir = os.path.dirname(db_dir)
    candidate_dirs = [db_dir]
    if parent_dir and parent_dir != db_dir:
        candidate_dirs.append(parent_dir)
    return db_dir, candidate_dirs


def _read_sidecar_metadata(pf, model_name, representation_type, precision):
    """Fills model_name, representation_type and precision from Parquet columns when unset."""
    if model_name is None and "model_name" in pf.schema_arrow.names:
        try:
            m_val = pf.read(columns=["model_name"])["model_name"].to_pylist()[0]
            if m_val:
                model_name = m_val
        except Exception:
            pass

    if (
        representation_type is None or representation_type == "cls"
    ) and "representation_type" in pf.schema_arrow.names:
        try:
            r_val = pf.read(columns=["representation_type"])[
                "representation_type"
            ].to_pylist()[0]
            if r_val:
                representation_type = r_val
        except Exception:
            pass

    if precision is None and "precision" in pf.schema_arrow.names:
        try:
            p_val = pf.read(columns=["precision"])["precision"].to_pylist()[0]
            if p_val:
                precision = p_val
        except Exception:
            pass

    return model_name, representation_type, precision


def _strip_provenance_tags(base_name, model_slug, representation_type, precision):
    """Removes model and representation tags that may be baked into the file name."""
    stripped_base = base_name
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
        if tag in stripped_base:
            stripped_base = stripped_base.split(tag)[0]
    return stripped_base


def _find_candidate_bases(
    candidate_dirs, base_name, stripped_base, core_name_resolver=None
):
    """Lists base Parquet stems in the candidate dirs that could own these embeddings, best first."""
    if core_name_resolver is None:
        core_name_resolver = get_core_base_name
    candidate_bases = []
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
                    base_name.startswith(cand_stem)
                    or stripped_base.startswith(cand_stem)
                    or cand_stem.startswith(stripped_base)
                    or core_name_resolver(stripped_base)
                    == core_name_resolver(cand_stem)
                ):
                    candidate_bases.append(cand_stem)

    def rank_base_stem(stem):
        is_cleaned = "cleaned" in stem
        is_dedup = "deduplicated" in stem
        clean_score = 2 if is_cleaned else (0 if is_dedup else 1)
        exact_match = stem == stripped_base or stem == base_name
        return (clean_score, exact_match, len(stem))

    candidate_bases.sort(key=rank_base_stem, reverse=True)
    return candidate_bases


def _unique_bases(candidates):
    """Drops empty and repeated names, keeping the first occurrence order."""
    bases = []
    for candidate in candidates:
        if candidate and candidate not in bases:
            bases.append(candidate)
    return bases


def _npy_path(base, model, column, representation_type, candidate_dirs, db_dir):
    """Builds the companion .npy name for a base and model, preferring an existing file."""
    model_suf = f"_{model.replace('/', '_')}" if model else ""
    if column == "embedding":
        name = f"{base}{model_suf}_{representation_type}_embeddings.npy"
    elif column == "patch_embedding":
        name = f"{base}{model_suf}_patch_embeddings.npy"
    else:
        name = f"{base}{model_suf}_{column}_embeddings.npy"
    for d in candidate_dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(db_dir, name)


def _suffix_matches(filename, req_rep):
    has_cls_avg = "cls_avg_patch" in filename
    has_avg = "avg_patch" in filename and not has_cls_avg
    has_cls = "cls" in filename and not has_cls_avg

    if req_rep == "cls_avg_patch":
        return has_cls_avg
    elif req_rep == "avg_patch":
        return has_avg
    elif req_rep == "cls":
        # Allow fallback if the filename doesn't contain any known suffix
        if not has_cls_avg and not has_avg:
            return True
        return has_cls
    return True


def _resolve_npy_path(
    column,
    representation_type,
    model_name,
    is_tipsv2,
    base_name,
    bases_to_check,
    candidate_dirs,
    pf,
    db_dir,
):
    """Finds the companion .npy file, trying the fallbacks in order. Returns the best guess if none exists."""

    def npy_for(base, model):
        return _npy_path(
            base, model, column, representation_type, candidate_dirs, db_dir
        )

    # 1. Try resolving with model_name if provided
    npy_path = npy_for(base_name, model_name)
    if model_name:
        for b in bases_to_check:
            path = npy_for(b, model_name)
            if os.path.exists(path):
                npy_path = path
                break

    # 2. Fallback to model-agnostic (legacy) path resolution ONLY for TIPSv2 models
    # (simplifying assumption: if the embedding file does not contain a model name, it's a TIPSv2 model)
    if not os.path.exists(npy_path) and is_tipsv2:
        for b in bases_to_check:
            path = npy_for(b, None)
            if os.path.exists(path):
                npy_path = path
                break

    # 3. Fallback: check for shared deduplicated.npy if base file is cleaned.parquet
    if not os.path.exists(npy_path) and any("cleaned" in b for b in bases_to_check):
        npy_path = _find_deduplicated_npy(
            npy_path,
            npy_for=npy_for,
            bases_to_check=bases_to_check,
            column=column,
            model_name=model_name,
            is_tipsv2=is_tipsv2,
            candidate_dirs=candidate_dirs,
        )

    # 4. Wildcard search fallback for different column suffixes (e.g. cls_embeddings)
    if not os.path.exists(npy_path):
        npy_path = _wildcard_npy_search(
            npy_path,
            bases_to_check=bases_to_check,
            candidate_dirs=candidate_dirs,
            base_name=base_name,
            column=column,
            representation_type=representation_type,
            model_name=model_name,
            is_tipsv2=is_tipsv2,
        )

    # 5. Sidecar fallback: inspect companion keys files in candidate directories
    if not os.path.exists(npy_path):
        npy_path = _sidecar_npy_search(
            npy_path,
            pf=pf,
            candidate_dirs=candidate_dirs,
            representation_type=representation_type,
            model_name=model_name,
            is_tipsv2=is_tipsv2,
        )

    return npy_path


def _find_deduplicated_npy(
    npy_path,
    npy_for,
    bases_to_check,
    column,
    model_name,
    is_tipsv2,
    candidate_dirs,
):
    """Looks for the deduplicated companion file behind a cleaned base name."""
    for b in bases_to_check:
        if "cleaned" not in b:
            continue
        fallback_base = b.replace("cleaned", "deduplicated")
        if model_name:
            p = npy_for(fallback_base, model_name)
            if os.path.exists(p):
                return p
        if is_tipsv2:
            p = npy_for(fallback_base, None)
            if os.path.exists(p):
                return p
            fallback_name = (
                f"{fallback_base}.npy"
                if column == "embedding"
                else f"{fallback_base}_{column}.npy"
            )
            for d in candidate_dirs:
                p_legacy = os.path.join(d, fallback_name)
                if os.path.exists(p_legacy):
                    return p_legacy
    return npy_path


def _wildcard_npy_search(
    npy_path,
    bases_to_check,
    candidate_dirs,
    base_name,
    column,
    representation_type,
    model_name,
    is_tipsv2,
):
    """Globs for any <base>*.npy file that fits the requested column, representation and model."""
    bases = list(bases_to_check)
    for b in bases_to_check:
        if "cleaned" in b:
            bases.append(b.replace("cleaned", "deduplicated"))

    for d in candidate_dirs:
        for b in bases:
            pattern = os.path.join(d, f"{b}*.npy")
            matches = glob.glob(pattern)
            if matches:
                # Exclude checkpoint files if base_name does not contain checkpoint
                if "checkpoint" not in base_name:
                    matches = [
                        m for m in matches if "checkpoint" not in os.path.basename(m)
                    ]
                # Filter matches by representation type if default 'embedding' column is requested
                if column == "embedding" and representation_type:
                    matches = [
                        m
                        for m in matches
                        if _suffix_matches(os.path.basename(m), representation_type)
                    ]
                # For non-TIPSv2 models, NEVER pick up files without the model name!
                if not is_tipsv2:
                    model_clean = model_name.replace("/", "_")
                    matches = [m for m in matches if model_clean in os.path.basename(m)]
                elif model_name:
                    # For TIPSv2 with explicit model name, prefer matching model name if available
                    model_clean = model_name.replace("/", "_")
                    model_matches = [
                        m for m in matches if model_clean in os.path.basename(m)
                    ]
                    if model_matches:
                        matches = model_matches

                if not matches:
                    continue

                # If there's a file matching the specific column name, use it
                col_match = [m for m in matches if column in os.path.basename(m)]
                if col_match:
                    return col_match[0]
                # Otherwise, if default 'embedding' was requested, try finding cls_embeddings or similar
                if column == "embedding":
                    preferred = [
                        m
                        for m in matches
                        if "cls_embeddings" in os.path.basename(m)
                        or "embedding" in os.path.basename(m)
                    ]
                    if preferred:
                        return preferred[0]
                # Fallback to the first match
                return matches[0]
    return npy_path


def _sidecar_npy_search(
    npy_path, pf, candidate_dirs, representation_type, model_name, is_tipsv2
):
    """For sidecar Parquet files, finds a .npy whose sibling .keys.parquet exists."""
    is_sidecar = (
        "cluster_id" in pf.schema_arrow.names
        and "Latitude" not in pf.schema_arrow.names
        and "Longitude" not in pf.schema_arrow.names
    )
    if not is_sidecar:
        return npy_path

    for d in candidate_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".keys.parquet"):
                candidate_npy = os.path.join(d, f.replace(".keys.parquet", ".npy"))
                if not os.path.exists(candidate_npy):
                    continue
                if representation_type and not _suffix_matches(f, representation_type):
                    continue
                if not is_tipsv2:
                    model_clean = model_name.replace("/", "_")
                    if model_clean not in f:
                        continue
                return candidate_npy
    return npy_path


def _read_companion_embeddings(
    pf, npy_path, base_name, db_dir, core_name_resolver=None
):
    """Reads a companion .npy and reorders its rows to match the Parquet metadata."""
    emb = np.load(npy_path, mmap_mode="r")

    # If stable photo_key or platform/ID is available, and companion keys file exists, use it
    keys_path = npy_path.replace(".npy", ".keys.parquet")
    has_photo_key = "photo_key" in pf.schema_arrow.names
    has_platform_and_id = (
        "Platform" in pf.schema_arrow.names and "Photo_ID" in pf.schema_arrow.names
    )

    if (has_photo_key or has_platform_and_id) and os.path.exists(keys_path):
        return _gather_by_companion_keys(pf, emb, keys_path, has_photo_key)

    # Fallback to older embedding_idx mapping logic
    return _gather_by_embedding_idx(
        pf,
        emb,
        base_name,
        db_dir,
        core_name_resolver=core_name_resolver,
    )


def _gather_by_companion_keys(pf, emb, keys_path, has_photo_key):
    """Picks the companion rows whose keys match the Parquet rows, zero-filling misses."""
    print(f" -> Resolving embeddings via companion keys index: {keys_path}")
    meta_keys = _read_meta_keys(pf, has_photo_key)
    indices = _match_keys_to_rows(meta_keys, keys_path)

    valid_mask = indices >= 0
    if not valid_mask.all():
        missing_samples = list(meta_keys[~valid_mask][:10])
        print(
            f"Warning: Found {np.sum(~valid_mask):,} missing keys in embeddings keys index. Zero-filling..."
        )
        print(f" -> Sample missing keys: {missing_samples}")
        safe_indices = np.clip(indices, 0, len(emb) - 1)
        sliced_emb = emb[safe_indices].astype(np.float32)
        sliced_emb[~valid_mask] = 0.0
        return sliced_emb

    return emb[indices].astype(np.float32)


def _read_meta_keys(pf, has_photo_key):
    """Returns one lowercased lookup key per Parquet row."""
    if has_photo_key:
        meta_keys_table = pf.read(columns=["photo_key"])
        return meta_keys_table["photo_key"].to_pandas().astype(str).str.lower().values
    meta_keys_table = pf.read(columns=["Platform", "Photo_ID"])
    df_temp = meta_keys_table.to_pandas()
    return (
        df_temp["Platform"].astype(str).str.lower()
        + "_"
        + df_temp["Photo_ID"].astype(str)
    ).values


def _match_keys_to_rows(meta_keys, keys_path):
    """Maps each lookup key to its row in the companion keys file, or -1 when absent."""
    master_keys = pd.Index(
        pd.read_parquet(keys_path, columns=["photo_key"])["photo_key"]
        .astype(str)
        .str.lower()
    )
    if master_keys.is_unique:
        return master_keys.get_indexer(meta_keys)

    # If keys contain duplicates, resolve each key to its first occurrence in the matrix
    pos_series = pd.Series(np.arange(len(master_keys)), index=master_keys)
    pos_series = pos_series[~pos_series.index.duplicated(keep="first")]
    return pos_series.reindex(meta_keys, fill_value=-1).values


def _gather_by_embedding_idx(pf, emb, base_name, db_dir, core_name_resolver=None):
    """Uses the embedding_idx column (in this file or its base file) to pick rows from the matrix."""
    if core_name_resolver is None:
        core_name_resolver = get_core_base_name
    has_embedding_idx = "embedding_idx" in pf.schema_arrow.names
    pf_for_idx = pf

    if not has_embedding_idx and "Latitude" not in pf.schema_arrow.names:
        # Resolve to base metadata file if this is a sidecar file
        core_name = core_name_resolver(base_name)
        for fallback in [
            f"{base_name}_cleaned.parquet",
            f"{core_name}_cleaned.parquet",
            f"{base_name}_deduplicated.parquet",
            f"{core_name}_deduplicated.parquet",
            f"{base_name}.parquet",
            f"{core_name}.parquet",
        ]:
            fallback_path = os.path.join(db_dir, fallback)
            if os.path.exists(fallback_path):
                pf_base = pq.ParquetFile(fallback_path)
                if "embedding_idx" in pf_base.schema_arrow.names:
                    has_embedding_idx = True
                    pf_for_idx = pf_base
                break

    if not has_embedding_idx:
        return emb.astype(np.float32)

    idx_table = pf_for_idx.read(columns=["embedding_idx"])
    indices = idx_table["embedding_idx"].to_numpy()

    # Check bounds safety against the actual loaded matrix
    valid_mask = (indices >= 0) & (indices < len(emb))
    if not valid_mask.all():
        print(
            f"Warning: Found {np.sum(~valid_mask):,} out-of-bounds indices in embedding_idx. Clamping and zero-filling..."
        )
        safe_indices = np.clip(indices, 0, len(emb) - 1)
        sliced_emb = emb[safe_indices].astype(np.float32)
        sliced_emb[~valid_mask] = 0.0
        return sliced_emb

    return emb[indices].astype(np.float32)


def load_embeddings(
    parquet_path,
    column="embedding",
    representation_type="cls",
    model_name=None,
    precision=None,
):
    """Load embeddings from inline columns or companion NumPy files."""
    return _load_embeddings(
        parquet_path,
        column=column,
        representation_type=representation_type,
        model_name=model_name,
        precision=precision,
        core_name_resolver=get_core_base_name,
    )
