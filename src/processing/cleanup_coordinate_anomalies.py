import os
import shutil
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.credentials import get_mapillary_token
from src.utils.io import get_core_base_name, get_parquet_writer
from src.utils.sync_offline_dataset import (
    compute_photo_key,
    discover_companion_embedding_files,
)


def parse_platform_arg(
    platform_arg: Optional[Union[str, Sequence[str]]],
) -> Set[str]:
    """Parse `--platform` CLI argument (string or list of strings) into a set of lowercase platform names."""
    if not platform_arg:
        return set()
    if isinstance(platform_arg, str):
        raw_items = [platform_arg]
    else:
        raw_items = [str(x) for x in platform_arg if x is not None]

    parsed: Set[str] = set()
    for item in raw_items:
        for token in item.replace(",", " ").split():
            clean = token.strip().lower()
            if clean and clean not in ("all", "*"):
                parsed.add(clean)
    return parsed


def fetch_mapillary_sequences(
    photo_ids: list, token: str, batch_size: int = 100
) -> dict:
    """
    Batch-queries Mapillary Graph API v4 to map photo_ids to sequence_ids.
    Returns dict: {photo_id: sequence_id}
    """
    if not token or not photo_ids:
        return {}

    import requests

    headers = {"Authorization": f"OAuth {token}"}
    seq_map = {}
    clean_ids = [str(pid).strip().removesuffix(".0") for pid in photo_ids if pid]

    for i in range(0, len(clean_ids), batch_size):
        chunk = clean_ids[i : i + batch_size]
        url = f"https://graph.mapillary.com/?ids={','.join(chunk)}&fields=id,sequence"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    for pid, info in data.items():
                        if isinstance(info, dict) and "sequence" in info:
                            seq_map[str(pid)] = str(info["sequence"])
                    break
                elif r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    break
            except Exception:
                if attempt == 2:
                    break
                time.sleep(1.0)
    return seq_map


def fetch_sequence_all_image_ids(sequence_ids: set, token: str) -> set:
    """
    Queries Mapillary Graph API v4 to retrieve all image IDs belonging to the given sequence IDs.
    Returns set of image ID strings.
    """
    if not token or not sequence_ids:
        return set()

    import requests

    headers = {"Authorization": f"OAuth {token}"}
    all_image_ids = set()

    for seq_id in sequence_ids:
        if not seq_id:
            continue
        url = f"https://graph.mapillary.com/image_ids?sequence_id={seq_id}&limit=2000"
        while url:
            success = False
            for attempt in range(3):
                try:
                    r = requests.get(url, headers=headers, timeout=15)
                    if r.status_code == 200:
                        data = r.json()
                        for item in data.get("data", []):
                            if "id" in item:
                                all_image_ids.add(
                                    str(item["id"]).strip().removesuffix(".0")
                                )
                        url = data.get("paging", {}).get("next")
                        success = True
                        break
                    elif r.status_code == 429:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        break
                except Exception:
                    if attempt == 2:
                        break
                    time.sleep(1.0)
            if not success:
                break
    return all_image_ids


def resolve_companion_embeddings(
    parquet_path: str,
    model_name: Optional[str] = None,
    representation_type: str = "cls",
    embeddings_npy: Optional[str] = None,
    embeddings_keys: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Locate companion (.npy, .keys.parquet) files for the input dataset if present."""
    if embeddings_npy and embeddings_keys:
        if os.path.isfile(embeddings_npy) and os.path.isfile(embeddings_keys):
            return embeddings_npy, embeddings_keys

    db_dir = os.path.dirname(os.path.abspath(parquet_path))
    raw_base = os.path.splitext(os.path.basename(parquet_path))[0]
    core_base = get_core_base_name(raw_base)
    rep = representation_type or "cls"

    # 1. Try exact model + representation companion paths
    if model_name and model_name != "google/tipsv2-b14":
        model_slug = model_name.replace("/", "_")
        npy_p = os.path.join(db_dir, f"{core_base}_{model_slug}_{rep}_embeddings.npy")
        keys_p = npy_p.replace(".npy", ".keys.parquet")
        if os.path.isfile(npy_p) and os.path.isfile(keys_p):
            return npy_p, keys_p

    # 2. Try default companion paths (e.g. geo_space_cls_embeddings.npy)
    npy_p = os.path.join(db_dir, f"{core_base}_{rep}_embeddings.npy")
    keys_p = npy_p.replace(".npy", ".keys.parquet")
    if os.path.isfile(npy_p) and os.path.isfile(keys_p):
        return npy_p, keys_p

    # 3. Discover any companion pair in the same directory
    pairs = discover_companion_embedding_files(parquet_path)
    if pairs:
        return pairs[0]

    return None


def detect_remote_biome_knn_anomalies(
    df_meta: pd.DataFrame,
    target_scope_mask: np.ndarray,
    target_platforms: Set[str],
    npy_path: str,
    keys_path: str,
    k_neighbors: int = 25,
    incompat_threshold: float = 0.84,
    ref_per_group: int = 15000,
) -> Set[int]:
    """
    Detects misplaced images in remote/extreme regions (Polar EF/ET/Antarctica and
    remote Arid Deserts BWh/BWk) using Leave-One-Cell-Out Visual k-NN Climate Consensus.

    Platforms not in `target_platforms` (when `target_platforms` is non-empty) are
    treated as trusted non-target platforms and prioritized when constructing the
    Clean Reference Bank alongside multi-platform H3 cells.

    Returns a set of integer row indices in `df_meta` flagged as visual-climate anomalies.
    """
    if "Koppen_Code" not in df_meta.columns:
        print(
            " -> Notice: 'Koppen_Code' column not present; skipping visual-climate k-NN check."
        )
        return set()

    t0 = time.time()
    print(
        f"\n🌍 Running Leave-One-Cell-Out Visual k-NN Biome Consistency Check..."
        f"\n -> Embeddings : {os.path.basename(npy_path)}"
        f"\n -> Keys       : {os.path.basename(keys_path)}"
    )

    keys_df = pd.read_parquet(keys_path)
    emb_mmap = np.load(npy_path, mmap_mode="r")

    if "photo_key" in keys_df.columns:
        key_series = keys_df["photo_key"].astype(str)
    elif "Platform" in keys_df.columns and "Photo_ID" in keys_df.columns:
        key_series = pd.Series(
            [
                compute_photo_key(p, i)
                for p, i in zip(keys_df["Platform"], keys_df["Photo_ID"])
            ]
        )
    else:
        print(" -> Warning: Unrecognized schema in keys file; skipping k-NN check.")
        return set()

    key_to_idx = pd.Series(
        np.arange(len(key_series), dtype=np.int32), index=key_series.values
    ).to_dict()

    if "photo_key" in df_meta.columns:
        dataset_keys = df_meta["photo_key"].astype(str)
    elif "Photo_ID" in df_meta.columns:
        dataset_keys = pd.Series(
            [
                compute_photo_key(p, i)
                for p, i in zip(df_meta["Platform"], df_meta["Photo_ID"])
            ]
        )
    else:
        return set()

    emb_indices = dataset_keys.map(key_to_idx)
    valid_emb_mask = emb_indices.notna().to_numpy()
    if not np.any(valid_emb_mask):
        print(" -> Notice: No matching embedding keys found; skipping k-NN check.")
        return set()

    df_work = df_meta[valid_emb_mask].copy()
    df_work["orig_row_idx"] = np.where(valid_emb_mask)[0]
    df_work["emb_idx"] = emb_indices[valid_emb_mask].astype(np.int32).to_numpy()
    df_work["in_target_scope"] = target_scope_mask[valid_emb_mask]
    df_work["plat_lower"] = df_work["Platform"].str.lower()
    df_work["koppen_clean"] = df_work["Koppen_Code"].fillna("").astype(str)
    df_work["koppen_major"] = df_work["koppen_clean"].str[:1]

    # Coarsen H3 cells to Resolution 3 (~400km) and Resolution 4 (~177km)
    if "H3_Cell" in df_work.columns and df_work["H3_Cell"].notna().any():
        unq_h3 = df_work["H3_Cell"].dropna().unique()
        h3_to_r3 = {}
        h3_to_r4 = {}
        for c in unq_h3:
            try:
                h3_to_r3[c] = h3.cell_to_parent(str(c), 3)
                h3_to_r4[c] = h3.cell_to_parent(str(c), 4)
            except Exception:
                continue
        df_work["h3_r3"] = df_work["H3_Cell"].map(h3_to_r3).fillna("")
        df_work["h3_r4"] = df_work["H3_Cell"].map(h3_to_r4).fillna("")
    else:
        lats_w = df_work["Latitude"].to_numpy()
        lons_w = df_work["Longitude"].to_numpy()
        r3_list = []
        r4_list = []
        for la, lo in zip(lats_w, lons_w):
            if np.isfinite(la) and np.isfinite(lo):
                r3_list.append(h3.latlng_to_cell(float(la), float(lo), 3))
                r4_list.append(h3.latlng_to_cell(float(la), float(lo), 4))
            else:
                r3_list.append("")
                r4_list.append("")
        df_work["h3_r3"] = r3_list
        df_work["h3_r4"] = r4_list

    # Any platform not in target_platforms is a trusted non-target platform
    is_non_target_plat = (
        ~df_work["plat_lower"].isin(target_platforms)
        if target_platforms
        else pd.Series(False, index=df_work.index)
    )

    cell_plat_nunique = df_work.groupby("h3_r3")["plat_lower"].nunique()
    cell_img_counts = df_work.groupby("h3_r3").size()
    trusted_cells = set(
        cell_plat_nunique[(cell_plat_nunique >= 2) & (cell_img_counts >= 20)].index
    )
    trusted_cells.update(df_work.loc[is_non_target_plat, "h3_r3"].unique())
    trusted_cells.discard("")

    polar_cells = set(cell_img_counts[cell_img_counts >= 5].index).union(trusted_cells)
    polar_cells.discard("")

    rng = np.random.default_rng(42)
    ref_parts = []
    for major in ["A", "B", "C", "D", "E"]:
        if major == "E":
            pool = df_work[
                (df_work["koppen_major"] == major)
                & (df_work["h3_r3"].isin(polar_cells))
            ]
        else:
            pool = df_work[
                (df_work["koppen_major"] == major)
                & (df_work["h3_r3"].isin(trusted_cells))
            ]
        if pool.empty:
            pool = df_work[df_work["koppen_major"] == major]
        if pool.empty:
            continue

        # Prioritize non-target (trusted) platforms first when available, then sample from multi-platform pool
        if target_platforms:
            trusted_pool = pool[~pool["plat_lower"].isin(target_platforms)]
            other_pool = pool[pool["plat_lower"].isin(target_platforms)]
        else:
            trusted_pool = pool.iloc[0:0]
            other_pool = pool

        n_take = min(ref_per_group, len(pool))
        if len(trusted_pool) >= n_take:
            chosen = trusted_pool.iloc[
                rng.choice(len(trusted_pool), size=n_take, replace=False)
            ]
        else:
            n_rem = n_take - len(trusted_pool)
            chosen_other = other_pool.iloc[
                rng.choice(len(other_pool), size=n_rem, replace=False)
            ]
            chosen = pd.concat([trusted_pool, chosen_other], ignore_index=True)
        ref_parts.append(chosen)

    if len(ref_parts) < 2:
        print(" -> Notice: Insufficient climate diversity in dataset for k-NN check.")
        return set()

    ref_df = pd.concat(ref_parts, ignore_index=True)

    # Normalize reference vectors and trim bottom 10% outlier tail per Koppen group
    ref_emb = np.array(emb_mmap[ref_df["emb_idx"].to_numpy()], dtype=np.float32)
    norms = np.linalg.norm(ref_emb, axis=1, keepdims=True)
    ref_emb = ref_emb / np.maximum(norms, 1e-8)

    keep_ref = np.zeros(len(ref_df), dtype=bool)
    for major in ["A", "B", "C", "D", "E"]:
        m_idx = np.where(ref_df["koppen_major"].to_numpy() == major)[0]
        if len(m_idx) == 0:
            continue
        sub_vecs = ref_emb[m_idx]
        mean_vec = sub_vecs.mean(axis=0, keepdims=True)
        mean_vec = mean_vec / np.maximum(np.linalg.norm(mean_vec), 1e-8)
        sims = (sub_vecs @ mean_vec.T).squeeze(1)
        thresh = np.percentile(sims, 10) if len(sims) >= 10 else -1.0
        keep_ref[m_idx[sims >= thresh]] = True

    ref_df = ref_df[keep_ref].reset_index(drop=True)
    ref_emb = ref_emb[keep_ref]

    effective_k = min(k_neighbors, len(ref_df))
    if effective_k < 5:
        return set()

    print(
        f" -> Built Clean Reference Bank: {len(ref_df):,} anchor vectors across {ref_df['koppen_major'].nunique()} Köppen groups."
    )

    ref_h3_r3 = ref_df["h3_r3"].to_numpy()
    ref_koppen = ref_df["koppen_clean"].to_numpy()
    ref_lat = ref_df["Latitude"].to_numpy()
    ref_lon = ref_df["Longitude"].to_numpy()

    # Optional Torch GPU acceleration if available
    use_torch = False
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            use_torch = True
            device = "cuda"
            ref_emb_t = torch.from_numpy(ref_emb).to(device)
    except Exception:
        use_torch = False

    def evaluate_candidates(
        candidates: pd.DataFrame, incompat_mask_ref: np.ndarray, label: str
    ) -> Set[int]:
        if candidates.empty:
            return set()

        q_emb = np.array(emb_mmap[candidates["emb_idx"].to_numpy()], dtype=np.float32)
        q_norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb = q_emb / np.maximum(q_norms, 1e-8)

        q_h3 = candidates["h3_r3"].to_numpy()
        q_lat = candidates["Latitude"].to_numpy()
        q_lon = candidates["Longitude"].to_numpy()
        q_orig_idx = candidates["orig_row_idx"].to_numpy()

        flagged_indices: Set[int] = set()
        batch_size = 2048

        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            if use_torch:
                import torch

                q_batch_t = torch.from_numpy(q_emb[start:end]).to(device)
                sims_batch = (q_batch_t @ ref_emb_t.T).cpu().numpy()
            else:
                sims_batch = q_emb[start:end] @ ref_emb.T

            for i in range(end - start):
                idx = start + i
                # Leave-One-Cell-Out: exclude reference vectors in the same H3_r3 cell or within 3 deg
                spatial_far = (np.abs(ref_lat - q_lat[idx]) > 3.0) | (
                    np.abs(ref_lon - q_lon[idx]) > 3.0
                )
                valid_ref = (ref_h3_r3 != q_h3[idx]) & spatial_far
                if np.sum(valid_ref) < effective_k:
                    valid_ref = ref_h3_r3 != q_h3[idx]
                if np.sum(valid_ref) < effective_k:
                    valid_ref = np.ones(len(ref_df), dtype=bool)

                row_sims = np.where(valid_ref, sims_batch[i], -2.0)
                top_k_idx = np.argpartition(row_sims, -effective_k)[-effective_k:]
                incompat_ratio = float(np.mean(incompat_mask_ref[top_k_idx]))
                if incompat_ratio >= incompat_threshold:
                    flagged_indices.add(int(q_orig_idx[idx]))

        print(
            f" -> [{label}] Evaluated {len(candidates):,} candidate(s) -> Flagged {len(flagged_indices):,} visual-climate anomaly image(s)."
        )
        return flagged_indices

    flagged_all: Set[int] = set()

    # 1. Polar / Antarctica Candidates (EF, ET, or Antarctica)
    has_cont = "continent" in df_work.columns
    is_antarctica = (
        df_work["continent"].astype(str).str.lower() == "antarctica"
        if has_cont
        else pd.Series(False, index=df_work.index)
    )
    polar_mask = df_work["in_target_scope"] & (
        is_antarctica | df_work["koppen_clean"].isin(["EF", "ET"])
    )
    polar_incompat_ref = np.array(
        [
            (k[:1] in ("A", "B"))
            or (
                k
                in (
                    "Cfa",
                    "Cfb",
                    "Csa",
                    "Csb",
                    "Cwa",
                    "Cwb",
                    "Dfa",
                    "Dfb",
                )
            )
            for k in ref_koppen
        ],
        dtype=bool,
    )
    if np.any(polar_mask) and np.any(polar_incompat_ref):
        flagged_all.update(
            evaluate_candidates(
                df_work[polar_mask], polar_incompat_ref, "Polar / Antarctica (EF/ET)"
            )
        )

    # 2. Remote Arid Desert Candidates (BWh, BWk in non-metropolitan H3_r4 cells)
    r4_counts = df_work.groupby("h3_r4").size()
    r4_plats = df_work.groupby("h3_r4")["plat_lower"].nunique()
    dense_urban_desert_r4 = set(r4_counts[(r4_counts >= 500) & (r4_plats >= 3)].index)
    desert_mask = (
        df_work["in_target_scope"]
        & df_work["koppen_clean"].isin(["BWh", "BWk"])
        & (~df_work["h3_r4"].isin(dense_urban_desert_r4))
        & (~polar_mask)
    )
    desert_incompat_ref = np.array(
        [
            k
            in (
                "Af",
                "Am",
                "Cfa",
                "Cfb",
                "Cfc",
                "Dfa",
                "Dfb",
                "Dfc",
                "Dfd",
                "EF",
                "ET",
            )
            for k in ref_koppen
        ],
        dtype=bool,
    )
    if np.any(desert_mask) and np.any(desert_incompat_ref):
        flagged_all.update(
            evaluate_candidates(
                df_work[desert_mask],
                desert_incompat_ref,
                "Remote Arid Deserts (BWh/BWk)",
            )
        )

    print(
        f" -> Completed Visual k-NN Biome Check in {time.time() - t0:.2f}s (Total flagged: {len(flagged_all):,})."
    )
    return flagged_all


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Purge locked-latitude coordinate anomalies and remote-biome GPS glitches from the database."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="full_pipeline_output/geo_space_deduplicated.parquet",
        help="Path to the Parquet dataset.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="full_pipeline_output/geo_space_deduplicated.csv",
        help="Path to the CSV metadata file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the cleaned Parquet dataset. Defaults to input path.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to write the cleaned CSV metadata file. Defaults to input CSV path.",
    )
    parser.add_argument(
        "--platform",
        nargs="*",
        default=None,
        help=(
            "Restrict anomaly detection and cleanup to specific platform(s) "
            "(accepts space-separated list or comma-separated string, e.g. 'flickr mapillary kartaview')."
        ),
    )
    parser.add_argument(
        "--continent",
        type=str,
        default=None,
        help="Restrict anomaly detection and cleanup to this specific continent (e.g. 'africa').",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional vision backbone name to locate matching companion embeddings.",
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        default="cls",
        help="Representation type for companion embeddings (default: 'cls').",
    )
    parser.add_argument(
        "--embeddings_npy",
        type=str,
        default=None,
        help="Optional explicit path to companion embeddings .npy file.",
    )
    parser.add_argument(
        "--embeddings_keys",
        type=str,
        default=None,
        help="Optional explicit path to companion embeddings .keys.parquet file.",
    )
    parser.add_argument(
        "--disable_biome_knn",
        action="store_true",
        help="Disable Leave-One-Cell-Out visual k-NN climate consistency check.",
    )
    parser.add_argument(
        "--incompat_threshold",
        type=float,
        default=0.84,
        help="Fraction of incompatible visual neighbors required to flag a remote-biome anomaly (default: 0.84).",
    )
    args = parser.parse_args()

    parquet_path = args.input
    csv_path = args.csv
    output_parquet_path = args.output if args.output else parquet_path
    output_csv_path = args.output_csv if args.output_csv else csv_path

    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}")
        return

    target_platforms = parse_platform_arg(args.platform)

    print(
        "================================================================================"
    )
    print("🧹 GEOSPATIAL COORDINATE & REMOTE BIOME ANOMALY CLEANUP (STREAMING)")
    print(
        "================================================================================"
    )

    t0 = time.time()
    print("Opening Parquet file in streaming mode...")
    pf = pq.ParquetFile(parquet_path)
    num_row_groups = pf.num_row_groups
    print(f" -> Found {num_row_groups} row groups.")

    schema = pf.schema_arrow
    has_continent = "continent" in schema.names or "Continent" in schema.names
    continent_col = (
        "continent"
        if "continent" in schema.names
        else ("Continent" if "Continent" in schema.names else None)
    )
    has_photo_id = "Photo_ID" in schema.names or "photo_id" in schema.names
    photo_id_col = (
        "Photo_ID"
        if "Photo_ID" in schema.names
        else ("photo_id" if "photo_id" in schema.names else None)
    )
    has_koppen = "Koppen_Code" in schema.names
    has_h3 = "H3_Cell" in schema.names
    has_photo_key = "photo_key" in schema.names

    # 1. Read coordinates and group/biome columns only
    load_cols = ["Latitude", "Longitude", "Platform"]
    if has_photo_id and photo_id_col:
        load_cols.append(photo_id_col)
    if has_continent and continent_col:
        load_cols.append(continent_col)
    if has_koppen:
        load_cols.append("Koppen_Code")
    if has_h3:
        load_cols.append("H3_Cell")
    if has_photo_key:
        load_cols.append("photo_key")

    col_chunks: Dict[str, List[np.ndarray]] = {c: [] for c in load_cols}
    for rg in range(num_row_groups):
        tbl_rg = pf.read_row_group(rg, columns=load_cols)
        for c in load_cols:
            if c in ("Latitude", "Longitude"):
                col_chunks[c].append(tbl_rg[c].to_numpy())
            else:
                col_chunks[c].append(tbl_rg[c].to_numpy().astype(str))

    meta_dict = {c: np.concatenate(col_chunks[c]) for c in load_cols}
    if has_photo_id and photo_id_col and photo_id_col != "Photo_ID":
        meta_dict["Photo_ID"] = meta_dict.pop(photo_id_col)

    df_meta = pd.DataFrame(meta_dict)
    df_meta["Latitude"] = pd.to_numeric(df_meta["Latitude"], errors="coerce")
    df_meta["Longitude"] = pd.to_numeric(df_meta["Longitude"], errors="coerce")
    df_meta["Platform"] = df_meta["Platform"].fillna("").astype(str)
    if "Photo_ID" in df_meta.columns:
        df_meta["Photo_ID"] = df_meta["Photo_ID"].fillna("").astype(str)
    if has_continent and continent_col:
        df_meta[continent_col] = df_meta[continent_col].fillna("").astype(str)

    print(
        f" -> Loaded coordinates metadata for {len(df_meta):,} records in {time.time() - t0:.2f}s."
    )

    # Filter target platform/continent before analyzing anomalies if requested
    scope_mask = np.ones(len(df_meta), dtype=bool)
    if target_platforms:
        print(
            f"Filtering detection scope to platform(s): {', '.join(sorted(target_platforms))}"
        )
        scope_mask &= df_meta["Platform"].str.lower().isin(target_platforms).to_numpy()
    if args.continent:
        if has_continent and continent_col:
            print(f"Filtering detection scope to continent: {args.continent}")
            scope_mask &= (
                df_meta[continent_col].str.lower() == args.continent.lower()
            ).to_numpy()
        else:
            print(
                f"Warning: --continent '{args.continent}' specified but no continent column was found in schema. Skipping."
            )

    df_meta_for_stats = df_meta[scope_mask].copy()

    # =========================================================================
    # Detector 1a: Locked-Latitude Coordinate Parallels
    # =========================================================================
    df_meta_for_stats["lat_round"] = df_meta_for_stats["Latitude"].round(5)

    groupby_cols = ["Platform", "lat_round"]
    if has_continent and continent_col:
        groupby_cols.insert(1, continent_col)

    if not df_meta_for_stats.empty:
        stats = (
            df_meta_for_stats.groupby(groupby_cols)
            .agg(
                count=("Longitude", "count"),
                min_lon=("Longitude", "min"),
                max_lon=("Longitude", "max"),
            )
            .reset_index()
        )
        stats["span"] = stats["max_lon"] - stats["min_lon"]
        anomalies = stats[(stats["count"] > 10) & (stats["span"] > 1.0)]
    else:
        anomalies = pd.DataFrame()

    if not anomalies.empty:
        print(f"\n🚨 Found {len(anomalies)} locked-latitude coordinate anomaly lines:")
        print(
            "================================================================================"
        )
        print(anomalies.to_string(index=False))
        print(
            "================================================================================"
        )
        if has_continent and continent_col:
            flagged_keys = set(
                zip(
                    anomalies["Platform"],
                    anomalies[continent_col],
                    anomalies["lat_round"],
                )
            )
        else:
            flagged_keys = set(zip(anomalies["Platform"], anomalies["lat_round"]))
    else:
        print("✅ No locked-latitude coordinate parallel lines found in scope.")
        flagged_keys = set()

    # =========================================================================
    # Detector 1b: Null Island & Diagonal Coordinate Glitches (Lat == Lon)
    # =========================================================================
    lats_all = df_meta["Latitude"].to_numpy()
    lons_all = df_meta["Longitude"].to_numpy()
    null_island_mask = (
        scope_mask & (np.abs(lats_all) < 0.05) & (np.abs(lons_all) < 0.05)
    )
    diagonal_lock_mask = (
        scope_mask & (np.abs(lats_all - lons_all) < 1e-5) & (np.abs(lats_all) > 1.0)
    )
    coord_glitch_indices = set(
        np.where(null_island_mask | diagonal_lock_mask)[0].tolist()
    )
    if coord_glitch_indices:
        print(
            f"🚨 Found {len(coord_glitch_indices):,} Null-Island / Diagonal-Lock (Lat==Lon) coordinate glitches."
        )

    # =========================================================================
    # Detector 2: Leave-One-Cell-Out Visual k-NN Biome Consistency Check
    # =========================================================================
    biome_flagged_indices: Set[int] = set()
    if not args.disable_biome_knn and has_koppen:
        companion_paths = resolve_companion_embeddings(
            parquet_path=parquet_path,
            model_name=args.model_name,
            representation_type=args.representation_type,
            embeddings_npy=args.embeddings_npy,
            embeddings_keys=args.embeddings_keys,
        )
        if companion_paths:
            npy_p, keys_p = companion_paths
            biome_flagged_indices = detect_remote_biome_knn_anomalies(
                df_meta=df_meta,
                target_scope_mask=scope_mask,
                target_platforms=target_platforms,
                npy_path=npy_p,
                keys_path=keys_p,
                incompat_threshold=args.incompat_threshold,
            )
        else:
            print(
                " -> Notice: No companion embeddings (.npy) found for input dataset; skipping visual k-NN biome check."
            )

    flagged_row_indices: Set[int] = coord_glitch_indices.union(biome_flagged_indices)

    if not flagged_keys and not flagged_row_indices:
        print("\n✅ No coordinate or biome anomalies found in the dataset scope.")
        if parquet_path != output_parquet_path:
            print(f"Copying clean Parquet database to output: {output_parquet_path}")
            shutil.copy2(parquet_path, output_parquet_path)
        if csv_path != output_csv_path and os.path.exists(csv_path):
            print(f"Copying clean CSV metadata to output: {output_csv_path}")
            shutil.copy2(csv_path, output_csv_path)
        return

    # Collect flagged Photo_IDs for CSV synchronization and Mapillary sequence expansion
    flagged_photo_ids_by_plat: Set[Tuple[str, str]] = set()
    if "Photo_ID" in df_meta.columns and flagged_row_indices:
        for r_idx in flagged_row_indices:
            p_str = str(df_meta.at[r_idx, "Platform"]).lower()
            pid_str = str(df_meta.at[r_idx, "Photo_ID"]).strip().removesuffix(".0")
            if pid_str:
                flagged_photo_ids_by_plat.add((p_str, pid_str))

    # =========================================================================
    # Mapillary Sequence Expansion (Across Detector 1 & Detector 2)
    # =========================================================================
    violating_mapillary_image_ids: Set[str] = set()
    if has_photo_id and "Photo_ID" in df_meta.columns:
        m_flagged_pids: Set[str] = set()

        if flagged_keys:
            m_sub = df_meta_for_stats[
                df_meta_for_stats["Platform"].str.lower() == "mapillary"
            ]
            if not m_sub.empty:
                if has_continent and continent_col:
                    m_keys = list(
                        zip(
                            m_sub["Platform"],
                            m_sub[continent_col],
                            m_sub["lat_round"],
                        )
                    )
                else:
                    m_keys = list(zip(m_sub["Platform"], m_sub["lat_round"]))
                anom_mask = [k in flagged_keys for k in m_keys]
                for p in m_sub.loc[anom_mask, "Photo_ID"].dropna().astype(str):
                    clean_p = p.strip().removesuffix(".0")
                    if clean_p:
                        m_flagged_pids.add(clean_p)

        for p_str, pid_str in flagged_photo_ids_by_plat:
            if p_str == "mapillary" and pid_str:
                m_flagged_pids.add(pid_str)

        if m_flagged_pids:
            token = get_mapillary_token()
            if token:
                print(
                    f"\n🔍 Expanding {len(m_flagged_pids):,} flagged Mapillary anomaly photo(s) to full sequences..."
                )
                seq_map = fetch_mapillary_sequences(sorted(m_flagged_pids), token)
                unique_seqs = set(seq_map.values())
                print(
                    f" -> Flagged images belong to {len(unique_seqs)} distinct Mapillary sequence(s)."
                )
                if unique_seqs:
                    print(
                        f" -> Retrieving all image IDs for {len(unique_seqs)} sequence(s)..."
                    )
                    violating_mapillary_image_ids = fetch_sequence_all_image_ids(
                        unique_seqs, token
                    )
                    print(
                        f" -> Expanded to {len(violating_mapillary_image_ids):,} total Mapillary sequence image(s) to purge."
                    )
            else:
                print(
                    "\nNotice: MAPILLARY_TOKEN not found in environment or .env. Skipping Mapillary sequence expansion."
                )

    # =========================================================================
    # 3. Stream write filtered row groups to a temporary Parquet file
    # =========================================================================
    temp_parquet_path = output_parquet_path + ".tmp"
    schema = pf.schema_arrow

    print("\nStreaming row groups and filtering in C++ memory...")
    t_stream = time.time()
    removed_count = 0
    global_row_offset = 0

    with get_parquet_writer(temp_parquet_path, schema) as writer:
        for rg in range(num_row_groups):
            tbl_rg = pf.read_row_group(rg)
            rg_len = len(tbl_rg)
            rg_lat = tbl_rg["Latitude"].to_numpy()
            rg_lat_round = np.round(rg_lat, 5)
            rg_plat = tbl_rg["Platform"].to_numpy().astype(str)

            if has_photo_id and photo_id_col:
                rg_pids = tbl_rg[photo_id_col].to_numpy().astype(str)
                rg_pids_clean = [p.strip().removesuffix(".0") for p in rg_pids]
            else:
                rg_pids_clean = [""] * rg_len

            if has_continent and continent_col:
                rg_cont = tbl_rg[continent_col].to_numpy().astype(str)
                keys = list(zip(rg_plat, rg_cont, rg_lat_round))
            else:
                keys = list(zip(rg_plat, rg_lat_round))

            keep_mask = []
            for local_i, (k, plat, pid) in enumerate(zip(keys, rg_plat, rg_pids_clean)):
                global_idx = global_row_offset + local_i
                plat_l = plat.lower()
                if k in flagged_keys:
                    keep_mask.append(False)
                elif global_idx in flagged_row_indices:
                    keep_mask.append(False)
                elif plat_l == "mapillary" and pid in violating_mapillary_image_ids:
                    keep_mask.append(False)
                elif (plat_l, pid) in flagged_photo_ids_by_plat:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)

            global_row_offset += rg_len
            filtered_tbl = tbl_rg.filter(pa.array(keep_mask))
            writer.write_table(filtered_tbl)

            removed_count += rg_len - len(filtered_tbl)

    os.replace(temp_parquet_path, output_parquet_path)
    print(f" -> Saved cleaned Parquet in {time.time() - t_stream:.2f}s.")

    # =========================================================================
    # 4. Clean matching CSV if it exists
    # =========================================================================
    if os.path.exists(csv_path):
        print("Cleaning CSV file...")
        t_csv = time.time()
        temp_csv_path = output_csv_path + ".tmp"
        first = True
        csv_offset = 0
        for chunk in pd.read_csv(
            csv_path, chunksize=100000, dtype={"Platform": str, "Photo_ID": str}
        ):
            chunk_len = len(chunk)
            chunk["Latitude"] = pd.to_numeric(chunk["Latitude"], errors="coerce")
            chunk["Longitude"] = pd.to_numeric(chunk["Longitude"], errors="coerce")
            chunk["lat_round"] = chunk["Latitude"].round(5)
            chunk["Platform"] = chunk["Platform"].fillna("").astype(str)

            chunk_pids_clean = (
                chunk["Photo_ID"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.removesuffix(".0")
                if "Photo_ID" in chunk.columns
                else [""] * chunk_len
            )

            if has_continent and continent_col and continent_col in chunk.columns:
                chunk[continent_col] = chunk[continent_col].fillna("").astype(str)
                keys = list(
                    zip(chunk["Platform"], chunk[continent_col], chunk["lat_round"])
                )
            else:
                keys = list(zip(chunk["Platform"], chunk["lat_round"]))

            keep_mask = []
            for local_i, (k, plat, pid) in enumerate(
                zip(keys, chunk["Platform"], chunk_pids_clean)
            ):
                global_idx = csv_offset + local_i
                plat_l = plat.lower()
                if k in flagged_keys:
                    keep_mask.append(False)
                elif (plat_l, pid) in flagged_photo_ids_by_plat:
                    keep_mask.append(False)
                elif not pid and global_idx in flagged_row_indices:
                    keep_mask.append(False)
                elif plat_l == "mapillary" and pid in violating_mapillary_image_ids:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)

            csv_offset += chunk_len
            cleaned_chunk = chunk[keep_mask].drop(columns=["lat_round"])

            cleaned_chunk.to_csv(
                temp_csv_path,
                mode="a" if not first else "w",
                index=False,
                header=first,
            )
            first = False

        os.replace(temp_csv_path, output_csv_path)
        print(f" -> Saved CSV in {time.time() - t_csv:.2f}s.")

    print(
        f"\n🎉 Streaming cleanup completed successfully! Purged {removed_count:,} records."
    )


if __name__ == "__main__":
    main()
