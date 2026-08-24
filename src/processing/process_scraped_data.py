import argparse
import glob
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel

from src.models.vision_model_inference import extract_model_embeddings
from src.utils.io import (
    download_image,
    get_core_base_name,
    get_parquet_writer,
    load_dataframe,
    load_embeddings,
    save_dataframe,
)
from src.utils.licensing import FLICKR_LICENSE_MAP

# TIPSv2 specific transform
tips_transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]
)


def clean_photo_id(val):
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    val_str = val_str.removesuffix(".0")
    return val_str


def standardize_timestamps_vectorized(ts_raw):
    """Standardizes a Pandas Series of timestamps (numeric or string) to ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)."""
    # Initialize output Series with None as default
    standardized = pd.Series([None] * len(ts_raw), index=ts_raw.index, dtype=object)

    # Identify Unix epoch numeric timestamps vs. string representations
    numeric_ts = pd.to_numeric(ts_raw, errors="coerce")
    is_numeric = numeric_ts.notna() & (numeric_ts > 1e8)

    # Parse Unix numeric timestamps
    if is_numeric.any():
        is_ms_mask = is_numeric & (numeric_ts > 5e10)
        is_s_mask = is_numeric & ~is_ms_mask

        if is_ms_mask.any():
            parsed_ms = pd.to_datetime(
                numeric_ts[is_ms_mask], unit="ms", utc=True, errors="coerce"
            )
            try:
                valid_ms = (parsed_ms.dt.year >= 1) & (parsed_ms.dt.year <= 9999)
            except Exception:
                valid_ms = pd.Series(True, index=parsed_ms.index)

            formatted_ms = pd.Series(
                [None] * len(parsed_ms), index=parsed_ms.index, dtype=object
            )
            if valid_ms.any():
                formatted_ms[valid_ms] = parsed_ms[valid_ms].dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            standardized.loc[is_ms_mask] = formatted_ms

        if is_s_mask.any():
            parsed_s = pd.to_datetime(
                numeric_ts[is_s_mask], unit="s", utc=True, errors="coerce"
            )
            try:
                valid_s = (parsed_s.dt.year >= 1) & (parsed_s.dt.year <= 9999)
            except Exception:
                valid_s = pd.Series(True, index=parsed_s.index)

            formatted_s = pd.Series(
                [None] * len(parsed_s), index=parsed_s.index, dtype=object
            )
            if valid_s.any():
                formatted_s[valid_s] = parsed_s[valid_s].dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            standardized.loc[is_s_mask] = formatted_s

    # Parse string representations
    is_string = ts_raw.notna() & ~is_numeric
    if is_string.any():
        str_vals = ts_raw[is_string].astype(str).str.strip()
        has_colon_date = str_vals.str.match(r"^\d{4}:\d{2}:\d{2}")
        if has_colon_date.any():
            str_vals.loc[has_colon_date] = str_vals[has_colon_date].str.replace(
                ":", "-", n=2
            )

        parsed_str = pd.to_datetime(str_vals, errors="coerce", utc=True, format="mixed")
        try:
            valid_str = (parsed_str.dt.year >= 1) & (parsed_str.dt.year <= 9999)
        except Exception:
            valid_str = pd.Series(True, index=parsed_str.index)

        formatted_str = pd.Series(
            [None] * len(parsed_str), index=parsed_str.index, dtype=object
        )
        if valid_str.any():
            formatted_str[valid_str] = parsed_str[valid_str].dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        standardized.loc[is_string] = formatted_str

    # Fallback to original strings if parsing failed but was not null/empty
    ts_series = ts_raw.astype(str).str.strip()
    is_parsed = standardized.notna()
    invalid_mask = ~is_parsed & ts_raw.notna() & (ts_raw != "")
    standardized[invalid_mask] = ts_series[invalid_mask]

    return standardized


def get_tips_embeddings(
    images, model, device, batch_size=32, representation_type="cls"
):
    """Computes TIPSv2 embeddings for a list of PIL images in batches using a single forward pass."""
    if not images:
        return None

    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_tensors = torch.stack([tips_transform(img) for img in batch]).to(
                device
            )
            features = extract_model_embeddings(
                model, batch_tensors, representation_type=representation_type
            )
            all_features.append(features)

    return np.concatenate(all_features, axis=0)


def process_cell(
    cell_id,
    metadata_list,
    model,
    device,
    sim_threshold,
    executor,
    text_features=None,
    existing_items=None,
    cell_chunk_size=128,
    tips_batch_size=32,
    macro_idx=-1,
    sky_idx=-1,
    offline_dirs=None,
    representation_type="cls",
    mapillary_token=None,
):
    """Filters indoor images (Flickr only) and deduplicates images within an H3 cell in chunks."""
    results = existing_items.copy() if existing_items else []
    processed_embeddings = [item["embedding"] for item in results]
    download_fn = partial(
        download_image, offline_dirs=offline_dirs, mapillary_token=mapillary_token
    )

    # Process new images in chunks to limit peak memory usage
    for chunk_start in range(0, len(metadata_list), cell_chunk_size):
        chunk_metadata = metadata_list[chunk_start : chunk_start + cell_chunk_size]

        # Separate items that already have pre-computed embeddings from those that need them
        to_compute_indices = []
        precomputed_embeddings = {}  # maps chunk_metadata index -> embedding vector
        for idx, m in enumerate(chunk_metadata):
            if m.get("embedding") is not None:
                precomputed_embeddings[idx] = m["embedding"]
            else:
                to_compute_indices.append(idx)

        # Download and compute embeddings only for those that need it
        computed_embeddings = None
        valid_dl_indices = []
        if to_compute_indices:
            urls = [chunk_metadata[idx]["Image_URL"] for idx in to_compute_indices]
            pids = [chunk_metadata[idx]["Photo_ID"] for idx in to_compute_indices]
            plats = [chunk_metadata[idx].get("Platform") for idx in to_compute_indices]

            # Download images in parallel for this chunk
            imgs = list(executor.map(download_fn, urls, pids, plats))

            valid_dl_indices = [i for i, img in enumerate(imgs) if img is not None]
            if valid_dl_indices:
                valid_imgs = [imgs[i] for i in valid_dl_indices]

                # Compute embeddings for this chunk using configured tips_batch_size
                computed_embeddings = get_tips_embeddings(
                    valid_imgs,
                    model,
                    device,
                    batch_size=tips_batch_size,
                    representation_type=representation_type,
                )

                # Explicitly close PIL images immediately to free RAM
                for img in valid_imgs:
                    try:
                        img.close()
                    except Exception:
                        pass

        # Build unified lists of valid indices and embeddings for this chunk
        valid_indices = []
        all_embeddings_list = []

        # 1. Add precomputed ones
        for idx, emb in precomputed_embeddings.items():
            valid_indices.append(idx)
            all_embeddings_list.append(emb)

        # 2. Add newly computed ones
        if computed_embeddings is not None:
            for i, dl_idx in enumerate(valid_dl_indices):
                global_idx = to_compute_indices[dl_idx]
                valid_indices.append(global_idx)
                all_embeddings_list.append(computed_embeddings[i])

        if not all_embeddings_list:
            continue

        all_embeddings = np.vstack(all_embeddings_list).astype(np.float32)

        # Matrix multiply for indoor/outdoor zero-shot classification
        if text_features is not None:
            emb_norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            emb_norms[emb_norms == 0] = 1.0
            norm_embeddings = all_embeddings / emb_norms

            text_norms = np.linalg.norm(text_features, axis=1, keepdims=True)
            text_norms[text_norms == 0] = 1.0
            norm_text = text_features / text_norms

            all_io_sims = np.dot(norm_embeddings, norm_text.T)

        for i, idx in enumerate(valid_indices):
            metadata = chunk_metadata[idx]
            embedding = all_embeddings[i]

            # Zero-shot filtering (Indoor, Macro, and Sky/Flying)
            if text_features is not None:
                sims = all_io_sims[i]
                best_class = np.argmax(sims)

                # 1. Flickr & Wikimedia/GoogleLandmarks Indoor Filter (always Class 0)
                if (
                    str(metadata["Platform"]).lower()
                    in ["flickr", "wikimedia", "googlelandmarks"]
                    and best_class == 0
                ):
                    continue

                # 2. iNaturalist-only Macro/Close-up Filter
                if macro_idx != -1 and best_class == macro_idx:
                    if str(metadata["Platform"]).lower() == "inaturalist":
                        continue

                # 3. iNaturalist-only Sky/Flying Filter
                if sky_idx != -1 and best_class == sky_idx:
                    if str(metadata["Platform"]).lower() == "inaturalist":
                        continue

            # Deduplication check
            is_duplicate = False
            if processed_embeddings:
                curr_norm = embedding / (np.linalg.norm(embedding) or 1.0)
                kept_embs = np.array(processed_embeddings)
                kept_norms = np.linalg.norm(kept_embs, axis=1, keepdims=True)
                kept_norms[kept_norms == 0] = 1.0
                norm_kept = kept_embs / kept_norms

                sims = np.dot(norm_kept, curr_norm)
                if np.any(sims > sim_threshold):
                    is_duplicate = True

            if not is_duplicate:
                processed_embeddings.append(embedding)
                item = metadata.copy()
                item["embedding"] = embedding
                results.append(item)

    return results


def stream_update_parquet(
    input_path,
    output_path,
    df_new,
    active_cells,
    representation_type="cls",
    precision="float32",
):
    """
    Standardized stream updater that copies/appends data chunk-by-chunk using PyArrow.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    pf = pq.ParquetFile(input_path)

    # Build the output schema
    new_fields = []
    for f in pf.schema_arrow:
        if f.name in ("embedding", "embedding_idx"):
            continue
        new_fields.append(f)

    # Ensure photo_key is in the output schema
    if "photo_key" not in [f.name for f in new_fields]:
        new_fields.append(pa.field("photo_key", pa.string()))

    # Upgrade schema to include License if it is present in the new df
    # but not in the existing parquet database
    has_license_in_new = "License" in df_new.columns if df_new is not None else False
    has_license_in_existing = "License" in pf.schema_arrow.names
    if has_license_in_new and not has_license_in_existing:
        new_fields.append(pa.field("License", pa.string()))

    schema = pa.schema(new_fields)

    # Load existing embeddings (aligned with the input Parquet's rows)
    full_embeddings = load_embeddings(
        input_path, representation_type=representation_type
    )

    # Load the keys of the input database to perform a key-based index lookup
    print("Reading master key index from input database...")
    if "photo_key" in pf.schema_arrow.names:
        input_keys = pf.read(columns=["photo_key"])["photo_key"].to_pandas().values
    else:
        df_temp = pf.read(columns=["Platform", "Photo_ID"]).to_pandas()
        input_keys = (
            df_temp["Platform"].astype(str) + "_" + df_temp["Photo_ID"].astype(str)
        ).values

    input_keys_index = pd.Index(input_keys)

    tmp_output = f"{output_path}.tmp_stream"
    try:
        active_arr = pa.array(list(active_cells))
        inactive_embs_list = []
        inactive_keys_list = []

        with get_parquet_writer(tmp_output, schema) as writer:
            # 1. Stream copy inactive rows from original parquet
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)

                # Filter out active cells
                h3_col = table["H3_Cell"]
                mask = pc.invert(pc.is_in(h3_col, value_set=active_arr))
                filtered_table = table.filter(mask)

                # Drop legacy embedding_idx if present
                if "embedding_idx" in filtered_table.column_names:
                    idx_pos = filtered_table.column_names.index("embedding_idx")
                    filtered_table = filtered_table.remove_column(idx_pos)

                if len(filtered_table) > 0:
                    # Append a null column for License if we upgraded the schema
                    if (
                        has_license_in_new
                        and "License" not in filtered_table.column_names
                    ):
                        null_col = pa.array(
                            [None] * len(filtered_table), type=pa.string()
                        )
                        filtered_table = filtered_table.append_column(
                            "License", null_col
                        )

                    # Standardize Platform to lowercase
                    if "Platform" in filtered_table.column_names:
                        plat_pos = filtered_table.column_names.index("Platform")
                        plat_col = (
                            filtered_table.column(plat_pos)
                            .to_pandas()
                            .astype(str)
                            .str.lower()
                        )
                        filtered_table = filtered_table.set_column(
                            plat_pos,
                            pa.field("Platform", pa.string()),
                            pa.array(plat_col),
                        )

                    # Ensure photo_key is populated and standardized to lowercase
                    df_temp_rg = filtered_table.select(
                        ["Platform", "Photo_ID"]
                    ).to_pandas()
                    keys_arr = pa.array(
                        df_temp_rg["Platform"].astype(str).str.lower()
                        + "_"
                        + df_temp_rg["Photo_ID"].astype(str)
                    )
                    if "photo_key" in filtered_table.column_names:
                        pk_pos = filtered_table.column_names.index("photo_key")
                        filtered_table = filtered_table.set_column(
                            pk_pos, pa.field("photo_key", pa.string()), keys_arr
                        )
                    else:
                        filtered_table = filtered_table.append_column(
                            "photo_key", keys_arr
                        )

                    chunk_keys = keys_arr.to_numpy()
                    inactive_keys_list.append(chunk_keys)

                    # Retrieve matching embeddings using stable keys Indexer
                    if input_keys_index.is_unique:
                        indices = input_keys_index.get_indexer(chunk_keys)
                    else:
                        pos_series = pd.Series(
                            np.arange(len(input_keys_index)), index=input_keys_index
                        )
                        pos_series = pos_series[
                            ~pos_series.index.duplicated(keep="first")
                        ]
                        indices = pos_series.reindex(chunk_keys, fill_value=-1).values
                    embs = full_embeddings[indices]
                    inactive_embs_list.append(embs)

                    writer.write_table(filtered_table)

            # 2. Write the new/updated active rows
            new_embs = np.empty((0, 768), dtype=np.float32)
            new_keys = np.empty(0, dtype=object)
            if df_new is not None and not df_new.empty:
                df_new_aligned = df_new.copy()

                # Extract new embeddings
                new_embs = np.vstack(df_new_aligned["embedding"].values).astype(
                    np.float32
                )

                # Generate stable photo_key
                df_new_aligned["photo_key"] = (
                    df_new_aligned["Platform"].astype(str).str.lower()
                    + "_"
                    + df_new_aligned["Photo_ID"].astype(str)
                )

                new_keys = df_new_aligned["photo_key"].values
                df_new_aligned = df_new_aligned.drop(
                    columns=["embedding", "embedding_idx"], errors="ignore"
                )

                for name in schema.names:
                    if name not in df_new_aligned.columns:
                        df_new_aligned[name] = None
                df_new_aligned = df_new_aligned[schema.names]

                new_table = pa.Table.from_pandas(
                    df_new_aligned, schema=schema, preserve_index=False
                )
                writer.write_table(new_table)

        # 3. Concatenate and save all embeddings to output .npy
        if inactive_embs_list:
            all_inactive_embs = np.concatenate(inactive_embs_list, axis=0)
            if len(new_embs) > 0:
                final_embs = np.concatenate([all_inactive_embs, new_embs], axis=0)
            else:
                final_embs = all_inactive_embs
        else:
            final_embs = new_embs

        # 4. Concatenate and save all keys to output .keys.parquet
        if inactive_keys_list:
            all_inactive_keys = np.concatenate(inactive_keys_list, axis=0)
            if len(new_keys) > 0:
                final_keys = np.concatenate([all_inactive_keys, new_keys], axis=0)
            else:
                final_keys = all_inactive_keys
        else:
            final_keys = new_keys

        # Save companion .npy file for the output parquet path
        db_dir = os.path.dirname(os.path.abspath(output_path))
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        if "_clustered_k_" in base_name:
            base_name = base_name.split("_clustered_k_")[0]
        core_name = get_core_base_name(base_name)

        npy_path = os.path.join(
            db_dir, f"{core_name}_{representation_type}_embeddings.npy"
        )
        dtype = np.float16 if precision == "float16" else np.float32
        final_embs_cast = final_embs.astype(dtype)

        print(
            f" -> Saving merged companion embeddings matrix: {npy_path} (dtype={dtype.__name__})"
        )
        np.save(npy_path, final_embs_cast)

        keys_path = npy_path.replace(".npy", ".keys.parquet")
        print(f" -> Saving companion keys index: {keys_path}")
        pd.DataFrame({"photo_key": final_keys}).to_parquet(
            keys_path, compression="zstd"
        )

        if os.path.exists(tmp_output):
            os.replace(tmp_output, output_path)
    except Exception as e:
        if os.path.exists(tmp_output):
            try:
                os.remove(tmp_output)
            except Exception:
                pass
        raise e


def save_checkpoint(
    final_data,
    processed_cells,
    checkpoint_path,
    checkpoint_meta_path,
    resume_from=None,
    active_cells=None,
    representation_type="cls",
    precision="float32",
):
    """Saves the intermediate state to checkpoint files atomically."""
    tmp_path = f"{checkpoint_path}.tmp"
    tmp_meta_path = f"{checkpoint_meta_path}.tmp"
    try:
        # Convert final_data to DataFrame and save to tmp parquet
        if not final_data:
            df = pd.DataFrame(
                columns=[
                    "Photo_ID",
                    "Platform",
                    "Latitude",
                    "Longitude",
                    "Image_URL",
                    "H3_Cell",
                    "embedding",
                    "Captured_At",
                    "License",
                ]
            )
        else:
            df = pd.DataFrame(final_data)

        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

        if resume_from and os.path.exists(resume_from) and active_cells:
            stream_update_parquet(
                resume_from,
                tmp_path,
                df,
                active_cells,
                representation_type=representation_type,
                precision=precision,
            )
        else:
            save_dataframe(
                df,
                tmp_path,
                representation_type=representation_type,
                precision=precision,
            )

        # Save processed cells to tmp meta
        with open(tmp_meta_path, "wb") as f:
            pickle.dump(processed_cells, f)

        # Atomic rename
        if os.path.exists(tmp_path):
            os.replace(tmp_path, checkpoint_path)
        if os.path.exists(tmp_meta_path):
            os.replace(tmp_meta_path, checkpoint_meta_path)
        print(
            f"\nCheckpoint saved: {len(final_data)} images kept, {len(processed_cells)} cells processed."
        )
    except Exception as e:
        print(f"\nError saving checkpoint: {e}")


def load_and_preprocess_csv(f, offline_dirs=None, representation_type="cls"):
    """Loads a single CSV or Parquet file, normalizes column names, and converts Mapillary/KartaView URLs."""
    try:
        if f.endswith(".parquet"):
            df = load_dataframe(f)
            # Ensure metadata columns are string for downstream code compatibility
            for col in df.columns:
                if col not in ["embedding", "Latitude", "Longitude"]:
                    df[col] = df[col].astype(str)
            # Load matching companion embeddings if present
            try:
                embeddings = load_embeddings(f, representation_type=representation_type)
                df["embedding"] = list(embeddings)
                print(
                    f" -> Successfully loaded precomputed '{representation_type}' embeddings for: {f}"
                )
            except FileNotFoundError:
                print(
                    f" -> No precomputed '{representation_type}' embeddings found for: {f} (will compute from scratch)"
                )
        else:
            df = pd.read_csv(f, dtype=str)
        if df.empty:
            return None

        # Check if this is the iWildCam CSV
        is_iwildcam = "iwildcam" in f.lower() or (
            "Platform" in df.columns
            and df["Platform"].astype(str).str.lower().eq("iwildcam").any()
        )

        if "uuid" in df.columns and "source" in df.columns and "orig_id" in df.columns:
            df["Platform"] = df["source"]
            df["Latitude"] = df["lat"]
            df["Longitude"] = df["lon"]
            df["Photo_ID"] = df["orig_id"]
            df["Captured_At"] = (
                df["datetime_local"] if "datetime_local" in df.columns else None
            )

            urls = df["url"] if "url" in df.columns else [None] * len(df)
            df["Image_URL"] = [
                f"mapillary://{oid}"
                if str(src).lower() == "mapillary"
                else (f"kartaview://{oid}" if str(src).lower() == "kartaview" else url)
                for oid, src, url in zip(df["orig_id"], df["source"], urls)
            ]
        elif is_iwildcam:
            platform = "iWildCam"
            col_map = {
                "latitude": "Latitude",
                "longitude": "Longitude",
                "photo_id": "Photo_ID",
                "ID": "Photo_ID",
                "captured_at": "Captured_At",
                "Captured_At": "Captured_At",
            }
            df = df.rename(
                columns={k: v for k, v in col_map.items() if k in df.columns}
            )
            # Map Image_URL fallback safely
            if "Image_URL" not in df.columns:
                for fallback in ["Image_Location", "Image_URL"]:
                    if fallback in df.columns:
                        df = df.rename(columns={fallback: "Image_URL"})
                        break
            df["Platform"] = platform
        else:
            if "inaturalist" in f.lower():
                platform = "inaturalist"
            elif "flickr" in f.lower():
                platform = "flickr"
            elif "iwildcam" in f.lower():
                platform = "iwildcam"
            else:
                platform = os.path.splitext(os.path.basename(f))[0].lower()

            col_map = {
                "latitude": "Latitude",
                "longitude": "Longitude",
                "photo_id": "Photo_ID",
                "ID": "Photo_ID",
                "captured_at": "Captured_At",
                "Captured_At": "Captured_At",
                "Date_Observed": "Captured_At",
                "observed_on_string": "Captured_At",
                "license": "License",
                "License": "License",
            }
            df = df.rename(
                columns={k: v for k, v in col_map.items() if k in df.columns}
            )
            # Map Image_URL fallback safely
            if "Image_URL" not in df.columns:
                for fallback in ["image_url", "Image_Location", "Image_URL"]:
                    if fallback in df.columns:
                        df = df.rename(columns={fallback: "Image_URL"})
                        break
            if "Platform" not in df.columns:
                df["Platform"] = platform
            else:
                df["Platform"] = df["Platform"].astype(str).str.strip().str.lower()

        # Determine if the file is from an offline directory
        is_offline = False
        if offline_dirs:
            abs_f = os.path.abspath(f)
            for od in offline_dirs:
                if abs_f.startswith(os.path.abspath(od)):
                    is_offline = True
                    break

        # Resolve any local image paths (Image_Location or Image_URL) to absolute paths relative to the CSV's directory
        image_col = (
            "Image_URL"
            if "Image_URL" in df.columns
            else ("Image_Location" if "Image_Location" in df.columns else None)
        )
        if image_col:
            csv_dir = os.path.dirname(os.path.abspath(f))

            def resolve_offline_path(loc):
                if not isinstance(loc, str):
                    return loc
                if is_offline:
                    if os.path.isabs(loc):
                        return loc
                    # Try direct join
                    p = os.path.abspath(os.path.join(csv_dir, loc))
                    if os.path.exists(p):
                        return p
                    # Try with "train" subdir (backward compatibility for flat iWildCam)
                    p_train = os.path.abspath(
                        os.path.join(csv_dir, "train", os.path.basename(loc))
                    )
                    if os.path.exists(p_train):
                        return p_train
                    return p
                else:
                    return loc

            df["Image_URL"] = [resolve_offline_path(loc) for loc in df[image_col]]

        required_cols = [
            "Photo_ID",
            "Platform",
            "Latitude",
            "Longitude",
            "Image_URL",
            "Captured_At",
            "License",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # Populate standard platform licenses if missing
        platform_lower = df["Platform"].astype(str).str.lower()
        mapillary_mask = (platform_lower == "mapillary") & (
            df["License"].isna() | (df["License"] == "")
        )
        if mapillary_mask.any():
            df.loc[mapillary_mask, "License"] = "CC BY-SA 4.0"

        kartaview_mask = (platform_lower == "kartaview") & (
            df["License"].isna() | (df["License"] == "")
        )
        if kartaview_mask.any():
            df.loc[kartaview_mask, "License"] = "CC BY-SA 4.0"

        iwildcam_mask = (platform_lower == "iwildcam") & (
            df["License"].isna() | (df["License"] == "")
        )
        if iwildcam_mask.any():
            df.loc[iwildcam_mask, "License"] = "CDLA-Permissive-1.0"

        inat_mask = (
            platform_lower.str.contains("inaturalist") | (platform_lower == "inat")
        ) & (df["License"].isna() | (df["License"] == ""))
        if inat_mask.any():
            df.loc[inat_mask, "License"] = "CC BY-NC 4.0"

        # Map Flickr numeric indexes to human-readable strings
        flickr_mask = (
            (platform_lower == "flickr") & df["License"].notna() & (df["License"] != "")
        )
        if flickr_mask.any():
            # Clean keys to handle potential floats like '4.0'
            clean_keys = (
                df.loc[flickr_mask, "License"].astype(str).str.split(".").str[0]
            )
            mapped = clean_keys.map(FLICKR_LICENSE_MAP)
            df.loc[flickr_mask, "License"] = mapped.fillna(
                df.loc[flickr_mask, "License"]
            )

        df = df[required_cols].copy()
        df["License"] = (
            df["License"].astype(str).replace({"nan": None, "None": None, "<NA>": None})
        )
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        return df
    except Exception as e:
        print(f"Error reading {f}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate, Filter, and Deduplicate Geo-Scraped Data using TIPSv2."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="List of directories containing chunked CSVs.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="Directory to save output files and data.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="geo_embedding_space",
        help="Base name for output files.",
    )
    parser.add_argument("--h3_res", type=int, default=11, help="H3 resolution (~25m).")
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.95,
        help="TIPSv2 cosine similarity threshold.",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Disable Flickr indoor/outdoor filtering.",
    )
    parser.add_argument(
        "--filter_macro",
        action="store_true",
        help="Filter out macro/close-up photos of leaves, flowers, bark, and insects using zero-shot embeddings.",
    )
    parser.add_argument(
        "--filter_sky",
        action="store_true",
        help="Filter out photos of the sky, clouds, and flying objects (birds/insects in flight) for iNaturalist.",
    )
    parser.add_argument(
        "--limit_cells",
        type=int,
        default=0,
        help="Limit number of cells to process (for testing).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a previously generated .pkl or parquet file to resume from.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1800,
        help="Interval in seconds to save checkpoints (0 to disable).",
    )
    parser.add_argument(
        "--cell_chunk_size",
        type=int,
        default=128,
        help="Number of images within a cell to download/process in a chunk.",
    )
    parser.add_argument(
        "--tips_batch_size",
        type=int,
        default=32,
        help="Batch size for TIPSv2 embedding inference.",
    )
    parser.add_argument(
        "--offline_dataset_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Base directories containing offline dataset CSV indexes and images folders.",
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        default="cls",
        choices=["cls", "avg_patch", "cls_avg_patch"],
        help="Type of representation embedding to extract (cls, avg_patch, or cls_avg_patch).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Floating point precision format for stored embeddings (float32 or float16).",
    )
    parser.add_argument(
        "--mapillary_token",
        type=str,
        default=None,
        help="Mapillary API token for downloading images.",
    )
    args = parser.parse_args()

    # Try to load .env variables if not already set
    if not os.environ.get("MAPILLARY_TOKEN") and os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "MAPILLARY_TOKEN":
                            os.environ["MAPILLARY_TOKEN"] = (
                                v.strip().strip('"').strip("'")
                            )
                            break
        except Exception:
            pass

    if not args.mapillary_token:
        args.mapillary_token = os.environ.get("MAPILLARY_TOKEN", "")

    # 1. Gather all CSVs and Parquets
    csv_files = []

    def is_valid_input_file(filepath):
        basename = os.path.basename(filepath)
        if filepath.endswith(".keys.parquet"):
            return False
        if "_checkpoint.parquet" in basename:
            return False
        if basename == f"{args.output_name}.parquet":
            return False
        return True

    for d in args.dirs:
        files = glob.glob(os.path.join(d, "*.csv")) + glob.glob(
            os.path.join(d, "*.parquet")
        )
        csv_files.extend([f for f in files if is_valid_input_file(f)])

    if args.offline_dataset_dirs:
        for d in args.offline_dataset_dirs:
            files = glob.glob(os.path.join(d, "*.csv")) + glob.glob(
                os.path.join(d, "*.parquet")
            )
            offline_files = [f for f in files if is_valid_input_file(f)]
            print(
                f"Found {len(offline_files)} CSV/Parquet files in offline directory '{d}'."
            )
            csv_files.extend(offline_files)

    print(f"Found {len(csv_files)} total input files (CSVs/Parquets) to process.")

    # 2. Check for resume files
    df_existing = None
    seen_keys = set()
    active_cells = set()

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from existing data: {args.resume_from}")
        if args.resume_from.endswith(".pkl"):
            with open(args.resume_from, "rb") as f:
                existing_data = pickle.load(f)
            df_existing = pd.DataFrame(existing_data)
            del existing_data  # Free list from RAM
            existing_embeddings = np.vstack(df_existing["embedding"].values).astype(
                np.float32
            )

            seen_keys = set(
                zip(
                    df_existing["Platform"].astype(str).str.lower(),
                    df_existing["Photo_ID"].apply(clean_photo_id),
                )
            )

            # Retroactively clean existing URLs to virtual format if they are Mapillary/KartaView (vectorized)
            if not df_existing.empty and "Image_URL" in df_existing.columns:
                platforms = df_existing["Platform"].to_numpy()
                photo_ids = df_existing["Photo_ID"].to_numpy()
                image_urls = df_existing["Image_URL"].to_numpy()
                df_existing["Image_URL"] = [
                    f"mapillary://{pid}"
                    if str(plat).lower() == "mapillary"
                    else (
                        f"kartaview://{pid}"
                        if str(plat).lower() == "kartaview"
                        else url
                    )
                    for plat, pid, url in zip(platforms, photo_ids, image_urls)
                ]
        else:
            # Load minimal metadata columns only (uses ~50MB RAM even for millions of rows)
            print("Loading existing dataset metadata using PyArrow...")
            df_existing = load_dataframe(
                args.resume_from, columns=["Photo_ID", "Platform", "H3_Cell"]
            )
            seen_keys = set(
                zip(
                    df_existing["Platform"].astype(str).str.lower(),
                    df_existing["Photo_ID"].apply(clean_photo_id),
                )
            )

        # Retroactively clean existing URLs to virtual format if they are Mapillary/KartaView (vectorized)
        if not df_existing.empty and "Image_URL" in df_existing.columns:
            platforms = df_existing["Platform"].to_numpy()
            photo_ids = df_existing["Photo_ID"].to_numpy()
            image_urls = df_existing["Image_URL"].to_numpy()
            df_existing["Image_URL"] = [
                f"mapillary://{pid}"
                if str(plat).lower() == "mapillary"
                else (f"kartaview://{pid}" if str(plat).lower() == "kartaview" else url)
                for plat, pid, url in zip(platforms, photo_ids, image_urls)
            ]
            df_existing["Latitude"] = pd.to_numeric(
                df_existing["Latitude"], errors="coerce"
            )
            df_existing["Longitude"] = pd.to_numeric(
                df_existing["Longitude"], errors="coerce"
            )
        print(
            f"Loaded {len(df_existing)} existing images across {df_existing['H3_Cell'].nunique()} cells."
        )

    # Read input files in parallel using ThreadPoolExecutor
    all_dfs = []
    if csv_files:
        print(f"Reading {len(csv_files)} files in parallel...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    load_and_preprocess_csv,
                    f,
                    offline_dirs=args.offline_dataset_dirs,
                    representation_type=args.representation_type,
                )
                for f in csv_files
            ]
            for fut in tqdm(
                as_completed(futures), total=len(csv_files), desc="Reading input files"
            ):
                res = fut.result()
                if res is not None:
                    all_dfs.append(res)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame(
            columns=[
                "Photo_ID",
                "Platform",
                "Latitude",
                "Longitude",
                "Image_URL",
                "Captured_At",
                "License",
            ]
        )
    all_dfs = []  # Free memory

    # Vectorized H3 cell computation and filtering
    if not df_all.empty:
        # Standardize timestamps across all platforms (Flickr, Mapillary, iNaturalist) in a vectorized way
        df_all["Captured_At"] = standardize_timestamps_vectorized(df_all["Captured_At"])
        df_all["Latitude"] = pd.to_numeric(df_all["Latitude"], errors="coerce")
        df_all["Longitude"] = pd.to_numeric(df_all["Longitude"], errors="coerce")

        # Vectorized H3 cell calculation
        lats = df_all["Latitude"].to_numpy()
        lons = df_all["Longitude"].to_numpy()
        h3_res = args.h3_res
        df_all["H3_Cell"] = [
            h3.latlng_to_cell(float(lat), float(lon), h3_res)
            if pd.notna(lat) and pd.notna(lon)
            else None
            for lat, lon in zip(lats, lons)
        ]

        df_all = df_all.dropna(subset=["H3_Cell", "Photo_ID"])
        df_all["Photo_ID"] = df_all["Photo_ID"].apply(clean_photo_id)

        # Convert any raw Mapillary/KartaView URLs to virtual URIs to prevent CDN expiration (vectorized)
        platforms = df_all["Platform"].to_numpy()
        photo_ids = df_all["Photo_ID"].to_numpy()
        image_urls = df_all["Image_URL"].to_numpy()
        df_all["Image_URL"] = [
            f"mapillary://{pid}"
            if str(plat).lower() == "mapillary"
            else (f"kartaview://{pid}" if str(plat).lower() == "kartaview" else url)
            for plat, pid, url in zip(platforms, photo_ids, image_urls)
        ]

        df_all = df_all.drop_duplicates(subset=["Platform", "Photo_ID"])
        if seen_keys:
            df_all["temp_key"] = list(zip(df_all["Platform"], df_all["Photo_ID"]))
            df_all = df_all[~df_all["temp_key"].isin(seen_keys)]
            df_all = df_all.drop(columns=["temp_key"])

    print(f"Total NEW raw images: {len(df_all)}")

    new_cells = set(df_all["H3_Cell"].unique()) if not df_all.empty else set()
    existing_cells = (
        set(df_existing["H3_Cell"].unique()) if df_existing is not None else set()
    )
    all_cells = new_cells | existing_cells
    active_cells = new_cells

    # Load embeddings only for active cells (saves massive RAM for millions of rows)
    df_existing_active = None
    existing_embeddings = None
    if (
        args.resume_from
        and os.path.exists(args.resume_from)
        and not args.resume_from.endswith(".pkl")
        and active_cells
    ):
        print("Loading active cell embeddings only using PyArrow Dataset...")
        t0 = time.time()

        dataset = ds.dataset(args.resume_from, format="parquet")
        active_cells_list = list(active_cells)

        filter_expr = ds.field("H3_Cell").isin(active_cells_list)

        has_decoupled = "embedding" not in dataset.schema.names

        cols_to_load = [
            "Photo_ID",
            "Platform",
            "Latitude",
            "Longitude",
            "H3_Cell",
            "Captured_At",
            "Image_URL",
        ]
        if not has_decoupled:
            cols_to_load.append("embedding")
        else:
            if "photo_key" in dataset.schema.names:
                cols_to_load.append("photo_key")
            if "embedding_idx" in dataset.schema.names:
                cols_to_load.append("embedding_idx")

        if "License" in dataset.schema.names:
            cols_to_load.append("License")

        table_active = dataset.to_table(filter=filter_expr, columns=cols_to_load)
        df_existing_active = table_active.to_pandas()

        if len(df_existing_active) > 0:
            if has_decoupled:
                # Load the full memory-mapped embedding matrix
                full_embeddings = load_embeddings(
                    args.resume_from, representation_type=args.representation_type
                )

                # Build master keys index from df_existing
                df_existing_keys = (
                    df_existing["Platform"].astype(str)
                    + "_"
                    + df_existing["Photo_ID"].astype(str)
                )
                master_keys = pd.Index(df_existing_keys)

                # Get keys of active cells
                active_keys = (
                    df_existing_active["Platform"].astype(str)
                    + "_"
                    + df_existing_active["Photo_ID"].astype(str)
                )

                # Resolve indices using pd.Index.get_indexer
                if master_keys.is_unique:
                    indices = master_keys.get_indexer(active_keys)
                else:
                    pos_series = pd.Series(
                        np.arange(len(master_keys)), index=master_keys
                    )
                    pos_series = pos_series[~pos_series.index.duplicated(keep="first")]
                    indices = pos_series.reindex(active_keys, fill_value=-1).values

                valid_mask = indices >= 0
                if not valid_mask.all():
                    print(
                        f"Warning: Found {np.sum(~valid_mask):,} unmatched active cells in master database index."
                    )
                    safe_indices = np.clip(indices, 0, len(full_embeddings) - 1)
                    existing_embeddings = full_embeddings[safe_indices]
                    existing_embeddings[~valid_mask] = 0.0
                else:
                    existing_embeddings = full_embeddings[indices]
            else:
                chunked_arr = table_active["embedding"]
                dim = len(chunked_arr.chunk(0)[0].as_py())
                existing_embeddings = np.empty(
                    (len(df_existing_active), dim), dtype=np.float32
                )
                current_row = 0
                for chunk in chunked_arr.chunks:
                    chunk_len = len(chunk)
                    flat_chunk = chunk.flatten().to_numpy()
                    existing_embeddings[current_row : current_row + chunk_len] = (
                        flat_chunk.reshape(chunk_len, dim)
                    )
                    current_row += chunk_len

            # Retroactively clean existing URLs to virtual format if they are Mapillary/KartaView (vectorized)
            if "Image_URL" in df_existing_active.columns:
                platforms = df_existing_active["Platform"].to_numpy()
                photo_ids = df_existing_active["Photo_ID"].to_numpy()
                image_urls = df_existing_active["Image_URL"].to_numpy()
                df_existing_active["Image_URL"] = [
                    f"mapillary://{pid}"
                    if str(plat).lower() == "mapillary"
                    else (
                        f"kartaview://{pid}"
                        if str(plat).lower() == "kartaview"
                        else url
                    )
                    for plat, pid, url in zip(platforms, photo_ids, image_urls)
                ]
        else:
            existing_embeddings = None
        print(
            f" -> Loaded {len(df_existing_active):,} active existing embeddings in {time.time() - t0:.2f}s."
        )
    elif df_existing is not None and not df_existing.empty:
        df_existing_active = df_existing[
            df_existing["H3_Cell"].isin(active_cells)
        ].copy()

    print(
        f"Total H3 cells: {len(all_cells)} ({len(active_cells)} active with new data, {len(all_cells) - len(active_cells)} inactive/skipped)"
    )

    # 3. Load TIPSv2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TIPSv2 model on {device}...")
    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
    model.eval().to(device)

    # Pre-compute text features for Zero-Shot filtering
    text_features = None
    macro_idx = -1
    sky_idx = -1
    if not args.no_filter:
        print("Pre-computing zero-shot filter text embeddings...")
        with torch.no_grad():
            prompts = ["An indoor scene", "An outdoor landscape or street view"]
            if args.filter_macro:
                prompts.append(
                    "A close-up macro photo of a single leaf, plant petal, flower, insect, mushroom, or tree bark"
                )
                macro_idx = len(prompts) - 1
            if args.filter_sky:
                prompts.append(
                    "A photo of the sky, a bird flying in the air, an insect in flight, an airplane, or a close-up of a cloud with no ground visible"
                )
                sky_idx = len(prompts) - 1
            text_features = model.encode_text(prompts).cpu().numpy()

    # 4. Process and Deduplicate
    checkpoint_path = os.path.join(
        args.save_path, f"{args.output_name}_checkpoint.parquet"
    )
    checkpoint_meta_path = os.path.join(
        args.save_path, f"{args.output_name}_checkpoint_meta.pkl"
    )

    final_data = []
    processed_cells = set()

    if (
        args.checkpoint_interval > 0
        and os.path.exists(checkpoint_path)
        and os.path.exists(checkpoint_meta_path)
    ):
        print(f"Found checkpoint files: {checkpoint_path}")
        print(
            "Resuming from checkpoint. (To start fresh, delete these checkpoint files or run with --checkpoint_interval 0)"
        )
        try:
            df_ckpt = load_dataframe(checkpoint_path)
            # Filter df_ckpt to only include active cells for final_data
            df_ckpt_active = df_ckpt[df_ckpt["H3_Cell"].isin(active_cells)]
            final_data = df_ckpt_active.to_dict("records")
            with open(checkpoint_meta_path, "rb") as f:
                processed_cells = pickle.load(f)
            print(
                f"Loaded {len(final_data)} images from checkpoint. {len(processed_cells)} cells already processed."
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch/resume_from.")
            final_data = []
            processed_cells = set()

    cells_to_process = list(active_cells)
    if args.limit_cells > 0:
        cells_to_process = cells_to_process[: args.limit_cells]
        print(f"Limiting to {args.limit_cells} cells for testing.")

    last_checkpoint_time = time.time()

    print("Grouping metadata by H3 cell...")
    new_metadata_dict = defaultdict(list)
    if not df_all.empty:
        new_records = df_all.to_dict("records")
        for r in new_records:
            cell = r["H3_Cell"]
            new_metadata_dict[cell].append(r)

    existing_items_dict = defaultdict(list)
    if df_existing_active is not None and not df_existing_active.empty:
        if "existing_embeddings" in locals() and existing_embeddings is not None:
            if args.resume_from.endswith(".pkl"):
                active_indices = df_existing_active.index.values
                embs = existing_embeddings[active_indices]
            else:
                embs = existing_embeddings
        else:
            embs = [None] * len(df_existing_active)

        existing_records = df_existing_active.to_dict("records")
        for r, emb in zip(existing_records, embs):
            cell = r["H3_Cell"]
            r["embedding"] = emb
            existing_items_dict[cell].append(r)

    with ThreadPoolExecutor(max_workers=64) as executor:
        for cell in tqdm(cells_to_process, desc="Processing cells"):
            if cell in processed_cells:
                continue

            new_metadata = new_metadata_dict.get(cell, [])
            existing_items = existing_items_dict.get(cell, [])

            # If there's no new data for this cell, just keep the existing data
            if not new_metadata:
                final_data.extend(existing_items)
                processed_cells.add(cell)
                continue

            deduped = process_cell(
                cell,
                new_metadata,
                model,
                device,
                args.sim_threshold,
                executor,
                text_features,
                existing_items,
                cell_chunk_size=args.cell_chunk_size,
                tips_batch_size=args.tips_batch_size,
                macro_idx=macro_idx,
                sky_idx=sky_idx,
                offline_dirs=args.offline_dataset_dirs,
                representation_type=args.representation_type,
                mapillary_token=args.mapillary_token,
            )
            final_data.extend(deduped)
            processed_cells.add(cell)

            # Periodic checkpoint saving
            if args.checkpoint_interval > 0:
                current_time = time.time()
                if current_time - last_checkpoint_time > args.checkpoint_interval:
                    save_checkpoint(
                        final_data,
                        processed_cells,
                        checkpoint_path,
                        checkpoint_meta_path,
                        resume_from=args.resume_from,
                        active_cells=active_cells,
                        representation_type=args.representation_type,
                        precision=args.precision,
                    )
                    last_checkpoint_time = current_time

    # Save a final checkpoint upon loop completion so that raw data is never lost if saving fails
    if args.checkpoint_interval > 0:
        print("\nSaving final completed checkpoint...")
        save_checkpoint(
            final_data,
            processed_cells,
            checkpoint_path,
            checkpoint_meta_path,
            resume_from=args.resume_from,
            active_cells=active_cells,
            representation_type=args.representation_type,
            precision=args.precision,
        )

    # 5. Save Results
    if not final_data:
        print("No data processed successfully.")
        return

    os.makedirs(args.save_path, exist_ok=True)
    out_df = pd.DataFrame(final_data)
    if "Captured_At" in out_df.columns:
        out_df["Captured_At"] = standardize_timestamps_vectorized(out_df["Captured_At"])
    out_df["Latitude"] = pd.to_numeric(out_df["Latitude"], errors="coerce")
    out_df["Longitude"] = pd.to_numeric(out_df["Longitude"], errors="coerce")
    csv_path = os.path.join(args.save_path, f"{args.output_name}.csv")
    parquet_path = os.path.join(args.save_path, f"{args.output_name}.parquet")

    # If resume_from exists, we write the final parquet by streaming
    if (
        args.resume_from
        and os.path.exists(args.resume_from)
        and not args.resume_from.endswith(".pkl")
    ):
        print("Writing final Parquet database using streaming update...")
        # 1. Parquet stream update
        stream_update_parquet(
            args.resume_from,
            parquet_path,
            out_df,
            active_cells,
            representation_type=args.representation_type,
            precision=args.precision,
        )
    else:
        # Save Full Data to Parquet (High-performance binary storage)
        save_dataframe(
            out_df,
            parquet_path,
            representation_type=args.representation_type,
            precision=args.precision,
        )

    # Clean up checkpoint files on successful completion
    for path_to_remove in [checkpoint_path, checkpoint_meta_path]:
        if os.path.exists(path_to_remove):
            try:
                os.remove(path_to_remove)
            except Exception:
                pass
        # Clean up any lingering .tmp files
        if os.path.exists(path_to_remove + ".tmp"):
            try:
                os.remove(path_to_remove + ".tmp")
            except Exception:
                pass

    # Clean up checkpoint companion files (.npy and .keys.parquet)
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    checkpoint_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    for ext in ["*.npy", "*.keys.parquet"]:
        for companion_file in glob.glob(
            os.path.join(checkpoint_dir, f"{checkpoint_base}{ext}")
        ):
            try:
                os.remove(companion_file)
            except Exception:
                pass

    print("\nProcessing Complete!")
    print(f"Unique images kept: {len(final_data)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Parquet saved to: {parquet_path}")


if __name__ == "__main__":
    main()
