# ruff: noqa: E402
import os
import sys

# Disable oneDNN optimizations and CUDA device conflicts before any third-party imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Prevent UMAP from eagerly importing TensorFlow / oneDNN via parametric_umap (unused and causes segfaults)
sys.modules["tensorflow"] = None

import argparse
import json
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import seaborn as sns
import umap
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.utils.credentials import get_mapillary_token
from src.utils.io import (
    iter_embeddings,
    load_dataset_with_clusters,
    resolve_offline_image_path,
)


def inject_hover_card(
    html_path,
    auto_detected_base="",
    replace_prefix="",
    mapillary_token="",
    bin_galleries=None,
):
    """Inject cave_umwelten-style floating hover tooltip card with photo preview into Plotly HTML."""
    template_path = os.path.join(
        os.path.dirname(__file__), "templates", "cluster_scatter_hover_card.html"
    )
    if not os.path.exists(template_path):
        print(f"Warning: Hover card template not found at {template_path}")
        return

    try:
        galleries_json = json.dumps(bin_galleries or {})
        with open(template_path, "r", encoding="utf-8") as f:
            tooltip_html = (
                f.read()
                .replace("__AUTO_DETECTED_BASE__", auto_detected_base or "")
                .replace("__DEFAULT_REPLACE_PREFIX__", replace_prefix or "")
                .replace("__MAPILLARY_TOKEN__", mapillary_token or "")
                .replace("__SPATIAL_BIN_GALLERIES__", galleries_json)
            )

        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "</body>" in content:
            content = content.replace("</body>", f"{tooltip_html}\n</body>")
        else:
            content += tooltip_html
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f" -> Successfully injected interactive hovercards into {html_path}.")
    except Exception as e:
        print(f"Warning: Could not inject hover card into HTML: {e}")


def create_scatter_plot(
    pkl_path,
    output_png,
    representation_type=None,
    model_name=None,
    precision=None,
    batch_size=500000,
    platforms=None,
    image_root_dir=None,
    find_prefix=None,
    replace_prefix=None,
    recompute_umap=False,
    spatial_bin_size=3.0,
    images_per_bin=4,
):
    print(f"Loading clustered data from {pkl_path}...")
    start_time = time.time()
    try:
        pf = pq.ParquetFile(pkl_path)
        available_cols = pf.schema_arrow.names
    except Exception as e:
        print(f"Error opening parquet file: {e}")
        sys.exit(1)

    db_dir = os.path.dirname(os.path.abspath(pkl_path))
    base_name = os.path.splitext(os.path.basename(pkl_path))[0]

    # Auto-extract k_clusters from filename
    k_clusters = 50000
    match = re.search(r"_k_(\d+)", pkl_path)
    if match:
        k_clusters = int(match.group(1))

    # Trim '_clustered_k_X' suffix if present to find base name
    clean_base_name = base_name
    if "_clustered_k_" in clean_base_name:
        clean_base_name = clean_base_name.split("_clustered_k_")[0]

    sidecar_name = f"{clean_base_name}_clustered_k_{k_clusters}.parquet"
    sidecar_path = os.path.join(db_dir, sidecar_name)
    has_decoupled_layout = "embedding" not in available_cols

    if not has_decoupled_layout:
        required_cols = ["cluster_id", "embedding"]
        for c in required_cols:
            if c not in available_cols:
                print(f"Error: Input dataset is missing required column: '{c}'")
    match = re.search(r"_clustered_k_(\d+)", os.path.basename(pkl_path))
    if match:
        k_clusters = int(match.group(1))

    match = re.search(r"_k_(\d+)", os.path.basename(pkl_path))
    if match:
        k_clusters = int(match.group(1))

    # Verify if it's decoupled parquet format
    is_pkl = pkl_path.endswith(".pkl")
    has_decoupled = False
    if not is_pkl:
        pf = pq.ParquetFile(pkl_path)
        has_decoupled = "embedding" not in pf.schema_arrow.names

    output_html = output_png.replace(".png", ".html")

    # 1. Aggregate embeddings to compute cluster centroids
    cluster_sums = {}
    cluster_counts = {}
    cluster_metadata = {}
    cluster_sample_images = {}
    dim = None

    if not is_pkl and not has_decoupled:
        # Case A: Combined parquet (read raw embeddings directly in chunks)
        print("Reading combined Parquet dataset...")
        pf = pq.ParquetFile(pkl_path)
        available_cols = pf.schema_arrow.names
        for rg in tqdm(range(pf.num_row_groups), desc="Processing row groups"):
            columns_to_read = ["cluster_id", "embedding"]
            for extra in [
                "cluster_label",
                "parent_cluster_label",
                "cluster_description",
                "Platform",
                "Image_URL",
                "Photo_ID",
            ]:
                if extra in available_cols:
                    columns_to_read.append(extra)

            table = pf.read_row_group(rg, columns=columns_to_read)
            df_rg = table.to_pandas()
            if len(df_rg) == 0:
                continue

            # Extract embeddings matrix for this chunk
            embs = np.vstack(df_rg["embedding"].values).astype(np.float32)
            if dim is None:
                dim = embs.shape[1]
                print(f"Detected embedding dimensionality: {dim}")

            c_ids = df_rg["cluster_id"].values

            # Accumulation by unique ID in the chunk
            valid_m = ~pd.isna(c_ids) & (c_ids >= 0)
            for unique_id in np.unique(c_ids[valid_m]):
                unique_id = int(unique_id)
                mask = c_ids == unique_id
                sum_emb = embs[mask].sum(axis=0)
                count = int(mask.sum())

                if unique_id not in cluster_sums:
                    cluster_sums[unique_id] = sum_emb
                    cluster_counts[unique_id] = count
                    idx = np.where(mask)[0][0]
                    cluster_metadata[unique_id] = {
                        "label": df_rg.iloc[idx].get(
                            "cluster_label", f"Cluster {unique_id}"
                        ),
                        "parent": df_rg.iloc[idx].get(
                            "parent_cluster_label", f"Parent {int(unique_id) // 80}"
                        ),
                        "description": df_rg.iloc[idx].get(
                            "cluster_description", "No description available"
                        ),
                    }
                else:
                    cluster_sums[unique_id] += sum_emb
                    cluster_counts[unique_id] += count

                # Sample thumbnail image for cluster hover tooltip
                if (
                    unique_id not in cluster_sample_images
                    and "Image_URL" in df_rg.columns
                ):
                    sub_c = df_rg[mask]
                    if platforms and "Platform" in sub_c.columns:
                        plat_set = {p.lower() for p in platforms}
                        sub_c = sub_c[
                            sub_c["Platform"].astype(str).str.lower().isin(plat_set)
                        ]
                    if not sub_c.empty and pd.notna(sub_c.iloc[0].get("Image_URL")):
                        first_row = sub_c.iloc[0]
                        raw_url = str(first_row["Image_URL"])
                        pid = str(first_row.get("Photo_ID", ""))
                        plat = str(first_row.get("Platform", ""))
                        if image_root_dir:
                            resolved = resolve_offline_image_path(
                                raw_url,
                                image_root_dir,
                                photo_id=pid,
                                platform=plat,
                            )
                            if resolved:
                                raw_url = "file://" + os.path.abspath(resolved)
                        cluster_sample_images[unique_id] = {
                            "url": raw_url,
                            "photo_id": pid,
                            "platform": plat,
                        }
    else:
        # Case B: New decoupled format (load metadata/clusters and memory-mapped embeddings)
        sample_cols = [
            "cluster_id",
            "cluster_label",
            "parent_cluster_label",
            "cluster_description",
            "Platform",
            "Image_URL",
            "Photo_ID",
            "model_name",
            "representation_type",
        ]
        try:
            df_meta = load_dataset_with_clusters(
                pkl_path,
                k_clusters=k_clusters,
                columns=sample_cols,
                representation_type=representation_type,
                model_name=model_name,
                precision=precision,
            )
        except Exception:
            df_meta = load_dataset_with_clusters(
                pkl_path,
                k_clusters=k_clusters,
                representation_type=representation_type,
                model_name=model_name,
                precision=precision,
            )

        if not model_name and "model_name" in df_meta.columns:
            valid_models = df_meta["model_name"].dropna()
            if len(valid_models) > 0:
                model_name = str(valid_models.iloc[0])

        if not representation_type and "representation_type" in df_meta.columns:
            valid_reps = df_meta["representation_type"].dropna()
            if len(valid_reps) > 0:
                representation_type = str(valid_reps.iloc[0])

        # Extract metadata per unique cluster_id
        print("Extracting unique cluster metadata...")
        meta_sub = df_meta.dropna(subset=["cluster_id"]).drop_duplicates(
            subset=["cluster_id"]
        )
        for _, row in meta_sub.iterrows():
            cid = int(row["cluster_id"])
            if cid >= 0:
                cluster_metadata[cid] = {
                    "label": row.get("cluster_label", f"Cluster {cid}"),
                    "parent": row.get("parent_cluster_label", f"Parent {cid // 80}"),
                    "description": row.get(
                        "cluster_description", "No description available"
                    ),
                }

        # Extract representative cluster thumbnail images for hover cards
        if "Image_URL" in df_meta.columns and "cluster_id" in df_meta.columns:
            print("Extracting representative cluster thumbnail images...")
            df_imgs = df_meta.dropna(subset=["Image_URL", "cluster_id"])
            if platforms and "Platform" in df_imgs.columns:
                plat_set = {p.lower() for p in platforms}
                df_imgs = df_imgs[
                    df_imgs["Platform"].astype(str).str.lower().isin(plat_set)
                ]
                print(
                    f" -> Filtered hover sample images to {platforms}: {len(df_imgs):,} available."
                )
            elif "Platform" in df_imgs.columns:
                # Default (all platforms): prioritize Flickr and iNaturalist if available
                is_persistent = (
                    df_imgs["Platform"]
                    .astype(str)
                    .str.lower()
                    .isin(["flickr", "inaturalist"])
                )
                df_imgs = pd.concat([df_imgs[is_persistent], df_imgs[~is_persistent]])

            sub_imgs = df_imgs.drop_duplicates(subset=["cluster_id"])
            for _, row in sub_imgs.iterrows():
                try:
                    cid = int(row["cluster_id"])
                    if cid >= 0:
                        raw_url = str(row["Image_URL"])
                        pid = str(row.get("Photo_ID", ""))
                        plat = str(row.get("Platform", ""))
                        if image_root_dir:
                            resolved = resolve_offline_image_path(
                                raw_url,
                                image_root_dir,
                                photo_id=pid,
                                platform=plat,
                            )
                            if resolved:
                                raw_url = "file://" + os.path.abspath(resolved)
                        cluster_sample_images[cid] = {
                            "url": raw_url,
                            "photo_id": pid,
                            "platform": plat,
                        }
                except (ValueError, TypeError):
                    continue
            print(
                f" -> Found thumbnail images for {len(cluster_sample_images):,} clusters."
            )

        c_ids = df_meta["cluster_id"].to_numpy()
        valid_mask = ~pd.isna(c_ids) & (c_ids >= 0)
        valid_c_ids = c_ids[valid_mask].astype(np.int64)
        max_cid = int(valid_c_ids.max()) if len(valid_c_ids) > 0 else 0

        chunk_size = batch_size
        total_len = len(c_ids)
        dim = None
        raw_centroids = None
        raw_counts = np.zeros(max_cid + 1, dtype=np.int64)

        print("Aggregating cluster centroids in fast streaming batches...")
        curr_idx = 0
        total_batches = max(1, (total_len + chunk_size - 1) // chunk_size)
        for chunk_raw in tqdm(
            iter_embeddings(
                pkl_path,
                batch_size=chunk_size,
                representation_type=representation_type,
                model_name=model_name,
                precision="float32",
            ),
            total=total_batches,
            desc="Computing cluster centroids",
        ):
            if dim is None:
                dim = chunk_raw.shape[1]
                print(f"Detected decoupled embedding matrix dimensionality: {dim}")
                raw_centroids = np.zeros((max_cid + 1, dim), dtype=np.float32)

            batch_len = len(chunk_raw)
            chunk_ids = c_ids[curr_idx : curr_idx + batch_len]
            m = ~pd.isna(chunk_ids) & (chunk_ids >= 0)
            valid_ids = chunk_ids[m].astype(np.int64)
            np.add.at(raw_centroids, valid_ids, chunk_raw[m])
            np.add.at(raw_counts, valid_ids, 1)
            curr_idx += batch_len

        active_cluster_ids = np.where(raw_counts > 0)[0]
        for cid in active_cluster_ids:
            cluster_sums[cid] = raw_centroids[cid]
            cluster_counts[cid] = int(raw_counts[cid])
            if cid not in cluster_metadata:
                cluster_metadata[cid] = {
                    "label": f"Cluster {cid}",
                    "parent": f"Parent {cid // 80}",
                    "description": "No description available",
                }

    num_clusters = len(cluster_sums)
    if num_clusters == 0:
        print("Error: No clusters found in the dataset.")
        sys.exit(1)

    print(
        f" -> Successfully aggregated {num_clusters} cluster centroids in {time.time() - start_time:.2f}s."
    )

    # 2. Compute exact mean centroids
    unique_ids = sorted(list(cluster_sums.keys()))
    centroids = np.empty((num_clusters, dim), dtype=np.float32)
    for i, c_id in enumerate(unique_ids):
        centroids[i] = cluster_sums[c_id] / cluster_counts[c_id]

    print("Normalizing centroids for cosine distance projection...")
    centroids_norm = normalize(centroids)

    # 3. Compute or load 2D UMAP projection
    umap_cache_path = output_png.replace(".png", "_umap_coords.npy")
    possible_caches = [
        umap_cache_path,
        os.path.splitext(pkl_path)[0] + "_umap_coords.npy",
        os.path.join(
            os.path.dirname(os.path.abspath(output_png)),
            f"{os.path.splitext(os.path.basename(pkl_path))[0]}_umap_coords.npy",
        ),
    ]
    umap_coords = None
    if not recompute_umap:
        for c_path in possible_caches:
            if os.path.exists(c_path):
                print(f"Loading cached 2D UMAP coordinates from {c_path}...")
                try:
                    cached = np.load(c_path)
                    if len(cached) == num_clusters:
                        umap_coords = cached
                        break
                    else:
                        print(
                            f"Cached coordinates size mismatch ({len(cached)} vs {num_clusters}), trying next cache..."
                        )
                except Exception as e:
                    print(f"Failed to load cached coordinates: {e}")

    if umap_coords is None:
        print("Computing 2D UMAP projection on centroids...")
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
        umap_coords = reducer.fit_transform(centroids_norm)
        try:
            np.save(umap_cache_path, umap_coords)
            print(f"Saved 2D UMAP coordinates cache to {umap_cache_path}.")
        except Exception as e:
            print(f"Warning: could not save UMAP cache ({e}).")

    # Determine auto-detected base and mapillary token for SSH tunneling / offline image resolution
    auto_detected_base = find_prefix or ""
    if not auto_detected_base:
        for cid, info in cluster_sample_images.items():
            u = info["url"] if isinstance(info, dict) else str(info)
            if u.startswith("file://") or u.startswith("/"):
                last_slash = u.rfind("/")
                if last_slash != -1:
                    auto_detected_base = u[: last_slash + 1]
                    break
    if not auto_detected_base and image_root_dir:
        first_dir = (
            image_root_dir[0] if isinstance(image_root_dir, list) else image_root_dir
        )
        auto_detected_base = "file://" + os.path.abspath(first_dir).rstrip("/") + "/"

    token = get_mapillary_token()
    if platforms and "mapillary" not in [p.lower() for p in platforms]:
        mapillary_token = ""
    else:
        mapillary_token = token

    # 4. Group clusters into spatial-semantic visual bins (delta distance on UMAP coordinates)
    bin_clusters = {}
    for i, c_id in enumerate(unique_ids):
        if spatial_bin_size and spatial_bin_size > 0:
            bx = int(np.floor(umap_coords[i][0] / spatial_bin_size))
            by = int(np.floor(umap_coords[i][1] / spatial_bin_size))
            b_key = (cluster_metadata[c_id]["label"], bx, by)
        else:
            b_key = c_id

        if b_key not in bin_clusters:
            bin_clusters[b_key] = []
        bin_clusters[b_key].append(c_id)

    bin_key_to_id = {}
    bin_galleries = {}
    cluster_to_bin_id = {}

    for b_key, c_list in bin_clusters.items():
        bin_id = len(bin_key_to_id)
        bin_key_to_id[b_key] = bin_id
        bin_galleries[bin_id] = []

        # Sort clusters in this bin by image_count descending for visual dominance
        c_list_sorted = sorted(
            c_list, key=lambda c: cluster_counts.get(c, 0), reverse=True
        )

        for c in c_list_sorted:
            cluster_to_bin_id[c] = bin_id
            if len(bin_galleries[bin_id]) < images_per_bin:
                img_info = cluster_sample_images.get(c)
                if img_info and isinstance(img_info, dict) and img_info.get("url"):
                    bin_galleries[bin_id].append(img_info)

    print(
        f" -> Partitioned {len(unique_ids):,} clusters into {len(bin_galleries):,} spatial-semantic visual bins (delta={spatial_bin_size})."
    )

    # 5. Prepare DataFrame for plotting
    plot_data = []
    for i, c_id in enumerate(unique_ids):
        meta = cluster_metadata[c_id]
        b_id = cluster_to_bin_id[c_id]

        plot_data.append(
            {
                "x": float(umap_coords[i][0]),
                "y": float(umap_coords[i][1]),
                "cluster_id": c_id,
                "cluster_label": meta["label"],
                "parent_cluster_label": meta["parent"],
                "cluster_description": meta["description"],
                "image_count": cluster_counts[c_id],
                "bin_id": b_id,
            }
        )
    df_plot = pd.DataFrame(plot_data)
    df_plot["parent_cluster_label"] = (
        df_plot["parent_cluster_label"].fillna("Unknown").astype(str)
    )
    df_plot["cluster_label"] = df_plot["cluster_label"].fillna("").astype(str)
    df_plot["cluster_description"] = (
        df_plot["cluster_description"].fillna("").astype(str)
    )

    # --- SAVE INTERACTIVE HTML PLOT (PLOTLY) ---
    output_html = output_png.replace(".png", ".html")
    print("Generating interactive WebGL scatter plot...")

    # Scale marker sizes: min size 5, max size 30
    max_count = df_plot["image_count"].max()
    min_count = df_plot["image_count"].min()
    if max_count == min_count:
        df_plot["marker_size"] = 10
    else:
        df_plot["marker_size"] = 6 + 24 * (df_plot["image_count"] - min_count) / (
            max_count - min_count
        )

    parent_categories = sorted(df_plot["parent_cluster_label"].unique())
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    color_map = {p: colors[i % len(colors)] for i, p in enumerate(parent_categories)}

    fig = go.Figure()

    for parent in parent_categories:
        df_sub = df_plot[df_plot["parent_cluster_label"] == parent]
        sub_customdata = df_sub[
            [
                "bin_id",
                "cluster_label",
                "parent_cluster_label",
                "image_count",
                "cluster_description",
                "cluster_id",
            ]
        ].to_numpy()

        fig.add_trace(
            go.Scattergl(
                x=df_sub["x"],
                y=df_sub["y"],
                mode="markers",
                marker=dict(
                    size=df_sub["marker_size"],
                    color=color_map[parent],
                    line=dict(width=0.5, color="rgba(255, 255, 255, 0.8)"),
                ),
                name=parent,
                customdata=sub_customdata,
                hoverinfo="none",
            )
        )

    fig.update_layout(
        title=dict(
            text="Geo-RAG: Semantic Visual Clusters (Centroid UMAP 2D Projection)",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title="UMAP Dimension 1", showgrid=True, zeroline=False),
        yaxis=dict(title="UMAP Dimension 2", showgrid=True, zeroline=False),
        legend=dict(
            title="Parent Categories",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        width=1350,
        height=850,
        margin=dict(r=250, t=60, b=40, l=40),
        template="plotly_white",
    )

    print(f"Saving interactive WebGL scatter plot to {output_html}...")
    fig.write_html(output_html)
    inject_hover_card(
        output_html,
        auto_detected_base=auto_detected_base,
        replace_prefix=replace_prefix or "",
        mapillary_token=mapillary_token,
        bin_galleries=bin_galleries,
    )

    # --- SAVE STATIC PLOT (MATPLOTLIB) ---
    print("Generating static scatter plot image...")
    plt.figure(figsize=(14, 9))
    sns.set_style("whitegrid")

    # Use tab20 colors or generic mapping for parents
    parent_categories = sorted(df_plot["parent_cluster_label"].unique())
    num_parents = len(parent_categories)
    if num_parents > 0:
        hue_order = parent_categories
        palette = (
            sns.color_palette("tab20", n_colors=num_parents)
            if num_parents <= 20
            else sns.color_palette("husl", num_parents)
        )
    else:
        hue_order = None
        palette = "viridis"

    # Sizing for matplotlib plot
    max_count = df_plot["image_count"].max()
    min_count = df_plot["image_count"].min()
    if max_count == min_count:
        sizes = 50
    else:
        sizes = 20 + 200 * (df_plot["image_count"] - min_count) / (
            max_count - min_count
        )

    scatter = sns.scatterplot(
        data=df_plot,
        x="x",
        y="y",
        hue="parent_cluster_label",
        hue_order=hue_order,
        size=sizes,
        sizes=(20, 250),
        palette=palette,
        alpha=0.8,
        edgecolor="grey",
        linewidth=0.5,
    )

    plt.title(
        "Geo-RAG: Semantic Visual Clusters (Centroid UMAP 2D Projection)", fontsize=15
    )
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        title="Parent Categories",
    )
    plt.tight_layout()

    print(f"Saving static scatter plot to {output_png}...")
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(
        f"🎉 Complete visualization processing finished in {time.time() - start_time:.2f}s."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize cluster centroids in an interactive 2D semantic scatter plot using UMAP."
    )
    parser.add_argument(
        "--pkl",
        type=str,
        required=True,
        help="Path to the clustered parquet or pkl file.",
    )
    parser.add_argument(
        "--out", type=str, default="cluster_scatter.png", help="Output PNG file name."
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        default="cls",
        choices=["cls", "avg_patch", "cls_avg_patch"],
        help="Type of representation embedding to load (cls, avg_patch, or cls_avg_patch).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Stored precision of companion binary file (float32 or float16).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional vision model name for companion embedding resolution.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500000,
        help="Streaming batch size for centroid aggregation (default: 500,000).",
    )
    parser.add_argument(
        "--platforms",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of platforms to filter hover preview images (default: None, all platforms).",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        nargs="+",
        default=None,
        help="Optional local offline image directories for resolving local image paths.",
    )
    parser.add_argument(
        "--find_prefix",
        type=str,
        default=None,
        help="Optional local path prefix to replace in hover images (auto-detected if omitted).",
    )
    parser.add_argument(
        "--replace_prefix",
        type=str,
        default=None,
        help="Optional replacement URL prefix for SSH tunneling (e.g. http://localhost:8000/).",
    )
    parser.add_argument(
        "--recompute_umap",
        action="store_true",
        help="Force recomputation of UMAP projection instead of using cached coordinates.",
    )
    parser.add_argument(
        "--spatial_bin_size",
        type=float,
        default=3.0,
        help="UMAP coordinate grid cell size for grouping clusters into representative photo galleries (default: 3.0).",
    )
    parser.add_argument(
        "--images_per_bin",
        type=int,
        default=4,
        help="Maximum representative images to show in hover card for each spatial-semantic bin (default: 4).",
    )
    args = parser.parse_args()

    create_scatter_plot(
        args.pkl,
        args.out,
        representation_type=args.representation_type,
        model_name=args.model_name,
        precision=args.precision,
        batch_size=args.batch_size,
        platforms=args.platforms,
        image_root_dir=args.image_root_dir,
        find_prefix=args.find_prefix,
        replace_prefix=args.replace_prefix,
        recompute_umap=args.recompute_umap,
        spatial_bin_size=args.spatial_bin_size,
        images_per_bin=args.images_per_bin,
    )
