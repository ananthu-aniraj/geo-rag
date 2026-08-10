import argparse
import os
import re
import sys
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

from src.utils.io import load_dataset_with_clusters, load_embeddings


def create_scatter_plot(pkl_path, output_png, representation_type=None):
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
    match = re.search(r'_k_(\d+)', pkl_path)
    if match:
        k_clusters = int(match.group(1))

    # Trim '_clustered_k_X' suffix if present to find base name
    clean_base_name = base_name
    if "_clustered_k_" in clean_base_name:
        clean_base_name = clean_base_name.split("_clustered_k_")[0]

    sidecar_name = f"{clean_base_name}_clustered_k_{k_clusters}.parquet"
    sidecar_path = os.path.join(db_dir, sidecar_name)
    has_decoupled_layout = 'embedding' not in available_cols

    if not has_decoupled_layout:
        required_cols = ['cluster_id', 'embedding']
        for c in required_cols:
            if c not in available_cols:
                print(f"Error: Input dataset is missing required column: '{c}'")
    match = re.search(r'_clustered_k_(\d+)', os.path.basename(pkl_path))
    if match:
        k_clusters = int(match.group(1))

    match = re.search(r'_k_(\d+)', os.path.basename(pkl_path))
    if match:
        k_clusters = int(match.group(1))

    # Verify if it's decoupled parquet format
    is_pkl = pkl_path.endswith('.pkl')
    has_decoupled = False
    if not is_pkl:
        pf = pq.ParquetFile(pkl_path)
        has_decoupled = 'embedding' not in pf.schema_arrow.names

    output_html = output_png.replace('.png', '.html')

    # 1. Aggregate embeddings to compute cluster centroids
    cluster_sums = {}
    cluster_counts = {}
    cluster_metadata = {}
    dim = None

    if not is_pkl and not has_decoupled:
        # Case A: Combined parquet (read raw embeddings directly in chunks)
        print("Reading combined Parquet dataset...")
        pf = pq.ParquetFile(pkl_path)
        available_cols = pf.schema_arrow.names
        for rg in tqdm(range(pf.num_row_groups), desc="Processing row groups"):
            columns_to_read = ['cluster_id', 'embedding']
            for extra in ['cluster_label', 'parent_cluster_label', 'cluster_description', 'Platform']:
                if extra in available_cols:
                    columns_to_read.append(extra)

            table = pf.read_row_group(rg, columns=columns_to_read)
            df_rg = table.to_pandas()
            if len(df_rg) == 0:
                continue

            # Extract embeddings matrix for this chunk
            embs = np.vstack(df_rg['embedding'].values).astype(np.float32)
            if dim is None:
                dim = embs.shape[1]
                print(f"Detected embedding dimensionality: {dim}")

            c_ids = df_rg['cluster_id'].values

            # Fast vectorized accumulation by unique ID in the chunk
            for unique_id in tqdm(np.unique(c_ids), desc="Processing clusters"):
                if unique_id is None or pd.isna(unique_id):
                    continue
                unique_id = int(unique_id)
                mask = (c_ids == unique_id)
                sum_emb = embs[mask].sum(axis=0)
                count = int(mask.sum())

                if unique_id not in cluster_sums:
                    cluster_sums[unique_id] = sum_emb
                    cluster_counts[unique_id] = count
                    idx = np.where(mask)[0][0]
                    cluster_metadata[unique_id] = {
                        'label': df_rg.iloc[idx].get('cluster_label', f"Cluster {unique_id}"),
                        'parent': df_rg.iloc[idx].get('parent_cluster_label', f"Parent {int(unique_id) // 80}"),
                        'description': df_rg.iloc[idx].get('cluster_description', "No description available")
                    }
                else:
                    cluster_sums[unique_id] += sum_emb
                    cluster_counts[unique_id] += count
    else:
        # Case B: New decoupled format (load metadata/clusters and memory-mapped embeddings)
        df_meta = load_dataset_with_clusters(pkl_path, k_clusters=k_clusters, representation_type=representation_type)
        embeddings = load_embeddings(pkl_path, representation_type=representation_type)

        dim = embeddings.shape[1]
        print(f"Detected decoupled embedding matrix dimensionality: {dim}")

        c_ids = df_meta['cluster_id'].values

        # Process in memory-safe chunks of 100,000 rows
        chunk_size = 100000
        for start_idx in tqdm(range(0, len(df_meta), chunk_size), desc="Processing chunks"):
            end_idx = min(start_idx + chunk_size, len(df_meta))
            chunk_df = df_meta.iloc[start_idx:end_idx]
            chunk_embs = embeddings[start_idx:end_idx]
            chunk_c_ids = c_ids[start_idx:end_idx]

            for unique_id in tqdm(np.unique(chunk_c_ids), desc="Processing clusters"):
                if unique_id is None or pd.isna(unique_id):
                    continue
                unique_id = int(unique_id)
                mask = (chunk_c_ids == unique_id)
                sum_emb = chunk_embs[mask].sum(axis=0)
                count = int(mask.sum())

                if unique_id not in cluster_sums:
                    cluster_sums[unique_id] = sum_emb
                    cluster_counts[unique_id] = count
                    idx = np.where(mask)[0][0]
                    cluster_metadata[unique_id] = {
                        'label': chunk_df.iloc[idx].get('cluster_label', f"Cluster {unique_id}"),
                        'parent': chunk_df.iloc[idx].get('parent_cluster_label', f"Parent {int(unique_id) // 80}"),
                        'description': chunk_df.iloc[idx].get('cluster_description', "No description available")
                    }
                else:
                    cluster_sums[unique_id] += sum_emb
                    cluster_counts[unique_id] += count

    num_clusters = len(cluster_sums)
    if num_clusters == 0:
        print("Error: No clusters found in the dataset.")
        sys.exit(1)

    print(f" -> Successfully aggregated {num_clusters} cluster centroids in {time.time() - start_time:.2f}s.")

    # 2. Compute exact mean centroids
    unique_ids = sorted(list(cluster_sums.keys()))
    centroids = np.empty((num_clusters, dim), dtype=np.float32)
    for i, c_id in enumerate(unique_ids):
        centroids[i] = cluster_sums[c_id] / cluster_counts[c_id]

    print("Normalizing centroids for cosine distance projection...")
    centroids_norm = normalize(centroids)

    # 3. Compute 2D UMAP projection
    print("Computing 2D UMAP projection on centroids...")
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
    umap_coords = reducer.fit_transform(centroids_norm)

    # 4. Prepare DataFrame for plotting
    plot_data = []
    for i, c_id in enumerate(unique_ids):
        meta = cluster_metadata[c_id]
        plot_data.append({
            'x': umap_coords[i][0],
            'y': umap_coords[i][1],
            'cluster_id': c_id,
            'cluster_label': meta['label'],
            'parent_cluster_label': meta['parent'],
            'cluster_description': meta['description'],
            'image_count': cluster_counts[c_id]
        })
    df_plot = pd.DataFrame(plot_data)
    df_plot['parent_cluster_label'] = df_plot['parent_cluster_label'].fillna("Unknown").astype(str)
    df_plot['cluster_label'] = df_plot['cluster_label'].fillna("").astype(str)
    df_plot['cluster_description'] = df_plot['cluster_description'].fillna("").astype(str)

    # --- SAVE INTERACTIVE HTML PLOT (PLOTLY) ---
    output_html = output_png.replace('.png', '.html')
    print("Generating interactive WebGL scatter plot...")

    # Scale marker sizes: min size 5, max size 30
    max_count = df_plot['image_count'].max()
    min_count = df_plot['image_count'].min()
    if max_count == min_count:
        df_plot['marker_size'] = 10
    else:
        df_plot['marker_size'] = 6 + 24 * (df_plot['image_count'] - min_count) / (max_count - min_count)

    parent_categories = sorted(df_plot['parent_cluster_label'].unique())
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    color_map = {p: colors[i % len(colors)] for i, p in enumerate(parent_categories)}

    # Build custom hover templates
    hover_templates = []
    for idx, row in df_plot.iterrows():
        tpl = (
            f"<b>Cluster {row['cluster_id']}: {row['cluster_label']}</b><br>"
            f"Parent: {row['parent_cluster_label']}<br>"
            f"Images count: {row['image_count']:,}<br>"
            f"Description: {row['cluster_description']}<extra></extra>"
        )
        hover_templates.append(tpl)

    fig = go.Figure()

    for parent in parent_categories:
        df_sub = df_plot[df_plot['parent_cluster_label'] == parent]
        sub_hover = [hover_templates[idx] for idx in df_sub.index]

        fig.add_trace(go.Scattergl(
            x=df_sub['x'],
            y=df_sub['y'],
            mode='markers',
            marker=dict(
                size=df_sub['marker_size'],
                color=color_map[parent],
                line=dict(width=0.5, color='rgba(255, 255, 255, 0.8)')
            ),
            name=parent,
            text=sub_hover,
            hovertemplate='%{text}'
        ))

    fig.update_layout(
        title=dict(
            text="Geo-RAG: Semantic Visual Clusters (Centroid UMAP 2D Projection)",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(title="UMAP Dimension 1", showgrid=True, zeroline=False),
        yaxis=dict(title="UMAP Dimension 2", showgrid=True, zeroline=False),
        legend=dict(
            title="Parent Categories",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        width=1350,
        height=850,
        margin=dict(r=250, t=60, b=40, l=40),
        template="plotly_white"
    )

    print(f"Saving interactive WebGL scatter plot to {output_html}...")
    fig.write_html(output_html)

    # --- SAVE STATIC PLOT (MATPLOTLIB) ---
    print("Generating static scatter plot image...")
    plt.figure(figsize=(14, 9))
    sns.set_style("whitegrid")

    # Use tab20 colors or generic mapping for parents
    parent_categories = sorted(df_plot['parent_cluster_label'].unique())
    num_parents = len(parent_categories)
    if num_parents > 0:
        hue_order = parent_categories
        palette = sns.color_palette("tab20", n_colors=num_parents) if num_parents <= 20 else sns.color_palette("husl",
                                                                                                               num_parents)
    else:
        hue_order = None
        palette = 'viridis'

    # Sizing for matplotlib plot
    max_count = df_plot['image_count'].max()
    min_count = df_plot['image_count'].min()
    if max_count == min_count:
        sizes = 50
    else:
        sizes = 20 + 200 * (df_plot['image_count'] - min_count) / (max_count - min_count)

    scatter = sns.scatterplot(
        data=df_plot,
        x='x', y='y',
        hue='parent_cluster_label',
        hue_order=hue_order,
        size=sizes,
        sizes=(20, 250),
        palette=palette,
        alpha=0.8,
        edgecolor='grey',
        linewidth=0.5
    )

    plt.title("Geo-RAG: Semantic Visual Clusters (Centroid UMAP 2D Projection)", fontsize=15)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=8, title="Parent Categories")
    plt.tight_layout()

    print(f"Saving static scatter plot to {output_png}...")
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"🎉 Complete visualization processing finished in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize cluster centroids in an interactive 2D semantic scatter plot using UMAP.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the clustered parquet or pkl file.")
    parser.add_argument("--out", type=str, default="cluster_scatter.png", help="Output PNG file name.")
    parser.add_argument("--representation_type", type=str, default="cls", choices=["cls", "avg_patch", "cls_avg_patch"],
                        help="Type of representation embedding to load (cls, avg_patch, or cls_avg_patch).")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16"],
                        help="Stored precision of companion binary file (float32 or float16).")
    args = parser.parse_args()

    create_scatter_plot(args.pkl, args.out, representation_type=args.representation_type)
