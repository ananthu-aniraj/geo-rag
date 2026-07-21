import concurrent.futures
import requests
import pickle
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import argparse
import time
import os
import glob
import math
import json
from collections import Counter
from tqdm import tqdm
try:
    import h3
except ImportError:
    h3 = None

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


def create_sample_grid(pkl_path, output_html, top_n=5, image_root_dir=None, target_h3_res=5):
    print(f"Loading clustered data from {pkl_path}...")

    if pkl_path.endswith('.pkl'):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        del data
        embeddings = np.vstack(df['embedding'].values).astype(np.float32)
    else:
        # Load metadata only (uses ~200MB RAM)
        parquet_file = pq.ParquetFile(pkl_path)
        metadata_cols = [c for c in parquet_file.schema_arrow.names if c != 'embedding']
        df = pd.read_parquet(pkl_path, columns=metadata_cols)

        # Load embeddings via PyArrow
        print("Loading raw embedding matrix using PyArrow...")
        t0 = time.time()
        table = pq.read_table(pkl_path, columns=["embedding"])

        num_rows = len(table)
        chunked_arr = table['embedding']
        dim = len(chunked_arr.chunk(0)[0].as_py())

        embeddings = np.empty((num_rows, dim), dtype=np.float32)
        current_row = 0
        for chunk in chunked_arr.chunks:
            chunk_len = len(chunk)
            flat_chunk = chunk.flatten().to_numpy()
            embeddings[current_row:current_row + chunk_len] = flat_chunk.reshape(chunk_len, dim)
            current_row += chunk_len

        del table
        print(f" -> Successfully loaded raw embedding matrix in {time.time() - t0:.2f}s.")

    if len(df) == 0 or 'cluster_id' not in df.columns:
        print("Error: Data is empty or not clustered.")
        return

    # Group data by cluster and prepare a compact JSON-like structure
    print("Aggregating cluster data for the dashboard...")

    # Load H3 res 4 parent cells from the pre-built spatial-semantic index
    cluster_h3_target_res = {}

    dir_name = os.path.dirname(os.path.abspath(pkl_path))
    index_candidates = glob.glob(os.path.join(dir_name, "*h3_semantic_index.parquet"))
    if index_candidates:
        index_path = index_candidates[0]
        print(f"Loading pre-built H3 index from {index_path}...")
        try:
            index_df = pd.read_parquet(index_path, columns=['resolution', 'cluster_id', 'query_cell'])
            target_h3_res_df = index_df[index_df['resolution'] == target_h3_res].dropna(subset=['query_cell'])
            target_h3_res_df = target_h3_res_df[['cluster_id', 'query_cell']].drop_duplicates()
            cluster_h3_target_res = target_h3_res_df.groupby('cluster_id')['query_cell'].agg(list).to_dict()
            print(f" -> Successfully loaded H3 res parent cells for clusters with resolution of {target_h3_res}")
        except Exception as e:
            print(f"Warning: Failed to load H3 index: {e}")

    # Group by cluster ID
    cluster_to_indices = df.groupby('cluster_id').groups
    dashboard_data = []
    sorted_ids = sorted(cluster_to_indices.keys())

    # Pre-extract numpy arrays from DataFrame for fast O(1) indexing (eliminates df.iloc overhead)
    print("Start pre-extracting columns for fast access...")
    col_photo_id = df['Photo_ID'].to_numpy() if 'Photo_ID' in df.columns else np.array([""] * len(df))
    col_platform = df['Platform'].to_numpy() if 'Platform' in df.columns else np.array([""] * len(df))
    col_lat = df['Latitude'].to_numpy() if 'Latitude' in df.columns else np.zeros(len(df))
    col_lon = df['Longitude'].to_numpy() if 'Longitude' in df.columns else np.zeros(len(df))
    col_url = df['Image_URL'].to_numpy() if 'Image_URL' in df.columns else np.array([""] * len(df))
    col_captured_at = df['Captured_At'].to_numpy() if 'Captured_At' in df.columns else np.array([""] * len(df))
    col_label = df['cluster_label'].to_numpy() if 'cluster_label' in df.columns else np.array([""] * len(df))
    col_desc = df['cluster_description'].to_numpy() if 'cluster_description' in df.columns else np.array([""] * len(df))
    col_parent_id = df['parent_cluster_id'].to_numpy() if 'parent_cluster_id' in df.columns else np.array([-1] * len(df))
    col_parent_label = df['parent_cluster_label'].to_numpy() if 'parent_cluster_label' in df.columns else np.array([""] * len(df))
    col_season = df['Season'].to_numpy() if 'Season' in df.columns else np.array(["Unknown"] * len(df))
    col_tod = df['Time_Of_Day'].to_numpy() if 'Time_Of_Day' in df.columns else np.array(["Unknown"] * len(df))
    col_h3 = df['H3_Cell'].to_numpy() if 'H3_Cell' in df.columns else np.array([""] * len(df))
    print("Successfully pre-extracted columns for fast access.")

    def make_sample(global_idx, sim_score, is_outlier, rank_label):
        pid = col_photo_id[global_idx]
        if pid is not None and not (isinstance(pid, float) and np.isnan(pid)):
            pid_str = str(int(pid)) if isinstance(pid, (float, int)) and not isinstance(pid, bool) else str(pid).strip()
            if pid_str.endswith('.0'):
                pid_str = pid_str[:-2]
        else:
            pid_str = ""

        cap = col_captured_at[global_idx]
        cap_str = str(cap) if cap is not None and not (isinstance(cap, float) and np.isnan(cap)) else ""

        return {
            "url": str(col_url[global_idx]),
            "id": pid_str,
            "sim": float(sim_score),
            "lat": float(col_lat[global_idx]) if pd.notna(col_lat[global_idx]) else 0.0,
            "lon": float(col_lon[global_idx]) if pd.notna(col_lon[global_idx]) else 0.0,
            "platform": str(col_platform[global_idx]).strip() if pd.notna(col_platform[global_idx]) else "",
            "captured_at": cap_str,
            "season": str(col_season[global_idx]) if pd.notna(col_season[global_idx]) else "Unknown",
            "time_of_day": str(col_tod[global_idx]) if pd.notna(col_tod[global_idx]) else "Unknown",
            "is_outlier": is_outlier,
            "rank_label": rank_label
        }

    for c_id in tqdm(sorted_ids, desc="Processing clusters"):
        indices = cluster_to_indices[c_id]
        embs = embeddings[indices]
        centroid = np.mean(embs, axis=0)

        # Cosine similarity
        norm_embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = np.dot(norm_embs, norm_centroid)

        sorted_indices = np.argsort(sims)[::-1][:top_n]

        idx0 = indices[0]
        first_label = str(col_label[idx0]) if pd.notna(col_label[idx0]) and str(col_label[idx0]) != "" else f"Cluster {c_id}"
        first_parent_id = int(col_parent_id[idx0]) if pd.notna(col_parent_id[idx0]) else -1
        first_parent_label = str(col_parent_label[idx0]) if pd.notna(col_parent_label[idx0]) and str(col_parent_label[idx0]) != "" else "Unlabeled Parent"

        # Keep representative samples
        samples = []
        for rank, local_idx in enumerate(sorted_indices):
            label_text = "Centroid Image" if rank == 0 else f"Representative Sample {rank}"
            samples.append(make_sample(indices[local_idx], float(sims[local_idx]), False, label_text))

        # Add outlier samples if cluster is larger than top_n
        if len(sims) > top_n:
            lowest_indices = [idx for idx in np.argsort(sims)[:2] if idx not in sorted_indices]
            for i, local_idx in enumerate(lowest_indices):
                samples.append(make_sample(indices[local_idx], float(sims[local_idx]), True, f"Furthest Outlier {i + 1}"))

        # Centroid description
        centroid_g_idx = indices[sorted_indices[0]]
        centroid_desc = str(col_desc[centroid_g_idx]) if pd.notna(col_desc[centroid_g_idx]) else ""

        # Vectorized geographic center calculation
        c_lats = col_lat[indices]
        c_lons = col_lon[indices]
        valid_mask = ~np.isnan(c_lats) & ~np.isnan(c_lons)
        v_lats = c_lats[valid_mask]
        v_lons = c_lons[valid_mask]

        if len(v_lats) > 0:
            center_lat = float(np.mean(v_lats))
            rad_lons = np.radians(v_lons)
            x = float(np.mean(np.cos(rad_lons)))
            y = float(np.mean(np.sin(rad_lons)))
            center_lon = float(np.degrees(np.arctan2(y, x)))
        else:
            center_lat, center_lon = 0.0, 0.0

        # Fast H3 cell frequency calculation using Counter
        c_h3s = col_h3[indices]
        h3_counts = Counter(cell for cell in c_h3s if cell and pd.notna(cell))
        unique_h3_count = len(h3_counts)

        h3_centroids = []
        if h3_counts and h3 is not None:
            try:
                top_h3 = h3_counts.most_common(50)
                for cell, count in top_h3:
                    try:
                        h3_lat, h3_lon = h3.cell_to_latlon(cell)
                        h3_centroids.append([float(h3_lat), float(h3_lon), int(count), cell])
                    except Exception:
                        continue
            except Exception:
                pass

        dashboard_data.append({
            "id": int(c_id),
            "label": first_label,
            "parent_id": first_parent_id,
            "parent_label": first_parent_label,
            "size": len(indices),
            "count": len(indices),
            "description": centroid_desc,
            "unique_h3_count": unique_h3_count,
            "center_lat": float(center_lat),
            "center_lon": float(center_lon),
            "h3_centroids": h3_centroids,
            "h3_res_tgt": [str(x) for x in cluster_h3_target_res.get(c_id, [])],
            "samples": samples
        })
    # Pre-resolve local image paths if image_root_dir is supplied or they already exist
    for item in tqdm(dashboard_data, desc="Resolving image paths"):
        for sample in item["samples"]:
            url = sample["url"]
            resolved_path = None
            if image_root_dir:
                dirs = [image_root_dir] if isinstance(image_root_dir, str) else image_root_dir
                for d in dirs:
                    if not d:
                        continue
                    path1 = os.path.join(d, url)
                    if os.path.exists(path1):
                        resolved_path = path1
                        break
                    path2 = os.path.join(d, os.path.basename(url))
                    if os.path.exists(path2):
                        resolved_path = path2
                        break
                    path3 = os.path.join(d, "train", os.path.basename(url))
                    if os.path.exists(path3):
                        resolved_path = path3
                        break
            if not resolved_path and os.path.exists(url):
                resolved_path = url

            if resolved_path:
                sample["url"] = "file://" + os.path.abspath(resolved_path)

    # Collect samples to check: first two representatives, last two representatives, and outliers
    samples_to_check = []
    for item in tqdm(dashboard_data, desc="Collecting samples to check"):
        cluster_samples = item["samples"]
        reps = [s for s in cluster_samples if not s.get("is_outlier")]
        outliers = [s for s in cluster_samples if s.get("is_outlier")]

        selected_for_check = []
        # First two representatives
        if len(reps) > 0:
            selected_for_check.append(reps[0])
        if len(reps) > 1:
            selected_for_check.append(reps[1])
        # Last two representatives
        if len(reps) > 2:
            selected_for_check.append(reps[-1])
            if len(reps) > 3:
                selected_for_check.append(reps[-2])
        # Outliers
        selected_for_check.extend(outliers)

        # Deduplicate using python object identity to avoid checking the same sample multiple times
        seen_ids = set()
        for s in selected_for_check:
            if id(s) not in seen_ids:
                seen_ids.add(id(s))
                samples_to_check.append(s)

    def check_and_resolve_sample(sample, timeout=5):
        url = sample["url"]
        photo_id = sample["id"]
        platform = sample["platform"]

        if url.startswith("file://"):
            return

        if os.path.exists(url):
            return

        # 1. Quick HEAD check to see if the URL signature has expired
        try:
            res = requests.head(url, timeout=timeout, allow_redirects=True)
            if res.status_code == 200:
                return  # URL is still valid!
        except Exception:
            pass

        if not photo_id or not platform:
            return

        platform_lower = str(platform).strip().lower()
        photo_str = str(photo_id).strip()
        if photo_str.endswith('.0'):
            photo_str = photo_str[:-2]

        is_mapillary = platform_lower == 'mapillary' or 'mapillary' in url or 'fbcdn.net' in url
        is_kartaview = platform_lower == 'kartaview' or 'kartaview' in url or 'openstreetcam' in url

        if not (is_mapillary or is_kartaview):
            return

        # 2. Resolve expired Mapillary or Kartaview URLs dynamically
        try:
            if is_mapillary:
                api_url = f"https://graph.mapillary.com/{photo_str}?fields=thumb_1024_url"
                headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
                res = requests.get(api_url, headers=headers, timeout=timeout)
                if res.status_code == 200:
                    fresh_url = res.json().get("thumb_1024_url")
                    if fresh_url:
                        sample["url"] = fresh_url
            elif is_kartaview:
                api_url = f"https://api.openstreetcam.org/2.0/photo/{photo_str}"
                res = requests.get(api_url, timeout=timeout)
                if res.status_code == 200:
                    data = res.json().get("result", {}).get("data", {})
                    fresh_url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")
                    if fresh_url:
                        sample["url"] = fresh_url
        except Exception:
            pass

    print(f"Checking and resolving signatures for {len(samples_to_check)} critical cluster images in parallel...")
    max_workers = min(32, (len(samples_to_check) + 4) // 5 or 1)

    completed_count = 0
    total_count = len(samples_to_check)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_and_resolve_sample, sample): sample for sample in samples_to_check}
        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == total_count:
                print(
                    f"  Progress: {completed_count}/{total_count} images checked ({completed_count * 100 // total_count}%)...")

    print("Signature check and resolution complete.")

    # Generate the Dynamic Dashboard HTML
    json_data = json.dumps(dashboard_data)

    # Save data to an external JS file to prevent browser freezing on massive inline scripts
    data_js_path = output_html.replace('.html', '_data.js')
    print(f"Writing data payload to {data_js_path}...")
    with open(data_js_path, 'w', encoding='utf-8') as f:
        f.write(f"var CLUSTER_DATA = {json_data};\nvar TARGET_H3_RES = {target_h3_res};")

    data_js_filename = os.path.basename(data_js_path)

    # Load HTML template from root templates/cluster_dashboard.html
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    template_path = os.path.join(root_dir, "templates", "cluster_dashboard.html")
    if not os.path.exists(template_path):
        template_path = os.path.join(os.path.dirname(__file__), "templates", "cluster_dashboard.html")
    if not os.path.exists(template_path):
        template_path = "templates/cluster_dashboard.html"

    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()

    html_content = html_template.replace("{{DATA_JS_FILENAME}}", data_js_filename).replace("{{TOTAL_CLUSTERS}}", str(len(dashboard_data)))

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Scalable Dashboard saved to: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an HTML grid of representative samples for each cluster.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the clustered .pkl file.")
    parser.add_argument("--out", type=str, default="cluster_samples.html", help="Output HTML file name.")
    parser.add_argument("--top_n", type=int, default=6, help="Number of samples to show per cluster.")
    parser.add_argument("--image_root_dir", type=str, nargs="+", default=None,
                        help="Optional root directories containing local images (for offline datasets).")
    parser.add_argument("--target_h3_res", type=int, default=5, help="Target H3 resolution for spatial aggregation (default: 5).")
    args = parser.parse_args()

    create_sample_grid(args.pkl, args.out, args.top_n, args.image_root_dir, target_h3_res=args.target_h3_res)
