import pickle
import numpy as np
import argparse
import time
import os

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


def create_sample_grid(pkl_path, output_html, top_n=5):
    print(f"Loading clustered data from {pkl_path}...")
    import pandas as pd
    import pyarrow.parquet as pq
    import numpy as np
    
    if pkl_path.endswith('.pkl'):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        del data
        embeddings = np.vstack(df['embedding'].values).astype(np.float32)
    else:
        # Load metadata only (uses ~200MB RAM)
        df = pd.read_parquet(pkl_path, columns=['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'Captured_At', 'cluster_id', 'cluster_label', 'cluster_description', 'parent_cluster_id', 'parent_cluster_label', 'parent_cluster_description', 'Season', 'Time_Of_Day', 'H3_Cell'])
        
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
    cluster_h3_res4 = {}
    import glob
    dir_name = os.path.dirname(os.path.abspath(pkl_path))
    index_candidates = glob.glob(os.path.join(dir_name, "*h3_semantic_index.parquet"))
    if index_candidates:
        index_path = index_candidates[0]
        print(f"Loading pre-built H3 index from {index_path}...")
        try:
            index_df = pd.read_parquet(index_path)
            res4_df = index_df[index_df['resolution'] == 4]
            cluster_h3_res4 = res4_df.groupby('cluster_id')['query_cell'].apply(lambda x: list(set(x.dropna()))).to_dict()
        except Exception as e:
            print(f"Warning: Failed to load H3 index: {e}")
            
    # Group by cluster ID
    cluster_to_indices = df.groupby('cluster_id').groups
    dashboard_data = []
    sorted_ids = sorted(cluster_to_indices.keys())

    import math
    for c_id in sorted_ids:
        indices = cluster_to_indices[c_id]
        embs = embeddings[indices]
        centroid = np.mean(embs, axis=0)

        # Cosine similarity
        norm_embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = np.dot(norm_embs, norm_centroid)

        sorted_indices = np.argsort(sims)[::-1][:top_n]
        
        # Load first row to pull cluster metadata
        first_row = df.iloc[indices[0]]

        # Keep only the representative samples to save space in the HTML
        samples = []
        for rank, local_idx in enumerate(sorted_indices):
            global_idx = indices[local_idx]
            img = df.iloc[global_idx]
            
            photo_id = img.get('Photo_ID')
            if photo_id is not None:
                if isinstance(photo_id, float):
                    import math as py_math
                    if not py_math.isnan(photo_id):
                        photo_id = str(int(photo_id))
                    else:
                        photo_id = ""
                else:
                    photo_id = str(photo_id).strip()
                    if photo_id.endswith('.0'):
                        photo_id = photo_id[:-2]
            else:
                photo_id = ""
                
            samples.append({
                "url": img["Image_URL"],
                "id": photo_id,
                "sim": float(sims[local_idx]),
                "lat": float(img.get("Latitude", 0.0)),
                "lon": float(img.get("Longitude", 0.0)),
                "platform": str(img.get("Platform", "")).strip(),
                "captured_at": img.get("Captured_At", ""),
                "season": img.get("Season", "Unknown"),
                "time_of_day": img.get("Time_Of_Day", "Unknown"),
                "is_outlier": False,
                "rank_label": "Centroid Image" if rank == 0 else f"Representative Sample {rank}"
            })

        # Add the least representative (outlier) images if the cluster is larger than top_n
        if len(sims) > top_n:
            lowest_indices = np.argsort(sims)[:2]  # Two absolute lowest similarity images
            # Deduplicate just in case any overlap
            lowest_indices = [idx for idx in lowest_indices if idx not in sorted_indices]
            for i, local_idx in enumerate(lowest_indices):
                global_idx = indices[local_idx]
                img = df.iloc[global_idx]
                
                photo_id = img.get('Photo_ID')
                if photo_id is not None:
                    if isinstance(photo_id, float):
                        import math as py_math
                        if not py_math.isnan(photo_id):
                            photo_id = str(int(photo_id))
                        else:
                            photo_id = ""
                    else:
                        photo_id = str(photo_id).strip()
                        if photo_id.endswith('.0'):
                            photo_id = photo_id[:-2]
                else:
                    photo_id = ""
                    
                samples.append({
                    "url": img["Image_URL"],
                    "id": photo_id,
                    "sim": float(sims[local_idx]),
                    "lat": float(img.get("Latitude", 0.0)),
                    "lon": float(img.get("Longitude", 0.0)),
                    "platform": str(img.get("Platform", "")).strip(),
                    "captured_at": img.get("Captured_At", ""),
                    "season": img.get("Season", "Unknown"),
                    "time_of_day": img.get("Time_Of_Day", "Unknown"),
                    "is_outlier": True,
                    "rank_label": f"Furthest Outlier {i+1}"
                })

        # Grab description from the centroid sample (highest similarity, sorted_indices[0])
        centroid_img = df.iloc[indices[sorted_indices[0]]]
        centroid_desc = centroid_img.get("cluster_description", "")

        # Calculate robust geographic center (handling wrap-around for longitude)
        cluster_rows = df.iloc[indices]
        lats = cluster_rows["Latitude"].dropna().tolist()
        lons = cluster_rows["Longitude"].dropna().tolist()

        if lats:
            center_lat = sum(lats) / len(lats)
            x = sum(math.cos(math.radians(lon)) for lon in lons) / len(lons)
            y = sum(math.sin(math.radians(lon)) for lon in lons) / len(lons)
            center_lon = math.degrees(math.atan2(y, x))
        else:
            center_lat, center_lon = 0.0, 0.0

        # Calculate unique H3 cells count
        h3_cells = set(cluster_rows["H3_Cell"].dropna())
        unique_h3_count = len(h3_cells)

        # Compute H3 cell frequency and centroids
        h3_counts = cluster_rows["H3_Cell"].dropna().value_counts().to_dict()

        h3_centroids = []
        try:
            import h3
            sorted_h3 = sorted(h3_counts.items(), key=lambda x: x[1], reverse=True)[:50]
            for cell, count in sorted_h3:
                try:
                    h3_lat, h3_lon = h3.cell_to_latlon(cell)
                    h3_centroids.append([float(h3_lat), float(h3_lon), int(count), cell])
                except Exception:
                    continue
        except ImportError:
            # Fallback if h3 library is not present: group by rounded lat/lon
            coord_counts = {}
            for img in samples:
                r_lat = round(img['lat'], 2)
                r_lon = round(img['lon'], 2)
                coord_counts[(r_lat, r_lon)] = coord_counts.get((r_lat, r_lon), 0) + 1
            sorted_coords = sorted(coord_counts.items(), key=lambda x: x[1], reverse=True)[:50]
            for (r_lat, r_lon), count in sorted_coords:
                h3_centroids.append([float(r_lat), float(r_lon), int(count), "N/A"])

        dashboard_data.append({
            "id": int(c_id),
            "label": str(first_row.get("cluster_label", f"Cluster {c_id}")),
            "parent_id": int(first_row.get("parent_cluster_id", -1)) if pd.notna(first_row.get("parent_cluster_id")) else -1,
            "parent_label": str(first_row.get("parent_cluster_label", "Unlabeled Parent")) if pd.notna(first_row.get("parent_cluster_label")) else "Unlabeled Parent",
            "description": centroid_desc,
            "count": int(len(indices)),
            "unique_h3_count": unique_h3_count,
            "center_lat": float(center_lat),
            "center_lon": float(center_lon),
            "h3_centroids": h3_centroids,
            "h3_res4": [str(x) for x in cluster_h3_res4.get(c_id, [])],
            "samples": samples
        })
    # Collect samples to check: first two representatives, last two representatives, and outliers
    samples_to_check = []
    for item in dashboard_data:
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

    import concurrent.futures
    import requests

    def check_and_resolve_sample(sample, timeout=5):
        url = sample["url"]
        photo_id = sample["id"]
        platform = sample["platform"]
        
        if os.path.exists(url):
            return
            
        # 1. Quick HEAD check to see if the URL signature has expired
        try:
            res = requests.head(url, timeout=timeout, allow_redirects=True)
            if res.status_code == 200:
                return # URL is still valid!
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
                print(f"  Progress: {completed_count}/{total_count} images checked ({completed_count * 100 // total_count}%)...")
                
    print("Signature check and resolution complete.")

    # Generate the Dynamic Dashboard HTML
    import json
    json_data = json.dumps(dashboard_data)

    # Save data to an external JS file to prevent browser freezing on massive inline scripts
    data_js_path = output_html.replace('.html', '_data.js')
    print(f"Writing data payload to {data_js_path}...")
    with open(data_js_path, 'w', encoding='utf-8') as f:
        f.write(f"const CLUSTER_DATA = {json_data};")

    data_js_filename = os.path.basename(data_js_path)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Geo-RAG Cluster Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script src="https://cdn.jsdelivr.net/npm/fuse.js/dist/fuse.basic.min.js"></script>
    <script src="https://unpkg.com/h3-js@3.7.2/dist/h3-js.umd.js"></script>
    <script src="{data_js_filename}"></script>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --accent: #f59e0b;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        }}

        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}

        .header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            color: white;
            padding: 32px 40px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-md);
        }}

        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 2.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}

        .header p {{
            margin: 0;
            color: #c7d2fe;
            font-size: 1.1rem;
            font-weight: 400;
        }}

        .controls {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
        }}

        input, select {{
            padding: 10px 16px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.95rem;
            font-family: inherit;
            color: var(--text-main);
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}

        input {{
            flex-grow: 1;
        }}

        input:focus, select:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }}

        .results-container {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        .cluster-card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .cluster-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        .cluster-header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .cluster-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #1e293b;
        }}

        .cluster-title span {{
            color: var(--primary);
        }}

        .stats-badges {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .tag {{
            background: #e0e7ff;
            color: #4338ca;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .tag-geo {{
            background: #fef3c7;
            color: #d97706;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .desc-box {{
            margin-bottom: 20px;
            color: var(--text-muted);
            font-size: 0.95rem;
            padding: 14px 18px;
            background: #f8fafc;
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            line-height: 1.5;
        }}

        .desc-box b {{
            color: var(--text-main);
        }}

        .map-toggle-btn {{
            background-color: #f1f5f9;
            color: #334155;
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            margin-bottom: 16px;
        }}

        .map-toggle-btn:hover {{
            background-color: #e2e8f0;
            color: #0f172a;
        }}

        .map-toggle-btn.active {{
            background-color: var(--primary);
            color: white;
            border-color: var(--primary);
        }}

        .map-container {{
            height: 350px;
            width: 100%;
            margin-bottom: 20px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 20px;
        }}

        .image-item {{
            display: flex;
            flex-direction: column;
            background: #f8fafc;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            transition: box-shadow 0.2s;
        }}

        .image-item:hover {{
            box-shadow: var(--shadow-sm);
        }}

        .image-role {{
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }}

        .image-item img {{
            width: 100%;
            height: 130px;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin-bottom: 8px;
        }}

        .image-meta {{
            font-size: 0.78rem;
            color: var(--text-muted);
            line-height: 1.4;
            margin-top: auto;
        }}

        .load-more {{
            background: var(--primary);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 8px;
            cursor: pointer;
            margin: 32px auto;
            display: block;
            font-size: 1rem;
            font-weight: 600;
            box-shadow: var(--shadow-sm);
            transition: background-color 0.2s;
        }}

        .load-more:hover {{
            background-color: var(--primary-hover);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Geo-RAG: Global Cluster Dashboard</h1>
        <p>Exploring {len(dashboard_data)} semantic clusters with geographic distributions across global streetscapes.</p>
    </div>

    <div class="controls" style="display: flex; gap: 16px; margin-bottom: 24px; background: var(--card-bg); padding: 20px; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: var(--shadow-sm); flex-wrap: wrap;">
        <div style="flex: 2; min-width: 250px; display: flex; flex-direction: column; gap: 6px;">
            <label for="searchInput" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Text Search</label>
            <input type="text" id="searchInput" placeholder="Search ID, label, or description..." onkeyup="filterDataCombined()">
        </div>
        <div style="flex: 2; min-width: 250px; display: flex; flex-direction: column; gap: 6px;">
            <label for="locationInput" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Geographic Filter</label>
            <input type="text" id="locationInput" placeholder="Filter by place/country (e.g. Alaska, Rome)..." onchange="searchLocation()">
            <span id="locationStatus" style="font-size: 0.8rem; font-weight: 500; min-height: 18px; margin-top: 2px;"></span>
        </div>
        <div style="flex: 1.5; min-width: 180px; display: flex; flex-direction: column; gap: 6px;">
            <label for="parentSelect" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Parent Category</label>
            <select id="parentSelect" onchange="filterDataCombined()" style="height: 44px; padding: 10px 16px; border: 1px solid var(--border-color); border-radius: 8px; font-family: inherit; font-size: 14px; background: white;">
                <option value="">All Parent Categories</option>
            </select>
        </div>
        <div style="flex: 1.2; min-width: 150px; display: flex; flex-direction: column; gap: 6px;">
            <label for="seasonSelect" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Season</label>
            <select id="seasonSelect" onchange="filterDataCombined()" style="height: 44px; padding: 10px 16px; border: 1px solid var(--border-color); border-radius: 8px; font-family: inherit; font-size: 14px; background: white;">
                <option value="">All Seasons</option>
                <option value="Spring">Spring</option>
                <option value="Summer">Summer</option>
                <option value="Autumn">Autumn</option>
                <option value="Winter">Winter</option>
                <option value="Wet Season">Wet Season</option>
                <option value="Dry Season">Dry Season</option>
            </select>
        </div>
        <div style="flex: 1.2; min-width: 150px; display: flex; flex-direction: column; gap: 6px;">
            <label for="todSelect" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Time of Day</label>
            <select id="todSelect" onchange="filterDataCombined()" style="height: 44px; padding: 10px 16px; border: 1px solid var(--border-color); border-radius: 8px; font-family: inherit; font-size: 14px; background: white;">
                <option value="">All Times</option>
                <option value="Dawn">Dawn</option>
                <option value="Morning">Morning</option>
                <option value="Afternoon">Afternoon</option>
                <option value="Dusk">Dusk</option>
                <option value="Night">Night</option>
            </select>
        </div>
        <div style="flex: 1.2; min-width: 140px; display: flex; flex-direction: column; gap: 6px;">
            <label for="minDateInput" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">From Date</label>
            <input type="date" id="minDateInput" onchange="filterDataCombined()" style="height: 44px; padding: 10px 16px; border: 1px solid var(--border-color); border-radius: 8px; font-family: inherit; font-size: 14px;">
        </div>
        <div style="flex: 1.2; min-width: 140px; display: flex; flex-direction: column; gap: 6px;">
            <label for="maxDateInput" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">To Date</label>
            <input type="date" id="maxDateInput" onchange="filterDataCombined()" style="height: 44px; padding: 10px 16px; border: 1px solid var(--border-color); border-radius: 8px; font-family: inherit; font-size: 14px;">
        </div>
        <div style="display: flex; flex-direction: column; gap: 6px; min-width: 140px;">
            <label for="sortSelect" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Sort Order</label>
            <select id="sortSelect" onchange="resetAndRender()" style="height: 44px; padding: 10px 16px; border: 1px solid var(--border-color); border-radius: 8px; font-family: inherit; font-size: 14px; background: white;">
                <option value="id">Sort by ID</option>
                <option value="count">Sort by Size</option>
                <option value="geo">Sort by Geo Spread</option>
            </select>
        </div>
    </div>

    <div id="results" class="results-container">
        <!-- Clusters rendered here via JS -->
    </div>
    
    <button id="loadMoreBtn" class="load-more" onclick="loadMore()" style="display: none;">Load More Clusters</button>

    <script>
        const MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER';
        const data = typeof CLUSTER_DATA !== 'undefined' ? CLUSTER_DATA : [];
        let filteredData = [...data];
        let renderLimit = 50;
        const activeMaps = {{}};

        function handleImageError(img, photoId, platform) {{
            // Avoid infinite loops if retry fails
            if (img.dataset.retryAttempt) return;
            img.dataset.retryAttempt = '1';
            
            const platformLower = String(platform).toLowerCase().trim();
            const isMapillary = platformLower === 'mapillary' || img.src.includes('mapillary') || img.src.includes('fbcdn.net');
            const isKartaview = platformLower === 'kartaview' || img.src.includes('kartaview') || img.src.includes('openstreetcam');
            
            // Clean up photoId to remove any trailing .0 or other formatting issues
            let cleanPhotoId = String(photoId).trim();
            if (cleanPhotoId.endsWith('.0')) {{
                cleanPhotoId = cleanPhotoId.slice(0, -2);
            }}
            
            if (isMapillary && cleanPhotoId && cleanPhotoId !== 'null' && cleanPhotoId !== 'undefined' && cleanPhotoId !== 'NaN') {{
                const apiUrl = `https://graph.mapillary.com/${{cleanPhotoId}}?fields=thumb_1024_url`;
                fetch(apiUrl, {{
                    headers: {{ 'Authorization': `OAuth ${MAPILLARY_TOKEN}` }}
                }})
                .then(res => res.json())
                .then(resData => {{
                    if (resData.thumb_1024_url) {{
                        img.src = resData.thumb_1024_url;
                        // Also update containing link if applicable
                        const link = img.closest('a');
                        if (link) link.href = resData.thumb_1024_url;
                    }}
                }})
                .catch(err => console.error('Error fetching Mapillary fresh URL:', err));
            }} else if (isKartaview && cleanPhotoId && cleanPhotoId !== 'null' && cleanPhotoId !== 'undefined' && cleanPhotoId !== 'NaN') {{
                const apiUrl = `https://api.openstreetcam.org/2.0/photo/${{cleanPhotoId}}`;
                fetch(apiUrl)
                .then(res => res.json())
                .then(resData => {{
                    const data = resData.result && resData.result.data;
                    if (data) {{
                        const freshUrl = data.fileurlLTh || data.fileurlTh || data.fileurl;
                        if (freshUrl) {{
                            img.src = freshUrl;
                            const link = img.closest('a');
                            if (link) link.href = freshUrl;
                        }}
                    }}
                }})
                .catch(err => console.error('Error fetching Kartaview fresh URL:', err));
            }}
        }}

        function renderResults(append = false) {{
            const container = document.getElementById('results');
            const sortVal = document.getElementById('sortSelect').value;
            
            let toRender = [...filteredData];
            if (sortVal === 'count') {{
                toRender.sort((a, b) => b.count - a.count);
            }} else if (sortVal === 'geo') {{
                toRender.sort((a, b) => b.unique_h3_count - a.unique_h3_count);
            }} else {{
                toRender.sort((a, b) => a.id - b.id);
            }}

            const slice = toRender.slice(append ? renderLimit - 50 : 0, renderLimit);
            
            const html = slice.map(c => `
                <div class="cluster-card" id="cluster-card-${{c.id}}">
                    <div class="cluster-header">
                        <div class="cluster-title">Cluster #${{c.id}}: <span>${{c.label}}</span></div>
                        <div class="stats-badges">
                            ${{c.parent_label ? `<span class="tag-parent" style="background-color: #e0f2fe; color: #0369a1; padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; margin-right: 6px;">${{c.parent_label}}</span>` : ''}}
                            <span class="tag">${{c.count.toLocaleString()}} images</span>
                            <span class="tag-geo">${{c.unique_h3_count.toLocaleString()}} H3 cells</span>
                        </div>
                    </div>
                    ${{c.description ? `
                    <div class="desc-box">
                        <b>VLM Prototypical Description:</b> ${{c.description}}
                    </div>` : ''}}
                    
                    <button class="map-toggle-btn" id="map-btn-${{c.id}}" onclick="toggleMap(${{c.id}})">
                        🗺️ Show Geographic Spread Map
                    </button>
                    
                    <div id="map-container-${{c.id}}" class="map-container" style="display: none;"></div>

                    <div class="image-grid">
                        ${{c.samples.filter(s => {{
                            const minDate = document.getElementById('minDateInput').value;
                            const maxDate = document.getElementById('maxDateInput').value;
                            const seasonVal = document.getElementById('seasonSelect').value;
                            const todVal = document.getElementById('todSelect').value;
                            
                            if (seasonVal && s.season !== seasonVal) return false;
                            if (todVal && s.time_of_day !== todVal) return false;
                            if (minDate || maxDate) {{
                                if (!s.captured_at) return false;
                                const dateStr = s.captured_at.substring(0, 10);
                                if (minDate && dateStr < minDate) return false;
                                if (maxDate && dateStr > maxDate) return false;
                            }}
                            return true;
                        }}).map((s, i) => {{
                            let roleColor = '#4f46e5';
                            if (s.is_outlier) {{
                                roleColor = '#e65100';
                            }} else if (s.rank_label === 'Centroid Image') {{
                                roleColor = '#d93025';
                            }}
                            return `
                                <div class="image-item" style="${{s.is_outlier ? 'border-color: #ffe0b2; background-color: #fffaf0;' : ''}}">
                                    <div class="image-role" style="color: ${{roleColor}}">
                                        ${{s.rank_label}}
                                    </div>
                                    <a href="${{s.url}}" target="_blank">
                                        <img src="${{s.url}}" onerror="handleImageError(this, '${{s.id}}', '${{s.platform}}')" loading="lazy">
                                    </a>
                                    <div class="image-meta">
                                        <b>ID:</b> ${{s.id}}<br>
                                        <b>Similarity:</b> ${{s.sim.toFixed(4)}}<br>
                                        <b>Lat/Lon:</b> ${{s.lat.toFixed(4)}}, ${{s.lon.toFixed(4)}}${{s.captured_at ? '<br><b>Taken:</b> ' + s.captured_at + (s.season && s.season !== 'Unknown' ? ' (' + s.season + ')' : '') + (s.time_of_day && s.time_of_day !== 'Unknown' ? ' [' + s.time_of_day + ']' : '') : ''}}
                                    </div>
                                </div>
                            `;
                        }}).join('')}}
                    </div>
                </div>
            `).join('');
            
            if (append) {{
                container.innerHTML += html;
            }} else {{
                // Cleanup maps that are no longer visible
                Object.keys(activeMaps).forEach(id => {{
                    activeMaps[id].remove();
                    delete activeMaps[id];
                }});
                container.innerHTML = html;
            }}
            
            const btn = document.getElementById('loadMoreBtn');
            if (toRender.length > renderLimit) {{
                btn.style.display = 'block';
            }} else {{
                btn.style.display = 'none';
            }}
        }}

        function toggleMap(clusterId) {{
            const container = document.getElementById(`map-container-${{clusterId}}`);
            const btn = document.getElementById(`map-btn-${{clusterId}}`);
            
            if (container.style.display === 'none') {{
                container.style.display = 'block';
                btn.innerHTML = '🗺️ Hide Geographic Map';
                btn.classList.add('active');
                initMap(clusterId);
            }} else {{
                container.style.display = 'none';
                btn.innerHTML = '🗺️ Show Geographic Spread Map';
                btn.classList.remove('active');
            }}
        }}

        function initMap(clusterId) {{
            if (activeMaps[clusterId]) {{
                activeMaps[clusterId].invalidateSize();
                return;
            }}

            const cluster = data.find(c => c.id === clusterId);
            if (!cluster) return;

            // Instantiate map
            const map = L.map(`map-container-${{clusterId}}`).setView([cluster.center_lat, cluster.center_lon], 2);
            activeMaps[clusterId] = map;

            // Load CartoDB Positron tiles
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
            }}).addTo(map);

            // Add circle for cluster geographical center
            L.circle([cluster.center_lat, cluster.center_lon], {{
                color: '#4f46e5',
                fillColor: '#4f46e5',
                fillOpacity: 0.1,
                radius: 100000 // 100km radius
            }}).addTo(map).bindPopup(`<b>Cluster center</b><br>Lat: ${{cluster.center_lat.toFixed(4)}}, Lon: ${{cluster.center_lon.toFixed(4)}}`);

            // Add density circle markers for H3 centroids
            let maxCount = 1;
            cluster.h3_centroids.forEach(pt => {{
                if (pt[2] > maxCount) maxCount = pt[2];
            }});

            cluster.h3_centroids.forEach(pt => {{
                const lat = pt[0];
                const lon = pt[1];
                const count = pt[2];
                const cell = pt[3];
                const percentage = (count / cluster.count) * 100;

                L.circleMarker([lat, lon], {{
                    radius: Math.max(5, Math.min(25, 5 + (count / maxCount) * 15)),
                    fillColor: '#f59e0b',
                    color: '#d97706',
                    weight: 1,
                    opacity: 0.8,
                    fillOpacity: 0.5
                }}).addTo(map).bindPopup(`
                    <b>Location Cluster (H3: ${{cell}})</b><br>
                    Images: ${{count.toLocaleString()}} (${{percentage.toFixed(2)}}% of cluster)
                `);
            }});

            // Add custom markers for the representative sample images
            cluster.samples.forEach((s, idx) => {{
                let markerColor = '#4f46e5';
                if (s.is_outlier) {{
                    markerColor = '#e65100';
                }} else if (s.rank_label === 'Centroid Image') {{
                    markerColor = '#d93025';
                }}
                const label = s.rank_label + ' location';
                
                L.circleMarker([s.lat, s.lon], {{
                    radius: 8,
                    fillColor: markerColor,
                    color: '#ffffff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.9
                }}).addTo(map).bindPopup(`
                    <div style="width: 140px; text-align: center;">
                        <b>${{label}}</b><br>
                        <img src="${{s.url}}" onerror="handleImageError(this, '${{s.id}}', '${{s.platform}}')" style="width: 100%; height: 90px; object-fit: cover; margin-top: 6px; border-radius: 4px; border: 1px solid #ddd;"><br>
                        ${{s.captured_at ? '<span style="font-size: 10px; color: #555;"><b>Taken:</b> ' + s.captured_at + (s.season && s.season !== 'Unknown' ? ' (' + s.season + ')' : '') + (s.time_of_day && s.time_of_day !== 'Unknown' ? ' [' + s.time_of_day + ']' : '') + '</span><br>' : ''}}
                        <a href="${{s.url}}" target="_blank" style="font-size: 11px; color: #4f46e5; font-weight: 600; text-decoration: none;">View Original</a>
                    </div>
                `);
            }});

            // Adjust viewport to show all geographic locations
            const group = new L.featureGroup();
            cluster.samples.forEach(s => {{
                group.addLayer(L.marker([s.lat, s.lon]));
            }});
            cluster.h3_centroids.forEach(pt => {{
                group.addLayer(L.marker([pt[0], pt[1]]));
            }});
            if (group.getLayers().length > 0) {{
                map.fitBounds(group.getBounds().pad(0.15));
            }}
        }}
        
        function loadMore() {{
            renderLimit += 50;
            renderResults(true);
        }}

        function resetAndRender() {{
            renderLimit = 50;
            renderResults(false);
        }}

        let locationBbox = null;

        function getHaversineDistance(lat1, lon1, lat2, lon2) {{
            const R = 6371; // Radius of Earth in km
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = 
                Math.sin(dLat/2) * Math.sin(dLat/2) +
                Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
                Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }}

        async function searchLocation() {{
            const query = document.getElementById('locationInput').value.trim();
            const statusLabel = document.getElementById('locationStatus');
            
            if (!query) {{
                locationBbox = null;
                statusLabel.innerHTML = "";
                statusLabel.style.color = "";
                filterDataCombined();
                return;
            }}
            
            statusLabel.innerHTML = "⏳ Geocoding location...";
            statusLabel.style.color = "var(--text-muted)";

            // Offline continent overrides
            const locClean = query.toLowerCase().trim();
            const CONTINENT_BOUNDS = {{
                "africa": [-35.0, 38.0, -26.0, 52.0],
                "europe": [35.0, 72.0, -25.0, 45.0],
                "asia": [1.0, 77.0, 26.0, 180.0],
                "north america": [7.0, 85.0, -168.0, -52.0],
                "south america": [-56.0, 13.0, -82.0, -34.0],
                "oceania": [-48.0, 0.0, 110.0, 180.0],
                "australia": [-48.0, -10.0, 110.0, 155.0],
                "antarctica": [-90.0, -60.0, -180.0, 180.0]
            }};

            if (CONTINENT_BOUNDS[locClean]) {{
                const bbox = CONTINENT_BOUNDS[locClean];
                locationBbox = {{
                    min_lat: bbox[0],
                    max_lat: bbox[1],
                    min_lon: bbox[2],
                    max_lon: bbox[3],
                    lat: (bbox[0] + bbox[1]) / 2,
                    lon: (bbox[2] + bbox[3]) / 2,
                    display_name: query.charAt(0).toUpperCase() + query.slice(1)
                }};
                statusLabel.innerHTML = `📍 Found: ${{locationBbox.display_name}}`;
                statusLabel.style.color = "var(--primary)";
                filterDataCombined();
                return;
            }}
            
            try {{
                const url = `https://nominatim.openstreetmap.org/search?q=${{encodeURIComponent(query)}}&format=json&limit=1`;
                const response = await fetch(url, {{
                    headers: {{ 'User-Agent': 'GeoRAG-Cluster-Visualizer/1.0' }}
                }});
                
                if (!response.ok) throw new Error("OSM Nominatim API request failed.");
                const results = await response.json();
                
                if (results && results.length > 0) {{
                    const place = results[0];
                    const bbox = place.boundingbox.map(Number); // [minlat, maxlat, minlon, maxlon]
                    locationBbox = {{
                        min_lat: bbox[0],
                        max_lat: bbox[1],
                        min_lon: bbox[2],
                        max_lon: bbox[3],
                        lat: Number(place.lat),
                        lon: Number(place.lon),
                        display_name: place.display_name
                    }};
                    statusLabel.innerHTML = `📍 Found: ${{place.display_name.split(',')[0]}}`;
                    statusLabel.style.color = "var(--primary)";
                }} else {{
                    locationBbox = null;
                    statusLabel.innerHTML = "❌ Location not found";
                    statusLabel.style.color = "red";
                }}
            }} catch (err) {{
                console.error(err);
                locationBbox = null;
                statusLabel.innerHTML = "❌ Connection error";
                statusLabel.style.color = "red";
            }}
            
            filterDataCombined();
        }}

        function latLngToH3Cell(lat, lon, res) {{
            if (typeof h3 !== 'undefined') {{
                if (typeof h3.latLngToCell === 'function') {{
                    return h3.latLngToCell(lat, lon, res);
                }} else if (typeof h3.geoToH3 === 'function') {{
                    return h3.geoToH3(lat, lon, res);
                }}
            }}
            return null;
        }}

        function getH3CellsInBbox(bbox, resolution = 4) {{
            const cells = new Set();
            const latStep = (bbox.max_lat - bbox.min_lat) / 8;
            const lonStep = (bbox.max_lon - bbox.min_lon) / 8;
            
            for (let i = 0; i <= 8; i++) {{
                for (let j = 0; j <= 8; j++) {{
                    const lat = bbox.min_lat + i * latStep;
                    const lon = bbox.min_lon + j * lonStep;
                    const cell = latLngToH3Cell(lat, lon, resolution);
                    if (cell) {{
                        cells.add(cell);
                    }}
                }}
            }}
            return Array.from(cells);
        }}

        function filterDataCombined() {{
            const searchVal = document.getElementById('searchInput').value.toLowerCase();
            const parentVal = document.getElementById('parentSelect').value;
            const minDate = document.getElementById('minDateInput').value;
            const maxDate = document.getElementById('maxDateInput').value;
            const seasonVal = document.getElementById('seasonSelect').value;
            const todVal = document.getElementById('todSelect').value;
            
            // 1. Text Search filtering
            let temp = data;
            if (searchVal) {{
                temp = temp.filter(c => 
                    c.id.toString().includes(searchVal) || 
                    c.label.toLowerCase().includes(searchVal) ||
                    (c.description && c.description.toLowerCase().includes(searchVal))
                );
            }}
            
            // 2. Parent Category filtering
            if (parentVal) {{
                temp = temp.filter(c => c.parent_label === parentVal);
            }}
            
            // 3. Location bounding box / proximity filtering
            if (locationBbox) {{
                let queryCells = [];
                if (typeof h3 !== 'undefined') {{
                    queryCells = getH3CellsInBbox(locationBbox, 4);
                }}
                
                temp = temp.filter(c => {{
                    // Check H3 overlap if H3 library is loaded and cluster has coverage
                    if (queryCells.length > 0 && c.h3_res4 && c.h3_res4.length > 0) {{
                        const hasOverlap = c.h3_res4.some(cell => queryCells.includes(cell));
                        if (hasOverlap) return true;
                    }}
                    
                    // Fallback to coordinates bounding box check
                    // Check if cluster center is inside geocoded bounding box
                    const centerInBbox = (
                        c.center_lat >= locationBbox.min_lat &&
                        c.center_lat <= locationBbox.max_lat &&
                        c.center_lon >= locationBbox.min_lon &&
                        c.center_lon <= locationBbox.max_lon
                    );
                    if (centerInBbox) return true;
                    
                    // Check if any cluster sample coordinates are inside bounding box
                    const sampleInBbox = c.samples.some(s => 
                        s.lat >= locationBbox.min_lat &&
                        s.lat <= locationBbox.max_lat &&
                        s.lon >= locationBbox.min_lon &&
                        s.lon <= locationBbox.max_lon
                    );
                    if (sampleInBbox) return true;

                    // Check if any density centroid points are inside bounding box
                    const h3InBbox = c.h3_centroids.some(pt => 
                        pt[0] >= locationBbox.min_lat &&
                        pt[0] <= locationBbox.max_lat &&
                        pt[1] >= locationBbox.min_lon &&
                        pt[1] <= locationBbox.max_lon
                    );
                    if (h3InBbox) return true;
                    
                    // Proximity fallback: Check if cluster center is within 60km of search point
                    const distance = getHaversineDistance(c.center_lat, c.center_lon, locationBbox.lat, locationBbox.lon);
                    if (distance <= 60.0) return true;
                    
                    return false;
                }});
            }}

            // 4. Season filtering
            if (seasonVal) {{
                temp = temp.filter(c => 
                    c.samples.some(s => s.season === seasonVal)
                );
            }}

            // 5. Time of Day filtering
            if (todVal) {{
                temp = temp.filter(c => 
                    c.samples.some(s => s.time_of_day === todVal)
                );
            }}

            // 6. Date Range filtering
            if (minDate || maxDate) {{
                temp = temp.filter(c => 
                    c.samples.some(s => {{
                        if (!s.captured_at) return false;
                        const dateStr = s.captured_at.substring(0, 10);
                        const isAfterMin = !minDate || dateStr >= minDate;
                        const isBeforeMax = !maxDate || dateStr <= maxDate;
                        return isAfterMin && isBeforeMax;
                    }})
                );
            }}
            
            filteredData = temp;
            resetAndRender();
        }}

        // Populate parent category filter dropdown dynamically
        const parentSelect = document.getElementById('parentSelect');
        const parents = [...new Set(data.map(c => c.parent_label).filter(Boolean))].sort();
        parents.forEach(p => {{
            const opt = document.createElement('option');
            opt.value = p;
            opt.textContent = p;
            parentSelect.appendChild(opt);
        }});

        // Initial render
        resetAndRender();
    </script>
</body>
</html>
"""

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Scalable Dashboard saved to: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an HTML grid of representative samples for each cluster.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the clustered .pkl file.")
    parser.add_argument("--out", type=str, default="cluster_samples.html", help="Output HTML file name.")
    parser.add_argument("--top_n", type=int, default=6, help="Number of samples to show per cluster.")
    args = parser.parse_args()

    create_sample_grid(args.pkl, args.out, args.top_n)
