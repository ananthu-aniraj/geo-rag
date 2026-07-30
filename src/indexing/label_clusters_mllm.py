import argparse
import base64
import os
import pickle
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from sklearn.preprocessing import normalize
from transformers import AutoModel
from urllib3.util import Retry

from src.utils.io import get_parquet_writer, load_dataframe, save_dataframe

# Shared LULC Vocabularies
from src.utils.lulc_vocab import MAN_MADE_LULC_VOCAB, NATURAL_LULC_VOCAB

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'

# Global connection pooled session configuration for thread-safe high-throughput downloads
http_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=64,
    pool_maxsize=64,
    max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
)
http_session.mount("https://", _adapter)
http_session.mount("http://", _adapter)


def resize_image_aspect(img, target_max=448):
    """Resizes a PIL image maintaining aspect ratio such that the largest dimension is target_max."""
    w, h = img.size
    if max(w, h) <= target_max:
        return img

    if w > h:
        new_w = target_max
        new_h = int(h * (target_max / w))
    else:
        new_h = target_max
        new_w = int(w * (target_max / h))
    resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', 1)
    return img.resize((new_w, new_h), resample)


def load_image(url, target_max=448, image_root_dir=None):
    """Loads an image from local path or downloads from Mapillary, Kartaview, or standard URL."""
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
        try:
            img = Image.open(resolved_path).convert("RGB")
            return resize_image_aspect(img, target_max)
        except Exception as e:
            print(f"Error loading local image {resolved_path}: {e}")
            return None
    try:
        if url.startswith("mapillary://"):
            orig_id = url.split("://")[1].strip()
            if orig_id.endswith('.0'):
                orig_id = orig_id[:-2]
            api_url = f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
            res = http_session.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
            else:
                return None
        elif url.startswith("kartaview://"):
            orig_id = url.split("://")[1].strip()
            if orig_id.endswith('.0'):
                orig_id = orig_id[:-2]
            api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
            res = http_session.get(api_url, timeout=10)
            if res.status_code == 200:
                data = res.json().get("result", {}).get("data", {})
                url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")
            else:
                return None

        if not url:
            return None

        response = http_session.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return resize_image_aspect(img, target_max)
    except Exception as e:
        print(f"Error loading image URL {url}: {e}")
    return None


def query_vlm_openai_api(image_base64, prompt_text, model_name, endpoint_url):
    """Queries an OpenAI-compatible VLM server (sglang, vllm, ollama) using HTTP requests."""
    headers = {
        "Content-Type": "application/json"
    }
    content = [{"type": "text", "text": prompt_text}]
    if image_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
        
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.2
    }

    if not endpoint_url.endswith("/v1/chat/completions"):
        endpoint_url = endpoint_url.rstrip("/") + "/v1/chat/completions"

    try:
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error from VLM API ({response.status_code}): {response.text}")
            return ""
    except Exception as e:
        print(f"Failed to query VLM API at {endpoint_url}: {e}")
        return ""


def build_prompt_templates(representative_item, prompt_step1_template, prompt_step2_template, lulc_list_str):
    lat = representative_item.get('Latitude', 'N/A')
    lon = representative_item.get('Longitude', 'N/A')
    location = f"Lat {lat}, Lon {lon}"
    country = representative_item.get('country', 'Unknown')
    if not country or pd.isna(country):
        country = 'Unknown'
    continent = representative_item.get('continent', 'Unknown')
    if not continent or pd.isna(continent):
        continent = 'Unknown'
    season = representative_item.get('Season', 'Unknown')
    if not season or pd.isna(season):
        season = 'Unknown'
    time_of_day = representative_item.get('Time_Of_Day', 'Unknown')
    if not time_of_day or pd.isna(time_of_day):
        time_of_day = 'Unknown'
    koppen_code = representative_item.get('Koppen_Code', 'Unknown')
    if not koppen_code or pd.isna(koppen_code):
        koppen_code = 'Unknown'
    koppen_desc = representative_item.get('Koppen_Desc', 'Unknown')
    if not koppen_desc or pd.isna(koppen_desc):
        koppen_desc = 'Unknown'

    step2_prompt = prompt_step2_template.format(
        location=location,
        country=country,
        continent=continent,
        season=season,
        time_of_day=time_of_day,
        koppen_code=koppen_code,
        koppen_desc=koppen_desc,
        visual_description="{visual_description}",
        lulc_list=lulc_list_str
    )
    return prompt_step1_template, step2_prompt


def label_clusters_mllm_batched(tasks, model_name, endpoint_url, chunk_size=128, img_max_dim=448, image_root_dir=None):
    """Runs VLM labeling in chunks to utilize batch inference via OpenAI-compatible API."""
    results = {}

    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i: i + chunk_size]
        print(f"Processing batch {i // chunk_size + 1}/{(len(tasks) - 1) // chunk_size + 1} ({len(chunk)} clusters)...")

        images_base64 = {}

        def prepare_image(task):
            cid = task['cid']
            img = load_image(task['img_url'], target_max=img_max_dim, image_root_dir=image_root_dir)
            if img is not None:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images_base64[cid] = img_str

        with ThreadPoolExecutor(max_workers=32) as executor:
            executor.map(prepare_image, chunk)

        valid_chunk = [t for t in chunk if t['cid'] in images_base64]
        if not valid_chunk:
            continue

        batch_responses = {}

        def query_task(t):
            cid = t['cid']
            # Step 1: Vision query (needs image)
            desc_text = query_vlm_openai_api(images_base64[cid], t['prompt_step1'], model_name, endpoint_url)
            if not desc_text:
                batch_responses[cid] = None
                return
            
            # Step 2: Text-only query (no image)
            step2_prompt = t['prompt_step2_template'].format(visual_description=desc_text)
            classification_text = query_vlm_openai_api(None, step2_prompt, model_name, endpoint_url)
            batch_responses[cid] = (desc_text, classification_text)

        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(query_task, valid_chunk)

        for t in chunk:
            cid = t['cid']
            if cid in batch_responses and batch_responses[cid]:
                desc_text, response_text = batch_responses[cid]
                label = "Unlabeled"
                description = response_text
                if "LABEL:" in response_text and "DESCRIPTION:" in response_text:
                    parts = response_text.split("DESCRIPTION:")
                    label = parts[0].replace("LABEL:", "").strip()
                    description = parts[1].strip()
                elif "LABEL:" in response_text:
                    label = response_text.replace("LABEL:", "").strip()

                label = label.replace("**", "").replace("*", "").replace("`", "").strip()
                label = label.strip('"\'*#-\t ')
                results[cid] = (label, description, desc_text)
            else:
                results[cid] = ("Error Labeling", "Inference failed or returned empty response.", "")

    return results


def label_clusters_zeroshot(centroids, text_features, categories, top_k=3):
    """Performs zero-shot labeling of cluster centroids using given text features."""
    centroids_norm = normalize(centroids)
    sims = np.dot(centroids_norm, text_features.T)
    results = []
    for i in range(len(centroids)):
        top_indices = np.argsort(sims[i])[::-1][:top_k]
        top_labels = [categories[idx] for idx in top_indices]
        results.append(", ".join(top_labels))
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal LLM Auto-Labeling for Clustered Geo-Images.")
    parser.add_argument("--in", "--input", dest="input_file", type=str, required=True, help="Path to input clustered Parquet or .pkl file.")
    parser.add_argument("--out", "--output", dest="output_file", type=str, default=None, help="Output path (defaults to overwriting input file).")
    parser.add_argument("--label_method", type=str, choices=["mllm", "zeroshot"], default="mllm", help="Labeling method.")
    parser.add_argument("--mllm_model", type=str, default="gemma4:e4b", help="VLM model identifier.")
    parser.add_argument("--mllm_backend", type=str, choices=["ollama", "sglang"], default="ollama", help="Backend server type.")
    parser.add_argument("--mllm_endpoint", type=str, default=None, help="Custom API URL for the VLM server.")
    parser.add_argument("--chunk_size", type=int, default=128, help="Batch chunk size for VLM API requests.")
    parser.add_argument("--img_max_dim", type=int, default=672, help="Target max dimension for images.")
    parser.add_argument("--image_root_dir", type=str, nargs="+", default=None, help="Optional root directories for local images.")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file

    # Load dataset
    print(f"Loading clustered dataset from {args.input_file}...")
    is_pkl = args.input_file.endswith('.pkl')
    if is_pkl:
        with open(args.input_file, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        embeddings = np.vstack(df['embedding'].values).astype(np.float32)
    else:
        try:
            parquet_file = pq.ParquetFile(args.input_file)
            metadata_cols = [c for c in parquet_file.schema_arrow.names if c != 'embedding']
            df = load_dataframe(args.input_file, columns=metadata_cols)
        except Exception:
            df = load_dataframe(args.input_file)

        print("Loading raw embedding matrix temporarily to compute centroids and representative images...")
        t0 = time.time()
        table = pq.read_table(args.input_file, columns=["embedding"])
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
        print(f" -> Temporarily loaded raw embedding matrix in {time.time() - t0:.2f}s.")
    if 'Latitude' in df.columns:
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    if 'Longitude' in df.columns:
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    if 'cluster_id' not in df.columns:
        raise ValueError("Input file missing 'cluster_id'. Please run cluster_images_global.py first.")

    child_ids = df['cluster_id'].values
    unique_child_ids = np.sort(np.unique(child_ids[child_ids >= 0]))
    k_clusters = len(unique_child_ids)

    print(f"Found {len(df):,} items across {k_clusters:,} child clusters.")

    # Re-compute child centroids
    d = embeddings.shape[1]
    embeddings_norm = normalize(embeddings).astype(np.float32)
    raw_centroids = np.zeros((k_clusters, d), dtype=np.float32)
    valid_mask = (child_ids >= 0)
    np.add.at(raw_centroids, child_ids[valid_mask], embeddings_norm[valid_mask])
    counts = np.bincount(child_ids[valid_mask], minlength=k_clusters)
    valid_counts = counts > 0
    raw_centroids[valid_counts] /= counts[valid_counts, None]

    # Parent cluster mapping
    has_parents = 'parent_cluster_id' in df.columns
    if has_parents:
        parent_ids = df.groupby('cluster_id')['parent_cluster_id'].first().to_dict()
        unique_parents = sorted(set(parent_ids.values()))
        k_parents = len(unique_parents)
        parent_centroids = np.zeros((k_parents, d), dtype=np.float32)
        for cid, pid in parent_ids.items():
            if cid < len(raw_centroids) and pid < k_parents:
                parent_centroids[pid] += raw_centroids[cid] * counts[cid]
        parent_counts = np.bincount(list(parent_ids.values()), minlength=k_parents)
        valid_p_counts = parent_counts > 0
        parent_centroids[valid_p_counts] /= parent_counts[valid_p_counts, None]
    else:
        parent_ids = {}
        k_parents = 0
        parent_centroids = np.array([])

    # Precompute representative image indices to free embeddings memory
    print("Finding closest representative images for each cluster...")
    parent_rep_indices = {}
    child_rep_indices = {}
    
    if has_parents and k_parents > 0:
        centroids_norm_hac = normalize(raw_centroids)
        for pid in range(k_parents):
            cids_in_parent = [cid for cid, p in parent_ids.items() if p == pid]
            if not cids_in_parent:
                continue
            p_centroid = parent_centroids[pid]
            p_centroid_norm = p_centroid / (np.linalg.norm(p_centroid) + 1e-9)
            child_embs = centroids_norm_hac[cids_in_parent]
            sims = np.dot(child_embs, p_centroid_norm)
            closest_child_cid = cids_in_parent[np.argmax(sims)]
            
            indices = np.where(child_ids == closest_child_cid)[0]
            if len(indices) == 0:
                continue
            child_centroid = raw_centroids[closest_child_cid]
            child_centroid_norm = child_centroid / (np.linalg.norm(child_centroid) + 1e-9)
            cluster_embs = embeddings_norm[indices]
            img_sims = np.dot(cluster_embs, child_centroid_norm)
            closest_img_idx = indices[np.argmax(img_sims)]
            parent_rep_indices[pid] = closest_img_idx

    for cid in range(k_clusters):
        indices = np.where(child_ids == cid)[0]
        if len(indices) == 0:
            continue
        centroid = raw_centroids[cid]
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        cluster_embs = embeddings_norm[indices]
        sims = np.dot(cluster_embs, centroid_norm)
        closest_idx = indices[np.argmax(sims)]
        child_rep_indices[cid] = closest_idx

    # Release heavy embedding matrices immediately
    del embeddings
    del embeddings_norm
    import gc
    gc.collect()
    print(" -> Released embedding matrices from memory to conserve RAM during VLM labeling.")

    noise_category = "None of the above / Noise"
    noise_prompt = "Noisy image, indoor scene, closeup object, selfie, text/graphic, or unrelated non-geographic photo."
    all_categories = list(NATURAL_LULC_VOCAB.keys()) + list(MAN_MADE_LULC_VOCAB.keys()) + [noise_category]
    all_prompts = list(NATURAL_LULC_VOCAB.values()) + list(MAN_MADE_LULC_VOCAB.values()) + [noise_prompt]

    parent_labels = {}
    parent_descriptions = {}
    parent_visual_descriptions = {}
    cluster_labels = {}
    cluster_descriptions = {}
    cluster_visual_descriptions = {}

    if args.label_method == "mllm":
        endpoint = args.mllm_endpoint
        if not endpoint:
            endpoint = "http://localhost:30000" if args.mllm_backend == "sglang" else "http://localhost:11434"

        # Pre-flight check
        print(f"Connecting to MLLM API server at {endpoint}...")
        server_running = False
        try:
            test_url = endpoint.rstrip("/") + ("/v1/models" if args.mllm_backend == "sglang" else "/api/tags")
            req = urllib.request.Request(test_url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    server_running = True
        except Exception:
            pass

        if not server_running:
            print(f"[WARNING] VLM server not reachable at {endpoint}. Falling back to zero-shot TIPSv2 labeling.")
            args.label_method = "zeroshot"

    if args.label_method == "mllm":
        # Load step 1 and step 2 prompt templates
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        prompt_step1_path = os.path.join(root_dir, "prompts", "shared", "prompt_step1.txt")
        prompt_step2_path = os.path.join(root_dir, "prompts", "shared", "prompt_step2.txt")
        if not os.path.exists(prompt_step1_path):
            prompt_step1_path = "prompts/shared/prompt_step1.txt"
        if not os.path.exists(prompt_step2_path):
            prompt_step2_path = "prompts/shared/prompt_step2.txt"

        print(f"Loading Step 1 prompt template from {prompt_step1_path}...")
        with open(prompt_step1_path, 'r', encoding='utf-8') as f:
            prompt_step1_template = f.read()

        print(f"Loading Step 2 prompt template from {prompt_step2_path}...")
        with open(prompt_step2_path, 'r', encoding='utf-8') as f:
            prompt_step2_template = f.read()

        lulc_list_str = "\n".join([f"- {k}: {v}" for k, v in zip(all_categories, all_prompts)])

        # Label Parent Clusters
        if has_parents and k_parents > 0:
            print(f"\nPreparing {k_parents} parent cluster tasks for MLLM labeling...")
            parent_tasks = []
            for pid in range(k_parents):
                closest_img_idx = parent_rep_indices.get(pid)
                if closest_img_idx is None:
                    continue
                representative_item = df.iloc[closest_img_idx]
                img_url = representative_item['Image_URL']

                photo_id = representative_item.get('Photo_ID')
                platform = str(representative_item.get('Platform', '')).lower()
                if photo_id:
                    photo_str = str(photo_id).strip()
                    if photo_str.endswith('.0'):
                        photo_str = photo_str[:-2]
                    if platform == 'mapillary' or 'mapillary' in img_url or 'fbcdn.net' in img_url:
                        img_url = f"mapillary://{photo_str}"
                    elif platform == 'kartaview' or 'kartaview' in img_url or 'openstreetcam' in img_url:
                        img_url = f"kartaview://{photo_str}"

                # Build templates
                p1_text, p2_text = build_prompt_templates(representative_item, prompt_step1_template, prompt_step2_template, lulc_list_str)
                parent_tasks.append({
                    "cid": pid,
                    "img_url": img_url,
                    "prompt_step1": p1_text,
                    "prompt_step2_template": p2_text
                })

            parent_results = label_clusters_mllm_batched(
                parent_tasks, args.mllm_model, endpoint,
                chunk_size=args.chunk_size, img_max_dim=args.img_max_dim,
                image_root_dir=args.image_root_dir
            )
            for pid, (lbl, desc, desc_vis) in parent_results.items():
                parent_labels[pid] = lbl
                parent_descriptions[pid] = desc
                parent_visual_descriptions[pid] = desc_vis

        # Label Child Clusters
        print(f"\nPreparing {k_clusters} child cluster tasks for MLLM labeling...")
        tasks = []
        for cid in range(k_clusters):
            closest_idx = child_rep_indices.get(cid)
            if closest_idx is None:
                continue
            representative_item = df.iloc[closest_idx]
            img_url = representative_item['Image_URL']

            photo_id = representative_item.get('Photo_ID')
            platform = str(representative_item.get('Platform', '')).lower()
            if photo_id:
                photo_str = str(photo_id).strip()
                if photo_str.endswith('.0'):
                    photo_str = photo_str[:-2]
                if platform == 'mapillary' or 'mapillary' in img_url or 'fbcdn.net' in img_url:
                    img_url = f"mapillary://{photo_str}"
                elif platform == 'kartaview' or 'kartaview' in img_url or 'openstreetcam' in img_url:
                    img_url = f"kartaview://{photo_str}"

            # Build templates
            p1_text, p2_text = build_prompt_templates(representative_item, prompt_step1_template, prompt_step2_template, lulc_list_str)
            tasks.append({
                "cid": cid,
                "img_url": img_url,
                "prompt_step1": p1_text,
                "prompt_step2_template": p2_text
            })

        results = label_clusters_mllm_batched(
            tasks, args.mllm_model, endpoint,
            chunk_size=args.chunk_size, img_max_dim=args.img_max_dim,
            image_root_dir=args.image_root_dir
        )
        for cid, (lbl, desc, desc_vis) in results.items():
            cluster_labels[cid] = lbl
            cluster_descriptions[cid] = desc
            cluster_visual_descriptions[cid] = desc_vis

    if args.label_method == "zeroshot":
        print("Encoding zero-shot LULC taxonomy prompts using TIPSv2...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).to(device)
        model.eval()
        with torch.no_grad():
            all_features = normalize(model.encode_text(all_prompts).cpu().numpy())
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if has_parents and k_parents > 0:
            p_labels_list = label_clusters_zeroshot(parent_centroids, all_features, all_categories)
            parent_labels = {pid: lbl for pid, lbl in enumerate(p_labels_list)}

        c_labels_list = label_clusters_zeroshot(raw_centroids, all_features, all_categories)
        cluster_labels = {cid: lbl for cid, lbl in enumerate(c_labels_list)}

    print("\nUpdating metadata with labels and descriptions...")
    
    # Initialize dictionaries if not present to avoid NameError
    if 'cluster_visual_descriptions' not in locals():
        cluster_visual_descriptions = {}
    if 'parent_visual_descriptions' not in locals():
        parent_visual_descriptions = {}
    if 'parent_labels' not in locals():
        parent_labels = {}
    if 'parent_descriptions' not in locals():
        parent_descriptions = {}

    if cluster_labels:
        df['cluster_label'] = df['cluster_id'].map(cluster_labels).combine_first(df['cluster_label'] if 'cluster_label' in df.columns else pd.Series(dtype=str))
    if cluster_descriptions:
        df['cluster_description'] = df['cluster_id'].map(cluster_descriptions).combine_first(df['cluster_description'] if 'cluster_description' in df.columns else pd.Series(dtype=str))
    if cluster_visual_descriptions:
        df['visual_description'] = df['cluster_id'].map(cluster_visual_descriptions).combine_first(df['visual_description'] if 'visual_description' in df.columns else pd.Series(dtype=str))

    if has_parents:
        if parent_labels:
            df['parent_cluster_label'] = df['parent_cluster_id'].map(parent_labels).combine_first(df['parent_cluster_label'] if 'parent_cluster_label' in df.columns else pd.Series(dtype=str))
        if parent_descriptions:
            df['parent_cluster_description'] = df['parent_cluster_id'].map(parent_descriptions).combine_first(df['parent_cluster_description'] if 'parent_cluster_description' in df.columns else pd.Series(dtype=str))
        if parent_visual_descriptions:
            df['parent_visual_description'] = df['parent_cluster_id'].map(parent_visual_descriptions).combine_first(df['parent_visual_description'] if 'parent_visual_description' in df.columns else pd.Series(dtype=str))

    print(f"Saving labeled dataset to {args.output_file}...")
    if 'Latitude' in df.columns:
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    if 'Longitude' in df.columns:
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    if is_pkl:
        data = df.to_dict('records')
        with open(args.output_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        print(f"Streaming and merging labels to output parquet: {args.output_file}...")
        
        import pyarrow as pa
        
        pf_in = pq.ParquetFile(args.input_file)
        schema_in = pf_in.schema_arrow
        
        meta_updates = {
            'cluster_label': cluster_labels,
            'cluster_description': cluster_descriptions,
            'visual_description': cluster_visual_descriptions
        }
        if has_parents:
            meta_updates.update({
                'parent_cluster_label': parent_labels,
                'parent_cluster_description': parent_descriptions,
                'parent_visual_description': parent_visual_descriptions
            })
            
        new_fields = list(schema_in)
        for col_name in meta_updates.keys():
            if col_name not in schema_in.names:
                new_fields.append(pa.field(col_name, pa.string()))
        schema_out = pa.schema(new_fields)
        
        temp_out = args.output_file + ".tmp_label"
        try:
            with get_parquet_writer(temp_out, schema_out) as writer:
                for rg in range(pf_in.num_row_groups):
                    table = pf_in.read_row_group(rg)
                    df_rg = table.to_pandas()
                    
                    for col_name, mapping_dict in meta_updates.items():
                        if mapping_dict:
                            df_rg[col_name] = df_rg['cluster_id'].map(mapping_dict).combine_first(
                                df_rg[col_name] if col_name in df_rg.columns else pd.Series(dtype=str)
                            )
                        else:
                            if col_name not in df_rg.columns:
                                df_rg[col_name] = None
                                
                    df_rg_aligned = df_rg[schema_out.names]
                    tbl_out = pa.Table.from_pandas(df_rg_aligned, schema=schema_out, preserve_index=False)
                    writer.write_table(tbl_out)
                    
            if os.path.exists(temp_out):
                os.replace(temp_out, args.output_file)
        except Exception as e:
            if os.path.exists(temp_out):
                os.remove(temp_out)
            raise e

    print(f"✅ Labeling Complete! Output saved to {args.output_file}.")


if __name__ == "__main__":
    main()
