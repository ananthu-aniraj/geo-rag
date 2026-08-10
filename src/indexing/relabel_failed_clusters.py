import argparse
import base64
import os
import pickle
import sys
import time
import urllib.request
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import yaml
from PIL import Image
from sklearn.preprocessing import normalize

from src.utils.io import save_dataframe

# Shared LULC Vocabularies
from src.utils.lulc_vocab import MAN_MADE_LULC_VOCAB, NATURAL_LULC_VOCAB

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


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


def load_image(url, target_max=448, timeout=15, image_root_dir=None, photo_id=None, platform=None):
    """Loads an image from local path or downloads from Mapillary, Kartaview, or standard URL."""
    from src.utils.io import resolve_offline_image_path
    
    resolved_path = None
    if image_root_dir:
        resolved_path = resolve_offline_image_path(url, image_root_dir, photo_id, platform)
    
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
            orig_id = url.split("://")[1]
            api_url = f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
            res = requests.get(api_url, headers=headers, timeout=timeout)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
            else:
                return None
        elif url.startswith("kartaview://"):
            orig_id = url.split("://")[1]
            api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
            res = requests.get(api_url, timeout=timeout)
            if res.status_code == 200:
                data = res.json().get("result", {}).get("data", {})
                url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")
            else:
                return None

        if not url:
            return None

        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return resize_image_aspect(img, target_max)
    except Exception as e:
        print(f"Error loading image URL {url}: {e}")
    return None


def load_image_with_retry(url, target_max=448, timeout=15, max_retries=3, image_root_dir=None, photo_id=None, platform=None):
    """Wrapper around load_image that retries with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            img = load_image(url, target_max=target_max, timeout=timeout, image_root_dir=image_root_dir, photo_id=photo_id, platform=platform)
            if img is not None:
                return img
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff (1s, 2s, 4s...)
    return None


def query_vlm_openai_api(image_base64, prompt_text, model_name, endpoint_url, timeout=60):
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

    # Ensure endpoint ends with /v1/chat/completions
    if not endpoint_url.endswith("/v1/chat/completions"):
        endpoint_url = endpoint_url.rstrip("/") + "/v1/chat/completions"

    try:
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
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


def save_dataset(data, final_results, parent_results, out_path):
    """Helper to update labels in the data list and write them to the output file."""
    updated_clusters = set()
    updated_parents = set()
    row_update_count = 0

    # Map child results
    cluster_labels = {}
    cluster_descriptions = {}
    cluster_visual_descriptions = {}
    for cid, (lbl, desc, desc_vis) in final_results.items():
        if lbl != "Error Labeling":
            cluster_labels[cid] = lbl
            cluster_descriptions[cid] = desc
            cluster_visual_descriptions[cid] = desc_vis
            updated_clusters.add(cid)

    # Map parent results
    parent_labels = {}
    parent_descriptions = {}
    parent_visual_descriptions = {}
    for pid, (lbl, desc, desc_vis) in parent_results.items():
        if lbl != "Error Labeling":
            parent_labels[pid] = lbl
            parent_descriptions[pid] = desc
            parent_visual_descriptions[pid] = desc_vis
            updated_parents.add(pid)

    for item in data:
        cid = item.get('cluster_id')
        try:
            cid_int = int(cid) if (cid is not None and cid == cid) else None
        except (ValueError, TypeError):
            cid_int = None
            
        if cid_int is not None and cid_int in cluster_labels:
            item['cluster_label'] = cluster_labels[cid_int]
            item['cluster_description'] = cluster_descriptions[cid_int]
            if cid_int in cluster_visual_descriptions:
                item['visual_description'] = cluster_visual_descriptions[cid_int]
            row_update_count += 1
            
        pid = item.get('parent_cluster_id')
        try:
            pid_int = int(pid) if (pid is not None and pid == pid) else None
        except (ValueError, TypeError):
            pid_int = None
            
        if pid_int is not None and pid_int in parent_labels:
            item['parent_cluster_label'] = parent_labels[pid_int]
            if pid_int in parent_descriptions:
                item['parent_cluster_description'] = parent_descriptions[pid_int]
            if int(pid_int) in parent_visual_descriptions:
                item['parent_visual_description'] = parent_visual_descriptions[pid_int]
            row_update_count += 1

    if out_path.endswith('.pkl'):
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        df = pd.DataFrame(data)
        if 'Latitude' in df.columns:
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        if 'Longitude' in df.columns:
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        save_dataframe(df, out_path)

    print(
        f"  -> Checkpoint: Saved {row_update_count} rows across {len(updated_clusters)} child clusters and {len(updated_parents)} parent clusters to {out_path}.")


def main():
    # 1. Load Defaults from params.yaml if available
    default_mllm_model = "google/gemma-4-E4B-it"
    default_mllm_backend = "sglang"
    default_output_dir = ""
    default_base_name = "geo_space"
    default_k = 40000

    if os.path.exists("params.yaml"):
        try:
            with open("params.yaml", "r") as f:
                params = yaml.safe_load(f)
                if 'pipeline' in params:
                    pipe = params['pipeline']
                    default_mllm_model = pipe.get('mllm_model', default_mllm_model)
                    default_mllm_backend = pipe.get('mllm_backend', default_mllm_backend)
                    default_output_dir = pipe.get('output_dir', default_output_dir)
                    default_base_name = pipe.get('base_name', default_base_name)
                    default_k = pipe.get('k_clusters', default_k)
        except Exception as e:
            print(f"Warning: Could not read params.yaml: {e}")

    default_in = ""
    if default_output_dir:
        default_in = os.path.join(default_output_dir, f"{default_base_name}_clustered_k_{default_k}.parquet")

    # 2. Parse CLI Arguments
    parser = argparse.ArgumentParser(description="Re-label failed clusters sequentially where images failed to load.")
    parser.add_argument("--file", "--in", dest="file", type=str, default=default_in, required=not bool(default_in),
                        help="Path to the clustered .pkl or .parquet file.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path to save the updated dataset. Defaults to overwriting the input file.")
    parser.add_argument("--cluster_ids", type=int, nargs="+", default=None,
                        help="Specific cluster IDs to force re-labeling. If not provided, automatically detects failed ones.")
    parser.add_argument("--mllm_model", type=str, default=default_mllm_model,
                        help="VLM model identifier.")
    parser.add_argument("--mllm_backend", type=str, choices=["ollama", "sglang"], default=default_mllm_backend,
                        help="Backend server type. Sets default endpoint port.")
    parser.add_argument("--mllm_endpoint", type=str, default=None,
                        help="Custom API URL for the VLM server.")
    parser.add_argument("--img_max_dim", type=int, default=672,
                        help="Target maximum dimension to resize images before VLM processing.")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Number of times to retry downloading an image.")
    parser.add_argument("--fallback_depth", type=int, default=20,
                        help="Number of top closest images in a cluster to check if the closest one fails to download.")
    parser.add_argument("--timeout", type=int, default=15,
                        help="Timeout in seconds for image download HTTP requests.")
    parser.add_argument("--image_root_dir", type=str, nargs="+", default=None,
                        help="Optional root directories containing local images (for offline datasets).")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Interval of successfully re-labeled clusters at which to save intermediate checkpoints.")
    parser.add_argument("--representation_type", type=str, default="cls", choices=["cls", "avg_patch", "cls_avg_patch"],
                        help="Type of representation embedding to load (cls, avg_patch, or cls_avg_patch).")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16"],
                        help="Stored precision of companion binary file (float32 or float16).")
    args = parser.parse_args()

    if not args.out:
        args.out = args.file

    print(f"Loading dataset from {args.file}...")
    if args.file.endswith('.pkl'):
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
    else:
        from src.utils.io import load_dataset_with_clusters
        df = load_dataset_with_clusters(args.file)
        if 'Latitude' in df.columns:
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        if 'Longitude' in df.columns:
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        data = df.to_dict('records')

    if not data:
        print("No data found.")
        sys.exit(1)

    print(f"Total records loaded: {len(data)}")

    # 3. Identify clusters needing re-labeling
    target_cluster_ids = set()
    target_parent_ids = set()
    
    has_parents = len(data) > 0 and 'parent_cluster_id' in data[0]

    for item in data:
        label = item.get('cluster_label')
        cid = item.get('cluster_id')
        if cid is not None:
            if label is None or label in ("Error Labeling", "Unlabeled", "", "None"):
                target_cluster_ids.add(int(cid))
                
        if has_parents:
            p_label = item.get('parent_cluster_label')
            pid = item.get('parent_cluster_id')
            if pid is not None:
                if p_label is None or p_label in ("Error Labeling", "Unlabeled Parent", "", "None", "Unlabeled"):
                    target_parent_ids.add(int(pid))

    if args.cluster_ids:
        # Override with forced list for child clusters
        target_cluster_ids = set(args.cluster_ids)
        print(f"Forced re-labeling for specific cluster IDs: {sorted(list(target_cluster_ids))}")
    else:
        print(f"Automatically detected {len(target_cluster_ids)} failed or unlabeled child clusters.")
        if has_parents:
            print(f"Automatically detected {len(target_parent_ids)} failed or unlabeled parent clusters.")

    if not target_cluster_ids and not target_parent_ids:
        print("No failed or unlabeled child or parent clusters found! Script will exit without modifying the file.")
        sys.exit(0)

    # 4. Pre-flight VLM Server Check
    endpoint = args.mllm_endpoint
    if not endpoint:
        if args.mllm_backend == "sglang":
            endpoint = "http://localhost:30000"
        else:
            endpoint = "http://localhost:11434"

    print(f"Pre-flight check: Verifying VLM server is running at {endpoint}...")
    server_running = False
    try:
        test_url = endpoint.rstrip("/")
        if args.mllm_backend == "sglang":
            test_url += "/v1/models"
        else:
            test_url += "/api/tags"

        req = urllib.request.Request(test_url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                server_running = True
                print("VLM server connection successful!")
    except Exception as e:
        print(f"Error checking VLM server: {e}")

    if not server_running:
        print(f"\n[ERROR] VLM server is not reachable at {endpoint}!")
        print("Please make sure your VLM server is running before executing this script.")
        sys.exit(1)

    # 5. Extract Cluster Mapping and Embeddings
    print("Extracting cluster assignments...")
    cluster_ids = np.array([item['cluster_id'] for item in data])

    print("Extracting embeddings...")
    from src.utils.io import load_embeddings
    try:
        embeddings = load_embeddings(args.file, representation_type=args.representation_type).squeeze()
    except Exception as e:
        print(f"Warning: Failed to load decoupled embeddings directly: {e}. Attempting fallback load...")
        embeddings = load_embeddings(args.file, column='embedding', representation_type=args.representation_type).squeeze()

    if len(data) > 0 and 'embedding_idx' in data[0]:
        print("Aligning raw embedding matrix with decoupled metadata using 'embedding_idx'...")
        idx_vals = np.array([item['embedding_idx'] for item in data], dtype=int)
        embeddings = embeddings[idx_vals]

    print("Normalizing embeddings...")
    embeddings_norm = normalize(embeddings)

    # 6. Construct prompt templates
    noise_category = "None of the above / Noise"
    noise_prompt = "Noisy image, indoor scene, closeup object, selfie, text/graphic, or unrelated non-geographic photo."
    all_categories = list(NATURAL_LULC_VOCAB.keys()) + list(MAN_MADE_LULC_VOCAB.keys()) + [noise_category]
    all_prompts = list(NATURAL_LULC_VOCAB.values()) + list(MAN_MADE_LULC_VOCAB.values()) + [noise_prompt]
    lulc_list_str = "\n".join([f"- {k}: {v}" for k, v in zip(all_categories, all_prompts)])

    # Load step 1 and step 2 prompt templates
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    prompt_step1_path = os.path.join(root_dir, "prompts", "shared", "prompt_step1.txt")
    prompt_step2_path = os.path.join(root_dir, "prompts", "shared", "prompt_step2.txt")
    if not os.path.exists(prompt_step1_path):
        prompt_step1_path = "prompts/shared/prompt_step1.txt"
    if not os.path.exists(prompt_step2_path):
        prompt_step2_path = "prompts/shared/prompt_step2.txt"

    print(f"Loading Step 1 prompt template from {prompt_step1_path}")
    with open(prompt_step1_path, 'r', encoding='utf-8') as f:
        prompt_step1_template = f.read()

    print(f"Loading Step 2 prompt template from {prompt_step2_path}")
    with open(prompt_step2_path, 'r', encoding='utf-8') as f:
        prompt_step2_template = f.read()

    # Helper function to prepare task for a single cluster (downloading with retry and fallback)
    def prepare_cluster_task(cid):
        indices = np.where(cluster_ids == cid)[0]
        if len(indices) == 0:
            return None

        # Calculate cluster centroid in 768D space
        cluster_embs = embeddings_norm[indices]
        centroid = np.mean(cluster_embs, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)

        # Sort members by cosine similarity to the centroid (descending)
        sims = np.dot(cluster_embs, centroid_norm)
        sorted_idx_in_cluster = np.argsort(sims)[::-1]

        # Try downloading images starting from the closest
        for rank in range(min(args.fallback_depth, len(sorted_idx_in_cluster))):
            item_idx = indices[sorted_idx_in_cluster[rank]]
            item = data[item_idx]
            img_url = item['Image_URL']
            photo_id = item.get('Photo_ID')
            platform = str(item.get('Platform', '')).lower()

            # Resolve potentially expired Mapillary or Kartaview URLs dynamically
            if photo_id:
                if platform == 'mapillary' or 'mapillary' in img_url or 'fbcdn.net' in img_url:
                    img_url = f"mapillary://{photo_id}"
                elif platform == 'kartaview' or 'kartaview' in img_url or 'openstreetcam' in img_url:
                    img_url = f"kartaview://{photo_id}"

            img = load_image_with_retry(img_url, target_max=args.img_max_dim, timeout=args.timeout,
                                        max_retries=args.max_retries, image_root_dir=args.image_root_dir,
                                        photo_id=photo_id, platform=platform)

            if img is not None:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return {
                    "cid": cid,
                    "img_str": img_str,
                    "img_url": img_url,
                    "rank": rank,
                    "item": item
                }
        return None

    # Helper function to prepare task for a single parent cluster
    def prepare_parent_task(pid):
        # We need to map row indices to parent IDs
        parent_ids_arr = np.array([item.get('parent_cluster_id', -1) for item in data])
        indices = np.where(parent_ids_arr == pid)[0]
        if len(indices) == 0:
            return None

        # Calculate parent centroid directly from raw image embeddings belonging to this parent
        parent_embs = embeddings_norm[indices]
        centroid = np.mean(parent_embs, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)

        # Sort parent members by similarity to parent centroid
        sims = np.dot(parent_embs, centroid_norm)
        sorted_idx_in_parent = np.argsort(sims)[::-1]

        # Try downloading images starting from the closest
        for rank in range(min(args.fallback_depth, len(sorted_idx_in_parent))):
            item_idx = indices[sorted_idx_in_parent[rank]]
            item = data[item_idx]
            img_url = item['Image_URL']
            photo_id = item.get('Photo_ID')
            platform = str(item.get('Platform', '')).lower()

            if photo_id:
                if platform == 'mapillary' or 'mapillary' in img_url or 'fbcdn.net' in img_url:
                    img_url = f"mapillary://{photo_id}"
                elif platform == 'kartaview' or 'kartaview' in img_url or 'openstreetcam' in img_url:
                    img_url = f"kartaview://{photo_id}"

            img = load_image_with_retry(img_url, target_max=args.img_max_dim, timeout=args.timeout,
                                        max_retries=args.max_retries, image_root_dir=args.image_root_dir,
                                        photo_id=photo_id, platform=platform)

            if img is not None:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return {
                    "pid": pid,
                    "img_str": img_str,
                    "img_url": img_url,
                    "rank": rank,
                    "item": item
                }
        return None

    # 7. Query the VLM sequentially for child clusters
    final_results = {}
    sorted_failed_ids = sorted(list(target_cluster_ids))
    total_clusters = len(sorted_failed_ids)

    if total_clusters > 0:
        print(f"\nStarting sequential VLM inference for {total_clusters} failed child clusters (batch size 1)...")
        print("Press Ctrl+C at any time to interrupt and save your progress.")

        try:
            for idx, cid in enumerate(sorted_failed_ids):
                print(f"[{idx + 1}/{total_clusters}] Processing Child Cluster #{cid}...")

                task = prepare_cluster_task(cid)
                if not task:
                    print(f"  [FAILED] Child Cluster #{cid}: Failed to download representative images.")
                    continue

                # Build templates
                representative_item = task['item']
                p1_text, p2_text = build_prompt_templates(representative_item, prompt_step1_template, prompt_step2_template, lulc_list_str)

                # Step 1: Vision
                desc_text = query_vlm_openai_api(task['img_str'], p1_text, args.mllm_model, endpoint,
                                                 timeout=120)

                if desc_text:
                    # Step 2: Text
                    step2_prompt_formatted = p2_text.format(visual_description=desc_text)
                    response_text = query_vlm_openai_api(None, step2_prompt_formatted, args.mllm_model, endpoint,
                                                         timeout=120)
                else:
                    response_text = ""

                if response_text:
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

                    final_results[cid] = (label, description, desc_text)
                    print(f"  [SUCCESS] Child Cluster #{cid} labeled: '{label}'")
                else:
                    final_results[cid] = ("Error Labeling", "Inference failed or returned empty response.", "")
                    print(f"  [FAILED] Child Cluster #{cid}: VLM returned empty response.")

                if len(final_results) % args.save_interval == 0:
                    save_dataset(data, final_results, {}, args.out)

        except KeyboardInterrupt:
            print("\nSequential child cluster labeling interrupted by user.")

    # 8. Query the VLM sequentially for parent clusters
    parent_results = {}
    sorted_failed_parent_ids = sorted(list(target_parent_ids))
    total_parents = len(sorted_failed_parent_ids)

    if total_parents > 0:
        print(f"\nStarting sequential VLM inference for {total_parents} failed parent clusters (batch size 1)...")
        print("Press Ctrl+C at any time to interrupt and save your progress.")

        try:
            for idx, pid in enumerate(sorted_failed_parent_ids):
                print(f"[{idx + 1}/{total_parents}] Processing Parent Cluster #{pid}...")

                task = prepare_parent_task(pid)
                if not task:
                    print(f"  [FAILED] Parent Cluster #{pid}: Failed to download representative images.")
                    continue

                # Build templates
                representative_item = task['item']
                p1_text, p2_text = build_prompt_templates(representative_item, prompt_step1_template, prompt_step2_template, lulc_list_str)

                # Step 1: Vision
                desc_text = query_vlm_openai_api(task['img_str'], p1_text, args.mllm_model, endpoint,
                                                 timeout=120)

                if desc_text:
                    # Step 2: Text
                    step2_prompt_formatted = p2_text.format(visual_description=desc_text)
                    response_text = query_vlm_openai_api(None, step2_prompt_formatted, args.mllm_model, endpoint,
                                                         timeout=120)
                else:
                    response_text = ""

                if response_text:
                    label = "Unlabeled Parent"
                    description = response_text

                    if "LABEL:" in response_text and "DESCRIPTION:" in response_text:
                        parts = response_text.split("DESCRIPTION:")
                        label = parts[0].replace("LABEL:", "").strip()
                        description = parts[1].strip()
                    elif "LABEL:" in response_text:
                        label = response_text.replace("LABEL:", "").strip()

                    label = label.replace("**", "").replace("*", "").replace("`", "").strip()
                    label = label.strip('"\'*#-\t ')

                    parent_results[pid] = (label, description, desc_text)
                    print(f"  [SUCCESS] Parent Cluster #{pid} labeled: '{label}'")
                else:
                    parent_results[pid] = ("Error Labeling", "Inference failed or returned empty response.", "")
                    print(f"  [FAILED] Parent Cluster #{pid}: VLM returned empty response.")

                if len(parent_results) % args.save_interval == 0:
                    save_dataset(data, final_results, parent_results, args.out)

        except KeyboardInterrupt:
            print("\nSequential parent cluster labeling interrupted by user.")

    # Final Save
    if final_results or parent_results:
        print("\nSaving final results...")
        save_dataset(data, final_results, parent_results, args.out)
        print("Re-labeling session finished.")
    else:
        print("\nNo clusters were successfully re-labeled.")


if __name__ == "__main__":
    main()
