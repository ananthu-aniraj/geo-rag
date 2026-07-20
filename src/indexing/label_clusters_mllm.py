import pickle
import numpy as np
import argparse
import time
import os
import base64
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import normalize
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModel
import requests
from io import BytesIO
from PIL import Image

# Shared LULC Vocabularies
from src.utils.lulc_vocab import NATURAL_LULC_VOCAB, MAN_MADE_LULC_VOCAB

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
            res = requests.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
            else:
                return None
        elif url.startswith("kartaview://"):
            orig_id = url.split("://")[1].strip()
            if orig_id.endswith('.0'):
                orig_id = orig_id[:-2]
            api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
            res = requests.get(api_url, timeout=10)
            if res.status_code == 200:
                data = res.json().get("result", {}).get("data", {})
                url = data.get("fileurlLTh") or data.get("fileurlTh") or data.get("fileurl")
            else:
                return None

        if not url:
            return None

        response = requests.get(url, timeout=10)
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
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
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

        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(prepare_image, chunk)

        valid_chunk = [t for t in chunk if t['cid'] in images_base64]
        if not valid_chunk:
            continue

        batch_responses = {}

        def query_task(t):
            cid = t['cid']
            response_text = query_vlm_openai_api(images_base64[cid], t['prompt'], model_name, endpoint_url)
            batch_responses[cid] = response_text

        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(query_task, valid_chunk)

        for t in chunk:
            cid = t['cid']
            if cid in batch_responses and batch_responses[cid]:
                response_text = batch_responses[cid]
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
                results[cid] = (label, description)
            else:
                results[cid] = ("Error Labeling", "Inference failed or returned empty response.")

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
    if args.input_file.endswith('.pkl'):
        with open(args.input_file, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        embeddings = np.vstack(df['embedding'].values).astype(np.float32)
    else:
        try:
            parquet_file = pq.ParquetFile(args.input_file)
            metadata_cols = [c for c in parquet_file.schema_arrow.names if c != 'embedding']
            df = pd.read_parquet(args.input_file, columns=metadata_cols)
        except Exception as e:
            df = pd.read_parquet(args.input_file)

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

    noise_category = "None of the above / Noise"
    noise_prompt = "Noisy image, indoor scene, closeup object, selfie, text/graphic, or unrelated non-geographic photo."
    all_categories = list(NATURAL_LULC_VOCAB.keys()) + list(MAN_MADE_LULC_VOCAB.keys()) + [noise_category]
    all_prompts = list(NATURAL_LULC_VOCAB.values()) + list(MAN_MADE_LULC_VOCAB.values()) + [noise_prompt]

    parent_labels = {}
    parent_descriptions = {}
    cluster_labels = {}
    cluster_descriptions = {}

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
        # Load prompt template
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        prompt_path = os.path.join(root_dir, "prompts", "shared", "prompt.txt")
        if not os.path.exists(prompt_path):
            prompt_path = "prompts/shared/prompt.txt"

        print(f"Loading shared prompt template from {prompt_path}...")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        lulc_list_str = "\n".join([f"- {k}" for k in all_categories])
        prompt_text = prompt_template.format(lulc_list=lulc_list_str)

        # Label Parent Clusters
        if has_parents and k_parents > 0:
            print(f"\nPreparing {k_parents} parent cluster tasks for MLLM labeling...")
            centroids_norm_hac = normalize(raw_centroids)
            parent_tasks = []
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

                parent_tasks.append({"cid": pid, "img_url": img_url, "prompt": prompt_text})

            parent_results = label_clusters_mllm_batched(
                parent_tasks, args.mllm_model, endpoint,
                chunk_size=args.chunk_size, img_max_dim=args.img_max_dim,
                image_root_dir=args.image_root_dir
            )
            for pid, (lbl, desc) in parent_results.items():
                parent_labels[pid] = lbl
                parent_descriptions[pid] = desc

        # Label Child Clusters
        print(f"\nPreparing {k_clusters} child cluster tasks for MLLM labeling...")
        tasks = []
        for cid in range(k_clusters):
            indices = np.where(child_ids == cid)[0]
            if len(indices) == 0:
                continue
            centroid = raw_centroids[cid]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            cluster_embs = embeddings_norm[indices]
            sims = np.dot(cluster_embs, centroid_norm)
            closest_idx = indices[np.argmax(sims)]
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

            tasks.append({"cid": cid, "img_url": img_url, "prompt": prompt_text})

        results = label_clusters_mllm_batched(
            tasks, args.mllm_model, endpoint,
            chunk_size=args.chunk_size, img_max_dim=args.img_max_dim,
            image_root_dir=args.image_root_dir
        )
        for cid, (lbl, desc) in results.items():
            cluster_labels[cid] = lbl
            cluster_descriptions[cid] = desc

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
    df['cluster_label'] = df['cluster_id'].map(cluster_labels)
    df['cluster_description'] = df['cluster_id'].map(cluster_descriptions)

    if has_parents:
        df['parent_cluster_label'] = df['parent_cluster_id'].map(parent_labels)
        df['parent_cluster_description'] = df['parent_cluster_id'].map(parent_descriptions)

    print(f"Saving labeled dataset to {args.output_file}...")
    if args.output_file.endswith('.pkl'):
        data = df.to_dict('records')
        for i, item in enumerate(data):
            item['embedding'] = embeddings[i]
        with open(args.output_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        df['embedding'] = list(embeddings)
        df.to_parquet(args.output_file, index=False)

    print(f"✅ Labeling Complete! Output saved to {args.output_file}.")


if __name__ == "__main__":
    main()
