import os
import sys
import time
import pickle
import argparse
import base64
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


# Shared LULC Vocabularies
from lulc_vocab import NATURAL_LULC_VOCAB, MAN_MADE_LULC_VOCAB

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
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    return img.resize((new_w, new_h), resample)


def load_image(url, target_max=448, timeout=15):
    """Loads an image from local path or downloads from Mapillary, Kartaview, or standard URL."""
    if os.path.exists(url):
        try:
            img = Image.open(url).convert("RGB")
            return resize_image_aspect(img, target_max)
        except Exception as e:
            print(f"Error loading local image {url}: {e}")
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


def load_image_with_retry(url, target_max=448, timeout=15, max_retries=3):
    """Wrapper around load_image that retries with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            img = load_image(url, target_max=target_max, timeout=timeout)
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


def save_dataset(data, final_results, out_path):
    """Helper to update labels in the data list and write them to the output file."""
    updated_clusters = set()
    row_update_count = 0

    # Map results
    cluster_labels = {}
    cluster_descriptions = {}
    for cid, (lbl, desc) in final_results.items():
        if lbl != "Error Labeling":
            cluster_labels[cid] = lbl
            cluster_descriptions[cid] = desc
            updated_clusters.add(cid)

    for item in data:
        cid = item.get('cluster_id')
        if cid is not None and int(cid) in cluster_labels:
            item['cluster_label'] = cluster_labels[int(cid)]
            item['cluster_description'] = cluster_descriptions[int(cid)]
            row_update_count += 1

    if out_path.endswith('.pkl'):
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        pd.DataFrame(data).to_parquet(out_path)

    print(
        f"  -> Checkpoint: Saved {row_update_count} rows across {len(updated_clusters)} updated clusters to {out_path}.")


def main():
    # 1. Load Defaults from params.yaml if available
    default_mllm_model = "google/gemma-4-E4B-it"
    default_mllm_backend = "sglang"
    default_output_dir = ""
    default_base_name = "geo_space"
    default_k = 40000

    if os.path.exists("params.yaml"):
        try:
            import yaml
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
    parser.add_argument("--fallback_depth", type=int, default=5,
                        help="Number of top closest images in a cluster to check if the closest one fails to download.")
    parser.add_argument("--timeout", type=int, default=15,
                        help="Timeout in seconds for image download HTTP requests.")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Interval of successfully re-labeled clusters at which to save intermediate checkpoints.")
    args = parser.parse_args()

    if not args.out:
        args.out = args.file

    print(f"Loading dataset from {args.file}...")
    if args.file.endswith('.pkl'):
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
    else:
        df = pd.read_parquet(args.file)
        data = df.to_dict('records')

    if not data:
        print("No data found.")
        sys.exit(1)

    print(f"Total records loaded: {len(data)}")

    # 3. Identify clusters needing re-labeling
    target_cluster_ids = set()
    if args.cluster_ids:
        target_cluster_ids = set(args.cluster_ids)
        print(f"Forced re-labeling for specific cluster IDs: {sorted(list(target_cluster_ids))}")
    else:
        for item in data:
            label = item.get('cluster_label')
            cid = item.get('cluster_id')
            if cid is not None:
                if label is None or label in ("Error Labeling", "Unlabeled", "", "None"):
                    target_cluster_ids.add(int(cid))
        print(f"Automatically detected {len(target_cluster_ids)} failed or unlabeled clusters.")

    if not target_cluster_ids:
        print("No failed or unlabeled clusters found! Script will exit without modifying the file.")
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
        import urllib.request
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
    embeddings = np.array([item['embedding'] for item in data]).squeeze()
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    print("Normalizing embeddings...")
    embeddings_norm = normalize(embeddings)

    # 6. Construct prompt templates
    noise_category = "None of the above / Noise"
    noise_prompt = "Noisy image, indoor scene, closeup object, selfie, text/graphic, or unrelated non-geographic photo."
    all_categories = list(NATURAL_LULC_VOCAB.keys()) + list(MAN_MADE_LULC_VOCAB.keys()) + [noise_category]
    lulc_list_str = "\n".join([f"- {k}" for k in all_categories])

    # Load prompt template from prompts/shared/prompt.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "shared", "prompt.txt")

    print(f"Loading shared prompt template from {prompt_path}")
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    prompt_text = prompt_template.format(lulc_list=lulc_list_str)

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
                                        max_retries=args.max_retries)

            if img is not None:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return {
                    "cid": cid,
                    "img_str": img_str,
                    "img_url": img_url,
                    "rank": rank
                }
        return None

    # 7. Query the VLM sequentially
    final_results = {}
    sorted_failed_ids = sorted(list(target_cluster_ids))
    total_clusters = len(sorted_failed_ids)

    print(f"\nStarting sequential VLM inference for {total_clusters} clusters (batch size 1)...")
    print("Press Ctrl+C at any time to interrupt and save your progress.")

    try:
        for idx, cid in enumerate(sorted_failed_ids):
            print(f"[{idx + 1}/{total_clusters}] Processing Cluster #{cid}...")

            # Download image sequentially
            task = prepare_cluster_task(cid)
            if not task:
                print(f"  [FAILED] Cluster #{cid}: Failed to download any of the top {args.fallback_depth} images.")
                continue

            # Query VLM sequentially
            response_text = query_vlm_openai_api(task['img_str'], prompt_text, args.mllm_model, endpoint,
                                                 timeout=args.timeout + 15)

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

                final_results[cid] = (label, description)
                print(f"  [SUCCESS] Cluster #{cid} labeled: '{label}' (using representative image rank {task['rank']})")
            else:
                final_results[cid] = ("Error Labeling", "Inference failed or returned empty response.")
                print(f"  [FAILED] Cluster #{cid}: VLM returned empty response.")

            # Periodically save intermediate checkpoints
            if len(final_results) % args.save_interval == 0:
                save_dataset(data, final_results, args.out)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user (Ctrl+C). Saving progress so far...")
    finally:
        if final_results:
            print("\nSaving final results...")
            save_dataset(data, final_results, args.out)
            print("Re-labeling session finished.")
        else:
            print("\nNo clusters were successfully re-labeled.")


if __name__ == "__main__":
    main()
