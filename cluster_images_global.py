import pickle
import numpy as np
import argparse
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
import os
import pandas as pd
import torch
from transformers import AutoModel
import requests
from io import BytesIO
from PIL import Image


try:
    import faiss
except ImportError:
    faiss = None

# Shared LULC Vocabularies
from lulc_vocab import NATURAL_LULC_VOCAB, MAN_MADE_LULC_VOCAB


def label_clusters(centroids, text_features, categories, top_k=3):
    """Performs zero-shot labeling of cluster centroids using given text features."""
    centroids_norm = normalize(centroids)
    sims = np.dot(centroids_norm, text_features.T)
    results = []
    for i in range(len(centroids)):
        top_indices = np.argsort(sims[i])[::-1][:top_k]
        top_labels = [categories[idx] for idx in top_indices]
        results.append(", ".join(top_labels))
    return results


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


def load_image(url, target_max=448):
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

    # Ensure endpoint ends with /v1/chat/completions
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


def label_clusters_mllm_batched(tasks, model_name, endpoint_url, chunk_size=128, img_max_dim=448):
    """Runs VLM labeling in chunks to utilize batch inference via OpenAI-compatible API."""
    results = {}

    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i: i + chunk_size]
        print(f"Processing batch {i // chunk_size + 1}/{(len(tasks) - 1) // chunk_size + 1} ({len(chunk)} clusters)...")

        from concurrent.futures import ThreadPoolExecutor
        import base64

        # Dictionary to store base64 encoded images
        images_base64 = {}

        def prepare_image(task):
            cid = task['cid']
            img = load_image(task['img_url'], target_max=img_max_dim)
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

        # Query the VLM server in parallel. The server handles parallel batching on the GPU.
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


def cluster_subset(subset_input, k_subset, gpu_enabled, minibatch_enabled, faiss_module):
    """Helper to run K-Means on a subset of embeddings."""
    if k_subset <= 0:
        return np.array([]), np.array([])
    if len(subset_input) < k_subset:
        k_subset = len(subset_input)

    if gpu_enabled:
        if faiss_module is None:
            raise ImportError("faiss is not installed. Please install it to use --gpu.")
        d = subset_input.shape[1]
        kmeans_faiss = faiss_module.Kmeans(d, k_subset, niter=20, verbose=True, gpu=True, seed=42)
        kmeans_faiss.train(subset_input.astype('float32'))
        _, cluster_ids = kmeans_faiss.index.search(subset_input.astype('float32'), 1)
        cluster_ids = cluster_ids.ravel()
        centroids = kmeans_faiss.centroids
    else:
        if minibatch_enabled:
            kmeans = MiniBatchKMeans(n_clusters=k_subset, random_state=42, n_init=3, batch_size=1024)
        else:
            kmeans = KMeans(n_clusters=k_subset, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(subset_input)
        centroids = kmeans.cluster_centers_

    return cluster_ids, centroids


def main():
    parser = argparse.ArgumentParser(description="Global Semantic Clustering of Geo-Images.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the .pkl or parquet file.")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters.")
    parser.add_argument("--k_parents", type=int, default=None, help="Number of parent clusters for hierarchical clustering (default: k // 80).")
    parser.add_argument("--use_umap", action="store_true", help="Use UMAP dimensionality reduction before clustering.")
    parser.add_argument("--reduce_dim", type=int, default=10,
                        help="Dimensions to reduce to via UMAP if --use_umap is set.")
    parser.add_argument("--minibatch", action="store_true",
                        help="Use MiniBatchKMeans for massive datasets (2M+ images).")
    parser.add_argument("--gpu", action="store_true",
                        help="Use FAISS GPU for massive datasets.")
    parser.add_argument("--no_label", action="store_true", help="Disable automatic cluster labeling.")
    parser.add_argument("--label_method", type=str, choices=["zeroshot", "mllm"], default="zeroshot",
                        help="Method to label clusters: 'zeroshot' (embedding-based) or 'mllm' (visual LLM on centroid image).")
    parser.add_argument("--mllm_model", type=str, default="gemma4:e4b",
                        help="VLM model identifier (Ollama model name or Hugging Face model path for SGLang).")
    parser.add_argument("--mllm_backend", type=str, choices=["ollama", "sglang"], default="ollama",
                        help="Backend server type. Sets the default endpoint port (Ollama: 11434, SGLang: 30000).")
    parser.add_argument("--mllm_endpoint", type=str, default=None,
                        help="Custom API URL for the VLM server. Overrides the default port assigned by --mllm_backend.")
    parser.add_argument("--chunk_size", type=int, default=128,
                        help="Batch chunk size for parallel VLM API requests.")
    parser.add_argument("--img_max_dim", type=int, default=672,
                        help="Target maximum dimension to resize images before VLM processing (default: 448). Prevents OOM on wide panoramic images.")
    parser.add_argument("--out", type=str, default="clustered_data.pkl", help="Output path.")
    args = parser.parse_args()

    if args.k_parents is None:
        args.k_parents = max(2, args.k // 80)

    # Pre-flight check for VLM server
    if args.label_method == "mllm" and not args.no_label:
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
        except Exception:
            pass

        if not server_running:
            print(f"\n[WARNING] VLM server is not reachable at {endpoint}!")
            print("Falling back to default embedding-based (zeroshot) cluster labeling.")
            args.label_method = "zeroshot"

    print(f"Loading data from {args.pkl}...")
    if args.pkl.endswith('.pkl'):
        with open(args.pkl, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        del data
        embeddings = np.vstack(df['embedding'].values).astype(np.float32)
    else:
        # Load only metadata columns first (uses ~200MB RAM)
        df = pd.read_parquet(args.pkl, columns=['Photo_ID', 'Platform', 'Latitude', 'Longitude', 'Image_URL', 'Captured_At'])
        
        # Load embeddings directly into numpy array using PyArrow
        print("Loading raw embedding matrix using PyArrow...")
        t0 = time.time()
        import pyarrow.parquet as pq
        table = pq.read_table(args.pkl, columns=["embedding"])
        
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

    if len(df) == 0:
        print("No data found.")
        return

    print(f"Loaded {len(df):,} images.")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TIPSv2 model on {device} to encode classification prompts...")
    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).to(device)
    model.eval()
    # # 1. Zero-shot classification (Natural vs Man-made)
    # SUBTYPE_PROMPTS = [
    #     "A natural landscape, wild nature, forest, grassland, desert, mountain, or water body.",
    #     "A man-made structure, city, road, building, industrial area, or transport infrastructure."
    # ]

    noise_category = "None of the above / Noise"
    noise_prompt = "Noisy image, indoor scene, closeup object, selfie, text/graphic, or unrelated non-geographic photo."

    all_categories = list(NATURAL_LULC_VOCAB.keys()) + list(MAN_MADE_LULC_VOCAB.keys()) + [noise_category]
    all_prompts = list(NATURAL_LULC_VOCAB.values()) + list(MAN_MADE_LULC_VOCAB.values()) + [noise_prompt]

    # Encode prompts
    with torch.no_grad():
        # subtype_features = normalize(model.encode_text(SUBTYPE_PROMPTS).cpu().numpy())
        all_features = normalize(model.encode_text(all_prompts).cpu().numpy())

    # Unload model to save memory
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print("Prompts embedded successfully. Model unloaded from memory.")

    print("Normalizing embeddings for cosine similarity classification...")
    embeddings_norm = normalize(embeddings)

    # Prepare inputs for clustering
    if args.use_umap:
        import umap
        print(f"\nReducing dimensions to {args.reduce_dim}D using UMAP...")
        reducer = umap.UMAP(n_components=args.reduce_dim, metric='cosine', random_state=42)
        cluster_input = reducer.fit_transform(embeddings_norm)
        clustering_mode = f"UMAP ({args.reduce_dim}D)"
    else:
        cluster_input = embeddings_norm
        clustering_mode = "Raw 768D (Normalized)"

    # Cluster subsets
    global_cluster_ids = np.zeros(len(df), dtype=int)

    # Cluster     
    print(f"\nClustering subset on {clustering_mode} space...")
    global_cluster_ids, _ = cluster_subset(cluster_input, args.k, args.gpu, args.minibatch, faiss)

    # Centroid derivation (run unconditionally to support hierarchical grouping)
    print("\nComputing centroids in original 768D space...")
    d = embeddings_norm.shape[1]
    raw_centroids = np.zeros((args.k, d), dtype=embeddings_norm.dtype)
    valid_mask = (global_cluster_ids >= 0)
    np.add.at(raw_centroids, global_cluster_ids[valid_mask], embeddings_norm[valid_mask])
    counts = np.bincount(global_cluster_ids[valid_mask], minlength=args.k)
    valid_counts = counts > 0
    raw_centroids[valid_counts] /= counts[valid_counts, None]

    # Hierarchical parent clustering (using Spherical K-Means for sub-second, interruptible execution)
    from sklearn.cluster import KMeans
    print(f"\nPerforming hierarchical clustering: grouping {args.k} centroids into {args.k_parents} parent clusters...")
    
    # Normalize centroids (K-Means on L2-normalized vectors is equivalent to Cosine-Similarity clustering)
    centroids_norm_hac = normalize(raw_centroids)
    parent_clustering = KMeans(
        n_clusters=args.k_parents,
        random_state=42,
        n_init=3
    )
    parent_ids = parent_clustering.fit_predict(centroids_norm_hac)

    # Compute parent centroids (weighted by flat cluster sizes)
    parent_centroids = np.zeros((args.k_parents, d), dtype=embeddings_norm.dtype)
    np.add.at(parent_centroids, parent_ids, raw_centroids * counts[:, None])
    parent_counts = np.bincount(parent_ids, minlength=args.k_parents)
    valid_parent_counts = parent_counts > 0
    parent_centroids[valid_parent_counts] /= parent_counts[valid_parent_counts, None]

    # Label parent clusters
    parent_labels = {}
    parent_descriptions = {}

    if not args.no_label and args.label_method == "mllm":
        print(f"\nPreparing parent clusters for MLLM labeling ({args.mllm_model} via {args.mllm_backend})...")
        parent_tasks = []
        
        # Load prompt template from prompts/shared/prompt.txt
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(script_dir, "prompts", "shared", "prompt.txt")

        print(f"Loading shared prompt template from {prompt_path}")
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()

        lulc_list_str = "\n".join([f"- {k}" for k in all_categories])
        prompt_text = prompt_template.format(lulc_list=lulc_list_str)

        for pid in range(args.k_parents):
            cids_in_parent = np.where(parent_ids == pid)[0]
            if len(cids_in_parent) == 0:
                continue

            # Find the closest child cluster centroid to the parent cluster centroid
            p_centroid = parent_centroids[pid]
            p_centroid_norm = p_centroid / (np.linalg.norm(p_centroid) + 1e-9)
            child_embs = centroids_norm_hac[cids_in_parent]
            sims = np.dot(child_embs, p_centroid_norm)
            closest_child_cid = cids_in_parent[np.argmax(sims)]

            # Find the closest image in that child cluster to use as parent's representative image
            indices = np.where(global_cluster_ids == closest_child_cid)[0]
            if len(indices) == 0:
                continue
            child_centroid = raw_centroids[closest_child_cid]
            child_centroid_norm = child_centroid / (np.linalg.norm(child_centroid) + 1e-9)
            cluster_embs = embeddings_norm[indices]
            img_sims = np.dot(cluster_embs, child_centroid_norm)
            closest_img_idx = indices[np.argmax(img_sims)]
            representative_item = df.iloc[closest_img_idx]
            img_url = representative_item['Image_URL']

            # Resolve potentially expired Mapillary or Kartaview URLs dynamically
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

            parent_tasks.append({
                "cid": pid,
                "img_url": img_url,
                "prompt": prompt_text
            })

        # Resolve API Endpoint
        endpoint = args.mllm_endpoint
        if not endpoint:
            if args.mllm_backend == "sglang":
                endpoint = "http://localhost:30000"
            else:
                endpoint = "http://localhost:11434"

        print(f"Total parent tasks prepared: {len(parent_tasks)}. Starting batch parent MLLM labeling...")
        parent_results = label_clusters_mllm_batched(
            parent_tasks, args.mllm_model, endpoint,
            chunk_size=args.chunk_size, img_max_dim=args.img_max_dim
        )
        for pid, (lbl, desc) in parent_results.items():
            parent_labels[pid] = lbl
            parent_descriptions[pid] = desc
    else:
        print("Labeling parent clusters zero-shot...")
        parent_labels_list = label_clusters(parent_centroids, all_features, all_categories)
        parent_labels = {pid: label for pid, label in enumerate(parent_labels_list)}

    # Label child clusters
    cluster_labels = {}
    cluster_descriptions = {}

    if not args.no_label:
        if args.label_method == "mllm":
            print(f"\nPreparing tasks for MLLM labeling ({args.mllm_model} via {args.mllm_backend})...")
            tasks = []
            for cid in range(args.k):
                indices = np.where(global_cluster_ids == cid)[0]
                if len(indices) == 0:
                    continue

                centroid = raw_centroids[cid]
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
                cluster_embs = embeddings_norm[indices]
                sims = np.dot(cluster_embs, centroid_norm)
                closest_idx = indices[np.argmax(sims)]
                representative_item = df.iloc[closest_idx]
                img_url = representative_item['Image_URL']

                # Resolve potentially expired Mapillary or Kartaview URLs dynamically
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

                prompt_text = prompt_template.format(lulc_list=lulc_list_str)

                tasks.append({
                    "cid": cid,
                    "img_url": img_url,
                    "prompt": prompt_text
                })

            # Resolve API Endpoint
            endpoint = args.mllm_endpoint
            if not endpoint:
                if args.mllm_backend == "sglang":
                    endpoint = "http://localhost:30000"
                else:
                    endpoint = "http://localhost:11434"

            print(
                f"Total tasks prepared: {len(tasks)}. Starting batch inference via VLM API at {endpoint} (chunk size {args.chunk_size})...")
            results = label_clusters_mllm_batched(
                tasks, args.mllm_model, endpoint,
                chunk_size=args.chunk_size, img_max_dim=args.img_max_dim
            )
            for cid, (lbl, desc) in results.items():
                cluster_labels[cid] = lbl
                cluster_descriptions[cid] = desc

        else:
            labels = label_clusters(raw_centroids, all_features, all_categories)
            for cid, label in enumerate(labels):
                cluster_labels[cid] = label

    print("\nUpdating metadata...")
    df['cluster_id'] = global_cluster_ids.astype(int)
    
    # Map child cluster metadata
    df['cluster_label'] = df['cluster_id'].map(cluster_labels)
    df['cluster_description'] = df['cluster_id'].map(cluster_descriptions)

    # Map parent cluster metadata
    parent_id_map = {cid: int(parent_ids[cid]) for cid in range(args.k) if cid < len(parent_ids)}
    df['parent_cluster_id'] = df['cluster_id'].map(parent_id_map)
    df['parent_cluster_label'] = df['parent_cluster_id'].map(parent_labels)
    df['parent_cluster_description'] = df['parent_cluster_id'].map(parent_descriptions)

    print(f"Saving to {args.out}...")
    if args.out.endswith('.pkl'):
        # For legacy compatibility, save as list of dicts if writing .pkl
        data = df.to_dict('records')
        # Add back embedding to the saved records
        for i, item in enumerate(data):
            item['embedding'] = embeddings[i]
        with open(args.out, 'wb') as f:
            pickle.dump(data, f)
    else:
        # Re-attach embedding column for the final Parquet save
        df['embedding'] = list(embeddings)
        df.to_parquet(args.out, index=False)

    print(f"\nClustering Complete!")
    unique, counts = np.unique(global_cluster_ids, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = f" ({cluster_labels[u]})" if u in cluster_labels else ""
        print(f"  Cluster {u}{lbl}: {c} images")


if __name__ == "__main__":
    main()
