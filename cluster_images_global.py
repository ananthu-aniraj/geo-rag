import pickle
import numpy as np
import argparse
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


# Exhaustive Natural LULC Vocabulary with Global Biomes
NATURAL_LULC_VOCAB = {
    "Broadleaved forest": "Deciduous or evergreen broad-leaf trees (oak, beech, maple, birch).",
    "Coniferous forest": "Evergreen needle-leaf trees (pine, spruce, fir, larch).",
    "Mixed forest": "Co-dominant broadleaved and coniferous trees.",
    "Tropical forest": "Equatorial rainforests, mangroves, or tropical dry forest.",
    "Sparsely wooded / Savanna": "Grassland with scattered trees (10-30% canopy cover).",
    "Natural grassland": "Meadows, wild steppes, alpine grasslands, or prairies.",
    "Temperate shrubland / Scrub": "Low woody scrub (heather, gorse, bramble).",
    "Arid shrubland": "Desert scrub, sagebrush, or dry savanna bushland.",
    "Tundra": "Low-growing polar vegetation (mosses, lichens, dwarf shrubs).",
    "Sandy desert / Dunes": "Sand sheets, active dunes, or sandy flats.",
    "Rocky desert / Gravel plains": "Stony hamadas, gravel plains, or barren volcanic ash fields.",
    "Barren soil / Badlands": "Highly eroded clay hills, bare dry earth, or dry salt flats.",
    "Bare rock / Cliffs": "Exposed bedrock, cliffs, scree slopes, or mountain peaks.",
    "Coastal beach / Spit": "Sandy or pebbly sea coast.",
    "Wetland / Marsh / Bog": "Marshes, peat bogs, fens, reed beds, or swamps.",
    "River / Stream": "Flowing freshwater channels, creeks, or canals.",
    "Lake / Pond": "Standing inland water bodies or reservoirs.",
    "Marine / Estuary": "Coastal saltwater, ocean surf, bays, or intertidal flats.",
    "Glacier / Permanent ice": "Glaciers, ice caps, or permanent snowfields.",
    "Other natural land cover": "Any other natural land cover or landscape."
}

# Exhaustive Man-made LULC Vocabulary
MAN_MADE_LULC_VOCAB = {
    "Forest plantation": "Evenly spaced rows of planted timber trees.",
    "Managed pasture": "Fenced grazing pastures or paddocks.",
    "Herbaceous cropland": "Annual cultivated field crops (cereal, corn, wheat, canola).",
    "Orchards & Vineyards": "Woody perennial row crops (vineyards, fruit/olive orchards, plantations).",
    "Rice paddies / Flooded crops": "Water-flooded agricultural basins.",
    "Covered agriculture": "Greenhouses, polytunnels, or nurseries.",
    "High-density built-up": "Skyscrapers, high-rise blocks, and dense urban centers.",
    "Suburban / Low-density residential": "Single-family houses, villas, private gardens, and streets.",
    "Industrial / Commercial zone": "Factories, warehouses, refineries, shopping centers, or office parks.",
    "Active construction site": "Earthworks, building foundations, cranes, and scaffolding.",
    "Transportation network": "Highways, railways, runways, or shipping ports.",
    "Mine / Quarry / Landfill": "Open-pit mines, gravel quarries, or landfill sites.",
    "Urban green space": "City parks, golf courses, botanical gardens, or sports fields.",
    "Other man-made surface": "Any other artificial or managed land cover or surface."
}


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
        chunk = tasks[i : i + chunk_size]
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
    else:
        df = pd.read_parquet(args.pkl)
        data = df.to_dict('records')

    if not data:
        print("No data found.")
        return

    print(f"Extracting embeddings for {len(data)} images...")
    embeddings = np.array([item['embedding'] for item in data]).squeeze()
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
    global_cluster_ids = np.zeros(len(data), dtype=int)
    
    # Cluster     
    print(f"\nClustering subset on {clustering_mode} space...")
    global_cluster_ids, _ = cluster_subset(cluster_input, args.k, args.gpu, args.minibatch, faiss)
        

    # Centroid derivation and labeling
    cluster_labels = {}
    cluster_descriptions = {}

    if not args.no_label:
        print("\nComputing centroids in original 768D space for labeling...")
        d = embeddings_norm.shape[1]
        raw_centroids = np.zeros((args.k, d), dtype=embeddings_norm.dtype)
        valid_mask = (global_cluster_ids >= 0)
        np.add.at(raw_centroids, global_cluster_ids[valid_mask], embeddings_norm[valid_mask])
        counts = np.bincount(global_cluster_ids[valid_mask], minlength=args.k)
        valid_counts = counts > 0
        raw_centroids[valid_counts] /= counts[valid_counts, None]

        if args.label_method == "mllm":
            print(f"\nPreparing tasks for MLLM labeling ({args.mllm_model} via {args.mllm_backend})...")
            # Build prompt templates
            prompt_template = (
                "Analyze the provided image with a strict focus on visual evidence. Do not guess or assume context outside the frame. Describe:\n"
                "1. visible_evidence: Primary objects, architectural elements, lighting, or natural formations clearly visible in the image. Base this strictly on visual facts.\n"
                "2. human_activities: Based ONLY on the visual evidence, what are people doing here, or what activities does the infrastructure support?\n"
                "3. land_cover_usage: Based ONLY on the visual evidence, what is on the ground (e.g., asphalt, grass, carpet) and how is the space utilized?\n"
                "4. type_of_vegetation: Describe the type of vegetation present, if applicable (e.g., grass, trees, shrubs). If none, state \"none\".\n\n"
                "Based ONLY on the visual evidence described above, classify this environment into EXACTLY one of the following Land Use / Land Cover (LULC) categories:\n"
                "{lulc_list}\n\n"
                "Format your output EXACTLY as follows:\n"
                "LABEL: <Insert EXACTLY one category from the list above>\n"
                "DESCRIPTION: <A detailed paragraph summarizing the visual evidence, human activities, land cover, and vegetation.>"
            )
            lulc_list_str = "\n".join([f"- {k}" for k in all_categories])

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
                representative_item = data[closest_idx]
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

            print(f"Total tasks prepared: {len(tasks)}. Starting batch inference via VLM API at {endpoint} (chunk size {args.chunk_size})...")
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
    for i, item in enumerate(data):
        cid = int(global_cluster_ids[i])
        item['cluster_id'] = cid
        if cid in cluster_labels:
            item['cluster_label'] = cluster_labels[cid]
        if cid in cluster_descriptions:
            item['cluster_description'] = cluster_descriptions[cid]

    print(f"Saving to {args.out}...")
    if args.out.endswith('.pkl'):
        with open(args.out, 'wb') as f:
            pickle.dump(data, f)
    else:
        pd.DataFrame(data).to_parquet(args.out)

    print(f"\nClustering Complete!")
    unique, counts = np.unique(global_cluster_ids, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = f" ({cluster_labels[u]})" if u in cluster_labels else ""
        print(f"  Cluster {u}{lbl}: {c} images")


if __name__ == "__main__":
    main()
