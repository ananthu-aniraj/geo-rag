import pickle
import numpy as np
import torch
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel


def dict_to_string(data):
    """Recursively converts a dictionary or list to a descriptive string."""
    if isinstance(data, dict):
        return ", ".join([f"{k}: {dict_to_string(v)}" for k, v in data.items()])
    elif isinstance(data, list):
        return ", ".join([dict_to_string(item) for item in data])
    else:
        return str(data)


def compute_metrics(sim_matrix, query_indices=None):
    """Computes Top-1, Top-5, and MRR from a similarity matrix.
    
    If query_indices is provided, only those queries (rows) are evaluated, 
    but they are searched against the FULL gallery (all columns).
    """
    if query_indices is None:
        query_indices = np.arange(sim_matrix.shape[0])
    
    num_eval = len(query_indices)
    if num_eval == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0}

    top1_hits = 0
    top5_hits = 0
    mrr_sum = 0.0

    for q_idx in query_indices:
        # q_idx is the index of the "correct" match in the gallery (columns)
        scores = sim_matrix[q_idx]
        rank_indices = np.argsort(scores)[::-1]

        # Find the rank of the correct item (1-based)
        rank = np.where(rank_indices == q_idx)[0][0] + 1

        if rank == 1:
            top1_hits += 1
        if rank <= 5:
            top5_hits += 1

        mrr_sum += 1.0 / rank

    return {
        "top1": (top1_hits / num_eval) * 100,
        "top5": (top5_hits / num_eval) * 100,
        "mrr": mrr_sum / num_eval
    }


def evaluate_retrieval(pickle_path, use_tips_embeddings=False):
    print(f"Loading retrieval data from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    if not data:
        print("No data found in pickle file.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Load Tips model")
    tips_model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
    tips_model.eval().to(device)

    print("Loading CLIP model for text encoding...")
    # Using the same CLIP model as the capture script to ensure embedding space alignment
    model = SentenceTransformer('clip-ViT-B-32')

    # Prepare data
    images = [item['image'] for item in data]
    
    # Define the components to evaluate
    component_keys = [
        'combined_caption',
        'visible_evidence',
        'human_activities',
        'land_cover_usage',
        'type_of_vegetation'
    ]

    # Ensure image embeddings are numpy arrays
    if use_tips_embeddings:
        image_embeddings = np.array([item['tips_image_embedding'] for item in data])
    else:
        image_embeddings = np.array([item["image_embedding"] for item in data])

    print(f"Dataset size: {len(images)} images")

    # Identify macro-category indices
    indoor_indices = [i for i, item in enumerate(data) if item.get('ground_truth_macro') == 'indoor']
    outdoor_indices = [i for i, item in enumerate(data) if str(item.get('ground_truth_macro', '')).startswith('outdoor')]
    
    groups = [("Overall", None)]
    if indoor_indices:
        groups.append(("Indoor", np.array(indoor_indices)))
    if outdoor_indices:
        groups.append(("Outdoor", np.array(outdoor_indices)))
    
    all_results = {}

    for key in component_keys:
        print(f"\n--- Evaluating Component: {key} ---")
        # Handle cases where value might be empty or missing
        raw_captions = [dict_to_string(item.get(key, "")) for item in data]
        
        # Format as "Category: Value" to provide context, especially for "none" values
        formatted_captions = []
        category_label = key.replace('_', ' ').capitalize()
        
        for c in raw_captions:
            val = c.strip() if c.strip() else "none"
            # Special case for combined_caption: don't prefix
            if key == 'combined_caption':
                formatted_captions.append(val)
            else:
                formatted_captions.append(f"{category_label}: {val}")
        
        print(f"Encoding '{key}'...")
        if use_tips_embeddings:
            text_embeddings = tips_model.encode_text(formatted_captions).detach().cpu().numpy()
        else:
            text_embeddings = model.encode(formatted_captions, show_progress_bar=True)

        # Calculate similarity matrix
        print(f"Calculating similarity matrix for {key}...")
        sim_matrix = cosine_similarity(text_embeddings, image_embeddings)

        all_results[key] = {}
        for group_name, group_indices in groups:
            print(f"Evaluating {group_name} retrieval for {key}...")
            t2i_metrics = compute_metrics(sim_matrix, group_indices)
            i2t_metrics = compute_metrics(sim_matrix.T, group_indices)
            
            all_results[key][group_name] = {
                "t2i": t2i_metrics,
                "i2t": i2t_metrics
            }

    # Final Comparison Tables
    for group_name, _ in groups:
        print("\n" + "=" * 115)
        print(f"RETRIEVAL PERFORMANCE: {group_name.upper()}")
        print("=" * 115)
        print(f"{'COMPONENT':<25} | {'T2I Top-1':<10} | {'T2I Top-5':<10} | {'T2I MRR':<10} | {'I2T Top-1':<10} | {'I2T Top-5':<10} | {'I2T MRR':<10}")
        print("-" * 115)
        for key in component_keys:
            res = all_results[key][group_name]
            print(f"{key:<25} | {res['t2i']['top1']:>9.2f}% | {res['t2i']['top5']:>9.2f}% | {res['t2i']['mrr']:>10.3f} | {res['i2t']['top1']:>9.2f}% | {res['i2t']['top5']:>9.2f}% | {res['i2t']['mrr']:>10.3f}")
        print("=" * 115)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Retrieval performance from VLM captions.")
    parser.add_argument("--pickle_file", type=str, help="Path to the .pkl file generated by caption_test.py")
    parser.add_argument("--use_tips_embeddings", action="store_true",
                        help="Whether to use TIPS image embeddings for evaluation")
    args = parser.parse_args()

    evaluate_retrieval(args.pickle_file, use_tips_embeddings=args.use_tips_embeddings)
