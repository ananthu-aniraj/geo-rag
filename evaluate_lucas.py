import os
import sys
import json
import re
import argparse
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import pickle
from PIL import Image
import torch
from transformers import AutoModel
from torchvision import transforms

tips_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


def load_lucas_metadata(csv_path):
    """Loads LUCAS metadata and maps the 8-digit point ID to its land cover, land use, and EUNIS class labels."""
    if not os.path.exists(csv_path):
        print(f"Error: Metadata file '{csv_path}' not found.")
        sys.exit(1)

    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    metadata = {}
    
    for _, row in df.iterrows():
        im_id = str(row['im_id']).strip()
        # Extract the 8-digit code to align with filenames
        match = re.search(r'(\d{8})', im_id)
        if match:
            clean_id = match.group(1)
        else:
            clean_id = im_id
            
        metadata[clean_id] = {
            'im_id': im_id,
            'lc_label': row.get('lc_label', ''),
            'lu_label': row.get('lu_label', ''),
            'eunis_class': row.get('eunis_class', '')
        }
        
    return metadata, df


def get_class_lists(df):
    """Retrieves class lists from lucas_class_mapping if available, falling back to unique values in the metadata CSV."""
    lc_list, lu_list, eunis_list = [], [], []
    try:
        import lucas_class_mapping
        lc_list = list(lucas_class_mapping.lc1_class_mapping.values())
        lu_list = list(lucas_class_mapping.lu1_class_mapping.values())
        eunis_list = list(lucas_class_mapping.eunis_mapping.values())
        print("Successfully loaded class lists from lucas_class_mapping.")
    except ImportError:
        print("Warning: lucas_class_mapping not found or could not be imported. Extracting unique classes from metadata CSV...")
        if 'lc_label' in df.columns:
            lc_list = sorted(df['lc_label'].dropna().unique().tolist())
        if 'lu_label' in df.columns:
            lu_list = sorted(df['lu_label'].dropna().unique().tolist())
        if 'eunis_class' in df.columns:
            eunis_list = sorted(df['eunis_class'].dropna().unique().tolist())
            
    return lc_list, lu_list, eunis_list


def extract_json_from_response(response_text):
    """Safely extracts and repairs JSON from the model's response using a tiered approach."""
    if not response_text:
        return None

    # 1. Greedy search for the outermost braces
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')

    if start_idx == -1 or end_idx == -1:
        return None

    json_str = response_text[start_idx:end_idx + 1]

    # --- ATTEMPT 1: Direct Parse ---
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        pass

    # --- ATTEMPT 2: Basic Cleanup (Backslashes & Commas) ---
    temp_json = json_str.replace('\\_', '_').replace('\\-', '-')
    temp_json = re.sub(r',\s*([\]}])', r'\1', temp_json)
    temp_json = re.sub(r':\s*\{\s*"([^"]*)"\s*\}', r': ["\1"]', temp_json)

    try:
        return json.loads(temp_json, strict=False)
    except json.JSONDecodeError:
        pass

    # --- ATTEMPT 3: Surgical Single Quote Repair ---
    if re.search(r"'\w+'\s*:", json_str):
        temp_json = re.sub(r"'(\w+)'\s*:", r'"\1":', json_str)
        temp_json = re.sub(r":\s*'([^']*)'\s*([,}])", r': "\1"\2', temp_json)
        temp_json = re.sub(r"'\s*([^']*)\s*'\s*([,\]])", r'"\1"\2', temp_json)
        temp_json = temp_json.replace('\\_', '_').replace('\\-', '-')
        temp_json = re.sub(r',\s*([\]}])', r'\1', temp_json)

        try:
            return json.loads(temp_json, strict=False)
        except json.JSONDecodeError:
            pass

    # --- FINAL FALLBACK: Strip Markdown ---
    try:
        clean_json = re.sub(r'```(?:json)?|```', '', json_str).strip()
        return json.loads(clean_json, strict=False)
    except:
        pass

    return None


def calculate_similarity(embedder, predicted, ground_truth):
    """Calculates cosine similarity between the model's prediction and the actual label."""
    if not predicted or not ground_truth:
        return 0.0

    embeddings = embedder.encode([str(predicted), str(ground_truth)])
    sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(sim_score)


def main():
    parser = argparse.ArgumentParser(description="Evaluate local VLMs against LUCAS 2018 using Ollama or sglang.")
    parser.add_argument("--backend", type=str, choices=["ollama", "sglang"], default="ollama", help="The inference backend to use.")
    parser.add_argument("--model", type=str, default="gemma4:e4b", help="The name of the model (Ollama tag or sglang model path).")
    parser.add_argument("--sgl_mem_fraction", type=float, default=0.8, help="Memory fraction for sglang (0.0 to 1.0).")
    parser.add_argument("--csv", type=str, default="/user/aaniraj/home/Documents/Projects/data/LUCAS2018/Sen4Map_Metadata_test.csv",
                        help="Path to the Sen4Map_Metadata_test.csv file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing LUCAS images.")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Maximum number of images to evaluate. Set to 0 for unlimited.")
    parser.add_argument("--prompt_version", type=str, default="v1", help="The version of the prompt to use (folder name in prompts_lucas/).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_prompts_dir = os.path.join(script_dir, "prompts_lucas")

    # Load prompts from external directory
    prompt_version_dir = os.path.join(default_prompts_dir, args.prompt_version)
    if not os.path.exists(prompt_version_dir):
        print(f"Error: Prompt version directory '{prompt_version_dir}' not found.")
        return
    
    step1_path = os.path.join(prompt_version_dir, "step1.txt")
    step2_path = os.path.join(prompt_version_dir, "step2.txt")
    
    if not os.path.exists(step1_path) or not os.path.exists(step2_path):
        print(f"Error: step1.txt or step2.txt missing in {prompt_version_dir}")
        return

    with open(step1_path, 'r') as f:
        prompt_step1 = f.read().strip()
    with open(step2_path, 'r') as f:
        prompt_step2 = f.read().strip()

    print(f"Using prompt version: {args.prompt_version}")

    # Load metadata and class definitions
    metadata_dict, df = load_lucas_metadata(args.csv)
    lc_list, lu_list, eunis_list = get_class_lists(df)

    lc_list_str = "\n".join([f"- {c}" for c in lc_list]) if lc_list else "None"
    lu_list_str = "\n".join([f"- {c}" for c in lu_list]) if lu_list else "None"
    eunis_list_str = "\n".join([f"- {c}" for c in eunis_list]) if eunis_list else "None"

    # Scan the image directory and match with metadata using point IDs
    print(f"Scanning for images in {args.img_dir}...")
    matched_images = []
    
    for root, _, files in os.walk(args.img_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                basename = os.path.splitext(filename)[0]
                # Look for an 8-digit sequence in the filename
                match = re.search(r'(\d{8})', basename)
                if match:
                    point_num = match.group(1)
                    # Detect direction suffix (e.g. W, E, N, S) immediately after the digits
                    dir_match = re.search(rf'{point_num}([WENS])', basename, re.IGNORECASE)
                    direction = dir_match.group(1).upper() if dir_match else "Unknown"
                    
                    if point_num in metadata_dict:
                        matched_images.append({
                            'path': os.path.join(root, filename),
                            'filename': filename,
                            'point_id': point_num,
                            'direction': direction,
                            'meta': metadata_dict[point_num]
                        })

    if not matched_images:
        print("No matching images found in the provided directory.")
        return

    print(f"Found {len(matched_images)} matched image files in the directory.")

    # Group matched images by point ID to preserve geographical diversity in sampling
    images_by_point = {}
    for item in matched_images:
        pt_id = item['point_id']
        if pt_id not in images_by_point:
            images_by_point[pt_id] = []
        images_by_point[pt_id].append(item)

    # Deterministically shuffle point IDs
    sorted_points = sorted(images_by_point.keys())
    rng = random.Random(42)
    rng.shuffle(sorted_points)

    # Interleave images across different points to maximize variety
    selected_images = []
    max_dirs = max(len(images_by_point[pt]) for pt in sorted_points)
    
    for dir_idx in range(max_dirs):
        for pt in sorted_points:
            if dir_idx < len(images_by_point[pt]):
                selected_images.append(images_by_point[pt][dir_idx])
                if args.max_images > 0 and len(selected_images) == args.max_images:
                    break
        if args.max_images > 0 and len(selected_images) == args.max_images:
            break

    print(f"Selected {len(selected_images)} images representing {len(set(img['point_id'] for img in selected_images))} unique locations.")

    # Initialize Backend
    sgl_runtime = None
    if args.backend == "sglang":
        print(f"Initializing sglang runtime with model: {args.model}")
        try:
            import sglang as sgl
            sgl_runtime = sgl.Runtime(
                model_path=args.model,
                mem_fraction_static=args.sgl_mem_fraction
            )
            sgl.set_default_backend(sgl_runtime)
        except ImportError:
            print("Error: sglang library not found. Please install it to use the sglang backend.")
            return
        except Exception as e:
            print(f"Error initializing sglang: {e}")
            return

    # Load modeling libraries for similarity checks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading TIPSv2 model...")
    tips_model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
    tips_model.eval().to(device)

    print("Loading lightweight text embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading CLIP model...")
    clip_model = SentenceTransformer("clip-ViT-B-32")

    results = []
    valid_evaluations = 0

    print(f"\nStarting evaluation using model: {args.model}")
    print("-" * 60)

    for item in selected_images:
        image_path = item['path']
        filename = item['filename']
        point_id = item['point_id']
        direction = item['direction']
        meta = item['meta']

        print(f"\nProcessing: {filename} (Point: {point_id}, Dir: {direction})")
        print(f"  GT Land Cover: {meta['lc_label']}")
        print(f"  GT Land Use: {meta['lu_label']}")
        print(f"  GT EUNIS Class: {meta['eunis_class']}")

        try:
            # Step 1: Extract visual observations from the image
            if args.backend == "ollama":
                response1 = ollama.generate(
                    model=args.model,
                    prompt=prompt_step1,
                    images=[image_path]
                )
                vlm_text1 = response1.get('response', '')
            else:
                import sglang as sgl
                @sgl.function
                def step1_fn(s, image_path, prompt):
                    s += s.image(image_path)
                    s += prompt + s.gen("response")
                
                res1 = step1_fn.run(image_path=image_path, prompt=prompt_step1)
                vlm_text1 = res1["response"]

            parsed_data1 = extract_json_from_response(vlm_text1)
            if parsed_data1 is None:
                print("  -> Failed to parse JSON from Step 1.")
                print(f"  -> Raw Output: {vlm_text1}")
                continue

            # If this is v3 (having new keys), map them to standard keys for database & retrieval compatibility
            if 'dominant_cover' in parsed_data1:
                visible_evidence = f"Dominant cover is {parsed_data1.get('dominant_cover', '')} ({parsed_data1.get('cover_fraction_estimate', '')}). Ground state: {parsed_data1.get('soil_surface_state', '')}. Soil/rock: {parsed_data1.get('surface_material_and_lithology', '')}. Pattern: {parsed_data1.get('structure_and_pattern', '')}. Context: {parsed_data1.get('context_background', '')}."
                human_activities = f"Human evidence: {parsed_data1.get('human_evidence', '')}."
                land_cover_usage = f"Dominant cover: {parsed_data1.get('dominant_cover', '')}. Ground state: {parsed_data1.get('soil_surface_state', '')}."
                type_of_vegetation = f"Veg: {parsed_data1.get('vegetation_detail', '')}. Cond: {parsed_data1.get('vegetation_condition', '')}. Phenology: {parsed_data1.get('phenological_stage', '')}. Canopy: {parsed_data1.get('canopy_structure', '')}."
                parsed_data1_mapped = {
                    "visible_evidence": visible_evidence,
                    "human_activities": human_activities,
                    "land_cover_usage": land_cover_usage,
                    "type_of_vegetation": type_of_vegetation
                }
            else:
                visible_evidence = parsed_data1.get('visible_evidence', '')
                human_activities = parsed_data1.get('human_activities', '')
                land_cover_usage = parsed_data1.get('land_cover_usage', '')
                type_of_vegetation = parsed_data1.get('type_of_vegetation', '')
                parsed_data1_mapped = parsed_data1

            # Step 2: Classify based on lists
            step2_prompt = prompt_step2
            if "{dominant_cover}" in step2_prompt:
                step2_prompt = step2_prompt.replace("{dominant_cover}", str(parsed_data1.get('dominant_cover', '')))
            
            step2_prompt = step2_prompt.replace("{visible_evidence}", str(visible_evidence)) \
                .replace("{human_activities}", str(human_activities)) \
                .replace("{land_cover_usage}", str(land_cover_usage)) \
                .replace("{type_of_vegetation}", str(type_of_vegetation)) \
                .replace("{lc_list}", lc_list_str) \
                .replace("{lu_list}", lu_list_str) \
                .replace("{eunis_list}", eunis_list_str)

            if args.backend == "ollama":
                response2 = ollama.generate(
                    model=args.model,
                    prompt=step2_prompt
                )
                vlm_text2 = response2.get('response', '')
            else:
                import sglang as sgl
                @sgl.function
                def step2_fn(s, prompt):
                    s += prompt + s.gen("response")
                
                res2 = step2_fn.run(prompt=step2_prompt)
                vlm_text2 = res2["response"]

            parsed_data2 = extract_json_from_response(vlm_text2)
            if parsed_data2 is None:
                print("  -> Failed to parse JSON from Step 2.")
                print(f"  -> Raw Output: {vlm_text2}")
                continue

            parsed_data = {**parsed_data1_mapped, **parsed_data2}

            # Embeddings and image-text similarities
            img = Image.open(image_path).convert('RGB')
            combined_caption = f"{visible_evidence}. {human_activities}. {land_cover_usage}. {type_of_vegetation}".replace("..", ".")

            img_transformed = tips_transform(img).unsqueeze(0).to(device)
            tips_img_features = tips_model.encode_image(img_transformed).cls_token.detach().cpu()
            tips_text_features = tips_model.encode_text([combined_caption]).detach().cpu()

            img_emb = clip_model.encode([img])
            text_emb = clip_model.encode([combined_caption])

            clip_similarity = float(cosine_similarity(img_emb, text_emb)[0][0])
            tips_similarity = float(cosine_similarity(tips_img_features[0], tips_text_features)[0][0])

            # Evaluate predictions (semantic similarity)
            pred_lc = parsed_data.get('lc_label', '')
            pred_lu = parsed_data.get('lu_label', '')
            pred_eunis = parsed_data.get('eunis_class', '')

            sim_lc = calculate_similarity(embedder, pred_lc, meta['lc_label'])
            sim_lu = calculate_similarity(embedder, pred_lu, meta['lu_label'])
            sim_eunis = calculate_similarity(embedder, pred_eunis, meta['eunis_class'])

            # Exact match check
            exact_lc = (str(pred_lc).strip().lower() == str(meta['lc_label']).strip().lower())
            exact_lu = (str(pred_lu).strip().lower() == str(meta['lu_label']).strip().lower())
            exact_eunis = (str(pred_eunis).strip().lower() == str(meta['eunis_class']).strip().lower())

            print(f"  -> Land Cover Sim: {sim_lc:.4f} (Exact: {exact_lc}) | Pred: {pred_lc}")
            print(f"  -> Land Use Sim: {sim_lu:.4f} (Exact: {exact_lu}) | Pred: {pred_lu}")
            print(f"  -> EUNIS Sim: {sim_eunis:.4f} (Exact: {exact_eunis}) | Pred: {pred_eunis}")
            print(f"  -> CLIP Sim: {clip_similarity:.4f} | TIPSv2 Sim: {tips_similarity:.4f}")

            results.append({
                'image': filename,
                'point_id': point_id,
                'direction': direction,
                'gt_lc': meta['lc_label'],
                'gt_lu': meta['lu_label'],
                'gt_eunis': meta['eunis_class'],
                'pred_lc': pred_lc,
                'pred_lu': pred_lu,
                'pred_eunis': pred_eunis,
                'sim_lc': sim_lc,
                'sim_lu': sim_lu,
                'sim_eunis': sim_eunis,
                'exact_lc': exact_lc,
                'exact_lu': exact_lu,
                'exact_eunis': exact_eunis,
                'clip_similarity': clip_similarity,
                'tips_similarity': tips_similarity,
                'visible_evidence': visible_evidence,
                'human_activities': human_activities,
                'land_cover_usage': land_cover_usage,
                'type_of_vegetation': type_of_vegetation,
                'image_embedding': img_emb[0],
                'tips_image_embedding': tips_img_features[0][0],
                'combined_caption': combined_caption,
                'prompt_version': args.prompt_version,
                'ground_truth_macro': 'outdoor'
            })

            valid_evaluations += 1

        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")

    # Build evaluation report
    summary_report = ["\n" + "=" * 60, "LUCAS 2018 EVALUATION COMPLETE", "=" * 60]

    if valid_evaluations > 0:
        results_df = pd.DataFrame(results)

        summary_report.append(f"Total Images Evaluated: {valid_evaluations}")
        summary_report.append(f"Average CLIP Similarity: {results_df['clip_similarity'].mean():.4f}")
        summary_report.append(f"Average TIPSv2 Similarity: {results_df['tips_similarity'].mean():.4f}")
        
        # Class-level scores
        summary_report.append("\n--- Metric Summaries ---")
        summary_report.append(f"Land Cover (lc_label) Semantic Sim: {results_df['sim_lc'].mean():.4f}")
        summary_report.append(f"Land Cover (lc_label) Exact Match: {(results_df['exact_lc'].sum() / len(results_df)) * 100:.2f}%")
        
        summary_report.append(f"Land Use (lu_label) Semantic Sim: {results_df['sim_lu'].mean():.4f}")
        summary_report.append(f"Land Use (lu_label) Exact Match: {(results_df['exact_lu'].sum() / len(results_df)) * 100:.2f}%")
        
        summary_report.append(f"EUNIS Class Semantic Sim: {results_df['sim_eunis'].mean():.4f}")
        summary_report.append(f"EUNIS Class Exact Match: {(results_df['exact_eunis'].sum() / len(results_df)) * 100:.2f}%")

        # Direction-wise Performance Breakdown
        summary_report.append("\n--- Direction-wise Performance ---")
        for direction_val, group in results_df.groupby('direction'):
            summary_report.append(f"\nDirection: {direction_val} ({len(group)} images)")
            summary_report.append(f"  Land Cover Sim: {group['sim_lc'].mean():.4f} (Exact: {(group['exact_lc'].sum() / len(group)) * 100:.2f}%)")
            summary_report.append(f"  Land Use Sim:   {group['sim_lu'].mean():.4f} (Exact: {(group['exact_lu'].sum() / len(group)) * 100:.2f}%)")
            summary_report.append(f"  EUNIS Class Sim: {group['sim_eunis'].mean():.4f} (Exact: {(group['exact_eunis'].sum() / len(group)) * 100:.2f}%)")
            summary_report.append(f"  CLIP Similarity: {group['clip_similarity'].mean():.4f}")

        # Save to output files
        model_tag = args.model.replace(':', '_').replace('/', '_')
        
        summary_file = f"vlm_lucas_evaluation_summary_{model_tag}_{args.prompt_version}.txt"
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_report))
        print("\n".join(summary_report))
        print(f"\nSummary report saved to: {summary_file}")

        output_csv = f"vlm_lucas_evaluation_results_{model_tag}_{args.prompt_version}.csv"
        cols_to_drop = ['image_embedding', 'combined_caption', 'tips_image_embedding']
        results_df.drop(columns=cols_to_drop).to_csv(output_csv, index=False)
        print(f"Detailed results CSV saved to: {output_csv}")

        retrieval_file = f"vlm_lucas_retrieval_data_{model_tag}_{args.prompt_version}.pkl"
        retrieval_data = results_df[[
            'image', 'point_id', 'direction', 'combined_caption', 'image_embedding', 'tips_image_embedding', 
            'prompt_version', 'visible_evidence', 'human_activities', 'land_cover_usage', 'type_of_vegetation',
            'ground_truth_macro'
        ]].to_dict('records')
        with open(retrieval_file, 'wb') as f:
            pickle.dump(retrieval_data, f)
        print(f"Retrieval embeddings pickle saved to: {retrieval_file}")
    else:
        print("No valid evaluations completed.")


if __name__ == "__main__":
    main()
