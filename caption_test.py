import os
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Load Tips model")
tips_model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
tips_model.eval().to(device)


# Initialize the lightweight embedding model for semantic similarity scoring
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading CLIP model...")
clip_model = SentenceTransformer('clip-ViT-B-32')

# ==========================================
# PROMPT DESIGN
# ==========================================
PROMPT_STEP1 = """
Analyze the provided image and output a JSON object with the following information:
1. Human Activities: What are people doing here, or what activities does the infrastructure support?
2. Land Cover/Usage: What is on the ground (e.g., asphalt, grass, carpet) and how is the space utilized?
3. Type of Vegetation (if any): Describe the type of vegetation present, if applicable (e.g., grass, trees, shrubs). If none, state "none".

You must respond ONLY with a valid JSON object. Do not include markdown formatting or backticks.
{
  "human_activities": "...",
  "land_cover_usage": "...",
   "type_of_vegetation": "..."
}
"""

PROMPT_STEP2 = """
Based on the following description of a scene, output a JSON object detailing the hierarchy:

Description:
Human Activities: {human_activities}
Land Cover/Usage: {land_cover_usage}
Type of Vegetation: {type_of_vegetation}

Hierarchical structure to derive:
1. Macro Category: Classify the scene into exactly one of these: 'indoor', 'outdoor natural', or 'outdoor man-made'.
2. Sub Category: Further categorize the scene. Choose the BEST fit from this list:
{sub_categories_list}

3. Environment/Landscape: Describe the physical surroundings (e.g., specific terrain, architectural style, or interior setting).
4. Type of Place: A concise label for the specific location. Choose the BEST fit from this list:
{type_of_places_list}

You must respond ONLY with a valid JSON object. Do not include markdown formatting or backticks.
{
  "macro_category": "...",
  "sub_category": "...",
  "environment_landscape": "...",
  "type_of_place": "..."
}
"""


def load_places365_labels(filepath):
    """Loads the valid Places365 categories from the provided Excel file and returns a mapping to macro and sub-categories, plus unique lists."""
    if not os.path.exists(filepath):
        print(f"Warning: Labels file '{filepath}' not found. Continuing without it.")
        return {}, [], []

    print(f"Loading labels from {filepath}...")
    try:
        # Load with multi-index header to handle the structure
        df = pd.read_excel(filepath, header=[0, 1])

        mapping = {}
        all_sub_cats = set()
        all_types = []

        # Get unique sub-categories from columns
        for col in df.columns:
            if col[0].startswith("Level 2"):
                all_sub_cats.add(col[1])

        for _, row in df.iterrows():
            # Extract category name and normalize it (e.g., '/a/airfield' -> 'airfield')
            raw_cat = row[('Unnamed: 0_level_0', 'category')]
            if not isinstance(raw_cat, str): continue

            # Clean category name: remove quotes and slashes
            clean_cat = raw_cat.strip("'").strip("/")
            # Remove single-letter prefix directory (e.g., 'a/', 'i/')
            if '/' in clean_cat:
                parts = clean_cat.split('/')
                if len(parts[0]) == 1:
                    clean_cat = "/".join(parts[1:])

            # Replace remaining '/' with '-' as per user instructions
            # Underscores are preserved to match folder names
            clean_cat = clean_cat.replace("/", "-")
            all_types.append(clean_cat)

            # Identify macro-category
            macro = "unknown"
            if row[('Level 1', 'indoor')] == 1:
                macro = "indoor"
            elif row[('Level 1', 'outdoor, natural')] == 1:
                macro = "outdoor natural"
            elif row[('Level 1', 'outdoor, man-made')] == 1:
                macro = "outdoor man-made"

            # Identify sub-category (Level 2)
            sub_cat = "unknown"
            for col in df.columns:
                if col[0].startswith("Level 2") and row[col] == 1:
                    sub_cat = col[1]
                    break

            mapping[clean_cat] = {"macro": macro, "sub_category": sub_cat}

        return mapping, sorted(list(all_sub_cats)), sorted(all_types)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}, [], []


def extract_json_from_response(response_text):
    """Safely extracts JSON from the model's response."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

    return None


def calculate_similarity(predicted_place, ground_truth):
    """Calculates cosine similarity between the model's guess and the actual label."""
    if not predicted_place or not ground_truth:
        return 0.0

    embeddings = embedder.encode([predicted_place, ground_truth])
    sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(sim_score)


def clean_macro_category(raw_string):
    """Normalizes the VLM's multiple choice answer to handle capitalization or weird punctuation."""
    clean_str = str(raw_string).lower().strip()
    if "indoor" in clean_str:
        return "indoor"
    elif "man-made" in clean_str or "man made" in clean_str:
        return "outdoor man-made"
    elif "natural" in clean_str:
        return "outdoor natural"
    else:
        return "unknown"  # Failsafe if the model hallucinates a 4th option


def main():
    parser = argparse.ArgumentParser(description="Evaluate local VLMs against Places365 using Ollama.")
    parser.add_argument("--model", type=str, default="llava:13b", help="The name of the model in Ollama.")
    parser.add_argument("--labels", type=str,
                        help="Path to the Places365 Scene hierarchy.xlsx file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the split directory.")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Maximum number of images to evaluate. Set to 0 for unlimited.")
    args = parser.parse_args()

    if not os.path.exists(args.img_dir):
        print(f"Error: Directory '{args.img_dir}' not found.")
        return

    # Attempt to load labels, but script will proceed even if it fails (using folder names as ground truth)
    labels_mapping, sub_categories_list, type_of_places_list = load_places365_labels(args.labels)

    sub_cats_str = "\n".join([f"- {c}" for c in sub_categories_list]) if sub_categories_list else "None"
    type_of_places_str = ", ".join(type_of_places_list) if type_of_places_list else "None"

    print(f"Scanning directory structure in {args.img_dir}...")
    images_by_class = {}

    for root, _, files in os.walk(args.img_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                class_folder_name = os.path.basename(root)
                ground_truth_label = class_folder_name
                if ground_truth_label not in images_by_class:
                    images_by_class[ground_truth_label] = []
                images_by_class[ground_truth_label].append((image_path, filename, ground_truth_label))

    if not images_by_class:
        print("No images found in the provided directory.")
        return

    # Sort classes and images within each class for determinism
    sorted_classes = sorted(images_by_class.keys())
    for cls in sorted_classes:
        images_by_class[cls].sort()

    # Shuffle each class with a fixed seed to maintain determinism but randomize selection within class
    rng = random.Random(42)
    for cls in sorted_classes:
        rng.shuffle(images_by_class[cls])

    # Interleave images to maximize class coverage
    all_images = []
    max_images_in_any_class = max(len(images_by_class[cls]) for cls in sorted_classes)

    for i in range(max_images_in_any_class):
        for cls in sorted_classes:
            if i < len(images_by_class[cls]):
                all_images.append(images_by_class[cls][i])

    total_found = len(all_images)
    if args.max_images > 0:
        all_images = all_images[:args.max_images]
        print(f"Found {total_found} images across {len(sorted_classes)} classes.")
        print(f"Limiting evaluation to {args.max_images} images with maximized class coverage.")

    results = []
    total_class_score = 0.0
    total_sub_score = 0.0
    macro_correct_count = 0
    valid_evaluations = 0

    print(f"\nStarting evaluation using model: {args.model}")
    print("-" * 50)

    for image_path, filename, ground_truth_label in all_images:
        print(f"\nProcessing: {filename}")
        print(f"Ground Truth Class: {ground_truth_label}")

        # Determine Ground Truth Hierarchy
        gt_info = labels_mapping.get(ground_truth_label, {"macro": "unknown", "sub_category": "unknown"})
        gt_macro = gt_info["macro"]
        gt_sub = gt_info["sub_category"]

        if gt_macro != "unknown":
            print(f"Ground Truth Macro Category: {gt_macro}")
        if gt_sub != "unknown":
            print(f"Ground Truth Sub Category: {gt_sub}")

        try:
            # Step 1: Extract Human Activities and Land Cover from Image
            response1 = ollama.generate(
                model=args.model,
                prompt=PROMPT_STEP1,
                images=[image_path]
            )
            vlm_text1 = response1.get('response', '')
            parsed_data1 = extract_json_from_response(vlm_text1)

            if parsed_data1 is None:
                print("  -> Failed to parse JSON from Step 1 model output.")
                continue

            human_activities = parsed_data1.get('human_activities', '')
            land_cover_usage = parsed_data1.get('land_cover_usage', '')
            type_of_vegetation = parsed_data1.get('type_of_vegetation', '')

            # Step 2: Derive Categories from Text
            step2_prompt = PROMPT_STEP2.replace("{human_activities}", str(human_activities)) \
                .replace("{land_cover_usage}", str(land_cover_usage)) \
                .replace("{type_of_vegetation}", str(type_of_vegetation)) \
                .replace("{sub_categories_list}", sub_cats_str) \
                .replace("{type_of_places_list}", type_of_places_str)
            response2 = ollama.generate(
                model=args.model,
                prompt=step2_prompt
            )
            vlm_text2 = response2.get('response', '')
            parsed_data2 = extract_json_from_response(vlm_text2)

            if parsed_data2 is None:
                print("  -> Failed to parse JSON from Step 2 model output.")
                continue

            # Merge results
            parsed_data = {**parsed_data1, **parsed_data2}

            # CLIP Similarity Calculation
            img = Image.open(image_path).convert('RGB')
            combined_caption = f"{human_activities}. {land_cover_usage}"

            # TIPSv2 similarity
            img_transformed = tips_transform(img).unsqueeze(0).to(device)  # Add batch dimension
            tips_img_features = tips_model.encode_image(img_transformed).cls_token.to('cpu')
            tips_text_features = tips_model.encode_text([combined_caption]).to('cpu')

            # Get embeddings (passing as list ensures 2D output)
            img_emb = clip_model.encode([img])
            text_emb = clip_model.encode([combined_caption])

            # Calculate similarity (already 2D, no need to wrap in [])
            clip_similarity = float(cosine_similarity(img_emb, text_emb)[0][0])
            print(f"  -> CLIP Similarity: {clip_similarity:.4f}")

            tips_similarity = float(cosine_similarity(tips_img_features, tips_text_features)[0][0])
            print(f"  -> TIPSv2 Similarity: {tips_similarity:.4f}")

            # 1. Evaluate the Class Name (Semantic Similarity)
            predicted_place = parsed_data.get('type_of_place', '')
            class_similarity = calculate_similarity(predicted_place, ground_truth_label)

            # 2. Extract and Evaluate the Macro Category
            raw_macro = parsed_data.get('macro_category', '')
            cleaned_macro = clean_macro_category(raw_macro)
            macro_correct = (cleaned_macro == gt_macro) if gt_macro != "unknown" else None

            # 3. Extract and Evaluate Sub Category (Semantic Similarity)
            predicted_sub = parsed_data.get('sub_category', '')
            sub_similarity = calculate_similarity(predicted_sub, gt_sub) if gt_sub != "unknown" else 0.0

            print(f"  -> Predicted Macro Category: {cleaned_macro}")
            if macro_correct is not None:
                print(f"  -> Macro Category Correct: {macro_correct}")
                if macro_correct: macro_correct_count += 1

            print(f"  -> Predicted Sub Category: {predicted_sub}")
            if gt_sub != "unknown":
                print(f"  -> Sub Category Similarity Score: {sub_similarity:.4f}")

            print(f"  -> Predicted Place: {predicted_place}")
            print(f"  -> Class Similarity Score: {class_similarity:.4f}")

            results.append({
                'image': filename,
                'ground_truth_class': ground_truth_label,
                'ground_truth_macro': gt_macro,
                'ground_truth_sub': gt_sub,
                'predicted_macro_category': cleaned_macro,
                'macro_correct': macro_correct,
                'predicted_sub_category': predicted_sub,
                'sub_category_similarity_score': sub_similarity,
                'predicted_place': predicted_place,
                'class_similarity_score': class_similarity,
                'clip_similarity': clip_similarity,
                'tips_similarity_score': tips_similarity,
                'environment_landscape': parsed_data.get('environment_landscape', ''),
                'human_activities': human_activities,
                'land_cover_usage': land_cover_usage,
                'type_of_vegetation': type_of_vegetation,
                'image_embedding': img_emb[0],
                'combined_caption': combined_caption})

            total_class_score += class_similarity
            if gt_sub != "unknown":
                total_sub_score += sub_similarity
            valid_evaluations += 1

        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)

    if valid_evaluations > 0:
        avg_class_score = total_class_score / valid_evaluations
        print(f"Total Images Evaluated: {valid_evaluations}")
        print(f"Average Class (Place) Similarity: {avg_class_score:.4f}")

        # Calculate scores for those with ground truth
        results_df = pd.DataFrame(results)

        avg_clip_score = results_df['clip_similarity'].mean()
        print(f"Average CLIP Similarity: {avg_clip_score:.4f}")

        macro_evaluable = results_df[results_df['ground_truth_macro'] != 'unknown']
        if not macro_evaluable.empty:
            macro_acc = (macro_evaluable['macro_correct'].sum() / len(macro_evaluable)) * 100
            print(
                f"Macro Category Accuracy: {macro_acc:.2f}% ({macro_evaluable['macro_correct'].sum()}/{len(macro_evaluable)})")

        sub_evaluable = results_df[results_df['ground_truth_sub'] != 'unknown']
        if not sub_evaluable.empty:
            avg_sub_score = sub_evaluable['sub_category_similarity_score'].mean()
            print(f"Average Sub Category Similarity: {avg_sub_score:.4f}")

        # Calculate how often the model picked each macro category
        macro_counts = results_df['predicted_macro_category'].value_counts().to_dict()
        print(f"Model Macro Category Distribution: {macro_counts}")

        output_csv = f"vlm_evaluation_results_{args.model.replace(':', '_')}.csv"
        # Drop the embedding column before saving CSV to keep it readable and small
        results_df.drop(columns=['image_embedding', 'combined_caption']).to_csv(output_csv, index=False)
        print(f"\nDetailed results saved to: {output_csv}")

        # Save retrieval data (filename, caption, embedding) to a pickle file
        retrieval_file = f"vlm_retrieval_data_{args.model.replace(':', '_')}.pkl"
        retrieval_data = results_df[['image', 'combined_caption', 'image_embedding']].to_dict('records')
        with open(retrieval_file, 'wb') as f:
            pickle.dump(retrieval_data, f)
        print(f"Retrieval data (embeddings + captions) saved to: {retrieval_file}")
    else:
        print("No valid evaluations were completed.")


if __name__ == "__main__":
    main()
