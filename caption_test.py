import os
import json
import re
import argparse
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Initialize the lightweight embedding model for semantic similarity scoring
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# PROMPT DESIGN
# ==========================================
PROMPT = """
Analyze the provided image and output a JSON object detailing the scene. 
You must respond ONLY with a valid JSON object. Do not include markdown formatting, backticks, or conversational text.
Include exactly these five keys:

{
  "macro_category": "Classify the image into exactly one of these three options: 'indoor', 'outdoor natural', or 'outdoor man-made'.",
  "environment_landscape": "Describe the physical surroundings in 1-2 sentences (e.g., mountains, urban skyline, coastal, arid).",
  "human_activities": "Describe what human activities are taking place or strongly hinted at by the infrastructure.",
  "type_of_place": "Categorize this location concisely in 1-3 words (e.g., auto showroom, wheat field, coast).",
  "land_cover_usage": "Describe the physical material on the surface and how the land is used."
}
"""

def load_places365_labels(filepath):
    """Loads the valid Places365 categories from the provided Excel file."""
    if not os.path.exists(filepath):
        print(f"Warning: Labels file '{filepath}' not found. Continuing without it.")
        return None
        
    print(f"Loading labels from {filepath}...")
    try:
        df = pd.read_excel(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_json_from_response(response_text):
    """Safely extracts JSON from the model's response."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
                
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
        return "unknown" # Failsafe if the model hallucinates a 4th option

def main():
    parser = argparse.ArgumentParser(description="Evaluate local VLMs against Places365 using Ollama.")
    parser.add_argument("--model", type=str, default="llava:13b", help="The name of the model in Ollama.")
    parser.add_argument("--labels", type=str, default="Scene hierarchy.xlsx", help="Path to the Places365 Scene hierarchy.xlsx file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the split directory.")
    parser.add_argument("--max_images", type=int, default=100, help="Maximum number of images to evaluate. Set to 0 for unlimited.")
    args = parser.parse_args()

    if not os.path.exists(args.img_dir):
        print(f"Error: Directory '{args.img_dir}' not found.")
        return

    # Attempt to load labels, but script will proceed even if it fails (using folder names as ground truth)
    labels_df = load_places365_labels(args.labels)


    print(f"Scanning directory structure in {args.img_dir}...")
    all_images = []
    
    for root, _, files in os.walk(args.img_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                class_folder_name = os.path.basename(root)
                ground_truth_label = class_folder_name.replace('_', ' ')
                all_images.append((image_path, filename, ground_truth_label))

    if not all_images:
        print("No images found in the provided directory.")
        return

    random.shuffle(all_images)
    if args.max_images > 0:
        all_images = all_images[:args.max_images]
        print(f"Found {len(all_images)} images. Limiting evaluation to {args.max_images} images.")

    results = []
    total_class_score = 0.0
    valid_evaluations = 0

    print(f"\nStarting evaluation using model: {args.model}")
    print("-" * 50)
    
    for image_path, filename, ground_truth_label in all_images:
        print(f"\nProcessing: {filename}")
        print(f"Ground Truth Class: {ground_truth_label}")

        try:
            response = ollama.generate(
                model=args.model,
                prompt=PROMPT,
                images=[image_path],
                format='json' 
            )
            
            vlm_text = response.get('response', '')
            parsed_data = extract_json_from_response(vlm_text)
            
            if parsed_data is None:
                print("  -> Failed to parse JSON from model output.")
                continue
                
            # 1. Evaluate the Class Name (Semantic Similarity)
            predicted_place = parsed_data.get('type_of_place', '')
            class_similarity = calculate_similarity(predicted_place, ground_truth_label)
            
            # 2. Extract the Macro Category (Indoor/Outdoor)
            raw_macro = parsed_data.get('macro_category', '')
            cleaned_macro = clean_macro_category(raw_macro)
            
            print(f"  -> Predicted Macro Category: {cleaned_macro}")
            print(f"  -> Predicted Place: {predicted_place}")
            print(f"  -> Class Similarity Score: {class_similarity:.4f}")
            
            results.append({
                'image': filename,
                'ground_truth_class': ground_truth_label,
                'predicted_macro_category': cleaned_macro,
                'predicted_place': predicted_place,
                'class_similarity_score': class_similarity,
                'environment_landscape': parsed_data.get('environment_landscape', ''),
                'human_activities': parsed_data.get('human_activities', ''),
                'land_cover_usage': parsed_data.get('land_cover_usage', '')
            })
            
            total_class_score += class_similarity
            valid_evaluations += 1
            
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    
    if valid_evaluations > 0:
        avg_class_score = total_class_score / valid_evaluations
        print(f"Total Images Evaluated: {valid_evaluations}")
        print(f"Average Class Similarity: {avg_class_score:.4f}")
        
        results_df = pd.DataFrame(results)
        
        # Calculate how often the model picked each macro category
        macro_counts = results_df['predicted_macro_category'].value_counts().to_dict()
        print(f"Macro Category Distribution: {macro_counts}")
        
        output_csv = f"vlm_evaluation_results_{args.model.replace(':', '_')}.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"\nDetailed results saved to: {output_csv}")
    else:
        print("No valid evaluations were completed.")

if __name__ == "__main__":
    main()