import os
import json
import re
import argparse
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the lightweight embedding model for semantic similarity scoring
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# PROMPT DESIGN
# ==========================================
PROMPT = """
Analyze the provided image and output a JSON object detailing the scene. 
You must respond ONLY with a valid JSON object. Do not include markdown formatting, backticks, or conversational text.
Include exactly these four keys:

{
  "environment_landscape": "Describe the physical surroundings in 1-2 sentences (e.g., mountains, urban skyline, coastal, arid).",
  "human_activities": "Describe what human activities are taking place or strongly hinted at by the infrastructure.",
  "type_of_place": "Categorize this location concisely in 1-3 words (e.g., auto showroom, wheat field, coast).",
  "land_cover_usage": "Describe the physical material on the surface and how the land is used."
}
"""

def load_places365_labels(filepath):
    """Loads the valid Places365 categories from the provided Excel file."""
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

def main():
    # Setup argparse
    parser = argparse.ArgumentParser(description="Evaluate local VLMs against Places365 using Ollama.")
    parser.add_argument("--model", type=str, default="llava:13b", help="The name of the model in Ollama (e.g., llava:13b, qwen2-vl).")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing your test images.")
    parser.add_argument("--labels", type=str, default="Scene hierarchy.xlsx", help="Path to the Places365 Scene hierarchy.xlsx file.")
    
    args = parser.parse_args()

    labels_df = load_places365_labels(args.labels)
    
    if not os.path.exists(args.img_dir):
        print(f"Error: Image directory '{args.img_dir}' not found.")
        return

    results = []
    total_score = 0.0
    valid_evaluations = 0

    print(f"\nStarting evaluation using model: {args.model}")
    print("-" * 50)
    
    for filename in os.listdir(args.img_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(args.img_dir, filename)
        ground_truth_label = filename.rsplit('_', 1)[0].replace('_', ' ') 
        
        print(f"\nProcessing: {filename}")
        print(f"Ground Truth: {ground_truth_label}")

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
                
            predicted_place = parsed_data.get('type_of_place', '')
            similarity = calculate_similarity(predicted_place, ground_truth_label)
            
            print(f"  -> Predicted Place: {predicted_place}")
            print(f"  -> Similarity Score: {similarity:.4f}")
            
            results.append({
                'image': filename,
                'ground_truth': ground_truth_label,
                'predicted_place': predicted_place,
                'similarity_score': similarity,
                'environment_landscape': parsed_data.get('environment_landscape', ''),
                'human_activities': parsed_data.get('human_activities', ''),
                'land_cover_usage': parsed_data.get('land_cover_usage', '')
            })
            
            total_score += similarity
            valid_evaluations += 1
            
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    
    if valid_evaluations > 0:
        average_score = total_score / valid_evaluations
        print(f"Total Images Evaluated: {valid_evaluations}")
        print(f"Average Semantic Similarity Score: {average_score:.4f}")
        
        results_df = pd.DataFrame(results)
        output_csv = f"vlm_evaluation_results_{args.model.replace(':', '_')}.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"\nDetailed results saved to: {output_csv}")
    else:
        print("No valid evaluations were completed.")

if __name__ == "__main__":
    main()