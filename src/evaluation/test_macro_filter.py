import argparse
import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'

tips_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


def download_image(url):
    """Downloads an image from URL (resolves mapillary:// and kartaview:// virtual URIs)."""
    try:
        if url.startswith("mapillary://"):
            orig_id = url.split("://")[1].strip()
            api_url = f"https://graph.mapillary.com/{orig_id}?fields=thumb_1024_url"
            headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
            res = requests.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                url = res.json().get("thumb_1024_url")
            else:
                return None
        elif url.startswith("kartaview://"):
            orig_id = url.split("://")[1].strip()
            api_url = f"https://api.openstreetcam.org/2.0/photo/{orig_id}"
            res = requests.get(api_url, timeout=10)
            if res.status_code == 200:
                url = res.json().get("result", {}).get("data", {}).get("thumb_1024_url")
            else:
                return None

        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return Image.open(BytesIO(res.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Test script to verify the zero-shot macro/close-up filter on iNaturalist observations.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the iNaturalist CSV file.")
    parser.add_argument("--limit", type=int, default=15, help="Number of observations to test.")
    parser.add_argument("--out_dir", type=str, default="macro_test_results",
                        help="Directory to save classified test images.")
    args = parser.parse_args()

    # Load CSV
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    if df.empty:
        print("CSV file is empty.")
        return

    # Select sample
    df_sample = df.head(args.limit)
    print(f"Testing filter on first {len(df_sample)} images...")

    # Load TIPSv2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TIPSv2 model on {device}...")
    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
    model.eval().to(device)

    # Pre-compute text features
    prompts = [
        "An indoor scene",
        "An outdoor landscape or street view",
        "A close-up macro photo of a animal, single leaf, plant petal, flower, insect, mushroom, or tree bark",
        "A photo of the sky, a bird flying in the air, an insect in flight, an airplane, or a close-up of a cloud with no ground visible"
    ]
    print("Pre-computing filter text embeddings...")
    with torch.no_grad():
        text_features = model.encode_text(prompts).cpu().numpy()

    # Setup output directories
    os.makedirs(args.out_dir, exist_ok=True)
    kept_dir = os.path.join(args.out_dir, "kept_landscape")
    macro_dir = os.path.join(args.out_dir, "dropped_macro")
    indoor_dir = os.path.join(args.out_dir, "dropped_indoor")
    sky_dir = os.path.join(args.out_dir, "dropped_sky")
    for d in [kept_dir, macro_dir, indoor_dir, sky_dir]:
        os.makedirs(d, exist_ok=True)

    results_summary = []

    print("\nProcessing images...")
    for idx, row in df_sample.iterrows():
        photo_id = str(row['Photo_ID'])
        common_name = row.get('Common_Name', 'Unknown')
        url = row['Image_URL']

        print(f"\n[{idx + 1}/{len(df_sample)}] Downloading Photo ID {photo_id} ({common_name})...")
        img = download_image(url)
        if img is None:
            print(" -> Failed to download image.")
            continue

        # Extract embedding
        with torch.no_grad():
            img_tensor = tips_transform(img).unsqueeze(0).to(device)
            embedding = model.encode_image(img_tensor).cls_token.squeeze(1).cpu().numpy()

        # Compute cosine similarities
        emb_norm = embedding / (np.linalg.norm(embedding) or 1.0)
        text_norms = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        sims = np.dot(emb_norm, text_norms.T)[0]

        best_class = np.argmax(sims)
        classes_map = {0: "Indoor", 1: "Landscape", 2: "Macro/Close-up", 3: "Sky/Flying"}
        prediction = classes_map[best_class]

        # Determine decision
        decision = "KEPT"
        save_target_dir = kept_dir
        if best_class == 0:
            decision = "DROPPED (Indoor)"
            save_target_dir = indoor_dir
        elif best_class == 2:
            decision = "DROPPED (Macro/Close-up)"
            save_target_dir = macro_dir
        elif best_class == 3:
            decision = "DROPPED (Sky/Flying)"
            save_target_dir = sky_dir

        # Save image to respective folder
        img.save(os.path.join(save_target_dir, f"{photo_id}_{common_name.replace(' ', '_')}.jpg"))

        results_summary.append({
            "Photo_ID": photo_id,
            "Common_Name": common_name,
            "Indoor_Sim": sims[0],
            "Landscape_Sim": sims[1],
            "Macro_Sim": sims[2],
            "Sky_Sim": sims[3],
            "Prediction": prediction,
            "Decision": decision
        })

        print(f" -> Cosine Sims: Indoor={sims[0]:.4f} | Landscape={sims[1]:.4f} | Macro={sims[2]:.4f} | Sky={sims[3]:.4f}")
        print(f" -> Predicted: {prediction} | Decision: {decision}")

    # Display final table
    df_res = pd.DataFrame(results_summary)
    print("\n" + "=" * 105)
    print("TEST FILTER RESULTS SUMMARY")
    print("=" * 105)
    print(df_res.to_string(index=False, columns=["Photo_ID", "Common_Name", "Prediction", "Decision"]))
    print("=" * 105)
    print(f"\nImages saved to dynamic classification folders under: {os.path.abspath(args.out_dir)}")
    print(" - kept_landscape/: True outdoor views")
    print(" - dropped_macro/: Plant/animal close-ups (discarded)")
    print(" - dropped_indoor/: Indoor scenes (discarded)")
    print(" - dropped_sky/: Sky-only or flying objects (discarded)")


if __name__ == "__main__":
    main()
