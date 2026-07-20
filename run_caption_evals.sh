#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ananthu_venv
# ==========================================
# CONFIGURATION
# ==========================================
# Update these variables to match your environment
PYTHON_SCRIPT="src.scrapers.caption_test"
IMG_DIR="/home/ananthu/DATA/data_ananthu/places365/versions/1/val"      # Path to your val/train directory
LABELS_FILE="/home/ananthu/DATA/data_ananthu/places365/versions/1/Scene_hierarchy.xlsx"
MAX_IMAGES=1000                    # Set to 0 to run the entire dataset

# Define the list of models you want to evaluate.
# Make sure these exact names match what you see when you type `ollama list`.
MODELS=(
    "gemma4:e4b"
    # "llava:7b"
    #"qwen3-vl:8b"
    # "llama3.2-vision:11b"
)

# Define the list of prompt versions you want to evaluate (folder names in prompts/)
VERSIONS=(
    "v1"
    "v2"
)

# ==========================================
# PRE-FLIGHT CHECKS
# ==========================================

if [ ! -d "$IMG_DIR" ]; then
    echo "❌ Error: Image directory '$IMG_DIR' not found!"
    exit 1
fi

echo "====================================================="
echo "🚀 Starting VLM Benchmark Batch Job"
echo "====================================================="
echo "📁 Image Directory: $IMG_DIR"
echo "🖼️  Max Images per run: $MAX_IMAGES"
echo "🤖 Models queued: ${#MODELS[@]}"
echo "📝 Versions queued: ${#VERSIONS[@]}"
echo "====================================================="

# ==========================================
# EXECUTION LOOP
# ==========================================
for MODEL in "${MODELS[@]}"; do
    for VERSION in "${VERSIONS[@]}"; do
        echo ""
        echo "▶️  [START] Evaluating Model: $MODEL | Version: $VERSION"
        echo "-----------------------------------------------------"

        # Run the python script with the arguments
        python3 -m "$PYTHON_SCRIPT" \
            --model "$MODEL" \
            --img_dir "$IMG_DIR" \
            --labels "$LABELS_FILE" \
            --max_images "$MAX_IMAGES" \
            --prompt_version "$VERSION"

        # Check if the python command executed successfully
        if [ $? -eq 0 ]; then
            echo "✅ [SUCCESS] Completed $MODEL with $VERSION"
        else
            echo "❌ [ERROR] An error occurred while evaluating $MODEL with $VERSION"
            # Optional: exit 1 
        fi
        echo "-----------------------------------------------------"
    done
done


echo ""
echo "🎉 All automated evaluations are complete!"
echo "Check your directory for the generated .csv result files."