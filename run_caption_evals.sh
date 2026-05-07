#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ananthu_venv
# ==========================================
# CONFIGURATION
# ==========================================
# Update these variables to match your environment
PYTHON_SCRIPT="caption_test.py"
IMG_DIR="/home/ananthu/DATA/data_ananthu/places365/versions/1/val"      # Path to your val/train directory
LABELS_FILE="/home/ananthu/DATA/data_ananthu/places365/versions/1/Scene_hierarchy.xlsx"
MAX_IMAGES=1000                    # Set to 0 to run the entire dataset

# Define the list of models you want to evaluate.
# Make sure these exact names match what you see when you type `ollama list`.
MODELS=(
    # "gemma4:e4b"
    # "llava:7b"
    "qwen3-vl:8b"
    # "llama3.2-vision:11b"
)

# ==========================================
# PRE-FLIGHT CHECKS
# ==========================================
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script '$PYTHON_SCRIPT' not found in the current directory!"
    exit 1
fi

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
echo "====================================================="

# ==========================================
# EXECUTION LOOP
# ==========================================
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "▶️  [START] Evaluating Model: $MODEL"
    echo "-----------------------------------------------------"
    
    # Run the python script with the arguments
    python3 "$PYTHON_SCRIPT" \
        --model "$MODEL" \
        --img_dir "$IMG_DIR" \
        --labels "$LABELS_FILE" \
        --max_images "$MAX_IMAGES"
        
    # Check if the python command executed successfully
    if [ $? -eq 0 ]; then
        echo "✅ [SUCCESS] Completed evaluation for $MODEL"
    else
        echo "❌ [ERROR] An error occurred while evaluating $MODEL"
        # Optional: exit 1 # Uncomment this if you want the bash script to stop entirely on a failure
    fi
    echo "-----------------------------------------------------"
done

echo ""
echo "🎉 All automated evaluations are complete!"
echo "Check your directory for the generated .csv result files."