#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

eval "$(conda shell.bash hook)"
conda activate ananthu_venv
# ==========================================
# CONFIGURATION
# ==========================================
YAML_PATH="config/evaluation/caption_evals.yaml"
# Helper function to read yaml values using Python (with local overrides)
get_param() {
    VAL=$(python3 -m src.utils.config get "$YAML_PATH" "caption_evals" "$1")
    if [ -z "$VAL" ]; then
        VAL=$(python3 -m src.utils.config get "$YAML_PATH" "eval" "$1")
    fi
    echo "$VAL"
}

PYTHON_SCRIPT=$(get_param "python_script")
[ -z "$PYTHON_SCRIPT" ] && PYTHON_SCRIPT="src.evaluation.caption_test"
IMG_DIR=$(get_param "img_dir")
LABELS_FILE=$(get_param "labels_file")
MAX_IMAGES=$(get_param "max_images")
[ -z "$MAX_IMAGES" ] && MAX_IMAGES=1000

# Load models and versions array
read -r -a MODELS <<< "$(get_param "models")"
read -r -a VERSIONS <<< "$(get_param "versions")"


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

        # Run the python script and check status directly
        if python3 -m "$PYTHON_SCRIPT" \
            --model "$MODEL" \
            --img_dir "$IMG_DIR" \
            --labels "$LABELS_FILE" \
            --max_images "$MAX_IMAGES" \
            --prompt_version "$VERSION"; then
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
