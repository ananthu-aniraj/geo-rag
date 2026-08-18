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
PYTHON_SCRIPT=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['eval'].get('python_script', 'src.evaluation.caption_test'))" 2>/dev/null)
IMG_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['eval'].get('img_dir', ''))" 2>/dev/null)
LABELS_FILE=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['eval'].get('labels_file', ''))" 2>/dev/null)
MAX_IMAGES=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['eval'].get('max_images', 1000))" 2>/dev/null)

# Load models array
MODELS_STR=$(python3 -c "
import yaml
with open('$YAML_PATH') as f:
    models = yaml.safe_load(f)['eval'].get('models', [])
    print(' '.join(['\"' + m + '\"' for m in models]))
" 2>/dev/null)
eval "MODELS=($MODELS_STR)"

# Load versions array
VERSIONS_STR=$(python3 -c "
import yaml
with open('$YAML_PATH') as f:
    versions = yaml.safe_load(f)['eval'].get('versions', [])
    print(' '.join(['\"' + v + '\"' for v in versions]))
" 2>/dev/null)
eval "VERSIONS=($VERSIONS_STR)"


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