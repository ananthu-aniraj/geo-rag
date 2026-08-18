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
YAML_PATH="config/evaluation/lucas_evals.yaml"
PYTHON_SCRIPT=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['eval'].get('python_script', 'src.evaluation.evaluate_lucas'))" 2>/dev/null)
IMG_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['eval'].get('img_dir', ''))" 2>/dev/null)
CSV_FILE=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['eval'].get('csv_file', ''))" 2>/dev/null)
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

# The user noted that images are not stored on this laptop, so warn and verify if path is set
if [ "$IMG_DIR" == "/path/to/lucas/images" ] || [ ! -d "$IMG_DIR" ]; then
    echo "⚠️  Warning: Image directory '$IMG_DIR' does not exist or is still set to default."
    echo "Please edit this script and set IMG_DIR to the actual path of the LUCAS images directory."
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "❌ Error: Metadata CSV file '$CSV_FILE' not found!"
    exit 1
fi

echo "====================================================="
echo "🚀 Starting LUCAS 2018 VLM Benchmark Batch Job"
echo "====================================================="
echo "📁 Image Directory: $IMG_DIR"
echo "📊 Metadata CSV:    $CSV_FILE"
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
            --csv "$CSV_FILE" \
            --max_images "$MAX_IMAGES" \
            --prompt_version "$VERSION"; then
            echo "✅ [SUCCESS] Completed $MODEL with $VERSION"
        else
            echo "❌ [ERROR] An error occurred while evaluating $MODEL with $VERSION"
        fi
        echo "-----------------------------------------------------"
    done
done

echo ""
echo "🎉 All automated evaluations are complete!"
echo "Check your directory for the generated .csv and .txt result files."
