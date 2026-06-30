#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ananthu_venv
# ==========================================
# CONFIGURATION
# ==========================================
# Update these variables to match your environment
PYTHON_SCRIPT="evaluate_lucas.py"
IMG_DIR="/home/ananthu/DATA/data_ananthu/LUCAS2018"                                             # Update this to the actual image directory path
CSV_FILE="/home/ananthu/DATA/data_ananthu/LUCAS2018/Sen4Map_Metadata_test.csv"
MAX_IMAGES=1000                                                             # Set to 0 to run the entire dataset

# Define the list of models you want to evaluate.
MODELS=(
    "gemma4:e4b"
    # "llava:7b"
    # "qwen3-vl:8b"
    # "llama3.2-vision:11b"
)

# Define the list of prompt versions you want to evaluate (folder names in prompts_lucas/)
VERSIONS=(
    "v1"
    "v2"
    "v3"
)

# ==========================================
# PRE-FLIGHT CHECKS
# ==========================================
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script '$PYTHON_SCRIPT' not found in the current directory!"
    exit 1
fi

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

        # Run the python script with the arguments
        python3 "$PYTHON_SCRIPT" \
            --model "$MODEL" \
            --img_dir "$IMG_DIR" \
            --csv "$CSV_FILE" \
            --max_images "$MAX_IMAGES" \
            --prompt_version "$VERSION"

        # Check if the python command executed successfully
        if [ $? -eq 0 ]; then
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
