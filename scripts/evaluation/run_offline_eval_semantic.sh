#!/usr/bin/env bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# Exit immediately if a command exits with a non-zero status
set -e

echo "================================================================================"
echo "      Starting Semantic Representation Evaluation (LUCAS & Places365)           "
echo "================================================================================"

# Helper function to read yaml values using Python
get_param() {
    python3 -c "import yaml; print(yaml.safe_load(open('config/evaluation/params_offline.yaml'))['$1']['$2'])"
}

# Load parameters from YAML
LUCAS_MODEL=$(get_param "lucas" "model_name")
LUCAS_CSV=$(get_param "lucas" "csv")
LUCAS_IMG_DIR=$(get_param "lucas" "img_dir")
LUCAS_QUERIES=$(get_param "lucas" "num_queries")
LUCAS_DB=$(get_param "lucas" "num_database")
LUCAS_BATCH=$(get_param "lucas" "batch_size")
LUCAS_SEED=$(get_param "lucas" "seed")
LUCAS_USE_SEG=$(get_param "lucas" "use_segformer")

PLACES_MODEL=$(get_param "places" "model_name")
PLACES_LABELS=$(get_param "places" "labels")
PLACES_IMG_DIR=$(get_param "places" "img_dir")
PLACES_QUERIES=$(get_param "places" "num_queries")
PLACES_DB=$(get_param "places" "num_database")
PLACES_BATCH=$(get_param "places" "batch_size")
PLACES_SEED=$(get_param "places" "seed")
PLACES_USE_SEG=$(get_param "places" "use_segformer")
PLACES_COMP_CLIP=$(get_param "places" "compare_clip")

OUTPUT_DIR="./benchmark_results"
mkdir -p "$OUTPUT_DIR"

# Validate directories
if [ ! -d "$LUCAS_IMG_DIR" ]; then
    echo "Warning: LUCAS image directory not found at '$LUCAS_IMG_DIR'."
fi
if [ ! -d "$PLACES_IMG_DIR" ]; then
    echo "Warning: Places365 validation image directory not found at '$PLACES_IMG_DIR'."
fi

# Optional raster parameters for spatial overlay in LUCAS benchmark (loaded from config/evaluation/params_online.yaml)
ENV_RASTER=$(python3 -c "import yaml; print(yaml.safe_load(open('config/evaluation/params_online.yaml'))['environmental_zones']['raster'])")
EUNIS_RASTER=$(python3 -c "import yaml; print(yaml.safe_load(open('config/evaluation/params_online.yaml'))['eunis']['raster'])")

LUCAS_SEG_FLAG=""
if [ "$LUCAS_USE_SEG" = "false" ] || [ "$LUCAS_USE_SEG" = "False" ]; then
    LUCAS_SEG_FLAG="--no_segformer"
fi

PLACES_SEG_FLAG=""
if [ "$PLACES_USE_SEG" = "false" ] || [ "$PLACES_USE_SEG" = "False" ]; then
    PLACES_SEG_FLAG="--no_segformer"
fi

PLACES_CLIP_FLAG=""
if [ "$PLACES_COMP_CLIP" = "true" ] || [ "$PLACES_COMP_CLIP" = "True" ]; then
    PLACES_CLIP_FLAG="--compare_clip"
fi

# Sanitize model names to prevent directory traversal issues in filenames
LUCAS_MODEL_CLEAN="${LUCAS_MODEL//\//_}"
PLACES_MODEL_CLEAN="${PLACES_MODEL//\//_}"

echo "Configuration (Loaded from config/evaluation/params_offline.yaml & config/evaluation/params_online.yaml):"
echo "- LUCAS Model: $LUCAS_MODEL"
echo "- LUCAS Metadata: $LUCAS_CSV"
echo "- LUCAS Image Dir: $LUCAS_IMG_DIR"
echo "- LUCAS SegFormer Active: $LUCAS_USE_SEG"
echo "- EUNIS Raster: $EUNIS_RASTER"
echo "- Env Zones Raster: $ENV_RASTER"
echo "- Places365 Model: $PLACES_MODEL"
echo "- Places365 Hierarchy: $PLACES_LABELS"
echo "- Places365 Image Dir: $PLACES_IMG_DIR"
echo "- Places365 SegFormer Active: $PLACES_USE_SEG"
echo "- Places365 Compare CLIP: $PLACES_COMP_CLIP"
echo "================================================================================"

# --- Step 1: Benchmark LUCAS ---
echo -e "\n[1/2] Running LUCAS Semantic Representation Benchmark..."
python3 -m src.evaluation.benchmark_lucas \
  --model_name "$LUCAS_MODEL" \
  --csv "$LUCAS_CSV" \
  --img_dir "$LUCAS_IMG_DIR" \
  --num_queries "$LUCAS_QUERIES" \
  --num_database "$LUCAS_DB" \
  --batch_size "$LUCAS_BATCH" \
  --seed "$LUCAS_SEED" \
  --eunis_raster "$EUNIS_RASTER" \
  --env_zones_raster "$ENV_RASTER" \
  $LUCAS_SEG_FLAG \
  --output_report "$OUTPUT_DIR/lucas_report_${LUCAS_MODEL_CLEAN}.txt" \
  --output_csv "$OUTPUT_DIR/lucas_results_${LUCAS_MODEL_CLEAN}.csv"

# --- Step 2: Benchmark Places365 ---
echo -e "\n[2/2] Running Places365 Semantic Representation Benchmark..."
python3 -m src.evaluation.benchmark_places \
  --model_name "$PLACES_MODEL" \
  --labels "$PLACES_LABELS" \
  --img_dir "$PLACES_IMG_DIR" \
  --num_queries "$PLACES_QUERIES" \
  --num_database "$PLACES_DB" \
  --batch_size "$PLACES_BATCH" \
  --seed "$PLACES_SEED" \
  $PLACES_CLIP_FLAG \
  $PLACES_SEG_FLAG \
  --output_report "$OUTPUT_DIR/places_report_${PLACES_MODEL_CLEAN}.txt" \
  --output_csv "$OUTPUT_DIR/places_results_${PLACES_MODEL_CLEAN}.csv"

echo -e "\n================================================================================"
echo "✅ Semantic evaluations completed successfully!"
echo "Reports saved to:"
echo "- $OUTPUT_DIR/lucas_report_${LUCAS_MODEL_CLEAN}.txt"
echo "- $OUTPUT_DIR/places_report_${PLACES_MODEL_CLEAN}.txt"
echo "================================================================================"
