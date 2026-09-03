#!/usr/bin/env bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# Exit immediately if a command exits with a non-zero status
set -e

# Source the .env file if it exists to load keys into environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "================================================================================"
echo "      Starting Spatial Representation Evaluation (Env Zones & EUNIS)            "
echo "================================================================================"

# Helper function to read yaml values using Python
get_param() {
    python3 -c "import yaml; print(yaml.safe_load(open('config/evaluation/params_online.yaml')).get('$1', {}).get('$2', ''))"
}

# Load parameters from YAML
ENV_MODEL=$(get_param "environmental_zones" "model_name")
ENV_CSV=$(get_param "environmental_zones" "csv_path")
ENV_RASTER=$(get_param "environmental_zones" "raster")
ENV_QUERIES=$(get_param "environmental_zones" "num_queries")
ENV_DB=$(get_param "environmental_zones" "num_database")
ENV_BATCH=$(get_param "environmental_zones" "batch_size")
ENV_SEED=$(get_param "environmental_zones" "seed")
ENV_QUERY_PLATFORM=$(get_param "environmental_zones" "query_platform")
ENV_VIZ_SAMPLES=$(get_param "environmental_zones" "visualize_samples")
ENV_OFFLINE=$(get_param "environmental_zones" "offline_dataset_dirs")
ENV_USE_SEG=$(get_param "environmental_zones" "use_segformer")

EUNIS_MODEL=$(get_param "eunis" "model_name")
EUNIS_CSV=$(get_param "eunis" "csv_path")
EUNIS_RASTER=$(get_param "eunis" "raster")
EUNIS_QUERIES=$(get_param "eunis" "num_queries")
EUNIS_DB=$(get_param "eunis" "num_database")
EUNIS_BATCH=$(get_param "eunis" "batch_size")
EUNIS_SEED=$(get_param "eunis" "seed")
EUNIS_QUERY_PLATFORM=$(get_param "eunis" "query_platform")
EUNIS_VIZ_SAMPLES=$(get_param "eunis" "visualize_samples")
EUNIS_OFFLINE=$(get_param "eunis" "offline_dataset_dirs")
EUNIS_USE_SEG=$(get_param "eunis" "use_segformer")

OUTPUT_DIR="./benchmark_results"
mkdir -p "$OUTPUT_DIR"

# Validate files
if [ ! -f "$ENV_CSV" ]; then
    echo "Warning: Database file not found at '$ENV_CSV'."
fi
if [ ! -f "$ENV_RASTER" ]; then
    echo "Warning: Environmental Zones raster file not found at '$ENV_RASTER'."
fi
if [ ! -f "$EUNIS_RASTER" ]; then
    echo "Warning: EUNIS raster file not found at '$EUNIS_RASTER'."
fi

ENV_SEG_FLAG=""
if [ "$ENV_USE_SEG" = "false" ] || [ "$ENV_USE_SEG" = "False" ]; then
    ENV_SEG_FLAG="--no_segformer"
fi

EUNIS_SEG_FLAG=""
if [ "$EUNIS_USE_SEG" = "false" ] || [ "$EUNIS_USE_SEG" = "False" ]; then
    EUNIS_SEG_FLAG="--no_segformer"
fi

ENV_PLATFORM_FLAG=""
if [ -n "$ENV_QUERY_PLATFORM" ] && [ "$ENV_QUERY_PLATFORM" != "None" ] && [ "$ENV_QUERY_PLATFORM" != "null" ] && [ "$ENV_QUERY_PLATFORM" != "all" ]; then
    ENV_PLATFORM_FLAG="--query_platform $ENV_QUERY_PLATFORM"
fi

EUNIS_PLATFORM_FLAG=""
if [ -n "$EUNIS_QUERY_PLATFORM" ] && [ "$EUNIS_QUERY_PLATFORM" != "None" ] && [ "$EUNIS_QUERY_PLATFORM" != "null" ] && [ "$EUNIS_QUERY_PLATFORM" != "all" ]; then
    EUNIS_PLATFORM_FLAG="--query_platform $EUNIS_QUERY_PLATFORM"
fi

ENV_VIZ_FLAG=""
if [ -n "$ENV_VIZ_SAMPLES" ] && [ "$ENV_VIZ_SAMPLES" != "None" ] && [ "$ENV_VIZ_SAMPLES" != "0" ] && [ "$ENV_VIZ_SAMPLES" != "false" ]; then
    ENV_VIZ_FLAG="--output_html $OUTPUT_DIR/environmental_zones_visualizer_${ENV_MODEL_CLEAN}.html --visualize_samples $ENV_VIZ_SAMPLES"
fi

EUNIS_VIZ_FLAG=""
if [ -n "$EUNIS_VIZ_SAMPLES" ] && [ "$EUNIS_VIZ_SAMPLES" != "None" ] && [ "$EUNIS_VIZ_SAMPLES" != "0" ] && [ "$EUNIS_VIZ_SAMPLES" != "false" ]; then
    EUNIS_VIZ_FLAG="--output_html $OUTPUT_DIR/eunis_visualizer_${EUNIS_MODEL_CLEAN}.html --visualize_samples $EUNIS_VIZ_SAMPLES"
fi

MAPILLARY_FLAG=""
if [ -n "$MAPILLARY_TOKEN" ]; then
    MAPILLARY_FLAG="--mapillary_token $MAPILLARY_TOKEN"
fi

# Sanitize model names to prevent directory traversal issues in filenames
ENV_MODEL_CLEAN="${ENV_MODEL//\//_}"
EUNIS_MODEL_CLEAN="${EUNIS_MODEL//\//_}"

echo "Configuration (Loaded from config/evaluation/params_online.yaml):"
echo "- Env Model: $ENV_MODEL"
echo "- Env Query Platform: ${ENV_QUERY_PLATFORM:-all}"
echo "- Env Visualizer Samples: ${ENV_VIZ_SAMPLES:-disabled}"
echo "- Env SegFormer Active: $ENV_USE_SEG"
echo "- EUNIS Model: $EUNIS_MODEL"
echo "- EUNIS Query Platform: ${EUNIS_QUERY_PLATFORM:-all}"
echo "- EUNIS Visualizer Samples: ${EUNIS_VIZ_SAMPLES:-disabled}"
echo "- EUNIS SegFormer Active: $EUNIS_USE_SEG"
echo "- Database path: $ENV_CSV"
echo "- Env Zones Raster: $ENV_RASTER"
echo "- EUNIS Raster: $EUNIS_RASTER"
echo "- Offline Dirs: $ENV_OFFLINE"
echo "- Mapillary Token Present: $([ -n "$MAPILLARY_TOKEN" ] && echo "Yes" || echo "No")"
echo "================================================================================"

# --- Step 1: Benchmark Environmental Zones ---
echo -e "\n[1/2] Running Environmental Zones of Europe Representation Benchmark..."
python3 -m src.evaluation.benchmark_environmental_zones \
  --model_name "$ENV_MODEL" \
  --csv_path "$ENV_CSV" \
  --raster "$ENV_RASTER" \
  --num_queries "$ENV_QUERIES" \
  --num_database "$ENV_DB" \
  --batch_size "$ENV_BATCH" \
  --seed "$ENV_SEED" \
  $ENV_PLATFORM_FLAG \
  $ENV_VIZ_FLAG \
  --offline_dataset_dirs "$ENV_OFFLINE" \
  $ENV_SEG_FLAG \
  $MAPILLARY_FLAG \
  --output_report "$OUTPUT_DIR/environmental_zones_report_${ENV_MODEL_CLEAN}.txt" \
  --output_csv "$OUTPUT_DIR/environmental_zones_results_${ENV_MODEL_CLEAN}.csv"

# --- Step 2: Benchmark EUNIS Ecosystems ---
echo -e "\n[2/2] Running EUNIS Ecosystems Representation Benchmark..."
python3 -m src.evaluation.benchmark_eunis \
  --model_name "$EUNIS_MODEL" \
  --csv_path "$EUNIS_CSV" \
  --raster "$EUNIS_RASTER" \
  --num_queries "$EUNIS_QUERIES" \
  --num_database "$EUNIS_DB" \
  --batch_size "$EUNIS_BATCH" \
  --seed "$EUNIS_SEED" \
  $EUNIS_PLATFORM_FLAG \
  $EUNIS_VIZ_FLAG \
  --offline_dataset_dirs "$EUNIS_OFFLINE" \
  $EUNIS_SEG_FLAG \
  $MAPILLARY_FLAG \
  --output_report "$OUTPUT_DIR/eunis_report_${EUNIS_MODEL_CLEAN}.txt" \
  --output_csv "$OUTPUT_DIR/eunis_results_${EUNIS_MODEL_CLEAN}.csv"

echo -e "\n================================================================================"
echo "✅ Spatial evaluations completed successfully!"
ENV_PLAT_LABEL="${ENV_QUERY_PLATFORM:-all}"
if [ "$ENV_PLAT_LABEL" = "None" ] || [ "$ENV_PLAT_LABEL" = "null" ]; then
    ENV_PLAT_LABEL="all"
fi
EUNIS_PLAT_LABEL="${EUNIS_QUERY_PLATFORM:-all}"
if [ "$EUNIS_PLAT_LABEL" = "None" ] || [ "$EUNIS_PLAT_LABEL" = "null" ]; then
    EUNIS_PLAT_LABEL="all"
fi

echo "Reports and visualizers saved to:"
echo "- $OUTPUT_DIR/environmental_zones_report_${ENV_MODEL_CLEAN}_s${ENV_SEED}_q${ENV_QUERIES}_plat-${ENV_PLAT_LABEL}.txt"
echo "- $OUTPUT_DIR/eunis_report_${EUNIS_MODEL_CLEAN}_s${EUNIS_SEED}_q${EUNIS_QUERIES}_plat-${EUNIS_PLAT_LABEL}.txt"
if [ -n "$ENV_VIZ_FLAG" ]; then
    echo "- $OUTPUT_DIR/environmental_zones_visualizer_${ENV_MODEL_CLEAN}_s${ENV_SEED}_q${ENV_QUERIES}_plat-${ENV_PLAT_LABEL}.html"
fi
if [ -n "$EUNIS_VIZ_FLAG" ]; then
    echo "- $OUTPUT_DIR/eunis_visualizer_${EUNIS_MODEL_CLEAN}_s${EUNIS_SEED}_q${EUNIS_QUERIES}_plat-${EUNIS_PLAT_LABEL}.html"
fi
echo "================================================================================"
