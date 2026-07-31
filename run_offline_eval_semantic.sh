#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Base directory setup
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "      Starting Semantic Representation Evaluation (LUCAS & Places365)           "
echo "================================================================================"

# Helper function to read yaml values using Python
get_param() {
    python3 -c "import yaml; print(yaml.safe_load(open('eval_params_offline.yaml'))['$1']['$2'])"
}

# Load parameters from YAML
LUCAS_CSV=$(get_param "lucas" "csv")
LUCAS_IMG_DIR=$(get_param "lucas" "img_dir")
LUCAS_QUERIES=$(get_param "lucas" "num_queries")
LUCAS_DB=$(get_param "lucas" "num_database")
LUCAS_BATCH=$(get_param "lucas" "batch_size")
LUCAS_SEED=$(get_param "lucas" "seed")

PLACES_LABELS=$(get_param "places" "labels")
PLACES_IMG_DIR=$(get_param "places" "img_dir")
PLACES_QUERIES=$(get_param "places" "num_queries")
PLACES_DB=$(get_param "places" "num_database")
PLACES_BATCH=$(get_param "places" "batch_size")
PLACES_SEED=$(get_param "places" "seed")

OUTPUT_DIR="./benchmark_results"
mkdir -p "$OUTPUT_DIR"

# Validate directories
if [ ! -d "$LUCAS_IMG_DIR" ]; then
    echo "Warning: LUCAS image directory not found at '$LUCAS_IMG_DIR'."
fi
if [ ! -d "$PLACES_IMG_DIR" ]; then
    echo "Warning: Places365 validation image directory not found at '$PLACES_IMG_DIR'."
fi

echo "Configuration (Loaded from eval_params_offline.yaml):"
echo "- LUCAS Metadata: $LUCAS_CSV"
echo "- LUCAS Image Dir: $LUCAS_IMG_DIR"
echo "- Places365 Hierarchy: $PLACES_LABELS"
echo "- Places365 Image Dir: $PLACES_IMG_DIR"
echo "================================================================================"

# --- Step 1: Benchmark LUCAS ---
echo -e "\n[1/2] Running LUCAS Semantic Representation Benchmark..."
python3 -m src.evaluation.benchmark_lucas \
  --csv "$LUCAS_CSV" \
  --img_dir "$LUCAS_IMG_DIR" \
  --num_queries "$LUCAS_QUERIES" \
  --num_database "$LUCAS_DB" \
  --batch_size "$LUCAS_BATCH" \
  --seed "$LUCAS_SEED" \
  --output_report "$OUTPUT_DIR/lucas_report.txt" \
  --output_csv "$OUTPUT_DIR/lucas_results.csv"

# --- Step 2: Benchmark Places365 ---
echo -e "\n[2/2] Running Places365 Semantic Representation Benchmark..."
python3 -m src.evaluation.benchmark_places \
  --labels "$PLACES_LABELS" \
  --img_dir "$PLACES_IMG_DIR" \
  --num_queries "$PLACES_QUERIES" \
  --num_database "$PLACES_DB" \
  --batch_size "$PLACES_BATCH" \
  --seed "$PLACES_SEED" \
  --compare_clip \
  --output_report "$OUTPUT_DIR/places_report.txt" \
  --output_csv "$OUTPUT_DIR/places_results.csv"

echo -e "\n================================================================================"
echo "✅ Semantic evaluations completed successfully!"
echo "Reports saved to:"
echo "- $OUTPUT_DIR/lucas_report.txt"
echo "- $OUTPUT_DIR/places_report.txt"
echo "================================================================================"
