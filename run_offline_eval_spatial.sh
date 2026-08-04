#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Base directory setup
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "      Starting Spatial Representation Evaluation (Env Zones & EUNIS)            "
echo "================================================================================"

# Helper function to read yaml values using Python
get_param() {
    python3 -c "import yaml; print(yaml.safe_load(open('eval_params_online.yaml'))['$1']['$2'])"
}

# Load parameters from YAML
ENV_CSV=$(get_param "environmental_zones" "csv_path")
ENV_SHP=$(get_param "environmental_zones" "countries_shp")
ENV_RASTER=$(get_param "environmental_zones" "raster")
ENV_QUERIES=$(get_param "environmental_zones" "num_queries")
ENV_DB=$(get_param "environmental_zones" "num_database")
ENV_BATCH=$(get_param "environmental_zones" "batch_size")
ENV_SEED=$(get_param "environmental_zones" "seed")

EUNIS_CSV=$(get_param "eunis" "csv_path")
EUNIS_SHP=$(get_param "eunis" "countries_shp")
EUNIS_RASTER=$(get_param "eunis" "raster")
EUNIS_QUERIES=$(get_param "eunis" "num_queries")
EUNIS_DB=$(get_param "eunis" "num_database")
EUNIS_BATCH=$(get_param "eunis" "batch_size")
EUNIS_SEED=$(get_param "eunis" "seed")

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

echo "Configuration (Loaded from eval_params_online.yaml):"
echo "- Database path: $ENV_CSV"
echo "- Countries SHP: $ENV_SHP"
echo "- Env Zones Raster: $ENV_RASTER"
echo "- EUNIS Raster: $EUNIS_RASTER"
echo "================================================================================"

# --- Step 1: Benchmark Environmental Zones ---
echo -e "\n[1/2] Running Environmental Zones of Europe Representation Benchmark..."
python3 -m src.evaluation.benchmark_environmental_zones \
  --csv_path "$ENV_CSV" \
  --countries_shp "$ENV_SHP" \
  --raster "$ENV_RASTER" \
  --num_queries "$ENV_QUERIES" \
  --num_database "$ENV_DB" \
  --batch_size "$ENV_BATCH" \
  --seed "$ENV_SEED" \
  --query_platform "flickr" \
  --output_report "$OUTPUT_DIR/environmental_zones_report.txt" \
  --output_csv "$OUTPUT_DIR/environmental_zones_results.csv"

# --- Step 2: Benchmark EUNIS Ecosystems ---
echo -e "\n[2/2] Running EUNIS Ecosystems Representation Benchmark..."
python3 -m src.evaluation.benchmark_eunis \
  --csv_path "$EUNIS_CSV" \
  --countries_shp "$EUNIS_SHP" \
  --raster "$EUNIS_RASTER" \
  --num_queries "$EUNIS_QUERIES" \
  --num_database "$EUNIS_DB" \
  --batch_size "$EUNIS_BATCH" \
  --seed "$EUNIS_SEED" \
  --query_platform "flickr" \
  --output_report "$OUTPUT_DIR/eunis_report.txt" \
  --output_csv "$OUTPUT_DIR/eunis_results.csv"

echo -e "\n================================================================================"
echo "✅ Spatial evaluations completed successfully!"
echo "Reports saved to:"
echo "- $OUTPUT_DIR/environmental_zones_report.txt"
echo "- $OUTPUT_DIR/eunis_report.txt"
echo "================================================================================"
