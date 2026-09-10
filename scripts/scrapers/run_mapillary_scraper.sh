#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# Source the .env file if it exists to load keys into environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$MAPILLARY_TOKEN" ]; then
    echo "❌ Error: MAPILLARY_TOKEN is not set in your environment or .env file."
    exit 1
fi

ACCESS_TOKEN="$MAPILLARY_TOKEN"

# Load parameters from config/scrapers/mapillary_scraper.yaml
YAML_PATH="config/scrapers/mapillary_scraper.yaml"

get_param() {
    VAL=$(python3 -m src.utils.config get "$YAML_PATH" "mapillary_scraper" "$1")
    if [ -z "$VAL" ]; then
        VAL=$(python3 -m src.utils.config get "$YAML_PATH" "scraper" "$1")
    fi
    echo "$VAL"
}

TOTAL_CHUNKS=$(get_param "total_chunks")
[ -z "$TOTAL_CHUNKS" ] && TOTAL_CHUNKS=10000

STEP_KM=$(get_param "step_km")
[ -z "$STEP_KM" ] && STEP_KM=5

MAX_PHOTOS_PER_BOX=$(get_param "max_photos_per_box")
[ -z "$MAX_PHOTOS_PER_BOX" ] && MAX_PHOTOS_PER_BOX=100

UNCOVERED_SHAPEFILE=$(get_param "uncovered_shapefile")
[ -z "$UNCOVERED_SHAPEFILE" ] && UNCOVERED_SHAPEFILE="shapefiles/uncovered_land_areas_test.shp"

BASE_DIR=$(get_param "base_dir")
[ -z "$BASE_DIR" ] && BASE_DIR="output/mapillary_scrape"

SCRIPT_NAME="src.scrapers.mapillary_scraper"
ORDER_FILE="$BASE_DIR/chunk_order.txt"

# Ensure base directory exists
mkdir -p "$BASE_DIR"

# 1. Generate the randomized order file if it doesn't exist
if [ ! -f "$ORDER_FILE" ]; then
    echo "🎲 Generating new randomized chunk order: $ORDER_FILE"
    shuf -i 0-$((TOTAL_CHUNKS - 1)) > "$ORDER_FILE"
else
    echo "📋 Using existing chunk order from: $ORDER_FILE"
fi

echo "Starting Mapillary grid search for $TOTAL_CHUNKS chunks..."

# 2. Loop through chunks in the saved order
while read -r i
do
    # Define the output file name (matching the Python script's logic)
    LOG_FILE="$BASE_DIR/mapillary_completed_boxes_chunk_${i}.txt"

    # Optimization: Skip chunk if the log file exists (indicating it was at least started/processed)
    if [ -f "$LOG_FILE" ]; then
        echo "⏩ Skipping chunk $i (already processed or in progress)."
        continue
    fi

    echo "========================================"
    echo "Processing chunk $i / $((TOTAL_CHUNKS - 1))"
    echo "========================================"

    # Run the Python script
    if ! python3 -m "$SCRIPT_NAME" --chunk "$i" --total_chunks "$TOTAL_CHUNKS" --base_dir "$BASE_DIR" --access_token "$ACCESS_TOKEN" --uncovered_shapefile "$UNCOVERED_SHAPEFILE" --step_km "$STEP_KM" --max_photos_per_box "$MAX_PHOTOS_PER_BOX"; then
        echo "CRITICAL ERROR: Script failed on chunk $i. Halting execution."
        exit 1
    fi
done < "$ORDER_FILE"

echo "All $TOTAL_CHUNKS chunks completed successfully!"
