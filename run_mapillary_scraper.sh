#!/bin/bash

# Configuration
TOTAL_CHUNKS=10000
SCRIPT_NAME="src.scrapers.mapillary_scraper"
BASE_DIR="/home/ananthu/Projects/data/mapillary_scrape_rand_8"
ORDER_FILE="$BASE_DIR/chunk_order.txt"
ACCESS_TOKEN='MAPILLARY_TOKEN_PLACEHOLDER'
STEP_KM=5
MAX_PHOTOS_PER_BOX=100
UNCOVERED_SHAPEFILE="shapefiles/uncovered_land_areas_test.shp"

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
