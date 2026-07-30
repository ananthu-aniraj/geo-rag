#!/bin/bash

# --- OSM Polygon Scraper Bash Runner ---

# Configuration
TOTAL_CHUNKS=200
SCRIPT_NAME="src.scrapers.osm_polygon_scraper"

# Scraper Target Selection (Uncomment/edit as needed)
# 1. By Query
OSM_TARGET_ARGS=(--osm_query "South America")
# 2. By Relation ID
# OSM_TARGET_ARGS=(--osm_relation 74263) # e.g. Paris

PLATFORMS="all" # Choices: wikimedia, kartaview, all
BASE_DIR="/home/ananthu/DATA/data_ananthu/osm_scrape_south_america"
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

echo "Starting OSM grid search for $TOTAL_CHUNKS chunks..."

# 2. Loop through chunks in the saved order
while read -r i
do
    LOG_FILE="$BASE_DIR/osm_completed_boxes_chunk_${i}.txt"

    # Skip chunk if the log file exists (indicating it was already processed)
    if [ -f "$LOG_FILE" ]; then
        echo "⏩ Skipping chunk $i (already processed or in progress)."
        continue
    fi

    echo "========================================"
    echo "Processing chunk $i / $((TOTAL_CHUNKS - 1))"
    echo "========================================"

    # Run the Python script
    if ! python3 -m "$SCRIPT_NAME" \
        "${OSM_TARGET_ARGS[@]}" \
        --chunk "$i" \
        --total_chunks "$TOTAL_CHUNKS" \
        --platforms "$PLATFORMS" \
        --base_dir "$BASE_DIR"; then
        echo "CRITICAL ERROR: Script failed on chunk $i. Halting execution."
        exit 1
    fi
done < "$ORDER_FILE"

echo "All $TOTAL_CHUNKS chunks completed successfully!"
