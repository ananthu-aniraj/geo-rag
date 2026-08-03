#!/bin/bash

# Configuration
TOTAL_CHUNKS=10000
SCRIPT_NAME="src.scrapers.flickr_5km_grid_search"
BASE_DIR="/home/ananthu/Projects/data/flickr_scrape_rand_6"
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

echo "Starting Flickr grid search for $TOTAL_CHUNKS chunks..."

# 2. Loop through chunks in the saved order
while read -r i
do
    # Define the output file name (matching the Python script's logic)
    # Note: Python script uses flickr_data_chunk_${i}.csv
    LOG_FILE="$BASE_DIR/flickr_completed_boxes_chunk_${i}.txt"

    # Optimization: Skip chunk if the log file exists (indicating it was at least started/processed)
    # Or skip if the data file exists and is non-empty.
    if [ -f "$LOG_FILE" ]; then
        echo "⏩ Skipping chunk $i (already processed or in progress)."
        continue
    fi

    echo "========================================"
    echo "Processing chunk $i / $((TOTAL_CHUNKS - 1))"
    echo "========================================"

    # Run the Python script
    if ! python3 -m "$SCRIPT_NAME" --chunk "$i" --total_chunks "$TOTAL_CHUNKS" --base_dir "$BASE_DIR"; then
        echo "CRITICAL ERROR: Script failed on chunk $i. Halting execution."
        exit 1
    fi
done < "$ORDER_FILE"

echo "All $TOTAL_CHUNKS chunks completed successfully!"