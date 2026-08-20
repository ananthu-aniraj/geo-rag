#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# Source the .env file if it exists to load keys into environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$FLICKR_API_KEY" ]; then
    echo "❌ Error: FLICKR_API_KEY is not set in your environment or .env file."
    exit 1
fi

API_KEY="$FLICKR_API_KEY"

# Load parameters from config/scrapers/flickr_scraper.yaml
YAML_PATH="config/scrapers/flickr_scraper.yaml"
TOTAL_CHUNKS=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('total_chunks', 10000))" 2>/dev/null)
STEP_KM=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('step_km', 5))" 2>/dev/null)
MAX_PHOTOS_PER_BOX=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('max_photos_per_box', 100))" 2>/dev/null)
UNCOVERED_SHAPEFILE=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('uncovered_shapefile', 'shapefiles/uncovered_land_areas_test.shp'))" 2>/dev/null)
BASE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('base_dir', 'output/flickr_scrape'))" 2>/dev/null)

SCRIPT_NAME="src.scrapers.flickr_5km_grid_search"
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
    if ! python3 -m "$SCRIPT_NAME" --chunk "$i" --total_chunks "$TOTAL_CHUNKS" --base_dir "$BASE_DIR" --api_key "$API_KEY" --uncovered_shapefile "$UNCOVERED_SHAPEFILE" --step_km "$STEP_KM" --max_photos_per_box "$MAX_PHOTOS_PER_BOX"; then
        echo "CRITICAL ERROR: Script failed on chunk $i. Halting execution."
        exit 1
    fi
done < "$ORDER_FILE"

echo "All $TOTAL_CHUNKS chunks completed successfully!"
