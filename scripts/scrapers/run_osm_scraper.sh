#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# --- OSM Polygon Scraper Bash Runner ---

# Load parameters from config/scrapers/osm_scraper.yaml
YAML_PATH="config/scrapers/osm_scraper.yaml"

get_param() {
    VAL=$(python3 -m src.utils.config get "$YAML_PATH" "osm_scraper" "$1")
    if [ -z "$VAL" ]; then
        VAL=$(python3 -m src.utils.config get "$YAML_PATH" "scraper" "$1")
    fi
    echo "$VAL"
}

MODE=$(get_param "mode")
[ -z "$MODE" ] && MODE="global"

OSM_QUERY=$(get_param "osm_query")
[ -z "$OSM_QUERY" ] && OSM_QUERY="Montpellier, France"

OSM_RELATION=$(get_param "osm_relation")

PLATFORMS=$(get_param "platforms")
[ -z "$PLATFORMS" ] && PLATFORMS="kartaview"

BASE_DIR=$(get_param "base_dir")
[ -z "$BASE_DIR" ] && BASE_DIR="output/osm_scrape"

TOTAL_CHUNKS=$(get_param "total_chunks")
[ -z "$TOTAL_CHUNKS" ] && TOTAL_CHUNKS=10000

SCRIPT_NAME="src.scrapers.osm_polygon_scraper"

# Setup target arguments based on MODE selection
OSM_TARGET_ARGS=()
if [ "$MODE" = "global" ]; then
    OSM_TARGET_ARGS=(--global_search)
else
    if [ -n "$OSM_RELATION" ]; then
        OSM_TARGET_ARGS=(--osm_relation "$OSM_RELATION")
    else
        OSM_TARGET_ARGS=(--osm_query "$OSM_QUERY")
    fi
fi

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
