#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# --- OSM Polygon Scraper Bash Runner ---

# Load parameters from config/scrapers/osm_scraper.yaml
YAML_PATH="config/scrapers/osm_scraper.yaml"
MODE=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('mode', 'global'))" 2>/dev/null)
OSM_QUERY=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('osm_query', 'Montpellier, France'))" 2>/dev/null)
OSM_RELATION=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('osm_relation', ''))" 2>/dev/null)
PLATFORMS=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('platforms', 'kartaview'))" 2>/dev/null)
BASE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('base_dir', 'output/osm_scrape'))" 2>/dev/null)
TOTAL_CHUNKS=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('total_chunks', 10000))" 2>/dev/null)

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
