#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Shell wrapper to execute mapillary_density_profiler.py

# Source the .env file if it exists to load keys into environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$MAPILLARY_TOKEN" ]; then
    echo "❌ Error: MAPILLARY_TOKEN is not set in your environment or .env file."
    exit 1
fi

ACCESS_TOKEN="$MAPILLARY_TOKEN"

# Load parameters from config/scrapers/mapillary_profiler.yaml
YAML_PATH="config/scrapers/mapillary_profiler.yaml"

get_param() {
    VAL=$(python3 -m src.utils.config get "$YAML_PATH" "mapillary_profiler" "$1")
    if [ -z "$VAL" ]; then
        VAL=$(python3 -m src.utils.config get "$YAML_PATH" "scraper" "$1")
    fi
    echo "$VAL"
}

LOCATION=$(get_param "location")
BBOX=$(get_param "bbox")

GRID_SIZE=$(get_param "grid_size")
[ -z "$GRID_SIZE" ] && GRID_SIZE=5.0

LIMIT_PER_BOX=$(get_param "limit_per_box")
[ -z "$LIMIT_PER_BOX" ] && LIMIT_PER_BOX=100

LIMIT_GLOBAL=$(get_param "limit_global")
[ -z "$LIMIT_GLOBAL" ] && LIMIT_GLOBAL=500

DELAY=$(get_param "delay")
[ -z "$DELAY" ] && DELAY=3.0

OUT_FILE=$(get_param "out_file")
[ -z "$OUT_FILE" ] && OUT_FILE="output/mapillary_density_profile.csv"

# Override LOCATION if a command line argument is provided
if [ -n "$1" ]; then
    LOCATION="$1"
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUT_FILE")"


# Build arguments array
ARGS=(
  --access_token "$ACCESS_TOKEN"
  --grid_size "$GRID_SIZE"
  --limit_per_box "$LIMIT_PER_BOX"
  --limit "$LIMIT_GLOBAL"
  --delay "$DELAY"
  --out "$OUT_FILE"
)

if [ -n "$BBOX" ]; then
    ARGS+=(--bbox "$BBOX")
    echo "Running Mapillary Density Profiler using manual bounding box: $BBOX..."
elif [ -n "$LOCATION" ]; then
    ARGS+=(--location "$LOCATION")
    echo "Running Mapillary Density Profiler for location: '$LOCATION'..."
else
    echo "Running Mapillary Density Profiler using preset Seven Wonders target landmarks..."
fi

# Run the profiler
python3 -m src.scrapers.mapillary_density_profiler "${ARGS[@]}"
