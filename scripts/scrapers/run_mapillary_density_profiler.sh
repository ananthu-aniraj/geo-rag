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
LOCATION=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('location', ''))" 2>/dev/null)
BBOX=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('bbox', ''))" 2>/dev/null)
GRID_SIZE=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('grid_size', 5.0))" 2>/dev/null)
LIMIT_PER_BOX=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('limit_per_box', 100))" 2>/dev/null)
LIMIT_GLOBAL=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('limit_global', 500))" 2>/dev/null)
DELAY=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('delay', 3.0))" 2>/dev/null)
OUT_FILE=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('out_file', 'output/mapillary_density_profile.csv'))" 2>/dev/null)

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
