#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Shell wrapper to execute flickr_density_profiler.py

# Configuration
# Set your API key here. If left empty, it will fall back to params.yaml or the FLICKR_API_KEY environment variable.
API_KEY="FLICKR_API_KEY_PLACEHOLDER"

# Scraping Targets
# Leave empty to load preset target landmarks (Seven Wonders) by default.
# You can override this by passing a location name as the first argument, e.g.: ./run_flickr_density_profiler.sh "Rome, Italy"
LOCATION=""
BBOX="" # Manual bounding box coords (min_lon,min_lat,max_lon,max_lat). E.g. "12.48,41.88,12.50,41.90" (Overrides LOCATION if set)

# Override LOCATION if a command line argument is provided
if [ -n "$1" ]; then
    LOCATION="$1"
fi

# Grid Settings
GRID_SIZE=5.0        # Grid size in km. Set to 0 to disable grid splitting.
LIMIT_PER_BOX=100    # Max photos to collect per grid sub-box
LIMIT_GLOBAL=500     # Max photos to collect overall (only used if GRID_SIZE=0)
DELAY=2.0            # Delay between API calls in seconds

# Output Path
OUT_FILE="output/flickr_density_profile.csv"

# Ensure output directory exists
mkdir -p "$(dirname "$OUT_FILE")"

# Fallback: Load API key if not set in configuration
if [ -z "$API_KEY" ]; then
    API_KEY=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('flickr_api_key', ''))" 2>/dev/null || echo "$FLICKR_API_KEY")
fi

if [ -z "$API_KEY" ]; then
    echo "❌ Error: Flickr API Key is not set in this script, params.yaml, or environment variables."
    exit 1
fi

# Build arguments array
ARGS=(
  --api_key "$API_KEY"
  --grid_size "$GRID_SIZE"
  --limit_per_box "$LIMIT_PER_BOX"
  --limit "$LIMIT_GLOBAL"
  --delay "$DELAY"
  --out "$OUT_FILE"
)

if [ -n "$BBOX" ]; then
    ARGS+=(--bbox "$BBOX")
    echo "Running Flickr Density Profiler using manual bounding box: $BBOX..."
elif [ -n "$LOCATION" ]; then
    ARGS+=(--location "$LOCATION")
    echo "Running Flickr Density Profiler for location: '$LOCATION'..."
else
    echo "Running Flickr Density Profiler using preset Seven Wonders target landmarks..."
fi

# Run the profiler
python3 -m src.scrapers.flickr_density_profiler "${ARGS[@]}"
