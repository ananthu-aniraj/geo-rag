#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Source the .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$WILDOBS_API_KEY" ]; then
    echo "❌ Error: WILDOBS_API_KEY is not set in your environment or .env file."
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

YAML_PATH="config/scrapers/wildobs_scraper.yaml"

if [ ! -f "$YAML_PATH" ]; then
    echo "❌ Error: Configuration file '$YAML_PATH' not found."
    exit 1
fi

get_param() {
    VAL=$(python3 -m src.utils.config get "$YAML_PATH" "wildobs_scraper" "$1")
    if [ -z "$VAL" ]; then
        VAL=$(python3 -m src.utils.config get "$YAML_PATH" "scraper" "$1")
    fi
    echo "$VAL"
}

echo "============================================================"
echo "WildObs Camera Trap Scraper Runner"
echo "============================================================"
echo "Config: $YAML_PATH"

ARGS=()

PROJECT_IDS=$(get_param "project_ids")
if [ -n "$PROJECT_IDS" ]; then
    read -r -a IDS_ARR <<< "$PROJECT_IDS"
    ARGS+=("--project_ids" "${IDS_ARR[@]}")
fi

ALL_OPEN=$(get_param "all_open")
if [ -z "$PROJECT_IDS" ] && [ "$ALL_OPEN" = "true" ]; then
    ARGS+=("--all_open")
fi

PROJECT_PREFIXES=$(get_param "project_prefixes")
if [ -n "$PROJECT_PREFIXES" ]; then
    read -r -a PREFIXES_ARR <<< "$PROJECT_PREFIXES"
    ARGS+=("--project_prefixes" "${PREFIXES_ARR[@]}")
fi

MAX_PER_CAM=$(get_param "max_images_per_camera")
[ -n "$MAX_PER_CAM" ] && ARGS+=("--max_images_per_camera" "$MAX_PER_CAM")

SAMPLES_PER_BIN=$(get_param "samples_per_bin")
[ -n "$SAMPLES_PER_BIN" ] && ARGS+=("--samples_per_bin" "$SAMPLES_PER_BIN")

TOD_MODE=$(get_param "tod_mode")
[ -n "$TOD_MODE" ] && ARGS+=("--tod_mode" "$TOD_MODE")

DAY_START=$(get_param "day_start")
[ -n "$DAY_START" ] && ARGS+=("--day_start" "$DAY_START")

DAY_END=$(get_param "day_end")
[ -n "$DAY_END" ] && ARGS+=("--day_end" "$DAY_END")

INCLUDE_BLANKS=$(get_param "include_blanks")
[ "$INCLUDE_BLANKS" = "true" ] && ARGS+=("--include_blanks")

TAXA=$(get_param "taxa")
if [ -n "$TAXA" ]; then
    read -r -a TAXA_ARR <<< "$TAXA"
    ARGS+=("--taxa" "${TAXA_ARR[@]}")
fi

OUTPUT=$(get_param "output")
[ -n "$OUTPUT" ] && ARGS+=("--output" "$OUTPUT")

NO_CSV=$(get_param "no_csv")
[ "$NO_CSV" = "true" ] && ARGS+=("--no_csv")

DOWNLOAD_IMAGES=$(get_param "download_images")
if [ "$DOWNLOAD_IMAGES" = "true" ]; then
    ARGS+=("--download_images")
    IMAGE_DIR=$(get_param "image_dir")
    [ -n "$IMAGE_DIR" ] && ARGS+=("--image_dir" "$IMAGE_DIR")
fi

THREADS=$(get_param "threads")
[ -n "$THREADS" ] && ARGS+=("--threads" "$THREADS")

echo "Executing: python3 -m src.scrapers.scrape_wildobs ${ARGS[*]} $*"
echo ""

exec python3 -m src.scrapers.scrape_wildobs "${ARGS[@]}" "$@"
