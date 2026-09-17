#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Source the .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

YAML_PATH="config/scrapers/wikiloc_scraper.yaml"

if [ ! -f "$YAML_PATH" ]; then
    echo "❌ Error: Configuration file '$YAML_PATH' not found."
    exit 1
fi

get_param() {
    VAL=$(python3 -m src.utils.config get "$YAML_PATH" "wikiloc_scraper" "$1")
    if [ -z "$VAL" ]; then
        VAL=$(python3 -m src.utils.config get "$YAML_PATH" "scraper" "$1")
    fi
    echo "$VAL"
}

echo "============================================================"
echo "Wikiloc Remote Region Scraper Runner"
echo "============================================================"
echo "Config: $YAML_PATH"

ARGS=()

REGIONS=$(get_param "regions")
if [ -n "$REGIONS" ]; then
    read -r -a REGIONS_ARR <<< "$REGIONS"
    ARGS+=("--regions" "${REGIONS_ARR[@]}")
fi

URLS=$(get_param "urls")
if [ -n "$URLS" ]; then
    read -r -a URLS_ARR <<< "$URLS"
    ARGS+=("--urls" "${URLS_ARR[@]}")
fi

ACTIVITY=$(get_param "activity")
[ -n "$ACTIVITY" ] && ARGS+=("--activity" "$ACTIVITY")

MAX_PAGES=$(get_param "max_pages_per_region")
[ -n "$MAX_PAGES" ] && ARGS+=("--max_pages_per_region" "$MAX_PAGES")

MAX_TRAILS=$(get_param "max_trails")
[ -n "$MAX_TRAILS" ] && ARGS+=("--max_trails" "$MAX_TRAILS")

MAX_PHOTOS=$(get_param "max_photos_per_trail")
[ -n "$MAX_PHOTOS" ] && ARGS+=("--max_photos_per_trail" "$MAX_PHOTOS")

DELAY=$(get_param "delay")
[ -n "$DELAY" ] && ARGS+=("--delay" "$DELAY")

OUTPUT=$(get_param "output")
[ -n "$OUTPUT" ] && ARGS+=("--output" "$OUTPUT")

NO_CSV=$(get_param "no_csv")
[ "$NO_CSV" = "true" ] && ARGS+=("--no_csv")

DOWNLOAD_IMAGES=$(get_param "download_images")
if [ "$DOWNLOAD_IMAGES" = "false" ]; then
    ARGS+=("--no_download_images")
else
    ARGS+=("--download_images")
    IMAGE_DIR=$(get_param "image_dir")
    [ -n "$IMAGE_DIR" ] && ARGS+=("--image_dir" "$IMAGE_DIR")
fi

THREADS=$(get_param "threads")
[ -n "$THREADS" ] && ARGS+=("--threads" "$THREADS")

echo "Executing: python3 -m src.scrapers.scrape_wikiloc ${ARGS[*]} $*"
echo ""

exec python3 -m src.scrapers.scrape_wikiloc "${ARGS[@]}" "$@"
