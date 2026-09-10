#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# ==============================================================================
# iNaturalist Biome Presets Batch Runner
# ==============================================================================

# Load parameters from config/scrapers/inaturalist_presets.yaml
YAML_PATH="config/scrapers/inaturalist_presets.yaml"

get_param() {
    VAL=$(python3 -m src.utils.config get "$YAML_PATH" "inaturalist_presets" "$1")
    if [ -z "$VAL" ]; then
        VAL=$(python3 -m src.utils.config get "$YAML_PATH" "scraper" "$1")
    fi
    echo "$VAL"
}

# Load presets array
read -r -a PRESETS <<< "$(get_param "presets")"

LIMIT=$(get_param "limit")
[ -z "$LIMIT" ] && LIMIT=200

EXCLUDE_FLYING=$(get_param "exclude_flying")
[ -z "$EXCLUDE_FLYING" ] && EXCLUDE_FLYING="false"

SCRAPE_WIKI=$(get_param "scrape_wiki")
[ -z "$SCRAPE_WIKI" ] && SCRAPE_WIKI="false"

OUT_DIR=$(get_param "out_dir")
[ -z "$OUT_DIR" ] && OUT_DIR="./inaturalist_preset_outputs"

mkdir -p "$OUT_DIR"

# ==============================================================================
# Executing Loop
# ==============================================================================

echo "============================================================"
echo "Starting iNaturalist presets batch scrape for ${#PRESETS[@]} biomes..."
echo "Parameters: Limit=$LIMIT | Scrape Wiki=$SCRAPE_WIKI"
echo "Out Directory: $OUT_DIR"
echo "============================================================"

for preset in "${PRESETS[@]}"; do
    out_file="$OUT_DIR/inaturalist_preset_${preset}.csv"

    echo -e "\n------------------------------------------------------------"
    echo "Processing preset biome: $preset"
    echo "Saving to: $out_file"
    echo "------------------------------------------------------------"

    # Build arguments dynamically
    ARGS=(
        "--preset" "$preset"
        "--limit" "$LIMIT"
        "--out" "$out_file"
    )

    if [ "$EXCLUDE_FLYING" = true ]; then
        ARGS+=("--exclude_flying")
    fi

    if [ "$SCRAPE_WIKI" = true ]; then
        ARGS+=("--scrape_wiki")
    fi

    # Run the python script
    python3 -m src.scrapers.fetch_inaturalist_data "${ARGS[@]}"

    # Sleep to be polite to iNaturalist API rate limits
    echo "Sleeping 5 seconds between requests..."
    sleep 5
done

echo -e "\n============================================================"
echo "Presets batch scrape completed! All outputs saved to $OUT_DIR."
echo "============================================================"
