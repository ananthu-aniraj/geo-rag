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

# Load presets array using Python helper
PRESETS_STR=$(python3 -c "
import yaml
with open('$YAML_PATH') as f:
    presets = yaml.safe_load(f)['scraper'].get('presets', [])
    print(' '.join(['\"' + p + '\"' for p in presets]))
" 2>/dev/null)
eval "PRESETS=($PRESETS_STR)"

LIMIT=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('limit', 200))" 2>/dev/null)
EXCLUDE_FLYING=$(python3 -c "import yaml; print(str(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('exclude_flying', False)).lower())" 2>/dev/null)
SCRAPE_WIKI=$(python3 -c "import yaml; print(str(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('scrape_wiki', False)).lower())" 2>/dev/null)
OUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('out_dir', './inaturalist_preset_outputs'))" 2>/dev/null)

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
