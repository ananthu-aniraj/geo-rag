#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# ==============================================================================
# iNaturalist Batch Scraping Runner
# ==============================================================================

# Load parameters from config/scrapers/inaturalist_scraper.yaml
YAML_PATH="config/scrapers/inaturalist_scraper.yaml"

get_param() {
    python3 -m src.utils.config get "$YAML_PATH" "scraper" "$1"
}

# Load countries array
read -r -a COUNTRIES <<< "$(get_param "countries")"

LIMIT=$(get_param "limit")
[ -z "$LIMIT" ] && LIMIT=5000

NUM_SPECIES=$(get_param "num_species")
[ -z "$NUM_SPECIES" ] && NUM_SPECIES=10

TARGET_TAXON=$(get_param "target_taxon")
[ -z "$TARGET_TAXON" ] && TARGET_TAXON="plants"

EXCLUDE_FLYING=$(get_param "exclude_flying")
[ -z "$EXCLUDE_FLYING" ] && EXCLUDE_FLYING="true"

SCRAPE_WIKI=$(get_param "scrape_wiki")
[ -z "$SCRAPE_WIKI" ] && SCRAPE_WIKI="false"

OUT_DIR=$(get_param "out_dir")
[ -z "$OUT_DIR" ] && OUT_DIR="./inaturalist_outputs"

mkdir -p "$OUT_DIR"

# ==============================================================================
# Executing Loop
# ==============================================================================

echo "============================================================"
echo "Starting iNaturalist batch scrape for ${#COUNTRIES[@]} regions..."
echo "Parameters: Limit=$LIMIT | Species=$NUM_SPECIES | Taxon=$TARGET_TAXON"
echo "Out Directory: $OUT_DIR"
echo "============================================================"

for country in "${COUNTRIES[@]}"; do
    # Replace spaces with underscores for clean file names
    country_clean=$(echo "$country" | tr ' ' '_')
    out_file="$OUT_DIR/inaturalist_${country_clean}_${TARGET_TAXON}.csv"

    echo -e "\n------------------------------------------------------------"
    echo "Processing region: $country"
    echo "Saving to: $out_file"
    echo "------------------------------------------------------------"

    # Build arguments dynamically
    ARGS=(
        "--country" "$country"
        "--limit" "$LIMIT"
        "--num_species" "$NUM_SPECIES"
        "--target_taxon" "$TARGET_TAXON"
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
echo "Batch scrape completed! All outputs saved to $OUT_DIR."
echo "============================================================"
