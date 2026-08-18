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

# Load countries array using Python helper
COUNTRIES_STR=$(python3 -c "
import yaml
with open('$YAML_PATH') as f:
    countries = yaml.safe_load(f)['scraper'].get('countries', [])
    print(' '.join(['\"' + c + '\"' for c in countries]))
" 2>/dev/null)
eval "COUNTRIES=($COUNTRIES_STR)"

LIMIT=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('limit', 5000))" 2>/dev/null)
NUM_SPECIES=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('num_species', 10))" 2>/dev/null)
TARGET_TAXON=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('target_taxon', 'plants'))" 2>/dev/null)
EXCLUDE_FLYING=$(python3 -c "import yaml; print(str(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('exclude_flying', True)).lower())" 2>/dev/null)
SCRAPE_WIKI=$(python3 -c "import yaml; print(str(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('scrape_wiki', False)).lower())" 2>/dev/null)
OUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML_PATH'))['scraper'].get('out_dir', './inaturalist_outputs'))" 2>/dev/null)

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
