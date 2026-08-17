#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# ==============================================================================
# iNaturalist Batch Scraping Runner
# ==============================================================================

# 1. Define list of countries/regions to loop over
# Wrapped in quotes to support places with spaces (e.g., "North Korea")
COUNTRIES=(
    "Angola"
    "North Korea"
    "Mongolia"
    "Australia"
    "Greenland"
    "Iceland"
    "Algeria"
    "Northwest Territories"
    "Western Sahara"
    "Sahara"
    "Alaska"
    "Siberia"
)

# 2. General parameters
LIMIT=5000                  # Max observations to download PER country
NUM_SPECIES=10             # Number of top native species to balance across
TARGET_TAXON="plants"      # Taxon to search (e.g., "plants", "animals", "birds")
EXCLUDE_FLYING=true        # Exclude flying animals (birds and insects)
SCRAPE_WIKI=false          # Set to true to scrape Wikipedia instead of native place counts

# 3. Output Configuration
OUT_DIR="./inaturalist_outputs"
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
