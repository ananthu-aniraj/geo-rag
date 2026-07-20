#!/bin/bash

# ==============================================================================
# iNaturalist Biome Presets Batch Runner
# ==============================================================================

# 1. Define list of presets to loop over
PRESETS=(
    "desert"
    "tundra"
    "wetland"
    "boreal"
    "rainforest"
    "polar"
)

# 2. General parameters
LIMIT=200                  # Max observations to download PER preset
EXCLUDE_FLYING=false       # Keep false for presets (allows hand-selected birds like penguins & kingfishers)
SCRAPE_WIKI=false          # Set to true to scrape Wikipedia for these biomes dynamically

# 3. Output Configuration
OUT_DIR="./inaturalist_preset_outputs"
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
