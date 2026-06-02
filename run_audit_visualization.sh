#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "==============================================="
echo "  Geo-RAG: Audit & Visualization Pipeline"
echo "==============================================="

# Define your data directories here. Modify these paths for your other machine.
FLICKR_DIR="/user/aaniraj/home/Documents/Projects/data/flickr_scrape_rand"
MAPILLARY_DIR="/user/aaniraj/home/Documents/Projects/data/mapillary_scrape_rand"
GS_DIR="/user/aaniraj/home/Documents/Projects/data/global-streetscapes/train"

# Output configuration
OUTPUT_DIR="./audit_output"
OUTPUT_NAME="audit_space"
LIMIT_CELLS=50

echo ""
echo "[Step 1] Running processing script on a subset ($LIMIT_CELLS cells)..."
# Note: Remove "$GS_DIR" from the --dirs argument below if you only want to test scraped data
python process_scraped_data.py \
  --dirs "$FLICKR_DIR" "$MAPILLARY_DIR" "$GS_DIR" \
  --save_path "$OUTPUT_DIR" \
  --output_name "$OUTPUT_NAME" \
  --limit_cells "$LIMIT_CELLS"

echo ""
echo "[Step 2] Generating interactive diagnostic map..."
PKL_FILE="$OUTPUT_DIR/$OUTPUT_NAME.pkl"
MAP_FILE="$OUTPUT_DIR/diagnostic_map.html"

python visualize_clusters.py \
  --pkl "$PKL_FILE" \
  --out "$MAP_FILE"

echo ""
echo "==============================================="
echo "  Pipeline Complete!"
echo "  - Data saved to: $OUTPUT_DIR"
echo "  - Open $MAP_FILE in your web browser to view."
echo "==============================================="
