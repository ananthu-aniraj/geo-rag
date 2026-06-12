#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "  Geo-RAG: Full Processing & Global Clustering Pipeline"
echo "=========================================================="

# --- CONFIGURATION ---
# Input Directories (Modify these for your machine)
FLICKR_DIR="/home/ananthu/DATA/data_ananthu/flickr_scrape_rand"
MAPILLARY_DIR="/home/ananthu/DATA/data_ananthu/mapillary_scrape_rand"
GS_DIR="/home/ananthu/DATA/data_ananthu/global-streetscapes/train"
FLICKR_DIR_2="/home/ananthu/DATA/data_ananthu/flickr_scrape"
MAPILLARY_DIR_2="/home/ananthu/DATA/data_ananthu/mapillary_scrape"

# Output Configuration
OUTPUT_DIR="/home/ananthu/DATA/data_ananthu/full_pipeline_output"
BASE_NAME="geo_space"
K_CLUSTERS=100  # Number of visual clusters to find
MAX_MARKERS=10000
LIMIT_CELLS=0  # Set to 0 to process EVERYTHING, or a small number for testing

# File Paths
RAW_PKL="$OUTPUT_DIR/${BASE_NAME}_deduplicated.pkl"
CLUSTERED_PKL="$OUTPUT_DIR/${BASE_NAME}_clustered_k_${K_CLUSTERS}.pkl"
MAP_FILE="$OUTPUT_DIR/global_cluster_map_markers_${MAX_MARKERS}_k_${K_CLUSTERS}.html"
SAMPLES_FILE="$OUTPUT_DIR/cluster_samples_k_${K_CLUSTERS}.html"
SCATTER_FILE="$OUTPUT_DIR/cluster_semantic_scatter_k_${K_CLUSTERS}.png"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo ""
echo "[Step 1/5] Spatial Deduplication (H3 + TIPSv2)..."
RESUME_FLAG=""
if [ -f "$RAW_PKL" ]; then
    echo "Found existing deduplicated data at $RAW_PKL. Incremental run enabled."
    RESUME_FLAG="--resume_from $RAW_PKL"
fi

python process_scraped_data.py \
  --dirs "$FLICKR_DIR" "$MAPILLARY_DIR" \
  --save_path "$OUTPUT_DIR" \
  --output_name "${BASE_NAME}_deduplicated" \
  --limit_cells "$LIMIT_CELLS" \
  $RESUME_FLAG

echo ""
echo "[Step 2/5] Global Unsupervised Clustering (K-Means)..."
# By default, uses Raw 768D Normalized embeddings.
# Add --use_umap to enable UMAP reduction (e.g. to 10D) before K-Means.
# Add --minibatch for faster processing of 2M+ images.
python cluster_images_global.py \
  --pkl "$RAW_PKL" \
  --k "$K_CLUSTERS" \
  --out "$CLUSTERED_PKL" \
  --minibatch

echo ""
echo "[Step 3/5] Generating Cluster Map..."
python visualize_clusters.py \
  --pkl_file "$CLUSTERED_PKL" \
  --output "$MAP_FILE" \
  --max_markers "$MAX_MARKERS"

echo ""
echo "[Step 4/5] Generating Cluster Representative Samples..."
python visualize_cluster_samples.py \
  --pkl "$CLUSTERED_PKL" \
  --out "$SAMPLES_FILE" \
  --top_n 6

echo ""
echo "[Step 5/5] Generating Semantic Scatter Plot (UMAP 2D)..."
python visualize_cluster_scatter.py \
  --pkl "$CLUSTERED_PKL" \
  --out "$SCATTER_FILE"

echo ""
echo "=========================================================="
echo "  Pipeline Complete!"
echo "  1. Deduplicated Space: $RAW_PKL"
echo "  2. Clustered Space:    $CLUSTERED_PKL"
echo "  3. Interactive Map:    $MAP_FILE"
echo "  4. Cluster Samples:     $SAMPLES_FILE"
echo "  5. Semantic Scatter:   $SCATTER_FILE"
echo "=========================================================="
