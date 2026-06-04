#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "  Geo-RAG: Full Processing & Global Clustering Pipeline"
echo "=========================================================="

# --- CONFIGURATION ---
# Input Directories (Modify these for your machine)
FLICKR_DIR="/user/aaniraj/home/Documents/Projects/data/flickr_scrape_rand"
MAPILLARY_DIR="/user/aaniraj/home/Documents/Projects/data/mapillary_scrape_rand"
GS_DIR="/user/aaniraj/home/Documents/Projects/data/global-streetscapes/train"

# Output Configuration
OUTPUT_DIR="./full_pipeline_output"
BASE_NAME="geo_space"
K_CLUSTERS=10  # Number of visual clusters to find
LIMIT_CELLS=0  # Set to 0 to process EVERYTHING, or a small number for testing

# File Paths
RAW_PKL="$OUTPUT_DIR/${BASE_NAME}_deduplicated.pkl"
CLUSTERED_PKL="$OUTPUT_DIR/${BASE_NAME}_clustered.pkl"
MAP_FILE="$OUTPUT_DIR/global_cluster_map.html"
SAMPLES_FILE="$OUTPUT_DIR/cluster_samples.html"
SCATTER_FILE="$OUTPUT_DIR/cluster_semantic_scatter.png"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo ""
echo "[Step 1/5] Spatial Deduplication (H3 + TIPSv2)..."
python process_scraped_data.py \
  --dirs "$FLICKR_DIR" "$MAPILLARY_DIR" "$GS_DIR" \
  --save_path "$OUTPUT_DIR" \
  --output_name "${BASE_NAME}_deduplicated" \
  --limit_cells "$LIMIT_CELLS"

echo ""
echo "[Step 2/5] Global Unsupervised Clustering (K-Means)..."
# By default, uses Raw 768D Normalized embeddings.
# Add --use_umap to enable UMAP reduction.
# Add --visualize to generate 2D coordinates for Step 5.
# Add --minibatch for faster processing of 2M+ images.
python cluster_images_global.py \
  --pkl "$RAW_PKL" \
  --k "$K_CLUSTERS" \
  --visualize \
  --out "$CLUSTERED_PKL"

echo ""
echo "[Step 3/5] Generating Cluster Map..."
python visualize_clusters.py \
  --pkl_file "$CLUSTERED_PKL" \
  --output "$MAP_FILE"

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
