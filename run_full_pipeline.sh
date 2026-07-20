#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "  Geo-RAG: Full Processing & Global Clustering Pipeline"
echo "=========================================================="

# 1. Load Parameters from params.yaml
echo "Loading parameters from params.yaml..."
K_CLUSTERS=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('k_clusters', 40000))" 2>/dev/null || echo "40000")
AUTO_FIND_K=$(python3 -c "import yaml; print(str(yaml.safe_load(open('params.yaml'))['pipeline'].get('auto_find_k', False)).lower())" 2>/dev/null || echo "false")
K_MIN=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('k_min', 10000))" 2>/dev/null || echo "10000")
K_MAX=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('k_max', 50000))" 2>/dev/null || echo "50000")
K_STEP=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('k_step', 10000))" 2>/dev/null || echo "10000")
CLEANUP_ANOMALIES=$(python3 -c "import yaml; print(str(yaml.safe_load(open('params.yaml'))['pipeline'].get('cleanup_anomalies', False)).lower())" 2>/dev/null || echo "false")
MAX_MARKERS=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('max_markers', 10000))" 2>/dev/null || echo "10000")
LIMIT_CELLS=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('limit_cells', 0))" 2>/dev/null || echo "0")
CHECKPOINT_INTERVAL=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('checkpoint_interval', 1800))" 2>/dev/null || echo "1800")
BATCH_SIZE=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('batch_size', 64))" 2>/dev/null || echo "64")
CELL_CHUNK_SIZE=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('cell_chunk_size', 64))" 2>/dev/null || echo "64")
BASE_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('base_name', 'geo_space'))" 2>/dev/null || echo "geo_space")
OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('output_dir', '/home/ananthu/DATA/data_ananthu/full_pipeline_output'))" 2>/dev/null || echo "/home/ananthu/DATA/data_ananthu/full_pipeline_output")

# Input dirs
INPUT_DIRS=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('input_dirs', ''))" 2>/dev/null || echo "")
IWILDCAM_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('iwildcam_dir', ''))" 2>/dev/null || echo "")

# MLLM config
LABEL_METHOD=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('label_method', 'mllm'))" 2>/dev/null || echo "mllm")
MLLM_BACKEND=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('mllm_backend', 'sglang'))" 2>/dev/null || echo "sglang")
MLLM_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('mllm_model', 'google/gemma-4-E4B-it'))" 2>/dev/null || echo "google/gemma-4-E4B-it")
CHUNK_SIZE=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('chunk_size', 64))" 2>/dev/null || echo "64")
FILTER_MACRO=$(python3 -c "import yaml; print(str(yaml.safe_load(open('params.yaml'))['pipeline'].get('filter_macro', False)).lower())" 2>/dev/null || echo "false")
FILTER_SKY=$(python3 -c "import yaml; print(str(yaml.safe_load(open('params.yaml'))['pipeline'].get('filter_sky', False)).lower())" 2>/dev/null || echo "false")

# File Paths
RAW_PARQUET="$OUTPUT_DIR/${BASE_NAME}_deduplicated.parquet"
CLEANED_PARQUET="$OUTPUT_DIR/${BASE_NAME}_cleaned.parquet"
CLEANED_CSV="$OUTPUT_DIR/${BASE_NAME}_cleaned.csv"
CLUSTERED_PARQUET="$OUTPUT_DIR/${BASE_NAME}_clustered_k_${K_CLUSTERS}.parquet"
H3_SEMANTIC_INDEX="$OUTPUT_DIR/${BASE_NAME}_h3_semantic_index.parquet"
MAP_FILE="$OUTPUT_DIR/global_cluster_map.html"
SAMPLES_FILE="$OUTPUT_DIR/cluster_samples_k_${K_CLUSTERS}.html"
SCATTER_FILE="$OUTPUT_DIR/cluster_semantic_scatter_k_${K_CLUSTERS}.png"
OCCUPANCY_MAP="$OUTPUT_DIR/global_h3_occupancy_map.html"
SEMANTIC_MAP="$OUTPUT_DIR/global_h3_semantic_map.html"
STATS_PLOT="$OUTPUT_DIR/global_dataset_stats.png"
STATS_TEXT="$OUTPUT_DIR/global_dataset_stats.txt"
STATS_MAP="$OUTPUT_DIR/global_dataset_map.html"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# 2. Check for resume files
echo ""
echo "[Step 1/5] Spatial Deduplication (H3 + TIPSv2)..."
RESUME_FLAG=""
if [ -f "$RAW_PARQUET" ]; then
    echo "Found existing deduplicated data at $RAW_PARQUET. Incremental run enabled."
    RESUME_FLAG="--resume_from $RAW_PARQUET"
elif [ -f "$OUTPUT_DIR/${BASE_NAME}_deduplicated.pkl" ]; then
    echo "Found existing legacy data at $OUTPUT_DIR/${BASE_NAME}_deduplicated.pkl. Converting to Parquet on this run."
    RESUME_FLAG="--resume_from $OUTPUT_DIR/${BASE_NAME}_deduplicated.pkl"
fi

FILTER_FLAGS=""
if [ "$FILTER_MACRO" = "true" ]; then
    echo " -> iNaturalist Macro Filter: Enabled"
    FILTER_FLAGS="$FILTER_FLAGS --filter_macro"
fi
if [ "$FILTER_SKY" = "true" ]; then
    echo " -> iNaturalist Sky/Flying Object Filter: Enabled"
    FILTER_FLAGS="$FILTER_FLAGS --filter_sky"
fi

IWILDCAM_FLAG=""
IMAGE_ROOT_FLAG=""
if [ -n "$IWILDCAM_DIR" ]; then
    echo " -> iWildCam Directory: $IWILDCAM_DIR"
    IWILDCAM_FLAG="--iwildcam_dir $IWILDCAM_DIR"
    IMAGE_ROOT_FLAG="--image_root_dir $IWILDCAM_DIR"
fi

python3 -m src.processing.process_scraped_data \
  --dirs $INPUT_DIRS \
  --save_path "$OUTPUT_DIR" \
  --output_name "${BASE_NAME}_deduplicated" \
  --limit_cells "$LIMIT_CELLS" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --tips_batch_size "$BATCH_SIZE" \
  --cell_chunk_size "$CELL_CHUNK_SIZE" \
  $RESUME_FLAG \
  $FILTER_FLAGS \
  $IWILDCAM_FLAG

echo ""
echo "[Step 1b/5] Standardizing Dataset Timestamps..."
python3 -m src.processing.standardize_timestamps --input "$RAW_PARQUET"
RAW_CSV="$OUTPUT_DIR/${BASE_NAME}_deduplicated.csv"
if [ -f "$RAW_CSV" ]; then
    python3 -m src.processing.standardize_timestamps --input "$RAW_CSV"
fi

echo ""
echo "[Step 1d/5] Cleaning Coordinate Anomalies (if enabled)..."
if [ "$CLEANUP_ANOMALIES" = "true" ]; then
    echo "Running coordinate anomaly cleanup..."
    python3 -m src.processing.cleanup_coordinate_anomalies --input "$RAW_PARQUET" --csv "$RAW_CSV" --output "$CLEANED_PARQUET" --output_csv "$CLEANED_CSV"
    INPUT_PARQUET="$CLEANED_PARQUET"
    INPUT_CSV="$CLEANED_CSV"
else
    echo "Coordinate anomaly cleanup is disabled."
    INPUT_PARQUET="$RAW_PARQUET"
    INPUT_CSV="$RAW_CSV"
fi

echo ""
echo "[Step 1c/5] Automatically Finding Optimal k (if enabled)..."
if [ "$AUTO_FIND_K" = "true" ]; then
    echo "Running spatial block validation to determine optimal k..."
    python3 -m src.utils.validate_cluster_count \
      --input "$INPUT_PARQUET" \
      --k_min "$K_MIN" \
      --k_max "$K_MAX" \
      --k_step "$K_STEP" \
      --update_params \
      --sample_limit 0
      
    # Reload the newly estimated K_CLUSTERS value from params.yaml
    echo "Reloading k_clusters parameter..."
    K_CLUSTERS=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('k_clusters', 40000))" 2>/dev/null || echo "40000")
    echo "Optimal k determined: $K_CLUSTERS"
    
    # Update target clustered parquet path with the new K_CLUSTERS
    CLUSTERED_PARQUET="$OUTPUT_DIR/${BASE_NAME}_clustered_k_${K_CLUSTERS}.parquet"
    SAMPLES_FILE="$OUTPUT_DIR/cluster_samples_k_${K_CLUSTERS}.html"
    SCATTER_FILE="$OUTPUT_DIR/cluster_semantic_scatter_k_${K_CLUSTERS}.png"
else
    echo "Auto-find k is disabled. Using current k_clusters value: $K_CLUSTERS"
fi

echo ""
echo "[Step 2/5] Global Unsupervised Clustering (K-Means)..."
python3 -m src.indexing.cluster_images_global \
  --pkl "$INPUT_PARQUET" \
  --k "$K_CLUSTERS" \
  --out "$CLUSTERED_PARQUET" \
  --minibatch \
  --label_method "$LABEL_METHOD" \
  --mllm_backend "$MLLM_BACKEND" \
  --mllm_model "$MLLM_MODEL" \
  --chunk_size "$CHUNK_SIZE" \
  --gpu \
  $IMAGE_ROOT_FLAG

echo ""
echo "[Step 2b/5] Re-labeling Failed Clusters (due to download timeouts)..."
python3 -m src.indexing.relabel_failed_clusters \
  --in "$CLUSTERED_PARQUET" \
  --mllm_model "$MLLM_MODEL" \
  --mllm_backend "$MLLM_BACKEND" \
  --fallback_depth 10 \
  $IMAGE_ROOT_FLAG

echo ""
echo "[Step 2c/5] Building H3 Spatial-Semantic Index..."
python3 -m src.indexing.build_spatial_semantic_index \
  --input "$CLUSTERED_PARQUET" \
  --output "$H3_SEMANTIC_INDEX"

echo ""
echo "[Step 3/5] Generating Cluster Map..."
python3 -m src.visualization.visualize_clusters \
  --pkl_file "$CLUSTERED_PARQUET" \
  --output "$MAP_FILE" \
  --max_markers "$MAX_MARKERS"

echo ""
echo "[Step 4/5] Generating Cluster Representative Samples..."
python3 -m src.visualization.visualize_cluster_samples \
  --pkl "$CLUSTERED_PARQUET" \
  --out "$SAMPLES_FILE" \
  --top_n 6 \
  $IMAGE_ROOT_FLAG

echo ""
echo "[Step 5/5] Generating Semantic Scatter Plot (UMAP 2D)..."
python3 -m src.visualization.visualize_cluster_scatter \
  --pkl "$CLUSTERED_PARQUET" \
  --out "$SCATTER_FILE"

echo ""
echo "Generating occupancy map"
python3 -m src.visualization.generate_h3_occupancy_map \
  --dirs "$OUTPUT_DIR" \
  --output "$OCCUPANCY_MAP"

echo ""
echo "Generating H3 spatial-semantic map"
python3 -m src.visualization.generate_h3_semantic_map \
  --index "$H3_SEMANTIC_INDEX" \
  --output "$SEMANTIC_MAP" \
  --res 5

echo ""
echo "Generating dataset statistics report, plots, and optimized H3 map..."
python3 -m src.utils.dataset_statistics \
  --input "$INPUT_PARQUET" \
  --spatial_index "$H3_SEMANTIC_INDEX" \
  --output_plot "$STATS_PLOT" \
  --output_text "$STATS_TEXT" \
  --output_map "$STATS_MAP"

echo ""
echo "=========================================================="
echo "  Pipeline Complete!"
echo "  Outputs located at: $OUTPUT_DIR"
echo "=========================================================="

# 3. Handle DVC Standalone Tracking (Option 2)
# Check if parent directory on HDD has DVC initialized
HDD_DIR=$(dirname "$OUTPUT_DIR")
if [ -d "$HDD_DIR/.dvc" ]; then
    echo ""
    echo "=========================================================="
    echo "  DVC Standalone Tracking & Upload"
    echo "=========================================================="
    echo "Updating DVC dataset version on HDD..."
    
    # Run DVC commands inside the HDD folder
    (cd "$HDD_DIR" && dvc add "$(basename "$OUTPUT_DIR")" && dvc push)
    
    # Copy the updated tracking .dvc file back to this SSD Git repository
    cp "$HDD_DIR/$(basename "$OUTPUT_DIR").dvc" .
    
    echo "Done! Commit '$(basename "$OUTPUT_DIR").dvc' to Git to share this version."
    echo "=========================================================="
fi

# Run git commands to add and commit the updated .dvc file
git add "$(basename "$OUTPUT_DIR").dvc"
git commit -m "Update DVC tracking for $(basename "$OUTPUT_DIR")"
git push