#!/bin/bash
# shellcheck disable=SC2086

# Exit immediately if a command exits with a non-zero status
set -e

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "=========================================================="
echo "  Geo-RAG: Streamlined Offline Clustering & Vis Pipeline"
echo "=========================================================="

# Load environment variables from .env if present
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
fi

# 1. Load Parameters from config/pipeline/params_offline.yaml
PARAMS_YAML="config/pipeline/params_offline.yaml"
if [ ! -f "$PARAMS_YAML" ]; then
    PARAMS_YAML="params_offline.yaml"
fi

get_param() {
    python3 -m src.utils.config get "$PARAMS_YAML" "offline_pipeline" "$1" 2>/dev/null || python3 -m src.utils.config get "$PARAMS_YAML" "pipeline" "$1" 2>/dev/null || echo ""
}

# Config defaults
INPUT_PARQUET=$(get_param "input_parquet")
[ -z "$INPUT_PARQUET" ] && INPUT_PARQUET="full_pipeline_output/geo_space_cleaned_offline.parquet"

IMAGE_ROOT_DIRS=$(get_param "image_root_dirs")

MODEL_NAME=$(get_param "model_name")
[ -z "$MODEL_NAME" ] && MODEL_NAME="google/tipsv2-b14"

REPRESENTATION_TYPE=$(get_param "representation_type")
[ -z "$REPRESENTATION_TYPE" ] && REPRESENTATION_TYPE="cls"

PRECISION=$(get_param "precision")
[ -z "$PRECISION" ] && PRECISION="float16"

RUN_BACKFILL_EMBEDDINGS=$(get_param "run_backfill_embeddings")
[ -z "$RUN_BACKFILL_EMBEDDINGS" ] && RUN_BACKFILL_EMBEDDINGS="false"

BACKFILL_BATCH_SIZE=$(get_param "backfill_batch_size")
[ -z "$BACKFILL_BATCH_SIZE" ] && BACKFILL_BATCH_SIZE=32

BACKFILL_CHUNK_SIZE=$(get_param "backfill_chunk_size")
[ -z "$BACKFILL_CHUNK_SIZE" ] && BACKFILL_CHUNK_SIZE=512

K_CLUSTERS=$(get_param "k_clusters")
[ -z "$K_CLUSTERS" ] && K_CLUSTERS=40000

AUTO_FIND_K=$(get_param "auto_find_k")
[ -z "$AUTO_FIND_K" ] && AUTO_FIND_K="false"

K_MIN=$(get_param "k_min")
[ -z "$K_MIN" ] && K_MIN=10000

K_MAX=$(get_param "k_max")
[ -z "$K_MAX" ] && K_MAX=50000

K_STEP=$(get_param "k_step")
[ -z "$K_STEP" ] && K_STEP=10000

RUN_COORDINATE_CLEANUP=$(get_param "run_coordinate_cleanup")
[ -z "$RUN_COORDINATE_CLEANUP" ] && RUN_COORDINATE_CLEANUP="false"

CLEANUP_PLATFORM=$(get_param "cleanup_platform")
CLEANUP_CONTINENT=$(get_param "cleanup_continent")

RUN_TIMESTAMP_STANDARDIZATION=$(get_param "run_timestamp_standardization")
[ -z "$RUN_TIMESTAMP_STANDARDIZATION" ] && RUN_TIMESTAMP_STANDARDIZATION="false"

KOPPEN_GEIGER_TIF=$(get_param "koppen_geiger_tif")
LAND_SHP=$(get_param "land_shp")

ENABLE_MLLM=$(get_param "enable_mllm")
[ -z "$ENABLE_MLLM" ] && ENABLE_MLLM="false"

MLLM_BACKEND=$(get_param "mllm_backend")
[ -z "$MLLM_BACKEND" ] && MLLM_BACKEND="sglang"

MLLM_MODEL=$(get_param "mllm_model")
[ -z "$MLLM_MODEL" ] && MLLM_MODEL="google/gemma-4-E4B-it"

CHUNK_SIZE=$(get_param "chunk_size")
[ -z "$CHUNK_SIZE" ] && CHUNK_SIZE=64

NUM_MEDOIDS=$(get_param "num_medoids")
[ -z "$NUM_MEDOIDS" ] && NUM_MEDOIDS=4

BASE_NAME=$(get_param "base_name")
[ -z "$BASE_NAME" ] && BASE_NAME="geo_space_offline"

OUTPUT_DIR=$(get_param "output_dir")

MAX_MARKERS=$(get_param "max_markers")
[ -z "$MAX_MARKERS" ] && MAX_MARKERS=10000

CAMERA_TRAP_PLATFORMS=$(get_param "camera_trap_platforms")
[ -z "$CAMERA_TRAP_PLATFORMS" ] && CAMERA_TRAP_PLATFORMS="iwildcam wildobs wildlife_insights snapshotusa"

GPU_FLAG="--gpu"

# CLI flag overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT_PARQUET="$2"
      shift 2
      ;;
    --image_root_dirs)
      IMAGE_ROOT_DIRS=""
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        IMAGE_ROOT_DIRS="$IMAGE_ROOT_DIRS $1"
        shift
      done
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --representation_type)
      REPRESENTATION_TYPE="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --run_backfill_embeddings|--backfill)
      RUN_BACKFILL_EMBEDDINGS="true"
      shift
      ;;
    --backfill_batch_size|--batch_size)
      BACKFILL_BATCH_SIZE="$2"
      shift 2
      ;;
    --backfill_chunk_size|--chunk_size)
      BACKFILL_CHUNK_SIZE="$2"
      shift 2
      ;;
    --k_clusters)
      K_CLUSTERS="$2"
      shift 2
      ;;
    --auto_find_k)
      AUTO_FIND_K="true"
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --base_name)
      BASE_NAME="$2"
      shift 2
      ;;
    --enable_mllm)
      ENABLE_MLLM="true"
      shift
      ;;
    --run_coordinate_cleanup)
      RUN_COORDINATE_CLEANUP="true"
      shift
      ;;
    --run_timestamp_standardization)
      RUN_TIMESTAMP_STANDARDIZATION="true"
      shift
      ;;
    --no-gpu|--no_gpu)
      GPU_FLAG="--no_gpu"
      shift
      ;;
    -h|--help)
      echo "Usage: ./scripts/pipeline/run_offline_pipeline.sh [options]"
      echo ""
      echo "Options:"
      echo "  --input PATH                 Path to input offline parquet file"
      echo "  --image_root_dirs DIRS...    Directories containing offline images"
      echo "  --model_name MODEL           Vision encoder (e.g. google/tipsv2-b14, facebook/dinov2-base)"
      echo "  --representation_type TYPE   Embedding type (cls, avg_patch, cls_avg_patch)"
      echo "  --precision PREC             Stored precision (float16, float32)"
      echo "  --run_backfill_embeddings    Compute/backfill embeddings before clustering"
      echo "  --backfill_batch_size BATCH  Batch size for feature extraction (default: 32)"
      echo "  --backfill_chunk_size CHUNK  Chunk size for parallel loading (default: 512)"
      echo "  --k_clusters K               Number of clusters (default: 40000)"
      echo "  --auto_find_k                Enable spatial block validation to find optimal k"
      echo "  --output_dir DIR             Directory for dataset & output storage (default: parent dir of --input)"
      echo "  --base_name NAME             Base name prefix (default: geo_space_offline)"
      echo "  --enable_mllm                Enable MLLM medoid labeling"
      echo "  --run_coordinate_cleanup     Run coordinate anomaly detection"
      echo "  --run_timestamp_standardization Run timestamp & solar time standardization"
      echo "  --no-gpu                     Disable GPU FAISS acceleration"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ ! -f "$INPUT_PARQUET" ]; then
    echo "❌ Error: Input parquet dataset not found at '$INPUT_PARQUET'."
    echo "Please provide a valid file via --input or configure input_parquet in $PARAMS_YAML."
    exit 1
fi

# Auto-detect output_dir from input_parquet if not explicitly specified
if [ -z "$OUTPUT_DIR" ] || [ "$OUTPUT_DIR" = "auto" ]; then
    OUTPUT_DIR=$(dirname "$INPUT_PARQUET")
fi
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="."

MODEL_SLUG=$(echo "$MODEL_NAME" | tr '/' '_')
EXP_TAG="${MODEL_SLUG}_${REPRESENTATION_TYPE}_${PRECISION}"
VIS_DIR="${OUTPUT_DIR}/vis_${K_CLUSTERS}_${EXP_TAG}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$VIS_DIR"

# Parquet files in main OUTPUT_DIR
CLUSTERED_PARQUET="$OUTPUT_DIR/${BASE_NAME}_${EXP_TAG}_clustered_k_${K_CLUSTERS}.parquet"
H3_SEMANTIC_INDEX="$OUTPUT_DIR/${BASE_NAME}_${EXP_TAG}_h3_semantic_index.parquet"

# Visualizations in VIS_DIR
MAP_FILE="$VIS_DIR/global_cluster_map.html"
SAMPLES_FILE="$VIS_DIR/cluster_samples_k_${K_CLUSTERS}.html"
SCATTER_FILE="$VIS_DIR/cluster_semantic_scatter_k_${K_CLUSTERS}.png"
OCCUPANCY_MAP="$VIS_DIR/global_h3_occupancy_map.html"
SEMANTIC_MAP="$VIS_DIR/global_h3_semantic_map.html"
STATS_PLOT="$VIS_DIR/global_dataset_stats.png"
STATS_TEXT="$VIS_DIR/global_dataset_stats.txt"
STATS_MAP="$VIS_DIR/global_dataset_map.html"
CLUSTER_COUNT_PLOT="$VIS_DIR/cluster_count_validation.png"

IMAGE_ROOT_FLAG=""
if [ -n "$IMAGE_ROOT_DIRS" ]; then
    IMAGE_ROOT_FLAG="--image_root_dir $IMAGE_ROOT_DIRS"
fi

LAND_SHP_FLAG=""
if [ -z "$LAND_SHP" ]; then
    if [ -f "shapefiles/ne_10m_admin_0_countries.shp" ]; then
        LAND_SHP="shapefiles/ne_10m_admin_0_countries.shp"
    elif [ -f "ne_10m_admin_0_countries.shp" ]; then
        LAND_SHP="ne_10m_admin_0_countries.shp"
    fi
fi
if [ -n "$LAND_SHP" ]; then
    LAND_SHP_FLAG="--land_shp $LAND_SHP"
fi

echo ""
echo "Configuration Summary:"
echo " - Input Dataset       : $INPUT_PARQUET"
echo " - Model Identifier    : $MODEL_NAME"
echo " - Representation Type : $REPRESENTATION_TYPE"
echo " - Precision           : $PRECISION"
echo " - Backfill Embeddings : $RUN_BACKFILL_EMBEDDINGS"
echo " - Clusters (k)        : $K_CLUSTERS"
echo " - Output Directory    : $OUTPUT_DIR"
echo " - Clustered Sidecar   : $CLUSTERED_PARQUET"
echo " - Visualizations Dir  : $VIS_DIR"
echo ""

# Optional Preprocessing Step: Timestamp Standardization
if [ "$RUN_TIMESTAMP_STANDARDIZATION" = "true" ]; then
    echo "Standardizing dataset timestamps and Köppen-Geiger mapping..."
    KOPPEN_FLAG=""
    if [ -n "$KOPPEN_GEIGER_TIF" ]; then
        KOPPEN_FLAG="--koppen_tif $KOPPEN_GEIGER_TIF"
    fi
    python3 -m src.processing.standardize_timestamps --input "$INPUT_PARQUET" $KOPPEN_FLAG $LAND_SHP_FLAG
fi

# Optional Preprocessing Step: Coordinate Anomaly Cleanup
if [ "$RUN_COORDINATE_CLEANUP" = "true" ]; then
    echo "Running coordinate anomaly cleanup..."
    CLEANUP_FLAGS=""
    if [ -n "$CLEANUP_PLATFORM" ]; then
        CLEANUP_FLAGS="$CLEANUP_FLAGS --platform $CLEANUP_PLATFORM"
    fi
    if [ -n "$CLEANUP_CONTINENT" ]; then
        CLEANUP_FLAGS="$CLEANUP_FLAGS --continent $CLEANUP_CONTINENT"
    fi
    CLEANED_OUTPUT="$OUTPUT_DIR/${BASE_NAME}_cleaned.parquet"
    python3 -m src.processing.cleanup_coordinate_anomalies --input "$INPUT_PARQUET" --output "$CLEANED_OUTPUT" $CLEANUP_FLAGS
    INPUT_PARQUET="$CLEANED_OUTPUT"
fi

# Optional Preprocessing Step: Backfill Embeddings for Specified Model
if [ "$RUN_BACKFILL_EMBEDDINGS" = "true" ]; then
    echo ""
    echo "[Preprocessing] Backfilling embeddings for model '$MODEL_NAME' ($REPRESENTATION_TYPE, $PRECISION)..."
    python3 -m src.processing.backfill_embeddings \
      --input "$INPUT_PARQUET" \
      --model_name "$MODEL_NAME" \
      --representation_type "$REPRESENTATION_TYPE" \
      --precision "$PRECISION" \
      --batch_size "$BACKFILL_BATCH_SIZE" \
      --chunk_size "$BACKFILL_CHUNK_SIZE" \
      $IMAGE_ROOT_FLAG
fi

# Step 1: Auto-find optimal k (if enabled)
if [ "$AUTO_FIND_K" = "true" ]; then
    echo ""
    echo "[Step 1/5] Running spatial block validation to determine optimal k..."
    python3 -m src.utils.validate_cluster_count \
      --input "$INPUT_PARQUET" \
      --k_min "$K_MIN" \
      --k_max "$K_MAX" \
      --k_step "$K_STEP" \
      --representation_type "$REPRESENTATION_TYPE" \
      --model_name "$MODEL_NAME" \
      --precision "$PRECISION" \
      --output_plot "$CLUSTER_COUNT_PLOT" \
      --sample_limit 0 \
      --params_path "$PARAMS_YAML"

    K_CLUSTERS=$(get_param "k_clusters")
    [ -z "$K_CLUSTERS" ] && K_CLUSTERS=40000
    echo "Optimal k determined: $K_CLUSTERS"

    # Refresh targets with updated k
    VIS_DIR="${OUTPUT_DIR}/vis_${K_CLUSTERS}_${EXP_TAG}"
    mkdir -p "$VIS_DIR"
    CLUSTERED_PARQUET="$OUTPUT_DIR/${BASE_NAME}_${EXP_TAG}_clustered_k_${K_CLUSTERS}.parquet"
    H3_SEMANTIC_INDEX="$OUTPUT_DIR/${BASE_NAME}_${EXP_TAG}_h3_semantic_index.parquet"
    MAP_FILE="$VIS_DIR/global_cluster_map.html"
    SAMPLES_FILE="$VIS_DIR/cluster_samples_k_${K_CLUSTERS}.html"
    SCATTER_FILE="$VIS_DIR/cluster_semantic_scatter_k_${K_CLUSTERS}.png"
fi

# Step 2: Semantic drift check
CLUSTERING_MODE="fit"
if [ -f "$CLUSTERED_PARQUET" ]; then
    echo ""
    echo "[Step 2/5] Pre-existing clustered sidecar found at $CLUSTERED_PARQUET."
    echo "Checking for semantic drift in the dataset..."
    DETECTOR_MODE=$(python3 -m src.utils.check_semantic_drift \
      --input "$INPUT_PARQUET" \
      --centroids_parquet "$CLUSTERED_PARQUET" \
      --representation_type "$REPRESENTATION_TYPE" \
      --model_name "$MODEL_NAME" \
      --precision "$PRECISION" \
      --k_clusters "$K_CLUSTERS" 2>/dev/null || echo "fit")

    if [ "$DETECTOR_MODE" = "assign" ]; then
        echo "Semantic representation is stable. Enabling ASSIGN mode (reusing existing centroids)."
        CLUSTERING_MODE="assign"
    else
        echo "Significant semantic drift or layout shift detected. Enabling FIT mode to re-cluster."
        CLUSTERING_MODE="fit"
    fi
else
    echo ""
    echo "[Step 2/5] No pre-existing clustered database found for k=$K_CLUSTERS. Enabling FIT mode."
    CLUSTERING_MODE="fit"
fi

# Step 3: Global clustering
echo ""
echo "[Step 3/5] Global Unsupervised FAISS Clustering (Mode: $CLUSTERING_MODE)..."
if [ "$CLUSTERING_MODE" = "assign" ]; then
    python3 -m src.indexing.cluster_images_global \
      --pkl "$INPUT_PARQUET" \
      --k "$K_CLUSTERS" \
      --out "$CLUSTERED_PARQUET.tmp" \
      --clustering_mode assign \
      --centroids_parquet "$CLUSTERED_PARQUET" \
      --model_name "$MODEL_NAME" \
      --representation_type "$REPRESENTATION_TYPE" \
      --precision "$PRECISION" \
      $GPU_FLAG
    mv "$CLUSTERED_PARQUET.tmp" "$CLUSTERED_PARQUET"
else
    python3 -m src.indexing.cluster_images_global \
      --pkl "$INPUT_PARQUET" \
      --k "$K_CLUSTERS" \
      --out "$CLUSTERED_PARQUET" \
      --model_name "$MODEL_NAME" \
      --representation_type "$REPRESENTATION_TYPE" \
      --precision "$PRECISION" \
      $GPU_FLAG
fi

# Step 4: Optional MLLM Labeling
if [ "$ENABLE_MLLM" = "true" ]; then
    echo ""
    echo "[Step 3b/5] MLLM Cluster Auto-Labeling..."
    # Detect AppArmor
    APPARMOR_FLAG=""
    if [ -f /sys/module/apparmor/parameters/enabled ] && [ "$(cat /sys/module/apparmor/parameters/enabled 2>/dev/null)" = "Y" ]; then
        APPARMOR_FLAG="--security-opt apparmor=unconfined"
    fi

    SGLANG_STARTED=false
    cleanup() {
        if [ "$SGLANG_STARTED" = "true" ]; then
            echo "Trap triggered: Cleaning up SGLang server container..."
            docker rm -f sglang-server >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT INT TERM ERR

    docker rm -f sglang-server >/dev/null 2>&1 || true
    echo "Launching SGLang server container..."
    docker run -d \
      --name sglang-server \
      --runtime nvidia \
      $APPARMOR_FLAG \
      -e NVIDIA_VISIBLE_DEVICES=0 \
      --shm-size 32g \
      -p 30000:30000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --env "HF_TOKEN=${HF_TOKEN}" \
      --ipc=host \
      lmsysorg/sglang:latest-runtime \
      bash -c "pip install distro && python3 -m sglang.launch_server --model-path $MLLM_MODEL --host 0.0.0.0 --port 30000 --mem-fraction-static 0.75"

    SGLANG_STARTED=true
    echo "Waiting for SGLang server to initialize..."
    until curl -s http://localhost:30000/health > /dev/null; do
        if ! docker ps -q --filter "name=sglang-server" | grep -q .; then
            echo "[ERROR] SGLang server container exited unexpectedly."
            docker logs --tail 20 sglang-server
            exit 1
        fi
        sleep 2
    done
    echo "SGLang server is live! Executing cluster labeling..."

    python3 -m src.indexing.label_clusters_mllm \
      --in "$CLUSTERED_PARQUET" \
      --label_method "mllm" \
      --mllm_backend "$MLLM_BACKEND" \
      --mllm_model "$MLLM_MODEL" \
      --chunk_size "$CHUNK_SIZE" \
      --representation_type "$REPRESENTATION_TYPE" \
      --precision "$PRECISION" \
      --num_medoids "$NUM_MEDOIDS" \
      $IMAGE_ROOT_FLAG

    python3 -m src.indexing.relabel_failed_clusters \
      --in "$CLUSTERED_PARQUET" \
      --mllm_model "$MLLM_MODEL" \
      --mllm_backend "$MLLM_BACKEND" \
      --representation_type "$REPRESENTATION_TYPE" \
      --precision "$PRECISION" \
      --num_medoids "$NUM_MEDOIDS" \
      $IMAGE_ROOT_FLAG

    docker rm -f sglang-server >/dev/null 2>&1 || true
    SGLANG_STARTED=false
fi

# Step 5: H3 Spatial-Semantic Index
echo ""
echo "[Step 4/5] Building H3 Spatial-Semantic Index..."
python3 -m src.indexing.build_spatial_semantic_index \
  --input "$CLUSTERED_PARQUET" \
  --output "$H3_SEMANTIC_INDEX"

# Step 6: Visualizations in VIS_DIR
echo ""
echo "[Step 5/5] Generating Visualizations in '$VIS_DIR'..."

echo " -> Generating Global Cluster Map..."
python3 -m src.visualization.visualize_clusters \
  --pkl_file "$CLUSTERED_PARQUET" \
  --output "$MAP_FILE" \
  --max_markers "$MAX_MARKERS" \
  $IMAGE_ROOT_FLAG

echo " -> Generating Cluster Representative Image Grid..."
python3 -m src.visualization.visualize_cluster_samples \
  --pkl "$CLUSTERED_PARQUET" \
  --out "$SAMPLES_FILE" \
  --top_n 6 \
  --model_name "$MODEL_NAME" \
  --representation_type "$REPRESENTATION_TYPE" \
  --precision "$PRECISION" \
  $IMAGE_ROOT_FLAG

echo " -> Generating 2D Semantic Scatter Plot (UMAP)..."
python3 -m src.visualization.visualize_cluster_scatter \
  --pkl "$CLUSTERED_PARQUET" \
  --out "$SCATTER_FILE" \
  --model_name "$MODEL_NAME" \
  --representation_type "$REPRESENTATION_TYPE" \
  --precision "$PRECISION"

echo " -> Generating H3 Occupancy Map..."
python3 -m src.visualization.generate_h3_occupancy_map \
  --dirs "$INPUT_PARQUET" \
  --output "$OCCUPANCY_MAP"

echo " -> Generating H3 Spatial-Semantic Map..."
python3 -m src.visualization.generate_h3_semantic_map \
  --index "$H3_SEMANTIC_INDEX" \
  --output "$SEMANTIC_MAP" \
  --res 5

echo " -> Generating Dataset Statistics & Coverage Report..."
CAMERA_TRAP_FLAG=""
if [ -n "$CAMERA_TRAP_PLATFORMS" ]; then
    CAMERA_TRAP_FLAG="--camera_trap_platforms $CAMERA_TRAP_PLATFORMS"
fi

python3 -m src.utils.dataset_statistics \
  --input "$CLUSTERED_PARQUET" \
  --spatial_index "$H3_SEMANTIC_INDEX" \
  --output_plot "$STATS_PLOT" \
  --output_text "$STATS_TEXT" \
  --output_map "$STATS_MAP" \
  --min_count 20 \
  $LAND_SHP_FLAG \
  $CAMERA_TRAP_FLAG

echo ""
echo "=========================================================="
echo "  Offline Pipeline Complete!"
echo "  Clustered Sidecar : $CLUSTERED_PARQUET"
echo "  H3 Spatial Index  : $H3_SEMANTIC_INDEX"
echo "  Visualizations    : $VIS_DIR"
echo "=========================================================="
