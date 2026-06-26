#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "  Geo-RAG: Full Processing & Global Clustering Pipeline"
echo "=========================================================="

# Read the label method from params.yaml using Python if PyYAML is installed
LABEL_METHOD=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('label_method', 'zeroshot'))" 2>/dev/null || echo "zeroshot")
MLLM_BACKEND=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('mllm_backend', 'ollama'))" 2>/dev/null || echo "ollama")
MLLM_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('params.yaml'))['pipeline'].get('mllm_model', 'gemma4:e4b'))" 2>/dev/null || echo "gemma4:e4b")

if [ "$LABEL_METHOD" = "mllm" ]; then
    echo "[Info] Pipeline configured to use MLLM cluster labeling (${MLLM_BACKEND})."
    echo "[Info] Please make sure your server is running in a screen session."
    if [ "$MLLM_BACKEND" = "sglang" ]; then
        echo "       Launch command: python -m sglang.launch_server --model ${MLLM_MODEL} --port 30000 --host 0.0.0.0 --mem-fraction-static 0.7"
    else
        echo "       Launch command: ollama run ${MLLM_MODEL}"
    fi
    echo ""
fi

echo "Running DVC pipeline..."
dvc repro
echo "=========================================================="
