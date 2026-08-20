#!/usr/bin/env bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


# Exit immediately if a command exits with a non-zero status
set -e

# Source the .env file if it exists to load keys into environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Read target benchmark from argument (defaults to lucas)
BENCHMARK="${1:-lucas}"

# Validate benchmark name
if [[ ! "$BENCHMARK" =~ ^(lucas|places|eunis|env_zones)$ ]]; then
    echo "Error: Invalid benchmark name '$BENCHMARK'."
    echo "Usage: $0 [lucas|places|eunis|env_zones]"
    exit 1
fi

# Load models list from YAML config
YAML_CONFIG="config/evaluation/compare_models.yaml"
if [ ! -f "$YAML_CONFIG" ]; then
    echo "Error: Configuration file not found at $YAML_CONFIG"
    exit 1
fi

MODELS=$(python3 -c "import yaml; print(' '.join(yaml.safe_load(open('$YAML_CONFIG'))['models']))")

if [ -z "$MODELS" ]; then
    echo "Error: No models found in $YAML_CONFIG"
    exit 1
fi

echo "================================================================================"
echo "   Launching Automated Model Comparison for: $BENCHMARK"
echo "   Config: $YAML_CONFIG"
echo "   Target Models: $MODELS"
echo "================================================================================"

python3 -m src.evaluation.compare_models --benchmark "$BENCHMARK" --models $MODELS
