#!/bin/bash
# Analyzer Inference Script
# Usage: ./scripts/inference_analyzer.sh --config <config_path> --dataset <dataset_path>

# Parse arguments
CONFIG=""
DATASET=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$CONFIG" ] || [ -z "$DATASET" ]; then
  echo "Usage: $0 --config <config_path> --dataset <dataset_path>"
  echo "Example: $0 --config scripts/cfg/inference/analyzer/analyzer.yaml --dataset scripts/cfg/dataset/mocheg.yaml"
  exit 1
fi

echo "Running analyzer inference on $(hostname)"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"

python verification/inference/inference_analyzer.py --config "$CONFIG" --dataset "$DATASET"

