#!/bin/bash
# Zero-Shot Inference Script
# Usage: ./scripts/inference_zeroshot.sh --config <config_path> --dataset <dataset_path>

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
  echo "Example: $0 --config scripts/cfg/inference/zero-shot/zero_shot.yaml --dataset scripts/cfg/dataset/mocheg.yaml"
  exit 1
fi

echo "Running zero-shot inference"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"

python verification/inference/inference_zeroshot.py --config "$CONFIG" --dataset "$DATASET"

