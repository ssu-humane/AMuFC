#!/bin/bash
# Training Script
# Usage: ./scripts/train.sh --config <config_path> --dataset <dataset_path>

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
  echo "Example: $0 --config scripts/cfg/train/verifier_qwen.yaml --dataset scripts/cfg/dataset/mocheg.yaml"
  exit 1
fi

echo "Running training on $(hostname)"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"

# Train the model
accelerate launch verification/train/train.py --config "$CONFIG" --dataset "$DATASET"

# Merge and save the model
python verification/train/merge_and_save.py --config "$CONFIG"

