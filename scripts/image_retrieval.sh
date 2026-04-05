#!/bin/bash
# Image Retrieval Script
# Usage: ./scripts/image_retrieval.sh --config <config_path>

# Parse arguments
CONFIG=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$CONFIG" ]; then
  echo "Usage: $0 --config <config_path>"
  echo "Example: $0 --config scripts/cfg/retrieval/image_retrieval.yaml"
  exit 1
fi

echo "Running image retrieval"
echo "Config: $CONFIG"

python retrieval/image_retrieval_inference.py --config "$CONFIG"

