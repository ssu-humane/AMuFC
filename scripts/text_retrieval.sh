#!/bin/bash
# Text Retrieval Script
# Usage: ./scripts/text_retrieval.sh --config <config_path>

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
  echo "Example: $0 --config scripts/cfg/retrieval/text_retrieval.yaml"
  exit 1
fi

echo "Running text retrieval"
echo "Config: $CONFIG"

python retrieval/text_retrieval_inference.py --config "$CONFIG"

