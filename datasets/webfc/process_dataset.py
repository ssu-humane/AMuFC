#!/usr/bin/env python
"""
Process WebFC Dataset
Converts relative image paths to absolute paths.

Usage:
    python datasets/webfc/process_dataset.py \
        --input /path/to/webfc.json \
        --output /path/to/webfc_processed.json \
        --image_dir /path/to/images/
"""

import json
import os
import argparse


def process_dataset(input_file: str, output_file: str, image_base_path: str):
    """Process dataset by converting relative image paths to absolute paths."""
    print(f"Loading dataset from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return

    print(f"Processing {len(data)} items...")

    # Ensure image_base_path ends with separator
    if not image_base_path.endswith(os.sep):
        image_base_path = image_base_path + os.sep

    processed_count = 0
    for item in data:
        if "image" in item and item["image"]:
            # Skip if already an absolute path
            if not os.path.isabs(item["image"]):
                item["image"] = os.path.join(image_base_path, item["image"])
                processed_count += 1

    print(f"Updated {processed_count} image paths")
    print(f"Saving processed dataset to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description='Process WebFC dataset image paths')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file (webfc.json)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Base directory containing images')
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.image_dir)


if __name__ == "__main__":
    main()
