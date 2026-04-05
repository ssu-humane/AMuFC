#!/usr/bin/env python
"""
Merge text-only and image claims into final FinFact dataset.

Usage:
    python datasets/finfact/merge_dataset.py \
        --text /path/to/finfact_topn_onlytxt_refined.json \
        --image /path/to/finfact_top1_onlyimg_refined.json \
        --output /path/to/finfact_final.json
"""

from pathlib import Path
import json
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Merge text and image claims')
    parser.add_argument('--text', type=str, required=True,
                        help='Path to text-only claims JSON')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image claims JSON')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for merged dataset')
    args = parser.parse_args()

    text_path = Path(args.text)
    image_path = Path(args.image)
    output_path = Path(args.output)

    # 1) Load text-only dataset
    try:
        with text_path.open('r', encoding='utf-8') as f:
            text_only_records = json.load(f)
        print(f"✅ Loaded {len(text_only_records)} text-only records from {text_path.name}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Failed to load {text_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Load image-only dataset
    try:
        with image_path.open('r', encoding='utf-8') as f:
            image_only_records = json.load(f)
        print(f"✅ Loaded {len(image_only_records)} image-only records from {image_path.name}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Failed to load {image_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Merge the datasets
    merged_records = text_only_records + image_only_records
    print(f"📦 Merged {len(merged_records)} total records")

    # 4) Write the merged dataset
    with output_path.open('w', encoding='utf-8') as out:
        json.dump(merged_records, out, ensure_ascii=False, indent=2)

    print(f"✅ Successfully wrote merged dataset to {output_path}")
    print(f"\nSummary:")
    print(f"  - Text-only records: {len(text_only_records)}")
    print(f"  - Image-only records: {len(image_only_records)}")
    print(f"  - Total merged: {len(merged_records)}")


if __name__ == "__main__":
    main()
