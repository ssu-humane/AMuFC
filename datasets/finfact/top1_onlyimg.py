#!/usr/bin/env python
"""
Extract image claims from FinFact dataset (top-1 image per claim).

Usage:
    python datasets/finfact/top1_onlyimg.py \
        --input /path/to/finfact.json \
        --output /path/to/finfact_top1_onlyimg.json
"""

from pathlib import Path
import json
import sys
import argparse
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Extract image claims from FinFact')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to processed finfact.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for image claims')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1) Load the entire JSON array
    try:
        with input_path.open('r', encoding='utf-8') as f:
            records = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) For each record, replace "image" list with its first element or empty string
    #    Also convert labels: true->supported, false->refuted, neutral->NEI
    #    And filter out images smaller than 100x100
    label_map = {
        "true": "supported",
        "false": "refuted",
        "neutral": "NEI"
    }

    cleaned_records = []
    
    print(f"Starting processing of {len(records)} records...")
    
    for item in records:
        imgs = item.get("image", [])
        if isinstance(imgs, list) and imgs:
            img_path_str = imgs[0]
            item["image"] = img_path_str
            
            # Convert label
            original_label = item.get("label", "").lower()
            if original_label in label_map:
                item["label"] = label_map[original_label]

            # Filter by image size
            try:
                with Image.open(img_path_str) as img:
                    width, height = img.size
                    if width >= 100 and height >= 100:
                        cleaned_records.append(item)
            except Exception as e:
                # If image cannot be opened or found, we might want to exclude it or just log it.
                # Here we assume if it's invalid, it doesn't meet the "valid image >= 100" criteria.
                print(f"Warning: Could not process image {img_path_str}: {e}")
                
    
    # Update records to only contain items with images
    records = cleaned_records

    # 3) Write the cleaned list back out as a JSON array
    with output_path.open('w', encoding='utf-8') as out:
        json.dump(records, out, ensure_ascii=False, indent=2)

    print(f"✅ Processed {len(records)} records and wrote to {output_path}")

if __name__ == "__main__":
    main()
