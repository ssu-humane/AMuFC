#!/usr/bin/env python
"""
Extract text-only claims from FinFact dataset.

Usage:
    python datasets/finfact/topn_onlytxt.py \
        --input /path/to/finfact.json \
        --original /path/to/finfact_hf.json \
        --output /path/to/finfact_topn_onlytxt.json
"""

from pathlib import Path
import json
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Extract text-only claims from FinFact')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to processed finfact.json')
    parser.add_argument('--original', type=str, required=True,
                        help='Path to original finfact_hf.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for text-only claims')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1) Load the original finfact_hf.json to identify text-only examples (empty image_data)
    try:
        with open(args.original, 'r', encoding='utf-8') as f:
            original_records = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse {args.original}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 2) Create a set of claim texts for text-only examples (where image_data is empty)
    text_only_claims = set()
    for record in original_records:
        image_data = record.get("image_data", [])
        if not image_data or len(image_data) == 0:
            text_only_claims.add(record.get("claim", "").strip())
    
    print(f"Found {len(text_only_claims)} text-only examples in original dataset")
    
    # 3) Load the processed JSON array
    try:
        with input_path.open('r', encoding='utf-8') as f:
            records = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # 4) Filter for text-only examples based on claim match
    #    Also convert labels: true->supported, false->refuted, neutral->NEI
    label_map = {
        "true": "supported",
        "false": "refuted",
        "neutral": "NEI"
    }

    text_only_records = []

    print(f"Starting processing of {len(records)} records...")

    for item in records:
        claim = item.get("claim", "").strip()

        # Check if this claim was originally text-only (no image_data)
        if claim in text_only_claims:
            # Convert label
            original_label = item.get("label", "").lower()
            if original_label in label_map:
                item["label"] = label_map[original_label]

            # Set image field to empty string - these are text-only records
            item["image"] = ""

            text_only_records.append(item)

    # Update records to only contain text-only items
    records = text_only_records

    # 5) Write the filtered list back out as a JSON array
    with output_path.open('w', encoding='utf-8') as out:
        json.dump(records, out, ensure_ascii=False, indent=2)

    print(f"✅ Processed {len(records)} text-only records and wrote to {output_path}")

if __name__ == "__main__":
    main()
