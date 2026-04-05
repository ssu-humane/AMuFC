#!/usr/bin/env python
"""
Merge Result Script
Merges text-only and multimodal verification results based on analyzer decisions.

Usage:
    python evaluation/merge_result.py --mode <mode> --text <text_path> --multimodal <multimodal_path> --output <output_path> [--analyzer <analyzer_path>]

Modes:
    - multimodal: Replace text results with multimodal results
    - analyzer: Use analyzer decisions to select between text and multimodal
    - retrieval: Merge text and image retrieval results
"""

import json
import argparse
from typing import Dict, List, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], path: str) -> None:
    """Save JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def create_claim_map(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create a mapping from claim_id to data item for faster lookup."""
    return {item['claim_id']: item for item in data}


def merge_multimodal(text_data: List[Dict], multimodal_data: List[Dict]) -> List[Dict]:
    """
    Merge multimodal results into text results.
    Replaces text predictions with multimodal predictions.
    """
    multimodal_map = create_claim_map(multimodal_data)
    
    for item in text_data:
        claim_id = item['claim_id']
        if claim_id in multimodal_map:
            mm_item = multimodal_map[claim_id]
            item['image'] = mm_item.get('image', '')
            item['gpt_output'] = mm_item.get('gpt_output', item.get('gpt_output', ''))
            item['label'] = mm_item['label']
    
    return text_data


def merge_with_analyzer(text_data: List[Dict], multimodal_data: List[Dict], 
                        analyzer_data: List[Dict]) -> tuple[List[Dict], Dict]:
    """
    Merge results based on analyzer decisions.
    If analyzer says 'Yes' (image needed), use multimodal; otherwise use text.
    """
    multimodal_map = create_claim_map(multimodal_data)
    analyzer_map = create_claim_map(analyzer_data)
    
    stats = {'yes': 0, 'no': 0, 'total': 0}
    
    for item in text_data:
        claim_id = item['claim_id']
        stats['total'] += 1
        
        if claim_id in analyzer_map:
            analyzer_label = analyzer_map[claim_id].get('label', 'No').strip().lower()
            
            if analyzer_label == 'yes' and claim_id in multimodal_map:
                # Use multimodal result
                mm_item = multimodal_map[claim_id]
                item['image'] = mm_item.get('image', '')
                item['gpt_output'] = mm_item.get('gpt_output', item.get('gpt_output', ''))
                item['label'] = mm_item['label']
                stats['yes'] += 1
            else:
                # Keep text result
                stats['no'] += 1
    
    return text_data, stats


def merge_retrieval(text_data: List[Dict], image_data: List[Dict]) -> List[Dict]:
    """Merge image retrieval results into text retrieval results."""
    image_map = create_claim_map(image_data)
    
    for item in text_data:
        claim_id = item['claim_id']
        if claim_id in image_map:
            item['image'] = image_map[claim_id].get('image', '')
    
    return text_data


def main():
    parser = argparse.ArgumentParser(description='Merge verification results')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['multimodal', 'analyzer', 'retrieval'],
                        help='Merge mode')
    parser.add_argument('--text', type=str, required=True,
                        help='Path to text-only results JSON')
    parser.add_argument('--multimodal', type=str, required=True,
                        help='Path to multimodal results JSON')
    parser.add_argument('--analyzer', type=str, default=None,
                        help='Path to analyzer results JSON (required for analyzer mode)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for merged results')
    args = parser.parse_args()

    # Load data
    text_data = load_json(args.text)
    multimodal_data = load_json(args.multimodal)

    if args.mode == 'multimodal':
        result = merge_multimodal(text_data, multimodal_data)
        print(f"Merged {len(result)} items (multimodal mode)")

    elif args.mode == 'analyzer':
        if not args.analyzer:
            raise ValueError("Analyzer mode requires --analyzer path")
        analyzer_data = load_json(args.analyzer)
        result, stats = merge_with_analyzer(text_data, multimodal_data, analyzer_data)
        print(f"Merged {stats['total']} items")
        print(f"  - Used multimodal (Yes): {stats['yes']} ({100*stats['yes']/stats['total']:.1f}%)")
        print(f"  - Used text (No): {stats['no']} ({100*stats['no']/stats['total']:.1f}%)")

    elif args.mode == 'retrieval':
        result = merge_retrieval(text_data, multimodal_data)
        print(f"Merged {len(result)} items (retrieval mode)")

    # Save result
    save_json(result, args.output)
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()

