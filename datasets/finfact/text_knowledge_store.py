#!/usr/bin/env python
"""
Create text knowledge store for evidence retrieval.

Usage:
    python datasets/finfact/text_knowledge_store.py \
        --input /path/to/finfact.json \
        --output /path/to/finfact_corpus_sentence_level.csv
"""

import json
import pandas as pd
import os
import re
import argparse
import nltk


def create_knowledge_store(input_file: str, output_file: str):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items.")

    # Prepare data for sentence-level CSV
    rows = []

    for idx, item in enumerate(data):
        # Extract evidence - it's a string with embedded "Evidence N: " prefixes
        evidence_text = item.get('evidence', '')

        # Remove all "Evidence N: " prefixes from the text
        cleaned_evidence = re.sub(r'Evidence \d+:\s*', '', evidence_text)

        # Strip leading/trailing whitespace
        cleaned_evidence = cleaned_evidence.strip()

        # Handle empty evidence
        if not cleaned_evidence:
            cleaned_evidence = "[No evidence available]"

        # Split evidence into sentences using nltk
        evidence_sent_list = nltk.tokenize.sent_tokenize(cleaned_evidence)

        # Create rows for each sentence
        for sent_idx, sentence in enumerate(evidence_sent_list):
            corpus_id = f"{idx}-{idx}-{sent_idx}"

            rows.append({
                'claim_id': idx,
                'relevant_document_id': idx,
                'paragraph_id': sent_idx,
                'corpus_id': corpus_id,
                'paragraph': sentence
            })

    # Create DataFrame with required columns
    df = pd.DataFrame(rows)

    print(f"Creating sentence-level CSV with {len(df)} sentences from {len(data)} evidence entries...")
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create text knowledge store for retrieval')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input finfact.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for sentence-level CSV')
    args = parser.parse_args()

    create_knowledge_store(args.input, args.output)


if __name__ == "__main__":
    main()
