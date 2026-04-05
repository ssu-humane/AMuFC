#!/usr/bin/env python
"""
Refine ambiguous claims using Gemini API.

Usage:
    python datasets/finfact/refine_claims_gemini.py \
        --input /path/to/claims.json \
        --output /path/to/claims_refined.json
"""

import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

from openai import OpenAI

# Model configuration
MODEL_NAME = "gemini-2.5-pro"

# Initialize Gemini Client via OpenAI-compatible API
if "GEMINI_API_KEY" not in os.environ:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit(1)

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

REFINE_PROMPT = """You are a fact-checking assistant.
Check if the following claim has complete intention and is clear.
If the claim is vague or incomplete (e.g., just a keyword like "Google PhoneBook"), refine it into a clear, complete sentence based on the Evidence, Justification, and Label.
If the claim is already clear and complete, return "not needed".

Claim: {claim}
Evidence: {evidence}
Justification: {justification}
Label: {label}

If you refine the claim, it should be clear and complete, that can be solved by the evidence. Return ONLY the refined claim or "not needed"."""

def refine_claim(sample):
    """
    Refines the claim using Gemini.
    """
    claim = sample.get("claim", "")
    evidence = sample.get("evidence", "")
    justification = sample.get("justification", "")
    label = sample.get("label", "")
    
    # Construct prompt content manually since logic differs from inference_model wrappers
    prompt_text = REFINE_PROMPT.format(
        claim=claim,
        evidence=evidence,
        justification=justification,
        label=label
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0,
        )
        output_text = response.choices[0].message.content.strip()
        
        # Clean up output
        if output_text.lower().startswith("not needed"):
            return sample["claim"], False
        elif output_text.lower().startswith("refined claim:"):
             output_text = output_text[len("refined claim:"):].strip()
        
        # Remove surrounding quotes if present
        if output_text.startswith('"') and output_text.endswith('"'):
            output_text = output_text[1:-1]
            
        return output_text, True

    except Exception as e:
        print(f"Error processing claim {sample.get('claim_id')}: {e}")
        return sample["claim"], False

def process_sample(sample):
    new_claim, changed = refine_claim(sample)
    
    # Create a copy to avoid mutating original list in place implicitly
    new_sample = sample.copy()
    
    if changed:
        new_sample["claim"] = new_claim
        
    return new_sample, changed, sample["claim_id"]

def main():
    parser = argparse.ArgumentParser(description='Refine ambiguous claims using Gemini')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input claims JSON')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for refined claims')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples.")

    new_data = [None] * len(data)
    changed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_sample, sample): idx for idx, sample in enumerate(data)}

        for fut in tqdm(as_completed(futures), total=len(data), desc="Refining Claims"):
            idx = futures[fut]
            try:
                processed_sample, changed, claim_id = fut.result()
                new_data[idx] = processed_sample
                if changed:
                    changed_count += 1
            except Exception as e:
                print(f"Critical error in worker: {e}")
                new_data[idx] = data[idx]  # Fallback to original

    print(f"Refinement complete. Modified {changed_count} claims.")

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
