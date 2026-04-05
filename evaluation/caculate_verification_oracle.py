#!/usr/bin/env python
"""
Calculate Oracle Verification Metrics
Evaluates text, multimodal, and oracle (upper bound) performance.

Oracle selects text result if correct, otherwise uses multimodal result.
This provides an upper bound for analyzer-based selection.

Usage:
    python evaluation/caculate_verification_oracle.py \
        --ground_truth <gt_path> \
        --text <text_path> \
        --multimodal <multimodal_path> \
        [--output <output_path>]
"""

import json
import argparse
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score


LABELS = ['Refuted', 'Supported', 'NEI']


def convert_label(label: str) -> int:
    """Convert string label to numeric."""
    label = label.strip().lower()
    if label in ['supported', 'true']:
        return 1
    elif label in ['refuted', 'false']:
        return 0
    else:
        return 2


def calc_metric(y_true: list, y_pred: list, verbose: bool = True) -> dict:
    """Calculate classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    crep = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)

    if verbose:
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (micro): {f1_micro:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")

    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_per_class': dict(zip(LABELS, f1_per_class.tolist())),
        'confusion_matrix': cm.tolist(),
        'classification_report': crep
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate oracle verification metrics')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth JSON')
    parser.add_argument('--text', type=str, required=True,
                        help='Path to text-only prediction JSON')
    parser.add_argument('--multimodal', type=str, required=True,
                        help='Path to multimodal prediction JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for metrics JSON')
    args = parser.parse_args()

    # Load data
    with open(args.ground_truth) as f:
        gt_data = json.load(f)
    with open(args.text) as f:
        text_data = json.load(f)
    with open(args.multimodal) as f:
        multimodal_data = json.load(f)

    # Convert labels
    y_true = [convert_label(d['label']) for d in gt_data]
    y_text = [convert_label(d['label']) for d in text_data]
    y_multimodal = [convert_label(d['label']) for d in multimodal_data]

    # Oracle: select text if correct, otherwise multimodal
    y_oracle = [
        y_text[i] if y_text[i] == y_true[i] else y_multimodal[i]
        for i in range(len(y_true))
    ]

    # Calculate metrics
    print("\n=== Text Performance ===")
    text_metrics = calc_metric(y_true, y_text)

    print("\n=== Multimodal Performance ===")
    multimodal_metrics = calc_metric(y_true, y_multimodal)

    print("\n=== Oracle Performance (Upper Bound) ===")
    oracle_metrics = calc_metric(y_true, y_oracle)

    # Selection statistics
    n_different = sum(1 for i in range(len(y_true)) if y_text[i] != y_multimodal[i])
    n_total = len(y_true)

    print(f"\n=== Selection Statistics ===")
    print(f"  Total samples: {n_total}")
    print(f"  Text != Multimodal: {n_different} ({100*n_different/n_total:.1f}%)")
    print(f"  Text == Multimodal: {n_total - n_different} ({100*(n_total-n_different)/n_total:.1f}%)")

    # Save results
    if args.output:
        results = {
            'text': text_metrics,
            'multimodal': multimodal_metrics,
            'oracle': oracle_metrics,
            'statistics': {
                'total': n_total,
                'different_predictions': n_different,
                'same_predictions': n_total - n_different
            }
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to: {args.output}")


if __name__ == '__main__':
    main()
    print("=====================================")