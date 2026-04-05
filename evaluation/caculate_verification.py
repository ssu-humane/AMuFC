#!/usr/bin/env python
"""
Calculate Verification Metrics
Evaluates verification results against ground truth.

Usage:
    python evaluation/caculate_verification.py --ground_truth <gt_path> --prediction <pred_path>
    python evaluation/caculate_verification.py --ground_truth <gt_path> --prediction_dir <pred_dir>
"""

import json
import glob
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
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (micro): {f1_micro:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        for lbl, score in zip(LABELS, f1_per_class):
            print(f"F1 ({lbl}): {score:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_per_class': dict(zip(LABELS, f1_per_class.tolist())),
        'confusion_matrix': cm.tolist(),
        'classification_report': crep
    }


def evaluate_file(gt_path: str, pred_path: str, verbose: bool = True) -> dict:
    """Evaluate a single prediction file."""
    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(pred_path) as f:
        pred_data = json.load(f)

    y_true = [convert_label(d['label']) for d in gt_data]
    y_pred = [convert_label(d['label']) for d in pred_data]

    return calc_metric(y_true, y_pred, verbose)


def main():
    parser = argparse.ArgumentParser(description='Calculate verification metrics')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth JSON')
    parser.add_argument('--prediction', type=str, default=None,
                        help='Path to prediction JSON')
    parser.add_argument('--prediction_dir', type=str, default=None,
                        help='Directory containing prediction JSONs (evaluates all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for metrics JSON')
    args = parser.parse_args()

    if not args.prediction and not args.prediction_dir:
        raise ValueError("Must provide --prediction or --prediction_dir")

    results = {}

    if args.prediction:
        # Single file evaluation
        exp_name = args.prediction.split('/')[-1].replace('.json', '')
        print(f"=== {exp_name} ===")
        results[exp_name] = evaluate_file(args.ground_truth, args.prediction)

    elif args.prediction_dir:
        # Batch evaluation
        json_files = glob.glob(f"{args.prediction_dir}/*.json")
        for pred_path in sorted(json_files):
            try:
                exp_name = pred_path.split('/')[-1].replace('.json', '')
                print(f"\n=== {exp_name} ===")
                results[exp_name] = evaluate_file(args.ground_truth, pred_path)
            except Exception as e:
                print(f"Error in {pred_path}: {e}")

    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to: {args.output}")


if __name__ == '__main__':
    main()