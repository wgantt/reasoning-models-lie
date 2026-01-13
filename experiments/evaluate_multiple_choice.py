"""Evaluate multiple choice question answering performance.

This script evaluates model performance on multiple choice questions using
bootstrap confidence intervals. It takes model responses in JSONL format
and computes accuracy metrics with statistical confidence bounds.

Example usage:
    python evaluate_multiple_choice.py responses.jsonl metrics.json --n_samples 10000
"""

import argparse
import json

from reasoning_models_lie.evaluation.multiple_choice import evaluate_multiple_choice

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GPQA results.")
    parser.add_argument(
        "input_file", type=str, help="Path to the input file containing GPQA results."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file to save evaluation metrics.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of bootstrap samples."
    )
    args = parser.parse_args()
    metrics = evaluate_multiple_choice(args.input_file, n_samples=args.n_samples)
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)
