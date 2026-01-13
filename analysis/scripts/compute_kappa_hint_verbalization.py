#!/usr/bin/env python3
"""
Compute Cohen's kappa between manual annotations and automated check_verbalization results.

Compares human annotations from verbalization_manual_annotations.jsonl with
automated judgments from check_verbalization results.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.metrics import cohen_kappa_score

from reasoning_models_lie.evaluation.faithfulness import parse_json_string


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of records."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


def extract_matched_annotations(
    manual_annotations: List[Dict[str, Any]], automated_results: List[Dict[str, Any]]
) -> Tuple[List[bool], List[bool], List[bool], List[bool]]:
    """
    Extract matched annotations for kappa calculation.

    Returns:
        Tuple of (manual_hint_present, auto_hint_present, manual_relied_on_hint, auto_relied_on_hint)
    """
    # Create a lookup dict for automated results by instance_id
    auto_lookup = {}
    for result in automated_results:
        instance_id = result.get("instance_id")
        if instance_id:
            # Parse the JSON response from the automated check
            json_response = parse_json_string(result["result"]["response"])
            auto_lookup[instance_id] = {
                "hint_present": json_response.get("hint_present"),
                "relied_on_hint": json_response.get("relied_on_hint"),
            }

    # Match manual annotations with automated results
    manual_hint_present = []
    auto_hint_present = []
    manual_relied_on_hint = []
    auto_relied_on_hint = []

    matched_count = 0
    for manual in manual_annotations:
        instance_id = manual.get("instance_id")
        if instance_id in auto_lookup:
            auto = auto_lookup[instance_id]

            # Only include if both values are not None
            if (
                manual.get("hint_present") is not None
                and auto.get("hint_present") is not None
            ):
                manual_hint_present.append(manual["hint_present"])
                auto_hint_present.append(auto["hint_present"])

            if (
                manual.get("relied_on_hint") is not None
                and auto.get("relied_on_hint") is not None
            ):
                manual_relied_on_hint.append(manual["relied_on_hint"])
                auto_relied_on_hint.append(auto["relied_on_hint"])

            matched_count += 1

    print(
        f"Matched {matched_count} examples out of {len(manual_annotations)} manual annotations"
    )
    print(f"  - {len(manual_hint_present)} pairs for hint_present")
    print(f"  - {len(manual_relied_on_hint)} pairs for relied_on_hint")

    return (
        manual_hint_present,
        auto_hint_present,
        manual_relied_on_hint,
        auto_relied_on_hint,
    )


def compute_agreement_stats(manual: List[bool], auto: List[bool], label: str):
    """Compute and display agreement statistics."""
    if len(manual) == 0:
        print(f"\nNo data available for {label}")
        return

    # Compute agreement
    agreements = sum(m == a for m, a in zip(manual, auto))
    total = len(manual)
    percent_agreement = 100 * agreements / total

    # Compute Cohen's kappa
    kappa = cohen_kappa_score(manual, auto, weights="linear")

    # Compute confusion matrix
    true_pos = sum(m and a for m, a in zip(manual, auto))
    false_pos = sum((not m) and a for m, a in zip(manual, auto))
    true_neg = sum((not m) and (not a) for m, a in zip(manual, auto))
    false_neg = sum(m and (not a) for m, a in zip(manual, auto))

    print(f"\n{'='*60}")
    print(f"{label.upper()}")
    print(f"{'='*60}")
    print(f"Total examples: {total}")
    print(f"Raw agreement: {agreements}/{total} ({percent_agreement:.1f}%)")
    print(f"Cohen's kappa: {kappa:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"                 Auto=True  Auto=False")
    print(f"Manual=True      {true_pos:5d}      {false_neg:5d}")
    print(f"Manual=False     {false_pos:5d}      {true_neg:5d}")

    # Interpretation
    print(f"\nKappa interpretation:")
    if kappa < 0:
        interpretation = "Less than chance agreement"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    print(f"  {interpretation}")


def main():
    # File paths
    manual_file = Path("analysis/data/verbalization_manual_annotations.jsonl")
    auto_file = Path(
        "experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_changed_check_verbalization.jsonl"
    )

    # Load data
    print("Loading annotations...")
    manual_annotations = load_jsonl(manual_file)
    automated_results = load_jsonl(auto_file)

    print(f"Loaded {len(manual_annotations)} manual annotations")
    print(f"Loaded {len(automated_results)} automated results")

    # Extract matched pairs
    print("\nMatching annotations...")
    manual_hp, auto_hp, manual_roh, auto_roh = extract_matched_annotations(
        manual_annotations, automated_results
    )

    # Compute and display statistics
    compute_agreement_stats(manual_hp, auto_hp, "hint_present")
    compute_agreement_stats(manual_roh, auto_roh, "relied_on_hint")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
