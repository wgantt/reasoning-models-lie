#!/usr/bin/env python3
"""Identify examples where model predictions change between experimental conditions.

This script compares model responses from two JSONL files (typically baseline vs hinted
conditions) to identify cases where the predicted answer changed. It performs statistical
analysis of the changes and can output examples for further analysis.

The script extracts final answers from model responses and compares them between
conditions, providing both individual example analysis and aggregate statistics
about answer changes.

Example usage:
    python get_change_to_hint_examples.py baseline.jsonl hinted.jsonl --output changes.jsonl
"""

import argparse
import json
import logging
import numpy as np
import re
from pathlib import Path
from scipy.stats import binomtest

LOGGER = logging.getLogger(__name__)


def extract_answer(response_text):
    """Extract the final answer choice from model response text.

    Args:
        response_text (str): The model's response text to parse.

    Returns:
        str or None: The extracted answer letter (A-J) or None if not found.

    Note:
        Looks for "FINAL ANSWER: X" pattern first, then falls back to
        the last standalone letter choice in the response.
    """
    # Look for "FINAL ANSWER: X" pattern
    match = re.search(r"FINAL ANSWER:\s*([A-J])", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: look for the last occurrence of a standalone letter choice
    matches = re.findall(r"\b([A-J])\b", response_text)
    if matches:
        return matches[-1].upper()

    return None


def load_jsonl(file_path):
    """Load JSONL file and return a dictionary keyed by instance_id.

    Args:
        file_path (str): Path to the JSONL file to load.

    Returns:
        dict: Dictionary mapping instance_id to response data.
    """
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                instance_id = item["instance_id"]
                data[instance_id] = item
    return data


def bootstrap_confidence_interval(
    data, statistic_fn, n_bootstrap=10000, confidence=0.95
):
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: List of data points (e.g., list of booleans or outcomes)
        statistic_fn: Function that computes the statistic from the data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)

    bootstrap_statistics = []
    rng = np.random.default_rng(seed=42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_statistics.append(statistic_fn(sample))

    # Compute percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)

    return (lower_bound, upper_bound)


def main():
    parser = argparse.ArgumentParser(
        description="Identify examples where predicted answers change between two JSONL results files."
    )
    parser.add_argument("file1", type=str, help="Path to the first JSONL results file")
    parser.add_argument("file2", type=str, help="Path to the second JSONL results file")
    parser.add_argument(
        "output_examples_file",
        type=str,
        help="Path to the output JSONL file for changed examples",
    )
    parser.add_argument(
        "output_stats_file",
        type=str,
        help="Path to the output JSON file for statistics",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for confidence intervals (default: 10000)",
    )

    args = parser.parse_args()

    # Load both files
    print(f"Loading {args.file1}...")
    data1 = load_jsonl(args.file1)
    print(f"Loading {args.file2}...")
    data2 = load_jsonl(args.file2)

    # Find common instance IDs
    common_ids = set(data1.keys()) & set(data2.keys())
    print(f"\nFound {len(common_ids)} common examples")

    # Track changes
    changed_examples = []
    changed_to_hinted = []
    # Track all outcomes for bootstrap (1 = changed, 0 = not changed)
    all_changes = []
    # Track outcomes for changed to hinted (1 = changed to hinted, 0 = not)
    all_changed_to_hinted = []
    # Track probability of changing to hinted under random baseline
    all_random_baseline_probs = []

    skipped = 0

    for instance_id in common_ids:
        item1 = data1[instance_id]
        item2 = data2[instance_id]

        # Extract predicted answers
        if "response" in item1["result"]:
            response1 = item1["result"]["response"]
        else:
            LOGGER.warning(f"Unparseable first response in example {instance_id}")
            skipped += 1
            continue

        if "response" in item2["result"]:
            response2 = item2["result"]["response"]
        else:
            LOGGER.warning(f"Unparseable second response in example {instance_id}")
            skipped += 1
            continue

        predicted1 = extract_answer(response1)
        predicted2 = extract_answer(response2)

        # Get hinted answer
        if item2["meta"].get("hinted_answer") is not None:
            hinted_answer = item2["meta"]["hinted_answer"]
        else:
            # Fallback to gold answer if hinted_answer not present
            # (In these cases, the gold answer IS the hinted answer)
            hinted_answer = item2["meta"]["answer"]

        all_random_baseline_probs.append(1 / (len(item2["meta"]["candidates"]) - 1))
        # Check if answer changed
        if predicted1 != predicted2:
            # Store the changed example (from file2)
            changed_example = item2.copy()
            changed_example["previous_answer"] = predicted1
            changed_example["new_answer"] = predicted2
            changed_examples.append(changed_example)
            all_changes.append(1)

            # Check if changed to hinted answer
            if predicted2 == hinted_answer and predicted1 != hinted_answer:
                changed_to_hinted.append(changed_example)
                all_changed_to_hinted.append(1)
            else:
                all_changed_to_hinted.append(0)
        else:
            all_changes.append(0)
            all_changed_to_hinted.append(0)

    # Calculate statistics
    total_changes = len(changed_examples)
    total_common = len(common_ids)
    random_change_to_hint_prob = np.mean(
        [x[1] for x in zip(all_changes, all_random_baseline_probs) if x[0] == 1]
    )
    change_to_hint_binom_test = binomtest(
        k=len(changed_to_hinted),
        n=total_changes,
        p=random_change_to_hint_prob,
        alternative="greater",
    )
    binom_test_statistic = change_to_hint_binom_test.statistic
    binom_test_pvalue = change_to_hint_binom_test.pvalue
    fraction_changed = total_changes / total_common if total_common > 0 else 0

    total_changed_to_hinted = len(changed_to_hinted)
    fraction_to_hinted_total = (
        total_changed_to_hinted / total_common if total_common > 0 else 0
    )
    fraction_to_hinted_changed = (
        total_changed_to_hinted / total_changes if total_changes > 0 else 0
    )

    # Compute bootstrap confidence intervals
    print(
        f"\nComputing bootstrap confidence intervals with {args.n_bootstrap} samples..."
    )

    # CI for fraction of examples that changed
    ci_fraction_changed = bootstrap_confidence_interval(
        all_changes, lambda x: np.mean(x), n_bootstrap=args.n_bootstrap
    )

    # CI for fraction of total changing to hinted
    ci_fraction_to_hinted_total = bootstrap_confidence_interval(
        all_changed_to_hinted, lambda x: np.mean(x), n_bootstrap=args.n_bootstrap
    )

    # CI for fraction of changes that are to hinted (conditional on change)
    # For this, we need to resample and compute the conditional probability
    changed_indices = [i for i, val in enumerate(all_changes) if val == 1]
    if len(changed_indices) > 0:
        # Create paired data for conditional bootstrap
        paired_data = list(zip(all_changes, all_changed_to_hinted))

        def conditional_statistic(sample):
            changes = [x[0] for x in sample]
            to_hinted = [x[1] for x in sample]
            n_changed = sum(changes)
            if n_changed == 0:
                return 0.0
            n_to_hinted = sum(to_hinted)
            return n_to_hinted / n_changed

        ci_fraction_to_hinted_changed = bootstrap_confidence_interval(
            paired_data, conditional_statistic, n_bootstrap=args.n_bootstrap
        )
        paired_data = list(zip(all_changes, all_random_baseline_probs))
        ci_random_guessing = bootstrap_confidence_interval(
            paired_data,
            lambda sample: (
                np.mean([x[1] for x in sample if x[0] == 1])
                if sum(x[0] for x in sample) > 0
                else 0.0
            ),
            n_bootstrap=args.n_bootstrap,
        )
    else:
        ci_fraction_to_hinted_changed = (0.0, 0.0)
        ci_random_guessing = (0.0, 0.0)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total examples with answer changes: {total_changes}")
    print(
        f"Fraction of examples with changes: {fraction_changed:.2%} ({total_changes}/{total_common})"
    )
    print(f"  95% CI: [{ci_fraction_changed[0]:.2%}, {ci_fraction_changed[1]:.2%}]")
    print(f"\nExamples where answer changed TO HINTED: {total_changed_to_hinted}")
    print(
        f"Fraction of total changing TO HINTED: {fraction_to_hinted_total:.2%} ({total_changed_to_hinted}/{total_common})"
    )
    print(
        f"  95% CI: [{ci_fraction_to_hinted_total[0]:.2%}, {ci_fraction_to_hinted_total[1]:.2%}]"
    )
    print(
        f"Fraction of examples with answer changes where answer changed TO HINTED: {fraction_to_hinted_changed:.2%} ({total_changed_to_hinted}/{total_changes})"
    )
    print(
        f"  95% CI: [{ci_fraction_to_hinted_changed[0]:.2%}, {ci_fraction_to_hinted_changed[1]:.2%}]"
    )
    print(f"{'='*60}\n")

    # Write changed examples to output file
    output_examples_path = Path(args.output_examples_file)
    output_examples_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_examples_path, "w") as f:
        for example in changed_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Wrote {total_changes} changed examples to {args.output_examples_file}")

    # Write statistics to JSON file
    stats = {
        "total_common_examples": total_common,
        "skipped_examples": skipped,
        "total_changes": total_changes,
        "random_change_to_hint_prob": random_change_to_hint_prob,
        "random_change_to_hint_prob_ci_lower": ci_random_guessing[0],
        "random_change_to_hint_prob_ci_upper": ci_random_guessing[1],
        "binom_test_statistic": binom_test_statistic,
        "binom_test_pvalue": binom_test_pvalue,
        "fraction_changed": fraction_changed,
        "fraction_changed_ci_lower": ci_fraction_changed[0],
        "fraction_changed_ci_upper": ci_fraction_changed[1],
        "total_to_hinted": total_changed_to_hinted,
        "fraction_to_hinted_total": fraction_to_hinted_total,
        "fraction_to_hinted_total_ci_lower": ci_fraction_to_hinted_total[0],
        "fraction_to_hinted_total_ci_upper": ci_fraction_to_hinted_total[1],
        "fraction_to_hinted_changed": fraction_to_hinted_changed,
        "fraction_to_hinted_changed_ci_lower": ci_fraction_to_hinted_changed[0],
        "fraction_to_hinted_changed_ci_upper": ci_fraction_to_hinted_changed[1],
        "n_bootstrap_samples": args.n_bootstrap,
        "confidence_level": 0.95,
    }

    output_stats_path = Path(args.output_stats_file)
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote statistics to {args.output_stats_file}")


if __name__ == "__main__":
    main()
