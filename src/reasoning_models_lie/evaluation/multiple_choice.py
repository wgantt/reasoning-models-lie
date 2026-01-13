"""Evaluation utilities for multiple-choice question answering tasks."""

import json
import logging
import numpy as np

from pprint import pprint
from random import choices
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# N.B.: GPQA multiple-choice options are A, B, C, or D
#       AQUA-RAT multiple-choice options are A, B, C, D, or E
#       MMLU Pro multiple-choice options are A, B, C, D, E, F, G, H, I, or J
MULTIPLE_CHOICE_OPTIONS = frozenset({"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"})


def parse_multiple_choice_answer(answer_str: str) -> str:
    """Parse the final answer from the model's response string.

    Extracts the answer letter from a model's response, including
    "Final Answer: X" patterns and handles bold formatting.

    Args:
        answer_str: The raw response string from the model.

    Returns:
        The extracted answer letter (A-J) if found, empty string otherwise.
    """
    answer_str = answer_str.strip()
    # Look for the pattern "Final Answer: X"
    if "FINAL ANSWER:" in answer_str:
        parts = answer_str.split("FINAL ANSWER:")
        final_part = parts[-1].strip()
        # The answer should be the first character after "Final Answer:"
        if final_part:
            # Check for bold formatting
            if final_part.startswith("**") and len(final_part) >= 4:
                answer_letter = final_part[2].upper()
                if answer_letter in MULTIPLE_CHOICE_OPTIONS:
                    return answer_letter
            else:
                answer_letter = final_part[0].upper()
                if answer_letter in MULTIPLE_CHOICE_OPTIONS:
                    return answer_letter
        else:
            return ""
    # If not found, return empty string
    return ""


def evaluate_multiple_choice(
    results_file: str, n_samples: int = 10000
) -> Dict[str, int | float]:
    """Evaluate multiple-choice results from a JSONL file with bootstrap confidence intervals.

    Computes accuracy metrics with bootstrap confidence intervals for model responses
    on multiple-choice questions.

    Args:
        results_file: Path to the JSONL file containing evaluation results.
        n_samples: Number of bootstrap samples for confidence interval computation.

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy (0.0-1.0)
        - accuracy_ci_lower: Lower bound of 95% confidence interval
        - accuracy_ci_upper: Upper bound of 95% confidence interval
        - total_examples: Total number of examples
        - unparseable_examples: Number of unparseable responses
    """
    with open(results_file, "r") as f:
        results = [json.loads(line) for line in f]

    correct = 0
    unparseable = 0
    total = len(results)
    for r in results:
        instance_id = r["instance_id"]
        if "meta" not in r or "answer" not in r["meta"]:
            LOGGER.warning(f"Missing gold answer for instance {instance_id}, skipping.")
            unparseable += 1
            continue

        gold_answer = r["meta"]["answer"]
        if gold_answer not in MULTIPLE_CHOICE_OPTIONS:
            LOGGER.warning(
                f"Invalid gold answer '{gold_answer}' for instance {instance_id}, skipping."
            )
            unparseable += 1
            continue

        if "result" not in r or "response" not in r["result"]:
            LOGGER.warning(
                f"Missing model response for instance {instance_id}, skipping."
            )
            unparseable += 1
            continue

        raw_response = r["result"]["response"]
        predicted_answer = parse_multiple_choice_answer(raw_response)
        if predicted_answer not in MULTIPLE_CHOICE_OPTIONS:
            unparseable += 1
        elif predicted_answer == gold_answer:
            correct += 1

    parsable_correct = correct
    parsable_incorrect = total - unparseable - parsable_correct
    items = (
        ["PC"] * parsable_correct + ["PI"] * parsable_incorrect + ["U"] * unparseable
    )
    parseable_correct_samples = []
    parseable_incorrect_samples = []
    unparseable_samples = []
    for _ in range(n_samples):
        sample = choices(items, k=total)
        sample_correct = sample.count("PC")
        sample_unparseable = sample.count("U")
        sample_parseable = total - sample_unparseable

        # Update totals
        parseable_correct_samples.append(sample_correct)
        parseable_incorrect_samples.append(sample_parseable - sample_correct)
        unparseable_samples.append(sample_unparseable)
    # Compute average metrics over samples
    avg_parseable_correct = np.mean(parseable_correct_samples)
    avg_unparseable = np.mean(unparseable_samples)
    # Compute 95% confidence intervals
    ci_lower = 2.5
    ci_upper = 97.5
    bootstrap_results = {
        "correct": float(avg_parseable_correct),
        "total": total,
        "parseable": float(total - avg_unparseable),
        "unparseable": float(avg_unparseable),
        "accuracy_total": float(avg_parseable_correct / total) if total > 0 else 0,
        "accuracy_parseable": (
            float(avg_parseable_correct / (total - avg_unparseable))
            if (total - avg_unparseable) > 0
            else 0
        ),
        "accuracy_total_ci": (
            float(
                np.percentile(
                    [
                        (pc) / total if total > 0 else 0
                        for pc in parseable_correct_samples
                    ],
                    ci_lower,
                )
            ),
            float(
                np.percentile(
                    [
                        (pc) / total if total > 0 else 0
                        for pc in parseable_correct_samples
                    ],
                    ci_upper,
                )
            ),
        ),
    }

    parseable = total - unparseable
    results = {
        "correct": correct,
        "total": total,
        "parseable": parseable,
        "unparseable": unparseable,
        "accuracy_total": correct / total if total > 0 else 0,
        "accuracy_parseable": correct / parseable if parseable > 0 else 0,
        "bootstrap": bootstrap_results,
    }
    LOGGER.info(f"GPQA Evaluation Results: {pprint(results)}")
    return results
