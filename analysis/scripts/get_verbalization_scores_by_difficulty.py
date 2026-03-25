import argparse
import json
import pandas as pd
import re

from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MMLU_PRO_RESULTS_DIR = "experiments/mmlu_pro/results/all_responses"
MMLU_PRO_RESULTS_FILES = [
    "mmlu_pro_test200_verbalize_baseline_claude_4.5_haiku_results.json",
    "mmlu_pro_test200_verbalize_baseline_kimi_k2_5_results.json",
    "mmlu_pro_test200_verbalize_baseline_qwen3_next_results.json",
]
GPQA_RESULTS_DIR = "experiments/gpqa/results/all_responses"
GPQA_RESULTS_FILES = [
    "gpqa_diamond_verbalize_baseline_claude_4.5_haiku_results.json",
    "gpqa_diamond_verbalize_baseline_kimi_k2_5_results.json",
    "gpqa_diamond_verbalize_baseline_qwen3_next_results.json",
]

GPQA_CHECK_VERBALIZATION_DIR = "experiments/gpqa/results/check_verbalization"
MMLU_PRO_CHECK_VERBALIZATION_DIR = "experiments/mmlu_pro/results/check_verbalization"
OUTPUT_PATH = "analysis/data/verbalization_scores_by_difficulty.csv"
EXCLUDED_GPQA_FILE_SUBSTRINGS = ("1024", "2000", "4000", "8000", "16000")

USED_HINT_OPTIONS = frozenset({"a", "b", "c"})
METRIC_FIELDS = [
    "hint_present",
    "hint_use",
    "hint_use_option",
    "hint_intent",
    "hint_use_or_intent",
    "aligned_intent_and_use",
    "changed_to_hinted",
]

MODEL_NAME_MAP = {
    "claude_4.5_haiku": "Claude 4.5 Haiku",
    "kimi_k2_5": "Kimi-K2.5",
    "qwen3_next": "Qwen3-Next-80B-A3B-Thinking",
}

HINT_TYPE_MAP = {
    "grader_hacking": "grader_hacking",
    "metadata": "metadata",
    "unethical_information": "unethical_information",
    "sycophancy_v1": "sycophancy",
}

GPQA_CHECK_VERBALIZATION_REGEX = re.compile(
    r"^gpqa_diamond_verbalize_"
    r"(grader_hacking|metadata|unethical_information|sycophancy_v1)_"
    r"(correct|incorrect)_"
    r"(claude_4\.5_haiku|kimi_k2_5|qwen3_next)_"
    r"changed_check_verbalization_results\.json$"
)
MMLU_PRO_CHECK_VERBALIZATION_REGEX = re.compile(
    r"^mmlu_pro_test200_verbalize_"
    r"(grader_hacking|metadata|unethical_information|sycophancy_v1)_"
    r"(correct|incorrect)_"
    r"(claude_4\.5_haiku|kimi_k2_5|qwen3_next)_"
    r"changed_check_verbalization_results\.json$"
)


def _group_dataset_examples_by_difficulty(
    results_dir: str,
    results_files: List[str],
) -> Dict[str, List[str]]:
    """Group examples into difficulty buckets from per-example correctness.

    Difficulty categories are based on how many result files contain an
    incorrect prediction for the same example:
    - EASY: 0 incorrect files
    - MEDIUM: 1 incorrect file
    - HARD: 2 incorrect files
    - HARDEST: all files incorrect
    """
    per_example_correctness = defaultdict(list)

    for results_file in results_files:
        path = Path(results_dir) / results_file
        with open(path, "r") as f:
            data = json.load(f)

        if "per_example_results" not in data:
            raise ValueError(f"Missing 'per_example_results' in results file: {path}")

        for result in data["per_example_results"]:
            if "instance_id" not in result or "is_correct" not in result:
                raise ValueError(
                    f"Missing required per-example keys in results file: {path}"
                )
            per_example_correctness[result["instance_id"]].append(result["is_correct"])

    expected_files_per_example = len(results_files)
    grouped_examples = {
        "EASY": [],
        "MEDIUM": [],
        "HARD": [],
        "HARDEST": [],
    }

    for instance_id, correctness_values in per_example_correctness.items():
        if len(correctness_values) != expected_files_per_example:
            raise ValueError(
                "Example appears in an unexpected number of result files: "
                f"{instance_id} has {len(correctness_values)} values, "
                f"expected {expected_files_per_example}"
            )

        incorrect_count = sum(not is_correct for is_correct in correctness_values)
        if incorrect_count == 0:
            grouped_examples["EASY"].append(instance_id)
        elif incorrect_count == 1:
            grouped_examples["MEDIUM"].append(instance_id)
        elif incorrect_count == 2:
            grouped_examples["HARD"].append(instance_id)
        elif incorrect_count == expected_files_per_example:
            grouped_examples["HARDEST"].append(instance_id)

    return grouped_examples


def group_examples_by_difficulty() -> Dict[str, Dict[str, List[str]]]:
    """Group MMLU-Pro and GPQA examples by baseline difficulty."""
    return {
        "mmlu_pro": _group_dataset_examples_by_difficulty(
            MMLU_PRO_RESULTS_DIR,
            MMLU_PRO_RESULTS_FILES,
        ),
        "gpqa": _group_dataset_examples_by_difficulty(
            GPQA_RESULTS_DIR,
            GPQA_RESULTS_FILES,
        ),
    }


def _parse_check_verbalization_filename(
    dataset: str,
    filename: str,
) -> Optional[Tuple[str, str, str]]:
    """Parse hint type, setting, and model from a check verbalization filename."""
    if dataset == "gpqa":
        match = GPQA_CHECK_VERBALIZATION_REGEX.match(filename)
    elif dataset == "mmlu_pro":
        match = MMLU_PRO_CHECK_VERBALIZATION_REGEX.match(filename)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _get_check_verbalization_files(dataset: str) -> List[str]:
    """Return standard check verbalization files for one dataset."""
    if dataset == "gpqa":
        check_dir = GPQA_CHECK_VERBALIZATION_DIR
    elif dataset == "mmlu_pro":
        check_dir = MMLU_PRO_CHECK_VERBALIZATION_DIR
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    candidates = glob(f"{check_dir}/*changed_check_verbalization_results.json")
    files = []
    for file_path in candidates:
        filename = Path(file_path).name
        if dataset == "gpqa" and any(
            s in filename for s in EXCLUDED_GPQA_FILE_SUBSTRINGS
        ):
            continue
        if _parse_check_verbalization_filename(dataset, filename) is not None:
            files.append(file_path)
    return sorted(files)


def get_verbalization_scores_by_difficulty() -> pd.DataFrame:
    """Compute true proportions by dataset, hint type, model, correctness, and difficulty."""
    grouped = group_examples_by_difficulty()
    difficulty_lookup = {
        dataset_name: {
            instance_id: difficulty
            for difficulty, instance_ids in grouped[dataset_name].items()
            for instance_id in instance_ids
        }
        for dataset_name in grouped
    }
    bin_sizes = {
        dataset_name: {
            difficulty: len(instance_ids)
            for difficulty, instance_ids in grouped[dataset_name].items()
        }
        for dataset_name in grouped
    }

    aggregated = defaultdict(
        lambda: {
            "n_examples": 0,
            "hint_present_true": 0,
            "hint_use_true": 0,
            "hint_use_option_true": 0,
            "hint_intent_true": 0,
            "hint_use_or_intent_true": 0,
            "aligned_intent_and_use_true": 0,
            "changed_to_hinted_true": 0,
        }
    )

    for dataset_name in ["mmlu_pro", "gpqa"]:
        for file_path in _get_check_verbalization_files(dataset_name):
            parsed = _parse_check_verbalization_filename(
                dataset_name,
                Path(file_path).name,
            )
            if parsed is None:
                continue

            hint_type_key, setting, model_key = parsed
            hint_type = HINT_TYPE_MAP[hint_type_key]
            hint_correctness = setting
            model_name = MODEL_NAME_MAP[model_key]

            with open(file_path, "r") as f:
                data = json.load(f)
            per_example_results = data.get("per_example_results", [])

            for result in per_example_results:
                instance_id = result.get("instance_id")
                difficulty = difficulty_lookup[dataset_name].get(instance_id)
                if difficulty is None:
                    continue

                key = (
                    dataset_name,
                    hint_type,
                    hint_correctness,
                    model_name,
                    difficulty,
                )
                aggregated[key]["n_examples"] += 1
                aggregated[key]["hint_present_true"] += int(
                    result.get("hint_present") is True
                )
                aggregated[key]["hint_use_true"] += int(result.get("hint_use") is True)
                aggregated[key]["hint_use_option_true"] += int(
                    result.get("hint_use_option") in USED_HINT_OPTIONS
                )
                aggregated[key]["hint_intent_true"] += int(
                    result.get("hint_intent") is True
                )
                aggregated[key]["hint_use_or_intent_true"] += int(
                    result.get("hint_use_or_intent") is True
                )
                aggregated[key]["aligned_intent_and_use_true"] += int(
                    result.get("aligned_intent_and_use") is True
                )
                aggregated[key]["changed_to_hinted_true"] += int(
                    result.get("changed_to_hinted") is True
                )

    rows = []
    dataset_label = {
        "mmlu_pro": "MMLU-Pro",
        "gpqa": "GPQA-Diamond",
    }
    hint_types = [
        "grader_hacking",
        "metadata",
        "unethical_information",
        "sycophancy",
    ]
    model_names = [
        "Claude 4.5 Haiku",
        "Kimi-K2.5",
        "Qwen3-Next-80B-A3B-Thinking",
    ]
    hint_correctness_values = ["correct", "incorrect"]
    difficulties = ["EASY", "MEDIUM", "HARD", "HARDEST"]

    for dataset_name in ["mmlu_pro", "gpqa"]:
        for hint_type in hint_types:
            for hint_correctness in hint_correctness_values:
                for model_name in model_names:
                    for difficulty in difficulties:
                        key = (
                            dataset_name,
                            hint_type,
                            hint_correctness,
                            model_name,
                            difficulty,
                        )
                        counts = aggregated[key]
                        n_examples = counts["n_examples"]

                        row = {
                            "dataset": dataset_label[dataset_name],
                            "hint_type": hint_type,
                            "hint_correctness": hint_correctness,
                            "model_name": model_name,
                            "difficulty": difficulty,
                            "n_examples_in_bin": bin_sizes[dataset_name][difficulty],
                            "n_examples_with_verbalization": n_examples,
                        }
                        for metric_field in METRIC_FIELDS:
                            true_count = counts[f"{metric_field}_true"]
                            row[f"{metric_field}_true_count"] = true_count
                            row[f"{metric_field}_true_proportion"] = (
                                true_count / n_examples if n_examples > 0 else None
                            )
                        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute verbalization metric proportions by model and difficulty bin."
        )
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_PATH,
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    output_df = get_verbalization_scores_by_difficulty()
    output_df.to_csv(args.output_path, index=False)
    print(f"Saved verbalization scores by difficulty to {args.output_path}")
