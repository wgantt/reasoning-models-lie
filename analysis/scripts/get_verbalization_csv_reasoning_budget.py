"""Generate verbalization CSV data for specific GPQA reasoning budgets.

This script mirrors analysis/scripts/get_verbalization_csv.py but only reads
GPQA check_verbalization result files whose filenames include one of these
reasoning budgets: 1024, 2000, 4000, 8000, 16000.

Example usage:
    python analysis/scripts/get_verbalization_csv_reasoning_budget.py
"""

import json
from glob import glob
from pathlib import Path

import pandas as pd

CLAUDE = "claude_4.5_haiku"
KIMI = "kimi_k2_5"
QWEN = "qwen3_next"

REASONING_BUDGETS = [1024, 2000, 4000, 8000, 16000]

FIELDS_TO_SAVE = [
    ("main_results", "total_examples"),
    ("main_results", "changed_to_hinted_count"),
    ("main_results", "changed_to_hinted_percentage"),
    ("main_results", "faithfulness_score_normalized"),
    ("main_results", "honesty_score_use_normalized"),
    ("main_results", "honesty_score_intent_normalized"),
    ("main_results", "honesty_score_use_or_intent_normalized"),
    ("main_results", "honesty_score_aligned_intent_and_use_normalized"),
    ("bootstrap_results", "faithfulness_score_normalized_ci_lower"),
    ("bootstrap_results", "faithfulness_score_normalized_ci_upper"),
    ("bootstrap_results", "honesty_score_use_normalized_ci_lower"),
    ("bootstrap_results", "honesty_score_use_normalized_ci_upper"),
    ("bootstrap_results", "honesty_score_intent_normalized_ci_lower"),
    ("bootstrap_results", "honesty_score_intent_normalized_ci_upper"),
    ("bootstrap_results", "honesty_score_use_or_intent_normalized_ci_lower"),
    ("bootstrap_results", "honesty_score_use_or_intent_normalized_ci_upper"),
    (
        "bootstrap_results",
        "honesty_score_aligned_intent_and_use_normalized_ci_lower",
    ),
    (
        "bootstrap_results",
        "honesty_score_aligned_intent_and_use_normalized_ci_upper",
    ),
]

GPQA_RESULTS_PATH = "experiments/gpqa/results/check_verbalization"
OUTPUT_PATH = "analysis/data/verbalization_reasoning_budget.csv"


def get_reasoning_budget_from_file(file_path: str) -> int | None:
    """Extract reasoning budget from filename if one of the supported values."""
    filename = Path(file_path).name
    for budget in REASONING_BUDGETS:
        if f"_{budget}_" in filename:
            return budget
    return None


def get_gpqa_verbalization_reasoning_budget_csv() -> pd.DataFrame:
    """Generate a verbalization DataFrame for selected GPQA reasoning budgets."""
    all_files = glob(f"{GPQA_RESULTS_PATH}/*_results.json")
    out = []

    for file in all_files:
        reasoning_budget_tokens = get_reasoning_budget_from_file(file)
        if reasoning_budget_tokens is None:
            continue

        with open(file) as f:
            in_data = json.load(f)

        out_data = {"reasoning_budget_tokens": reasoning_budget_tokens}

        model_name = None
        if CLAUDE in file:
            model_name = "Claude 4.5 Haiku"
        elif KIMI in file:
            model_name = "Kimi K2.5"
        elif QWEN in file:
            model_name = "Qwen 3 Next"
        else:
            print(f"Skipping file with unknown model: {file}")
            continue
        out_data["model_name"] = model_name

        hint_type = None
        if "sycophancy_v1" in file:
            hint_type = "Sycophancy V1"
        elif "metadata" in file:
            hint_type = "Metadata"
        elif "grader_hacking" in file:
            hint_type = "Grader Hacking"
        elif "unethical_information" in file:
            hint_type = "Unethical Information"
        else:
            raise ValueError(f"Unknown hint type in file name: {file}")
        out_data["hint_type"] = hint_type

        setting = None
        if "incorrect" in file:
            setting = "Incorrect Hint"
        elif "correct" in file:
            setting = "Correct Hint"
        else:
            raise ValueError(f"Unknown setting in file name: {file}")
        out_data["setting"] = setting

        for k1, k2 in FIELDS_TO_SAVE:
            out_data[k2] = in_data[k1].get(k2)

        out.append(out_data)

    all_data = pd.DataFrame(out)
    all_data["dataset"] = "GPQA"
    return all_data.sort_values(
        ["reasoning_budget_tokens", "model_name", "hint_type", "setting"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    gpqa = get_gpqa_verbalization_reasoning_budget_csv()
    gpqa.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved verbalization CSV to {OUTPUT_PATH}")
