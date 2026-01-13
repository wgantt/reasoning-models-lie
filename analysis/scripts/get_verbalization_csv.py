"""Generate verbalization CSV data from faithfulness evaluation results.

This script processes JSON files containing verbalization analysis results and
generates a CSV file with faithfulness and honesty metrics for different models,
datasets, and experimental conditions.

The script extracts metrics related to hint verbalization, faithfulness scores,
and confidence intervals from the evaluation results.

Example usage:
    python get_verbalization_csv.py
"""

import pandas as pd
import json

from glob import glob

CLAUDE = "claude_4.5_haiku"
KIMI = "kimi_k2_thinking"
QWEN = "qwen3_next"

FIELDS_TO_SAVE = [
    ("main_results", "total_examples"),
    ("main_results", "changed_to_hinted_count"),
    ("main_results", "changed_to_hinted_percentage"),
    ("main_results", "faithfulness_score_normalized"),
    ("main_results", "honesty_score_normalized"),
    ("bootstrap_results", "faithfulness_score_normalized_ci_lower"),
    ("bootstrap_results", "faithfulness_score_normalized_ci_upper"),
    ("bootstrap_results", "honesty_score_normalized_ci_lower"),
    ("bootstrap_results", "honesty_score_normalized_ci_upper"),
]
GPQA_RESULTS_PATH = "experiments/gpqa/results/check_verbalization/"
MMLU_PRO_RESULTS_PATH = "experiments/mmlu_pro/results/check_verbalization/"
OUTPUT_PATH = "analysis/data/verbalization.csv"


def get_hint_usage_csvs(ds: str = "gpqa") -> pd.DataFrame:
    """Generate verbalization DataFrame from evaluation results.

    Args:
        ds (str): Dataset name, either "gpqa" or "mmlu_pro".

    Returns:
        pd.DataFrame: DataFrame containing verbalization metrics for different
                      models and experimental conditions.

    Raises:
        ValueError: If dataset name is not supported.
    """
    if ds == "gpqa":
        results_path = GPQA_RESULTS_PATH
        all_files = glob(results_path + "/*.json")
    elif ds == "mmlu_pro":
        results_path = MMLU_PRO_RESULTS_PATH
        all_files = glob(results_path + "/mmlu_pro_test200*.json")
    else:
        raise ValueError(f"Unsupported dataset: {ds}")
    out = []
    for file in all_files:
        with open(file) as f:
            in_data = json.load(f)

        out_data = {}
        model_name = None
        if CLAUDE in file:
            model_name = "Claude 4.5 Haiku"
        elif KIMI in file:
            model_name = "Kimi K2 Thinking"
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
            val = in_data[k1].get(k2)
            out_data[k2] = val
        out.append(out_data)

    all_data = pd.DataFrame(out)
    all_data["dataset"] = ds.upper()
    return all_data


if __name__ == "__main__":
    gpqa = get_hint_usage_csvs("gpqa")
    mmlu_pro = get_hint_usage_csvs("mmlu_pro")
    final_df = pd.concat([gpqa, mmlu_pro], ignore_index=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved hint usage CSV to {OUTPUT_PATH}")
