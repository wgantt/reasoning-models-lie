"""Generate accuracy CSV data from evaluation results.

This script processes JSON files containing model evaluation results and
generates a CSV file with accuracy metrics for different models, datasets,
and experimental conditions.

The script supports both GPQA and MMLU Pro datasets and extracts accuracy
data with confidence intervals from the evaluation results.

Example usage:
    python get_accuracy_csv.py
"""

import numpy as np
import pandas as pd
import json

from glob import glob

CLAUDE = "claude_4.5_haiku"
KIMI = "kimi_k2_thinking"
QWEN = "qwen3_next"

GPQA_RESULTS_PATH = "experiments/gpqa/results/all_responses/"
MMLU_PRO_RESULTS_PATH = "experiments/mmlu_pro/results/all_responses/"
OUTPUT_PATH = "analysis/data/accuracy.csv"


def get_accuracy_csv(ds: str = "gpqa") -> pd.DataFrame:
    """Generate accuracy DataFrame from evaluation results.

    Args:
        ds (str): Dataset name, either "gpqa" or "mmlu_pro".

    Returns:
        pd.DataFrame: DataFrame containing accuracy metrics for different
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
        elif "baseline" in file:
            hint_type = "Baseline"
        else:
            raise ValueError(f"Unknown hint type in file name: {file}")
        out_data["hint_type"] = hint_type

        setting = None
        if "incorrect" in file:
            setting = "Incorrect Hint"
        elif "correct" in file:
            setting = "Correct Hint"
        elif "baseline" not in file:
            raise ValueError(f"Unknown setting in file name: {file}")
        out_data["setting"] = setting

        out_data["accuracy"] = np.round(in_data["accuracy_total"], 4)
        out_data["accuracy_ci_lower"] = np.round(
            in_data["bootstrap"]["accuracy_total_ci"][0], 4
        )
        out_data["accuracy_ci_upper"] = np.round(
            in_data["bootstrap"]["accuracy_total_ci"][1], 4
        )
        out.append(out_data)

    all_data = pd.DataFrame(out)
    all_data["dataset"] = ds.upper()
    return all_data


if __name__ == "__main__":
    gpqa = get_accuracy_csv("gpqa")
    mmlu_pro = get_accuracy_csv("mmlu_pro")
    final_df = pd.concat([gpqa, mmlu_pro], ignore_index=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved accuracy CSV to {OUTPUT_PATH}")
