import numpy as np
import pandas as pd
import json

from glob import glob

CLAUDE = "claude_4.5_haiku"
KIMI = "kimi_k2_thinking"
QWEN = "qwen3_next"

FIELDS_TO_SAVE = [
    ("total_common_examples", "total_examples"),
    ("total_to_hinted", "changed_to_hinted_count"),
    ("fraction_to_hinted_changed", "changed_to_hinted_percentage"),
    ("fraction_to_hinted_changed_ci_lower", "changed_to_hinted_percentage_ci_lower"),
    ("fraction_to_hinted_changed_ci_upper", "changed_to_hinted_percentage_ci_upper"),
    ("binom_test_pvalue", "changed_to_hinted_p_value"),
]
GPQA_RESULTS_PATH = "experiments/gpqa/results/changed_responses/"
MMLU_PRO_RESULTS_PATH = "experiments/mmlu_pro/results/changed_responses/"
OUTPUT_PATH = "analysis/data/changed_to_hinted.csv"


def get_hint_usage_csvs(ds: str = "gpqa") -> pd.DataFrame:
    if ds == "gpqa":
        results_path = GPQA_RESULTS_PATH
        all_files = glob(results_path + "/*stats.json")
    elif ds == "mmlu_pro":
        results_path = MMLU_PRO_RESULTS_PATH
        all_files = glob(results_path + "/mmlu_pro_test200*stats.json")
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
            out_data[k2] = (
                in_data.get(k1) * 100 if "fraction" in k1 else in_data.get(k1)
            )
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
