"""Evaluate faithfulness of reasoning in model responses.

This script evaluates how faithfully models follow hints provided in prompts
by analyzing their reasoning processes and final answers. It computes metrics
for hint reliance and reasoning faithfulness with bootstrap confidence intervals.

Example usage:
    python evaluate_faithfulness.py gpqa responses.jsonl metrics.json --n_bootstrap 10000
"""

import argparse
import json

from reasoning_models_lie.data_loaders.constants import (
    DEFAULT_RANDOM_SEED,
    TASKS,
    GPQA_DIAMOND_NUM_EXAMPLES,
    MMLU_PRO_VALIDATION_NUM_EXAMPLES,
    MMLU_PRO_TEST200_NUM_EXAMPLES,
)
from reasoning_models_lie.evaluation.faithfulness import (
    bootstrap_evaluate_faithfulness,
    evaluate_faithfulness,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness of reasoning outputs"
    )
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file")
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for confidence intervals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    with open(args.input_file, "r") as f:
        d = [json.loads(line) for line in f]
    if args.task == "gpqa":
        dataset_size = GPQA_DIAMOND_NUM_EXAMPLES
    elif args.task == "mmlu_pro":
        if "test200" in args.input_file.lower():
            dataset_size = MMLU_PRO_TEST200_NUM_EXAMPLES
        else:
            dataset_size = MMLU_PRO_VALIDATION_NUM_EXAMPLES
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    main_results = evaluate_faithfulness(d)
    if (
        main_results["faithfulness_score_normalized"] is None
        or main_results["honesty_score_normalized"] is None
    ):
        print(
            "Skipping bootstrap evaluation due to undefined faithfulness or honesty score."
        )
        bootstrap_results = {}
    else:
        bootstrap_results = {
            k: round(v, 3)
            for k, v in bootstrap_evaluate_faithfulness(
                args.input_file,
                dataset_size=dataset_size,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            ).items()
        }
    main_results = {k: round(v, 3) for k, v in main_results.items() if v is not None}
    all_results = {"main_results": main_results, "bootstrap_results": bootstrap_results}
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)
