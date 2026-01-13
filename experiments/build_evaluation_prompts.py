"""Build evaluation prompts for reasoning models lie experiments.

This script generates prompts for evaluating language models on multiple choice
questions from GPQA and MMLU Pro datasets, with optional hints for different
experimental conditions (baseline, sycophancy, grader hacking, etc.).

The script supports various hint types and strategies, different output formats
for different models, and can generate prompts for standard evaluation.

Example usage:
    python build_evaluation_prompts.py --task gpqa --setting baseline --split gpqa_main \\
           --hint_type grader_hacking --hint_strategy correct --output_dir ./prompts/
"""

import argparse
import json
import logging

from functools import partial
from pathlib import Path

from reasoning_models_lie.data_loaders.gpqa import (
    get_gpqa_baseline_eval_prompts,
    GPQA_SPLITS,
)
from reasoning_models_lie.data_loaders.mmlu_pro import (
    get_mmlu_pro_baseline_eval_prompts,
    MMLU_PRO_SPLITS,
)
from reasoning_models_lie.data_loaders.constants import (
    HINT_TYPES,
    HINT_STRATEGY,
    SETTINGS,
    TASKS,
    DEFAULT_RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build benchmark evaluation prompts and save to JSONL file."
    )
    parser.add_argument(
        "task",
        type=str,
        choices=TASKS,
        help="Task to build prompts for.",
    )
    parser.add_argument(
        "split",
        type=str,
        choices=GPQA_SPLITS | MMLU_PRO_SPLITS,
        help="Split to build prompts for.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=SETTINGS,
        default="baseline",
        required=True,
        help="Evaluation setting to use.",
    )
    parser.add_argument(
        "--format-for-deepseek",
        action="store_true",
        help="Whether to format prompts for DeepSeek model.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file to save the generated prompts.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to include in the output file.",
    )

    parser.add_argument(
        "--hint-type",
        type=str,
        choices=[None] + list(HINT_TYPES),
        default=None,
        help="Type of hint to include in the prompts (only used for 'baseline' setting).",
    )
    parser.add_argument(
        "--hint-strategy",
        type=str,
        choices=[None] + list(HINT_STRATEGY),
        default=None,
        help="Strategy for selecting the hint to include in the prompts (only used for 'baseline' setting).",
    )
    parser.add_argument(
        "--verbalize",
        action="store_true",
        help="Whether to include instructions to verbalize the hint (only used for 'baseline' setting).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of examples to sample from the dataset (for debugging purposes).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    # Validate split selection
    if args.task == "gpqa" and args.split not in GPQA_SPLITS:
        raise ValueError(
            f"Invalid split '{args.split}' for task 'gpqa'. Must be one of {GPQA_SPLITS}."
        )
    elif args.task == "mmlu_pro" and args.split not in MMLU_PRO_SPLITS:
        raise ValueError(
            f"Invalid split '{args.split}' for task 'mmlu_pro'. Must be one of {MMLU_PRO_SPLITS}."
        )

    # Select and execute appropriate prompt building function
    if args.setting == "baseline":
        if args.task == "gpqa":
            LOGGER.warning(
                "The --n-samples and --random-seed arguments are ignored for GPQA baseline evaluation."
            )
            get_baseline_eval_prompts = partial(
                get_gpqa_baseline_eval_prompts,
                verbalize=args.verbalize,
                hint_type=args.hint_type,
                hint_strategy=args.hint_strategy,
            )
        elif args.task == "mmlu_pro":
            get_baseline_eval_prompts = partial(
                get_mmlu_pro_baseline_eval_prompts,
                verbalize=args.verbalize,
                hint_type=args.hint_type,
                hint_strategy=args.hint_strategy,
                sample_size=args.n_samples,
                random_seed=args.seed,
            )
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        prompts = get_baseline_eval_prompts(
            split=args.split,
            format_for_deepseek=args.format_for_deepseek,
        )

        if args.hint_type is not None:
            if args.hint_strategy is None:
                raise ValueError(
                    "If --hint-type is provided, --hint-strategy must also be provided."
                )

    # Write prompts to output JSONL file
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for prompt in prompts[: args.max_examples]:
            f.write(json.dumps(prompt) + "\n")

    prompts_written = (
        len(prompts)
        if args.max_examples is None
        else min(len(prompts), args.max_examples)
    )
    LOGGER.info(f"Successfully wrote {prompts_written} prompts to {args.output_jsonl}")
