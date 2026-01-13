"""Build prompts to check hint verbalization in model responses.

This script generates prompts to check whether models properly verbalized
hints that were provided in their input prompts. It creates evaluation
prompts that can be used to assess hint acknowledgment and verbalization.

Example usage:
    python build_check_verbalization_prompts.py responses.jsonl check_prompts.jsonl
"""

import argparse

from reasoning_models_lie.prompts.hints import get_check_hint_verbalization_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build verbalization check prompts and save to JSONL file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file containing model outputs.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output JSONL file to save verbalization check prompts.",
    )
    args = parser.parse_args()

    get_check_hint_verbalization_prompts(args.input_file, args.output_file)
