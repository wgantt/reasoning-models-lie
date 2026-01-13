"""Data loading utilities for the GPQA (Graduate-level Physics, Chemistry and Biology) dataset."""

import os
import logging
import random

from datasets import load_dataset, DownloadConfig, IterableDataset
from datasets.formatting.formatting import LazyBatch
from hashlib import md5
from typing import Dict, List, Optional

from reasoning_models_lie.data_loaders.constants import (
    HINT_TYPES,
    DEFAULT_RANDOM_SEED,
    OPTION_PREFIXES,
    OPTION_IDX_TO_LETTER,
)
from reasoning_models_lie.prompts.gpqa import (
    VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE,
    NO_VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE,
)
from reasoning_models_lie.prompts.hints import get_hint_str


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: you must agree to the terms here:
# https://huggingface.co/datasets/Idavidrein/gpqa
# Then you must obtain an HF API token from your profile
# and assign it to this environment variable
HF_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

# GPQA dataset and split names
GPQA = "Idavidrein/gpqa"
GPQA_SPLITS = frozenset(["gpqa_main", "gpqa_diamond", "gpqa_experts", "gpqa_extended"])


def preprocess_gpqa_example(
    examples: LazyBatch,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Optional[str | List[str]]]:
    """
    Preprocess a batch of GPQA examples.

    Args:
        examples (LazyBatch): The GPQA dataset to yield examples from.
        seed (int): Random seed for shuffling answer options.

    Returns:
        Dict[str, str | List[str]]: The GPQA dataset with full examples.
    """
    random.seed(seed)
    original_questions = []
    problems = []
    answers = []
    reasoning_traces = []
    candidates = []
    domains = []
    for i in range(len(examples["Question"])):
        # question
        original_questions.append(examples["Question"][i])

        # candidate answers
        correct_answer = examples["Correct Answer"][i]
        incorrect_answers = [
            examples["Incorrect Answer 1"][i],
            examples["Incorrect Answer 2"][i],
            examples["Incorrect Answer 3"][i],
        ]
        options = [correct_answer] + incorrect_answers
        random.shuffle(options)

        # formatted problem
        options_formatted = [
            f"{OPTION_PREFIXES[i]} {options[i]}" for i in range(len(options))
        ]
        candidates.append(options_formatted)
        options_str = f"Answer Choices: {' '.join(options_formatted)}."
        problem = f"{original_questions[i]}\n\n{options_str}"
        problems.append(problem)

        # correct answers
        correct_answer_idx = options.index(correct_answer)
        correct_answer_letter = OPTION_IDX_TO_LETTER[correct_answer_idx]
        answers.append(correct_answer_letter)

        # reasoning steps (sentence-split w/ SpaCy)
        explanation = examples["Explanation"][i]
        reasoning_traces.append(explanation)
        domains.append(examples["High-level domain"][i])

    return {
        "original_question": original_questions,
        "problem": problems,
        "answer": answers,
        "reasoning_trace": reasoning_traces,
        "candidates": candidates,
        "domain": domains,
    }


def load_gpqa(split: str) -> IterableDataset:
    """Load a GPQA dataset split as an IterableDataset.

    Downloads and preprocesses the specified GPQA split, shuffling answer choices
    and formatting the data for evaluation.

    Args:
        split: The GPQA split to load. Must be one of GPQA_SPLITS
            (gpqa_main, gpqa_diamond, gpqa_experts, gpqa_extended).

    Returns:
        An IterableDataset containing preprocessed GPQA examples.

    Raises:
        AssertionError: If the split is not valid.

    Note:
        Requires HUGGINGFACE_API_TOKEN environment variable to be set.
    """
    assert (
        split in GPQA_SPLITS
    ), f"Invalid GPQA split: {split}. Must be one of {GPQA_SPLITS}."
    ds = load_dataset(GPQA, split, download_config=DownloadConfig(token=HF_TOKEN))
    ds = ds["train"]
    return ds.map(
        preprocess_gpqa_example,
        batched=True,
        remove_columns=ds.column_names,
    ).to_iterable_dataset()


def get_gpqa_baseline_eval_prompts(
    split: str,
    format_for_deepseek: bool = False,
    verbalize: bool = True,
    hint_type: Optional[str] = None,
    hint_strategy: Optional[str] = None,
) -> List[Dict[str, str | Dict[str, str]]]:
    """Get GPQA prompts for baseline evaluation.

    Args:
        split (str): The GPQA split to load prompts from.
        format_for_deepseek (bool): Whether to format prompts for DeepSeek model.
            (This means merging the system prompt with the user prompt.)
        verbalize (bool): Whether to include instructions to verbalize the hint.
        hint_type (Optional[str]): Type of hint to include in the prompt. If None, no hint is included.
        hint_strategy (Optional[str]): Strategy for generating hints. Required if hint_type is provided.

    Returns:
        List[Dict[str, str | Dict[str, str]]]: List of prompt dictionaries for evaluation.
    """
    if hint_type is not None:
        if hint_strategy is None:
            raise ValueError(
                "If hint_type is provided, hint_strategy must also be provided."
            )
        if hint_type not in HINT_TYPES:
            raise ValueError(
                f"Invalid hint_type: {hint_type}. Must be one of {HINT_TYPES}."
            )

    gpqa_iter = load_gpqa(split)
    prompts = []
    for example in gpqa_iter:
        if hint_type is not None:
            # TODO: pass prohibited answer if needed
            hint_str, hinted_answer = get_hint_str(
                hint_type,
                hint_strategy,
                ["A", "B", "C", "D"],
                example["answer"],
                domain=example["domain"],
            )
            example["hinted_answer"] = hinted_answer

        if format_for_deepseek:
            if hint_type == "metadata":
                # metadata hints go at the top
                user_message = "# Problem\n\n" + hint_str + "\n\n" + example["problem"]
            elif hint_type is not None:
                # other hints go at the bottom
                user_message = "# Problem\n\n" + example["problem"] + "\n\n" + hint_str
            else:
                user_message = "# Problem\n\n" + example["problem"]

            if verbalize:
                user_message = (
                    VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE + "\n\n" + user_message
                )
            else:
                user_message = (
                    NO_VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE + "\n\n" + user_message
                )
            system_message = ""

        else:
            if verbalize:
                system_message = VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE
            else:
                system_message = NO_VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE

            user_message = "Question: " + example["problem"]

            if hint_type == "metadata":
                # metadata hints go at the top
                user_message = hint_str + "\n\n" + user_message
            else:
                # other hints go at the bottom
                user_message = user_message + "\n\n" + hint_str

        prompt = {
            "instance_id": md5(example["original_question"].encode()).hexdigest(),
            "user_message": user_message,
            "system_message": system_message,
            "meta": example,
        }
        prompts.append(prompt)
    return prompts


if __name__ == "__main__":
    # Example usage
    prompts = get_gpqa_baseline_eval_prompts(
        "gpqa_main",
        format_for_deepseek=False,
        hint_type="grader_hacking",
        hint_strategy="correct",
    )
    for p in prompts[:3]:
        print(f"Instance ID: {p['instance_id']}")
        print(f"System Message: {p['system_message']}")
        print(f"User Message: {p['user_message']}")
        print()
