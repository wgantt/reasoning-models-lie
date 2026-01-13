"""MMLU Pro dataset loading and preprocessing utilities.

This module provides functions to load and preprocess the MMLU Pro dataset
for evaluation of large language models on multiple choice questions.

Dependencies:
    - datasets: For loading the MMLU Pro dataset from HuggingFace
    - reasoning_models_lie.prompts: For generating system prompts and hints
"""

import logging
import random

from datasets import load_dataset, IterableDataset
from datasets.formatting.formatting import LazyBatch
from hashlib import md5
from typing import Dict, List, Optional

from reasoning_models_lie.data_loaders.constants import (
    HINT_TYPES,
    DEFAULT_RANDOM_SEED,
)
from reasoning_models_lie.prompts.mmlu_pro import (
    VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE,
    NO_VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE,
)
from reasoning_models_lie.prompts.hints import get_hint_str


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset name
MMLU_PRO = "TIGER-Lab/MMLU-Pro"

# Data splits
MMLU_PRO_SPLITS = frozenset(["test", "validation"])

# multiple-choice options
MMLU_PRO_OPTION_PREFIXES = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J."]
MMLU_PRO_OPTION_IDX_TO_LETTER = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
}


def preprocess_mmlu_pro_example(
    examples: LazyBatch,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Optional[str | List[str]]]:
    """Preprocess a batch of MMLU Pro examples.

    Args:
        examples (LazyBatch): The MMLU Pro dataset batch to preprocess.
        seed (int): Random seed for reproducible behavior. Defaults to DEFAULT_RANDOM_SEED.

    Returns:
        Dict[str, Optional[str | List[str]]]: Dictionary containing:
            - original_question: List of original question texts
            - problem: List of formatted questions with answer choices
            - answer: List of correct answer letters
            - candidates: List of formatted answer choice lists
            - domain: List of question categories/domains
    """
    random.seed(seed)
    original_questions = []
    problems = []
    answers = []
    candidates = []

    for i in range(len(examples["question"])):
        # question
        original_questions.append(examples["question"][i])

        # correct answer
        correct_answer_letter = examples["answer"][i]
        answers.append(correct_answer_letter)

        # candidate answers - MMLU Pro has 10 answer choices
        choices = examples["options"][i]
        options_formatted = [
            f"{MMLU_PRO_OPTION_PREFIXES[j]} {choices[j]}" for j in range(len(choices))
        ]
        candidates.append(options_formatted)
        options_str = f"Answer Choices:\n{'\n'.join(options_formatted)}."
        problem = f"Question: {original_questions[i]}\n\n{options_str}"
        problems.append(problem)

    return {
        "original_question": original_questions,
        "problem": problems,
        "answer": answers,
        "candidates": candidates,
        "domain": examples["category"],
    }


def load_mmlu_pro(
    split: str = "validation",
    sample_size: Optional[int] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> IterableDataset:
    """Load MMLU Pro dataset and preprocess it.

    Args:
        split (str): The split to load. Must be one of MMLU_PRO_SPLITS. Defaults to "validation".
        sample_size (Optional[int]): If provided, randomly sample this many examples from the dataset.
        random_seed (int): Random seed for sampling.

    Returns:
        IterableDataset: The preprocessed MMLU Pro dataset for the given subject or all subjects.
    """
    assert (
        split in MMLU_PRO_SPLITS
    ), f"Invalid MMLU split: {split}. Must be one of {MMLU_PRO_SPLITS}."

    ds = load_dataset(MMLU_PRO, split=split)
    if sample_size is not None:
        assert sample_size <= len(ds), "Sample size cannot be larger than dataset size."
        ds = ds.shuffle(seed=random_seed).select(range(sample_size))
    return ds.map(
        preprocess_mmlu_pro_example,
        batched=True,
        remove_columns=ds.column_names,
    ).to_iterable_dataset()


def get_mmlu_pro_baseline_eval_prompts(
    split: str = "test",
    format_for_deepseek: bool = False,
    verbalize: bool = True,
    hint_type: Optional[str] = None,
    hint_strategy: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> List[Dict[str, str | Dict[str, str]]]:
    """Get MMLU Pro prompts for baseline evaluation.

    Args:
        split (str): The split to load. Defaults to "test".
        format_for_deepseek (bool): Whether to format prompts for DeepSeek model.
            (This means merging the system prompt with the user prompt.)
        verbalize (bool): Whether to use the verbalize hint prompt.
        hint_type (Optional[str]): Type of hint to include in the prompt. If None, no hint is included.
        hint_strategy (Optional[str]): Strategy for selecting the hinted answer. If None, no hint is included.
        sample_size (Optional[int]): If provided, randomly sample this many examples from the dataset.
        random_seed (int): Random seed for hint selection.

    Returns:
        List[Dict[str, str | Dict[str, str]]]: List of prompt dictionaries for evaluation.
    """
    random.seed(random_seed)
    if hint_type is not None:
        if hint_strategy is None:
            raise ValueError(
                "If hint_type is provided, hint_strategy must also be provided."
            )
        if hint_type not in HINT_TYPES:
            raise ValueError(
                f"Invalid hint_type: {hint_type}. Must be one of {HINT_TYPES}."
            )

    mmlu_pro_iter = load_mmlu_pro(
        split, sample_size=sample_size, random_seed=random_seed
    )
    prompts = []
    for example in mmlu_pro_iter:

        if hint_type is not None:
            # TODO: pass prohibited answer if needed
            letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            if len(letters) > len(example["candidates"]):
                logger.warning("Fewer candidates than expected; truncating letters.")
                candidates = example["candidates"][: len(letters)]
            elif len(letters) < len(example["candidates"]):
                raise ValueError(
                    "More candidates than expected; cannot assign letters."
                )
            else:
                candidates = example["candidates"]
            candidate_letters = []
            for letter, answer in zip(letters, candidates):
                if "N/A" not in answer:
                    candidate_letters.append(letter)

            assert (
                example["answer"] in candidate_letters
            ), "Correct answer must be in candidate letters."
            hint_str, hinted_answer = get_hint_str(
                hint_type,
                hint_strategy,
                candidate_letters,
                example["answer"],
                prohibited_answer=None,
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
            user_message = example["problem"]
            if verbalize:
                system_message = VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE
            else:
                system_message = NO_VERBALIZE_HINT_SYSTEM_PROMPT_CLAUDE

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
