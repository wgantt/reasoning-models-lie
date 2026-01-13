"""Constants and configuration values for data loading and evaluation.

This module contains global constants used throughout the reasoning models lie
evaluation pipeline, including supported datasets, hint types, evaluation settings,
and multiple choice answer formatting.
"""

# Default random seed
DEFAULT_RANDOM_SEED = 19106

# Hinted evaluations are done in the baseline setting
SETTINGS = frozenset(["baseline"])

HINT_TYPES = frozenset(
    [
        "metadata",
        "sycophancy_v1",
        "sycophancy_v2",
        "grader_hacking",
        "unethical_information",
    ]
)
# correct: always hints the correct answer
# random: hints a random answer (could be correct or incorrect)
# random_incorrect: hints a random incorrect answer
HINT_STRATEGY = frozenset(["correct", "random", "random_incorrect"])

# Supported datasets/tasks
TASKS = frozenset(["gpqa", "mmlu_pro"])
GPQA_DIAMOND_NUM_EXAMPLES = 198
MMLU_PRO_VALIDATION_NUM_EXAMPLES = 70
MMLU_PRO_TEST200_NUM_EXAMPLES = 200

# multiple-choice options
OPTION_PREFIXES = ["A.", "B.", "C.", "D."]
OPTION_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
