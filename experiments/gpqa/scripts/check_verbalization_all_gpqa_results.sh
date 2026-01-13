#!/bin/bash

# GRADER HACKING

echo "GRADER HACKING - CLAUDE - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_changed_check_verbalization_results.json

echo "GRADER HACKING - CLAUDE - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_claude_4.5_haiku_changed_check_verbalization_results_detailed.json

echo "GRADER HACKING - KIMI - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_kimi_k2_thinking_changed_check_verbalization_results.json

echo "GRADER HACKING - KIMI - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json


# METADATA

echo "METADATA - CLAUDE - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_claude_4.5_haiku_changed_check_verbalization_results.json

echo "METADATA - CLAUDE - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_claude_4.5_haiku_changed_check_verbalization_results.json

echo "METADATA - KIMI - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_kimi_k2_thinking_changed_check_verbalization_results.json

echo "METADATA - KIMI - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json


# SYCOPHANCY V1

echo "SYCOPHANCY V1 - CLAUDE - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_claude_4.5_haiku_changed_check_verbalization_results.json

echo "SYCOPHANCY V1 - CLAUDE - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_claude_4.5_haiku_changed_check_verbalization_results.json

echo "SYCOPHANCY V1 - KIMI - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_thinking_changed_check_verbalization_results.json

echo "SYCOPHANCY V1 - KIMI - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json


# UNETHICAL INFORMATION

echo "UNETHICAL INFORMATION - CLAUDE - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_claude_4.5_haiku_changed_check_verbalization_results.json

echo "UNETHICAL INFORMATION - CLAUDE - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_claude_4.5_haiku_changed_check_verbalization_results.json

echo "UNETHICAL INFORMATION - KIMI - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_kimi_k2_thinking_changed_check_verbalization_results.json

echo "UNETHICAL INFORMATION - KIMI - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json