#!/bin/bash

# GRADER HACKING

echo "GRADER HACKING - CLAUDE - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_changed_check_verbalization_results.json \
    --include_per_example_results

echo "GRADER HACKING - CLAUDE - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_claude_4.5_haiku_changed_check_verbalization_results.json \
    --include_per_example_results


echo "GRADER HACKING - KIMI - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results

echo "GRADER HACKING - KIMI - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results

echo "GRADER HACKING - QWEN3 NEXT - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results

echo "GRADER HACKING - QWEN3 NEXT - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_qwen3_next_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_grader_hacking_incorrect_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results


# METADATA

echo "METADATA - CLAUDE - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_claude_4.5_haiku_changed_check_verbalization_results.json \
    --include_per_example_results

echo "METADATA - CLAUDE - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_claude_4.5_haiku_changed_check_verbalization_results.json \
    --include_per_example_results


echo "METADATA - KIMI - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results


echo "METADATA - KIMI - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results


echo "METADATA - QWEN3 NEXT - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_qwen3_next_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_correct_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results


echo "METADATA - QWEN3 NEXT - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_qwen3_next_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_metadata_incorrect_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results



# SYCOPHANCY V1

echo "SYCOPHANCY V1 - CLAUDE - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_claude_4.5_haiku_changed_check_verbalization_results.json \
    --include_per_example_results

echo "SYCOPHANCY V1 - CLAUDE - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_claude_4.5_haiku_changed_check_verbalization_results.json \
    --include_per_example_results

echo "SYCOPHANCY V1 - KIMI - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results

echo "SYCOPHANCY V1 - KIMI - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results

echo "SYCOPHANCY V1 - QWEN3 NEXT - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_qwen3_next_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_correct_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results

echo "SYCOPHANCY V1 - QWEN3 NEXT - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_qwen3_next_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_sycophancy_v1_incorrect_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results


# UNETHICAL INFORMATION

echo "UNETHICAL INFORMATION - CLAUDE - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_claude_4.5_haiku_changed_check_verbalization_results.json \
    --include_per_example_results

echo "UNETHICAL INFORMATION - CLAUDE - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_claude_4.5_haiku_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_claude_4.5_haiku_changed_check_verbalization_results.json \
    --include_per_example_results

echo "UNETHICAL INFORMATION - KIMI - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results

echo "UNETHICAL INFORMATION - KIMI - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results

echo "UNETHICAL INFORMATION - QWEN3 NEXT - CORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_qwen3_next_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_correct_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results

echo "UNETHICAL INFORMATION - QWEN3 NEXT - INCORRECT"
python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_qwen3_next_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results
