#!/bin/bash

# Master script to run all check_verbalization scripts in mmlu_pro/scripts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Running all check_verbalization scripts for MMLU Pro"
echo "=========================================="
echo ""

# Array of all check_verbalization scripts
declare -a SCRIPTS=(
    "build_mmlu_pro_check_verbalization_prompts.sh"
    "grader_hacking/test200_check_verbalization_grader_hacking_correct_claude.sh"
    "grader_hacking/test200_check_verbalization_grader_hacking_correct_qwen.sh"
    "grader_hacking/test200_check_verbalization_grader_hacking_correct_kimi.sh"
    "grader_hacking/test200_check_verbalization_grader_hacking_incorrect_claude.sh"
    "grader_hacking/test200_check_verbalization_grader_hacking_incorrect_qwen.sh"
    "grader_hacking/test200_check_verbalization_grader_hacking_incorrect_kimi.sh"
    "metadata/test200_check_verbalization_metadata_correct_claude.sh"
    "metadata/test200_check_verbalization_metadata_correct_qwen.sh"
    "metadata/test200_check_verbalization_metadata_correct_kimi.sh"
    "metadata/test200_check_verbalization_metadata_incorrect_claude.sh"
    "metadata/test200_check_verbalization_metadata_incorrect_qwen.sh"
    "metadata/test200_check_verbalization_metadata_incorrect_kimi.sh"
    "sycophancy_v1/test200_check_verbalization_sycophancy_v1_correct_claude.sh"
    "sycophancy_v1/test200_check_verbalization_sycophancy_v1_correct_qwen.sh"
    "sycophancy_v1/test200_check_verbalization_sycophancy_v1_correct_kimi.sh"
    "sycophancy_v1/test200_check_verbalization_sycophancy_v1_incorrect_claude.sh"
    "sycophancy_v1/test200_check_verbalization_sycophancy_v1_incorrect_qwen.sh"
    "sycophancy_v1/test200_check_verbalization_sycophancy_v1_incorrect_kimi.sh"
    "unethical_information/test200_check_verbalization_unethical_information_correct_claude.sh"
    "unethical_information/test200_check_verbalization_unethical_information_correct_qwen.sh"
    "unethical_information/test200_check_verbalization_unethical_information_correct_kimi.sh"
    "unethical_information/test200_check_verbalization_unethical_information_incorrect_claude.sh"
    "unethical_information/test200_check_verbalization_unethical_information_incorrect_qwen.sh"
    "unethical_information/test200_check_verbalization_unethical_information_incorrect_kimi.sh"
)

# Run each script
count=0
total=${#SCRIPTS[@]}

for script in "${SCRIPTS[@]}"; do
    count=$((count + 1))
    echo "[$count/$total] Running: $script"
    
    script_path="${SCRIPT_DIR}/${script}"
    if [ -f "$script_path" ]; then
        bash "$script_path"
        echo "✓ Completed: $script"
    else
        echo "✗ Script not found: $script_path"
    fi
    echo ""
done

echo "=========================================="
echo "All check_verbalization scripts completed!"
echo "=========================================="
