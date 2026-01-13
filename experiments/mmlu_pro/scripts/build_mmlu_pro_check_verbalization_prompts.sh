#!/bin/bash

# Script to build check verbalization prompts for all changed response files
# Processes each *_changed.jsonl file in experiments/gpqa/results/changed_responses/
# and generates corresponding prompt files in experiments/gpqa/prompts/check_verbalization/

CHANGED_RESPONSES_DIR="experiments/mmlu_pro/results/changed_responses"
OUTPUT_DIR="experiments/mmlu_pro/prompts/check_verbalization"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each *_changed.jsonl file
for input_file in "$CHANGED_RESPONSES_DIR"/*_changed.jsonl; do
    # Extract the base filename without path
    basename=$(basename "$input_file")
    
    # Remove _changed.jsonl suffix and add _check_verbalization.jsonl
    output_basename="${basename%.jsonl}_check_verbalization.jsonl"
    output_file="$OUTPUT_DIR/$output_basename"
    
    echo "Processing $basename -> $output_basename"
    
    python experiments/build_check_verbalization_prompts.py \
        "$input_file" \
        "$output_file"
done

echo "Done! All check verbalization prompts have been generated."
