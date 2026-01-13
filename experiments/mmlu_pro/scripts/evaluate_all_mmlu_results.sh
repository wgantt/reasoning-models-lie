#!/bin/bash

# Script to evaluate all MMLU Pro results using evaluate_multiple_choice.py
# Processes all .jsonl files in experiments/mmlu_pro/results/all_responses
# and outputs results to the same directory with _results.json suffix

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the paths relative to the script location
RESULTS_DIR="${SCRIPT_DIR}/../results/all_responses"
EVALUATE_SCRIPT="${SCRIPT_DIR}/../../evaluate_multiple_choice.py"

# Check if the results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Check if the evaluation script exists
if [ ! -f "$EVALUATE_SCRIPT" ]; then
    echo "Error: Evaluation script not found: $EVALUATE_SCRIPT"
    exit 1
fi

# Process each .jsonl file in the results directory
for input_file in "$RESULTS_DIR"/*.jsonl; do
    # Check if any .jsonl files exist
    if [ ! -e "$input_file" ]; then
        echo "No .jsonl files found in $RESULTS_DIR"
        exit 0
    fi
    
    # Get the base name without extension
    base_name=$(basename "$input_file" .jsonl)
    
    # Create the output filename
    output_file="${RESULTS_DIR}/${base_name}_results.json"
    
    # Skip if the output file already exists
    # if [ -f "$output_file" ]; then
    #     echo "Skipping $base_name (output already exists)"
    #     continue
    # fi
    
    echo "Processing: $base_name"
    
    # Run the evaluation script
    python "$EVALUATE_SCRIPT" "$input_file" "$output_file"
    
    # Check if the evaluation was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated: $base_name"
    else
        echo "✗ Failed to evaluate: $base_name"
    fi
    
    echo ""
done

echo "All evaluations complete!"
