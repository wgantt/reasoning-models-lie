#!/bin/bash
PROJECT_ROOT=/home/hltcoe/wgantt/reasoning-models-lie
ALL_RESPONSES=$PROJECT_ROOT/experiments/gpqa/results/all_responses
CHANGED_RESPONSES=$PROJECT_ROOT/experiments/gpqa/results/changed_responses
GET_CHANGED_RESPONSES=$PROJECT_ROOT/experiments/get_change_to_hint_examples.py

CLAUDE_BASELINE=$ALL_RESPONSES/gpqa_diamond_verbalize_baseline_claude_4.5_haiku.jsonl
QWEN_BASELINE=$ALL_RESPONSES/gpqa_diamond_verbalize_baseline_qwen3_next.jsonl

BASELINES=("gpqa_diamond_verbalize_baseline_claude_4.5_haiku_1024.jsonl" "gpqa_diamond_verbalize_baseline_claude_4.5_haiku_2000.jsonl" "gpqa_diamond_verbalize_baseline_claude_4.5_haiku_4000.jsonl" "gpqa_diamond_verbalize_baseline_claude_4.5_haiku_8000.jsonl" "gpqa_diamond_verbalize_baseline_claude_4.5_haiku_16000.jsonl")

HINTED_RESPONSES=("gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_1024.jsonl" "gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_2000.jsonl" "gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_4000.jsonl" "gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_8000.jsonl" "gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_16000.jsonl")

QWEN_BASELINES=("gpqa_diamond_verbalize_baseline_qwen3_next_1024.jsonl" "gpqa_diamond_verbalize_baseline_qwen3_next_2000.jsonl" "gpqa_diamond_verbalize_baseline_qwen3_next_4000.jsonl" "gpqa_diamond_verbalize_baseline_qwen3_next_8000.jsonl" "gpqa_diamond_verbalize_baseline_qwen3_next_16000.jsonl")

QWEN_HINTED_RESPONSES=("gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_1024.jsonl" "gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_2000.jsonl" "gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_4000.jsonl" "gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_8000.jsonl" "gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_16000.jsonl")

# Claude 4.5 Haiku
for i in "${!BASELINES[@]}"; do
    BASELINE=$ALL_RESPONSES/${BASELINES[$i]}
    HINTED_RESPONSE=$ALL_RESPONSES/${HINTED_RESPONSES[$i]}
    # get file name
    filename=$(basename -- "$HINTED_RESPONSE")
    # remove extension
    filename="${filename%.*}"
    # ignore baseline
    if [[ "$filename" == *"verbalize_baseline_claude_4.5_haiku"* ]]; then
        continue
    fi
    python $GET_CHANGED_RESPONSES $BASELINE $HINTED_RESPONSE $CHANGED_RESPONSES/${filename}_changed.jsonl $CHANGED_RESPONSES/${filename}_stats.json
done

# Qwen3-Next-80B-A3B-Thinking
for i in "${!QWEN_BASELINES[@]}"; do
    BASELINE=$ALL_RESPONSES/${QWEN_BASELINES[$i]}
    HINTED_RESPONSE=$ALL_RESPONSES/${QWEN_HINTED_RESPONSES[$i]}
    # get file name
    filename=$(basename -- "$HINTED_RESPONSE")
    # remove extension
    filename="${filename%.*}"
    # ignore baseline
    if [[ "$filename" == *"verbalize_baseline_qwen3_next"* ]]; then
        continue
    fi
    python $GET_CHANGED_RESPONSES $BASELINE $HINTED_RESPONSE $CHANGED_RESPONSES/${filename}_changed.jsonl $CHANGED_RESPONSES/${filename}_stats.json
done