#!/bin/bash
PROJECT_ROOT=/home/hltcoe/wgantt/reasoning-models-lie
ALL_RESPONSES=$PROJECT_ROOT/experiments/mmlu_pro/results/all_responses
CHANGED_RESPONSES=$PROJECT_ROOT/experiments/mmlu_pro/results/changed_responses
GET_CHANGED_RESPONSES=$PROJECT_ROOT/experiments/get_change_to_hint_examples.py

CLAUDE_TEST200_BASELINE=$ALL_RESPONSES/mmlu_pro_test200_verbalize_baseline_claude_4.5_haiku.jsonl
KIMI_TEST200_BASELINE=$ALL_RESPONSES/mmlu_pro_test200_verbalize_baseline_kimi_k2_thinking.jsonl
QWEN_TEST200_BASELINE=$ALL_RESPONSES/mmlu_pro_test200_verbalize_baseline_qwen3_next.jsonl

# Claude 4.5 Haiku
for file in $ALL_RESPONSES/mmlu_pro_test200*haiku.jsonl; do
    # get file name
    filename=$(basename -- "$file")
    # remove extension
    filename="${filename%.*}"
    # ignore baseline file
    if [[ "$filename" == *"verbalize_baseline_claude_4.5_haiku"* ]]; then
        continue
    fi
    python $GET_CHANGED_RESPONSES $CLAUDE_TEST200_BASELINE $file $CHANGED_RESPONSES/${filename}_changed.jsonl $CHANGED_RESPONSES/${filename}_stats.json
done

# Kimi K2 Thinking
for file in $ALL_RESPONSES/mmlu_pro_test200*kimi_k2_thinking.jsonl; do
    # get file name
    filename=$(basename -- "$file")
    # remove extension
    filename="${filename%.*}"
    # ignore baseline file
    if [[ "$filename" == *"verbalize_baseline_kimi_k2_thinking"* ]]; then
        continue
    fi
    python $GET_CHANGED_RESPONSES $KIMI_TEST200_BASELINE $file $CHANGED_RESPONSES/${filename}_changed.jsonl $CHANGED_RESPONSES/${filename}_stats.json
done

# Qwen3-Next-80B-A3B-Thinking
for file in $ALL_RESPONSES/mmlu_pro_test200*qwen3_next.jsonl; do
    # get file name
    filename=$(basename -- "$file")
    # remove extension
    filename="${filename%.*}"
    # ignore baseline file
    if [[ "$filename" == *"verbalize_baseline_qwen3_next"* ]]; then
        continue
    fi
    python $GET_CHANGED_RESPONSES $QWEN_TEST200_BASELINE $file $CHANGED_RESPONSES/${filename}_changed.jsonl $CHANGED_RESPONSES/${filename}_stats.json
done