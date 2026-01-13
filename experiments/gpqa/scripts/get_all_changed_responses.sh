#!/bin/bash
PROJECT_ROOT=/home/hltcoe/wgantt/reasoning-models-lie
ALL_RESPONSES=$PROJECT_ROOT/experiments/gpqa/results/all_responses
CHANGED_RESPONSES=$PROJECT_ROOT/experiments/gpqa/results/changed_responses
GET_CHANGED_RESPONSES=$PROJECT_ROOT/experiments/get_change_to_hint_examples.py

CLAUDE_BASELINE=$ALL_RESPONSES/gpqa_diamond_verbalize_baseline_claude_4.5_haiku.jsonl
KIMI_BASELINE=$ALL_RESPONSES/gpqa_diamond_verbalize_baseline_kimi_k2_thinking.jsonl

# Claude 4.5 Haiku
for file in $ALL_RESPONSES/*haiku.jsonl; do
    # get file name
    filename=$(basename -- "$file")
    # remove extension
    filename="${filename%.*}"
    # ignore baseline
    if [[ "$filename" == *"verbalize_baseline_claude_4.5_haiku"* ]]; then
        continue
    fi
    python $GET_CHANGED_RESPONSES $CLAUDE_BASELINE $file $CHANGED_RESPONSES/${filename}_changed.jsonl $CHANGED_RESPONSES/${filename}_stats.json
done

# Kimi K2 Thinking
for file in $ALL_RESPONSES/*kimi_k2_thinking.jsonl; do
    # get file name
    filename=$(basename -- "$file")
    # remove extension
    filename="${filename%.*}"
    # ignore baseline
    if [[ "$filename" == *"verbalize_baseline_kimi_k2_thinking"* ]]; then
        continue
    fi
    python $GET_CHANGED_RESPONSES $KIMI_BASELINE $file $CHANGED_RESPONSES/${filename}_changed.jsonl $CHANGED_RESPONSES/${filename}_stats.json
done


# Qwen3-Next-80B-A3B-Thinking
for file in $ALL_RESPONSES/*qwen3_next.jsonl; do
    # get file name
    filename=$(basename -- "$file")
    # remove extension
    filename="${filename%.*}"
    # ignore baseline
    if [[ "$filename" == *"verbalize_baseline_qwen3_next"* ]]; then
        continue
    fi
    python $GET_CHANGED_RESPONSES $GPT_OSS_BASELINE $file $CHANGED_RESPONSES/${filename}_changed.jsonl $CHANGED_RESPONSES/${filename}_stats.json
done