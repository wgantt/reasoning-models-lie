#!/bin/bash
OPENAI_API_KEY=$OPENAI_API_KEY

OPENAI_TEMPERATURE=0.0
OPENAI_MAX_TOKENS=8192
OPENAI_MAX_THINKING_TOKENS=4096 # not actually consumed

python experiments/prompt.py \
    --input-jsonl experiments/mmlu_pro/prompts/check_verbalization/mmlu_pro_test200_verbalize_metadata_correct_qwen3_next_changed_check_verbalization.jsonl \
    --output-jsonl experiments/mmlu_pro/results/check_verbalization/mmlu_pro_test200_verbalize_metadata_correct_qwen3_next_changed_check_verbalization.jsonl \
    --model-type gpt-5.4-2026-03-05 \
    --api-key $OPENAI_API_KEY \
    --temperature $OPENAI_TEMPERATURE \
    --max-tokens $OPENAI_MAX_TOKENS \
    --max-thinking-tokens $OPENAI_MAX_THINKING_TOKENS \
    --reasoning-effort medium \
    --examples-per-write 200 \
    --max-concurrent-requests 20

python experiments/evaluate_faithfulness.py \
    mmlu_pro \
    experiments/mmlu_pro/results/check_verbalization/mmlu_pro_test200_verbalize_metadata_correct_qwen3_next_changed_check_verbalization.jsonl \
    experiments/mmlu_pro/results/check_verbalization/mmlu_pro_test200_verbalize_metadata_correct_qwen3_next_changed_check_verbalization_results.json \
    --include_per_example_results
