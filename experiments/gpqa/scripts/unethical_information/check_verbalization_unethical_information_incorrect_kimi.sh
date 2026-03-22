#!/bin/bash
OPENAI_API_KEY=$OPENAI_API_KEY

OPENAI_TEMPERATURE=0.0
OPENAI_MAX_TOKENS=8192
OPENAI_MAX_THINKING_TOKENS=4096 # not actually consumed

### Claude-4.5-Haiku: Grader Hacking Eval

python experiments/prompt.py \
    --input-jsonl experiments/gpqa/prompts/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    --output-jsonl experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    --model-type gpt-5.4-2026-03-05 \
    --api-key $OPENAI_API_KEY \
    --temperature $OPENAI_TEMPERATURE \
    --max-tokens $OPENAI_MAX_TOKENS \
    --max-thinking-tokens $OPENAI_MAX_THINKING_TOKENS \
    --reasoning-effort medium \
    --examples-per-write 200 \
    --max-concurrent-requests 3

python experiments/evaluate_faithfulness.py \
    gpqa \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/gpqa/results/check_verbalization/gpqa_diamond_verbalize_unethical_information_incorrect_kimi_k2_thinking_changed_check_verbalization_results.json \
    --include_per_example_results
