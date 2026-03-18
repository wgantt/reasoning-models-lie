#!/bin/bash

KIMI_TEMPERATURE=0.0
KIMI_MAX_TOKENS=12000
KIMI_MAX_THINKING_TOKENS=10000

### Kimi-K2-Thinking: Sycophancy V1 Eval
python experiments/prompt.py \
    --input-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_sycophancy_v1_correct_prompts_kimi.jsonl \
    --output-jsonl experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_5.jsonl \
    --model-type openrouter/moonshotai/kimi-k2.5 \
    --client-type openrouter \
    --api-key $OPENROUTER_API_KEY \
    --temperature $KIMI_TEMPERATURE \
    --max-tokens $KIMI_MAX_TOKENS \
    --max-thinking-tokens $KIMI_MAX_THINKING_TOKENS \
    --examples-per-write 200 \
    --max-concurrent-requests 20

python experiments/evaluate_multiple_choice.py \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_5.jsonl \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_5_results.json