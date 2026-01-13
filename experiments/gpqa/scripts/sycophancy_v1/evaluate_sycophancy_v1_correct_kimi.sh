#!/bin/bash

KIMI_TEMPERATURE=0.0
KIMI_MAX_TOKENS=12000
KIMI_MAX_THINKING_TOKENS=10000

### Kimi-K2-Thinking: Sycophancy V1 Eval
python experiments/prompt.py \
    --input-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_sycophancy_v1_correct_prompts_kimi.jsonl \
    --output-jsonl experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_thinking.jsonl \
    --model-type "moonshotai/Kimi-K2-Thinking" \
    --client-type together \
    --api-key $TOGETHER_API_KEY \
    --temperature $KIMI_TEMPERATURE \
    --max-tokens $KIMI_MAX_TOKENS \
    --max-thinking-tokens $KIMI_MAX_THINKING_TOKENS

python experiments/evaluate_multiple_choice.py \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_thinking.jsonl \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_sycophancy_v1_correct_kimi_k2_thinking_results.json