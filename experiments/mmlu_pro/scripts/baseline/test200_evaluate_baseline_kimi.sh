#!/bin/bash
KIMI_TEMPERATURE=0.0
KIMI_MAX_TOKENS=12000
KIMI_MAX_THINKING_TOKENS=10000

### Kimi K2 Thinking: Baseline Eval
python experiments/prompt.py \
    --input-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_baseline_prompts_kimi.jsonl \
    --output-jsonl experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_baseline_kimi_k2_5.jsonl \
    --model-type openrouter/moonshotai/kimi-k2.5 \
    --client-type openrouter \
    --api-key $OPENROUTER_API_KEY \
    --temperature $KIMI_TEMPERATURE \
    --max-tokens $KIMI_MAX_TOKENS \
    --max-thinking-tokens $KIMI_MAX_THINKING_TOKENS \
    --examples-per-write 200 \
    --max-concurrent-requests 20

python experiments/evaluate_multiple_choice.py \
    experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_baseline_kimi_k2_5.jsonl \
    experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_baseline_kimi_k2_5_results.json