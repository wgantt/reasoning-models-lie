#!/bin/bash
KIMI_API_KEY=$OPENROUTER_API_KEY

KIMI_TEMPERATURE=0.0
KIMI_MAX_TOKENS=12000
KIMI_MAX_THINKING_TOKENS=10000

### Kimi-K2-Thinking: Grader Hacking 
python experiments/prompt.py \
    --input-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_metadata_incorrect_prompts_kimi.jsonl \
    --output-jsonl experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_metadata_incorrect_kimi_k2_5.jsonl \
    --model-type openrouter/moonshotai/kimi-k2.5 \
    --client-type openrouter \
    --api-key $OPENROUTER_API_KEY \
    --temperature $KIMI_TEMPERATURE \
    --max-tokens $KIMI_MAX_TOKENS \
    --max-thinking-tokens $KIMI_MAX_THINKING_TOKENS \
    --examples-per-write 200 \
    --max-concurrent-requests 20

python experiments/evaluate_multiple_choice.py \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_metadata_incorrect_kimi_k2_5.jsonl \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_metadata_incorrect_kimi_k2_5_results.json