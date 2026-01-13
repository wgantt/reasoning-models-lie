#!/bin/bash
KIMI_TEMPERATURE=0.0
KIMI_MAX_TOKENS=12000
KIMI_MAX_THINKING_TOKENS=10000

### Kimi K2 Thinking: Metadata Eval
python experiments/prompt.py \
    --input-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_metadata_correct_prompts_kimi.jsonl \
    --output-jsonl experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_metadata_correct_kimi_k2_thinking.jsonl \
    --model-type moonshotai/Kimi-K2-Thinking \
    --client-type together \
    --api-key $TOGETHER_API_KEY \
    --temperature $KIMI_TEMPERATURE \
    --max-tokens $KIMI_MAX_TOKENS \
    --max-thinking-tokens $KIMI_MAX_THINKING_TOKENS \
    --examples-per-write 200 \
    --max-concurrent-requests 40

python experiments/evaluate_multiple_choice.py \
    experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_metadata_correct_kimi_k2_thinking.jsonl \
    experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_metadata_correct_kimi_k2_thinking_results.json