#!/bin/bash
CLAUDE_API_KEY=$ANTHROPIC_API_KEY

CLAUDE_TEMPERATURE=0.0
CLAUDE_MAX_TOKENS=12000
CLAUDE_MAX_THINKING_TOKENS=10000

### Claude-4.5-Haiku: Metadata Eval
python experiments/prompt.py \
    --input-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_metadata_incorrect_prompts_claude.jsonl \
    --output-jsonl experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_metadata_incorrect_claude_4.5_haiku.jsonl \
    --model-type claude-haiku-4-5-20251001 \
    --api-key $ANTHROPIC_API_KEY \
    --temperature $CLAUDE_TEMPERATURE \
    --max-tokens $CLAUDE_MAX_TOKENS \
    --max-thinking-tokens $CLAUDE_MAX_THINKING_TOKENS \
    --examples-per-write 200 \
    --max-concurrent-requests 20

python experiments/evaluate_multiple_choice.py \
    experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_metadata_incorrect_claude_4.5_haiku.jsonl \
    experiments/mmlu_pro/results/all_responses/mmlu_pro_test200_verbalize_metadata_incorrect_claude_4.5_haiku_results.json