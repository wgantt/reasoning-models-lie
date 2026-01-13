#!/bin/bash
CLAUDE_API_KEY=$ANTHROPIC_API_KEY

# Settings from the Anthropic paper
CLAUDE_TEMPERATURE=0.0
CLAUDE_MAX_TOKENS=12000
CLAUDE_MAX_THINKING_TOKENS=10000

### Claude 4.5 Haiku: Unethical Information
python experiments/prompt.py \
    --input-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_unethical_information_incorrect_prompts_claude.jsonl \
    --output-jsonl experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_unethical_information_incorrect_claude_4.5_haiku.jsonl \
    --model-type claude-haiku-4-5-20251001 \
    --api-key $ANTHROPIC_API_KEY \
    --temperature $CLAUDE_TEMPERATURE \
    --max-tokens $CLAUDE_MAX_TOKENS \
    --max-thinking-tokens $CLAUDE_MAX_THINKING_TOKENS

python experiments/evaluate_multiple_choice.py \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_unethical_information_incorrect_claude_4.5_haiku.jsonl \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_unethical_information_incorrect_claude_4.5_haiku_results.json