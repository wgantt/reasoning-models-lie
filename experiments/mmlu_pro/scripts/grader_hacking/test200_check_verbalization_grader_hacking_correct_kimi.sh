#!/bin/bash
CLAUDE_API_KEY=$ANTHROPIC_API_KEY

CLAUDE_TEMPERATURE=0.0
CLAUDE_MAX_TOKENS=2048
CLAUDE_MAX_THINKING_TOKENS=0 # no thinking tokens for this task


python experiments/prompt.py \
    --input-jsonl experiments/mmlu_pro/prompts/check_verbalization/mmlu_pro_test200_verbalize_grader_hacking_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    --output-jsonl experiments/mmlu_pro/results/check_verbalization/mmlu_pro_test200_verbalize_grader_hacking_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    --model-type claude-haiku-4-5-20251001 \
    --api-key $ANTHROPIC_API_KEY \
    --temperature $CLAUDE_TEMPERATURE \
    --max-tokens $CLAUDE_MAX_TOKENS \
    --max-thinking-tokens $CLAUDE_MAX_THINKING_TOKENS \
    --examples-per-write 100

python experiments/evaluate_faithfulness.py \
    mmlu_pro \
    experiments/mmlu_pro/results/check_verbalization/mmlu_pro_test200_verbalize_grader_hacking_correct_kimi_k2_thinking_changed_check_verbalization.jsonl \
    experiments/mmlu_pro/results/check_verbalization/mmlu_pro_test200_verbalize_grader_hacking_correct_kimi_k2_thinking_changed_check_verbalization_results.json