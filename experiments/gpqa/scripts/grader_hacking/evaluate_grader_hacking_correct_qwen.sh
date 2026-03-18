#!/bin/bash
QWEN_API_KEY=$OPENROUTER_API_KEY

QWEN_TEMPERATURE=0.0
QWEN_MAX_TOKENS=12000
QWEN_MAX_THINKING_TOKENS=10000

### Qwen3-Next: Grader Hacking Correct
python experiments/prompt.py \
    --input-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_correct_prompts_qwen.jsonl \
    --output-jsonl experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next.jsonl \
    --model-type openrouter/qwen/qwen3-next-80b-a3b-thinking \
    --client-type openrouter \
    --api-key $QWEN_API_KEY \
    --temperature $QWEN_TEMPERATURE \
    --max-tokens $QWEN_MAX_TOKENS \
    --max-thinking-tokens $QWEN_MAX_THINKING_TOKENS \
    --max-concurrent-requests 20 \
    --examples-per-write 200

python experiments/evaluate_multiple_choice.py \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next.jsonl \
    experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_results.json
