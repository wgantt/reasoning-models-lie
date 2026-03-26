#!/bin/bash
QWEN_API_KEY=$OPENROUTER_API_KEY

QWEN_TEMPERATURE=0.0

max_thinking_tokens_values=(1024 2000 4000 8000 16000 32000)
max_tokens_values=(1024 2000 4000 8000 16000 32000)

### Qwen3-Next: Grader Hacking Correct
for idx in "${!max_thinking_tokens_values[@]}";
do
    max_thinking_tokens=${max_thinking_tokens_values[$idx]}
    max_tokens=${max_tokens_values[$idx]}
    echo "Evaluating with max_thinking_tokens=$max_thinking_tokens and max_tokens=$max_tokens"

    python experiments/prompt.py \
        --input-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_correct_prompts_qwen.jsonl \
        --output-jsonl experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_${max_thinking_tokens}.jsonl \
        --model-type openrouter/qwen/qwen3-next-80b-a3b-thinking \
        --client-type openrouter \
        --api-key $QWEN_API_KEY \
        --temperature $QWEN_TEMPERATURE \
        --max-tokens $max_tokens \
        --max-thinking-tokens $max_thinking_tokens \
        --max-concurrent-requests 20 \
        --examples-per-write 200

    python experiments/evaluate_multiple_choice.py \
        experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_${max_thinking_tokens}.jsonl \
        experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_qwen3_next_${max_thinking_tokens}_results.json
done