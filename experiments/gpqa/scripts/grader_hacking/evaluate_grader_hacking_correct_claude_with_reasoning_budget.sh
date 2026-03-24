#!/bin/bash
CLAUDE_API_KEY=$ANTHROPIC_API_KEY

# Settings from the Anthropic paper
CLAUDE_TEMPERATURE=0.0
CLAUDE_MAX_TOKENS=12000
CLAUDE_MAX_THINKING_TOKENS=10000

max_thinking_tokens_values=(2000 4000 8000 16000)
max_tokens_values=(4000 8000 16000 32000)

### Claude 4.5 Haiku: Grader Hacking 
for idx in "${!max_thinking_tokens_values[@]}";
do
    max_thinking_tokens=${max_thinking_tokens_values[$idx]}
    max_tokens=${max_tokens_values[$idx]}
    echo "Evaluating with max_thinking_tokens=$max_thinking_tokens and max_tokens=$max_tokens"

    python experiments/prompt.py \
        --input-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_correct_prompts_claude.jsonl \
        --output-jsonl experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_${max_thinking_tokens}.jsonl \
        --model-type claude-haiku-4-5-20251001 \
        --api-key $ANTHROPIC_API_KEY \
        --temperature $CLAUDE_TEMPERATURE \
        --max-tokens $max_tokens \
        --max-thinking-tokens $max_thinking_tokens \
        --examples-per-write 200 \
        --max-concurrent-requests 20

    python experiments/evaluate_multiple_choice.py \
        experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_${max_thinking_tokens}.jsonl \
        experiments/gpqa/results/all_responses/gpqa_diamond_verbalize_grader_hacking_correct_claude_4.5_haiku_${max_thinking_tokens}_results.json
done