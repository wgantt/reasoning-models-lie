
# CLAUDE

## CORRECT HiNTS

### Baseline (no hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_baseline_prompts_claude.jsonl \
    --setting baseline \
    --verbalize

### Sycophancy v1 (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_sycophancy_v1_correct_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy correct

### Metadata (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_metadata_correct_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy correct

### Grader Hacking (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_correct_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy correct

### Unethical Information (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_unethical_information_correct_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy correct

## INCORRECT HINTS

### Sycophancy v1 (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_sycophancy_v1_incorrect_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy random_incorrect

### Metadata (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_metadata_incorrect_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy random_incorrect

### Grader Hacking (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_incorrect_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy random_incorrect

### Unethical Information (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_unethical_information_incorrect_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy random_incorrect

# KIMI

## CORRECT HINTS

### Baseline (no hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_baseline_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize

### Sycophancy v1 (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_sycophancy_v1_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy correct

### Metadata (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_metadata_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy correct

### Grader Hacking (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy correct

### Unethical Information (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_unethical_information_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy correct

## INCORRECT HINTS

### Sycophancy v1 (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_sycophancy_v1_incorrect_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy random_incorrect

### Metadata (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_metadata_incorrect_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy random_incorrect

### Grader Hacking (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_incorrect_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy random_incorrect

### Unethical Information (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_unethical_information_incorrect_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy random_incorrect

# QWEN

## CORRECT HINTS
### Baseline (no hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_baseline_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --format-for-deepseek

### Sycophancy v1 (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_sycophancy_v1_correct_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy correct \
    --format-for-deepseek

### Metadata (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_metadata_correct_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy correct \
    --format-for-deepseek

### Grader Hacking (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_correct_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy correct \
    --format-for-deepseek

### Unethical Information (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_unethical_information_correct_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy correct \
    --format-for-deepseek

## INCORRECT HINTS
### Sycophancy v1 (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_sycophancy_v1_incorrect_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy random_incorrect \
    --format-for-deepseek

### Metadata (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_metadata_incorrect_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy random_incorrect \
    --format-for-deepseek

### Grader Hacking (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_grader_hacking_incorrect_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy random_incorrect \
    --format-for-deepseek

### Unethical Information (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    gpqa \
    gpqa_diamond \
    --output-jsonl experiments/gpqa/prompts/evaluate/gpqa_diamond_verbalize_unethical_information_incorrect_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy random_incorrect \
    --format-for-deepseek