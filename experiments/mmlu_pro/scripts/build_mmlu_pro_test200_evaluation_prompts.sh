# CLAUDE

## CORRECT HINTS

### Baseline (no hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_baseline_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --n-samples 200

### Sycophancy v1 (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_sycophancy_v1_correct_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy correct \
    --n-samples 200

### Metadata (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_metadata_correct_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy correct \
    --n-samples 200

### Grader Hacking (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_grader_hacking_correct_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy correct \
    --n-samples 200

### Unethical Information (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_unethical_information_correct_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy correct \
    --n-samples 200

## INCORRECT HINTS

### Sycophancy v1 (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_sycophancy_v1_incorrect_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy random_incorrect \
    --n-samples 200

### Metadata (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_metadata_incorrect_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy random_incorrect \
    --n-samples 200

### Grader Hacking (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_grader_hacking_incorrect_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy random_incorrect \
    --n-samples 200

### Unethical Information (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_unethical_information_incorrect_prompts_claude.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy random_incorrect \
    --n-samples 200

# KIMI

## CORRECT HINTS

## Baseline (no hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_baseline_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --n-samples 200

## Sycophancy v1 (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_sycophancy_v1_correct_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy correct \
    --n-samples 200

## Metadata (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_metadata_correct_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy correct \
    --n-samples 200

## Grader Hacking (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_grader_hacking_correct_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy correct \
    --n-samples 200

## Unethical Information (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_unethical_information_correct_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy correct \
    --n-samples 200

## INCORRECT HINTS

### Sycophancy v1 (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_sycophancy_v1_incorrect_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy random_incorrect \
    --n-samples 200

### Metadata (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_metadata_incorrect_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy random_incorrect \
    --n-samples 200

### Grader Hacking (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_grader_hacking_incorrect_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy random_incorrect \
    --n-samples 200

### Unethical Information (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_unethical_information_incorrect_prompts_kimi.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy random_incorrect \
    --n-samples 200

# QWEN3-NEXT

## CORRECT HINTS

### Baseline (no hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_baseline_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --n-samples 200 \
    --format-for-deepseek

### Sycophancy v1 (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_sycophancy_v1_correct_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy correct \
    --n-samples 200 \
    --format-for-deepseek

### Metadata (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_metadata_correct_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy correct \
    --n-samples 200 \
    --format-for-deepseek

### Grader Hacking (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_grader_hacking_correct_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy correct \
    --n-samples 200 \
    --format-for-deepseek

### Unethical Information (correct hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_unethical_information_correct_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy correct \
    --n-samples 200 \
    --format-for-deepseek

## INCORRECT HINTS

### Sycophancy v1 (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_sycophancy_v1_incorrect_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type sycophancy_v1 \
    --hint-strategy random_incorrect \
    --n-samples 200 \
    --format-for-deepseek

### Metadata (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_metadata_incorrect_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type metadata \
    --hint-strategy random_incorrect \
    --n-samples 200 \
    --format-for-deepseek

### Grader Hacking (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_grader_hacking_incorrect_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type grader_hacking \
    --hint-strategy random_incorrect \
    --n-samples 200 \
    --format-for-deepseek

### Unethical Information (incorrect hint, with verbalization)
python experiments/build_evaluation_prompts.py \
    mmlu_pro \
    test \
    --output-jsonl experiments/mmlu_pro/prompts/evaluate/mmlu_pro_test200_verbalize_unethical_information_incorrect_prompts_qwen.jsonl \
    --setting baseline \
    --verbalize \
    --hint-type unethical_information \
    --hint-strategy random_incorrect \
    --n-samples 200 \
    --format-for-deepseek