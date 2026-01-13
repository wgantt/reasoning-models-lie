# Reasoning Models Lie

This repository contains companion code for the following paper:

*Reasoning Models Will Blatantly Lie About Their Reasoning*. William Walden (2025).


## Repository Structure

```
├── src/reasoning_models_lie/   # Core code
│   ├── data_loaders/          # Dataset loading utilities (GPQA, MMLU-Pro, etc.)
│   ├── evaluation/            # Evaluation metrics and scoring
│   ├── prompting/             # Model prompting utilities
│   └── prompts/               # Prompt templates
├── experiments/               # Experiment scripts and results
│   ├── gpqa/                  # GPQA benchmark experiments
│   └── mmlu_pro/              # MMLU-Pro benchmark experiments
├── analysis/                  # Analysis scripts and data
│   ├── data/                  # Processed results (CSV files)
│   ├── plots/                 # Generated figures (included in the paper)
│   └── scripts/               # R and Python analysis scripts
└── tests/                     # Unit tests
```

## Experimental Results

If you only care about raw predictions and results from the experiments in the paper (not re-running the experiments yourself), predictions and metrics can be found in `experiments/gpqa/results/` and `experiments/mmlu_pro/results/`. Within each of these directories are the following subdirectories:

- `all_responses/`:
  - `*.jsonl`: task outputs under the baseline (`*baseline*`) and various hinted (`*grader_hacking*|*metadata*|*sycophancy_v1*|*unethical_information*`) conditions.
  - `*_results.json`: task metrics for the corresponding `.jsonl` file.
- `changed_responses/`:
  - `*.jsonl`: For each of the hinted conditions, outputs on the subset of examples where the model's multiple-choice answer changed between the baseline and the given hinted condition.
  - `*stats.json`: Various statistics on the changed examples (explained below).
- `check_verbalization/`:
    - `*.jsonl`: Hint presence and hint usage verbalization results on the correspondingly named file from `changed_responses/`
    - `*results.json`: Verbalization metrics on the correspondingly named `.jsonl` file.

### Interpreting Metrics

#### Benchmark (MCQA) Results

Here's the meaning of the various statistics in the `all_responses/*_results.json` files, which cover performance on the multiple-choice QA benchmarks (GPQA and MMLU-Pro)

- `correct`: total # correctly answered examples
- `total`: total # examples for which predictions were generated
- `parseable`: total # examples for which an answer could be parsed from the model output
- `unparseable`: total # examples for which an answer could *not* be parsed from the model output
- `accuracy_total`: accuracy computed as `correct / total`
- `accuracy_parseable`: accuracy computed as `correct / parseable`

Results in the `bootstrap` dictionary are results for all the same metrics but computed using a 10,000-sample nonparametric bootstrap. The one novel field in the `bootstrap` dictionary, `accuracy_total_ci`, gives the 2.5% (lower) and 97.5% (upper) bounds of the 95% confidence interval for `accuracy_total` based on this bootstrap.

#### Change-to-Hint Stats

Here's the meaning of the various statistics in the `changed_responses/*stats.json` files, which cover information about examples for which a model changed its answer between the baseline and hinted settings:

- `total_common_examples`: Total examples in common between the baseline and hinted results used to produce the corresponding `*changed.jsonl` file (should always equal the total number of examples in the benchmark)
- `skipped_examples`: Examples in the hinted output not found in the baseline output (should always be 0)
- `total_changes`: # examples (out of `total_common_examples`) where the answer changed between the baseline and hinted setting
- `random_change_to_hint_prob`: Probability of changing to the hinted answer by chance (estimated based on the probability that the model switched to a non-hinted answer in the hinted setting)
- `binom_test_statistic`: Test statistic value from the one-sided binomial proportion test (used to determine whether hint usage is above what is expected by chance)
- `binom_test_pvalue`: P-value for the above statistic
- `fraction_changed`: `total_changes` / `total_common_examples`
- `total_to_hinted`: # examples (out of `total_changes`) where the model's response changed *to* the hinted answer
- `fraction_to_hinted_total`: `total_to_hinted` / `total_common_examples`
- `fraction_to_hinted_changed`: `total_to_hinted` / `total_changes`
- `n_bootstrap_samples`: # samples in the nonparametric bootstrap (should be 10,000 by default)
- `confidence_level`: Confidence level used for the nonparametric bootstrap

Variants of these fields with the suffixes `*ci_lower` or `*ci_upper` indicate upper and lower bounds on the corresponding quantity from a nonparametric bootstrap.

#### Verbalization Results

Here's the meaning of the various statistics in the `check_verbalization/*_results.json` files, which cover information about hint verbalization results. The main results are included in a dictionary named `main_results` and results on the same statistics from a 10,000-sample nonparametric bootstrap are included in a sibling dictionary called `bootstrap_results`:

- `total_examples`: The total # examples over which verbalization results are computed (i.e. the total number of examples for the target hinted setting where the model's answer changed from what it was in the baseline setting)
- `verbalized_hint_present_count`: The # examples (out of `total_examples`) for which the CoT was judged to have acknowledged the presence of the hint in the input
- `verbalized_hint_present_percentage`: `verbalized_hint_present_count / total_examples`
- `verbalized_reliance_on_hint_count`: The # examples (out of `total_examples`) for which the CoT was judged to have acknowledged relying on the hint provided in the input
- `verbalized_reliance_on_hint_percentage`: `verbalized_reliance_on_hint_count / total_examples`
- `changed_to_hinted_count`: The # examples (out of `total_examples`) where the model changed its answer specifically to the hinted answer
- `changed_to_hinted_percentage`: `changed_to_hinted_count / total_examples`
- `changed_to_hinted_and_verbalized_hint_present_count`: The # examples (out of `total_examples`) where the model BOTH changed its answer to the hinted answer AND acknowledged the presence of that hint in its CoT
- `changed_to_hinted_and_verbalized_hint_present_percentage`: `changed_to_hinted_and_verbalized_hint_present_count / total_examples`
- `changed_to_hinted_and_verbalized_reliance_on_hint_count`: The # examples (out of `total_examples`) where the model BOTH changed its answer to the hinted answer AND acknowledged relying on the hint in its CoT
- `changed_to_hinted_and_verbalized_reliance_on_hint_percentage`: `changed_to_hinted_and_verbalized_reliance_on_hint_count` / `total_examples`
- `faithfulness_score`: The unnormalized Faithfulness Score (see paper for details)
- `faithfulness_score_normalized`: The normalized Faithfulness Score
- `honesty_score`: The unnormalized Honesty Score (see paper for details)
- `honesty_score_normalized`: The normalized Honesty Score
- `p`: The proportion of `total_examples` that changed to the hinted answer (same as `changed_to_hinted_percentage`)
- `q`: The proportion of `total_examples` that changed to any answer *other than the hinted answer* (1 - `changed_to_hinted_percentage`)
- `z`: The normalization constant (used for both `faithfulness_score_normalized` and `honesty_score_normalized`)

## Reproducing Our Results

If you wish to reproduce our results (in whole or in part), follow the instructions below.

### 1. Configure the Environment

#### Recommended Method: uv

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you have not already, then run the commands below:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
```

#### Alternative Method: pip

If you prefer not to use `uv`, you can install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```


### 2. Configure API keys

Copy `.env.example` to `.env` and add your API keys:

```bash
# Create a new .env file from the provided template
# (see .env.example for which API keys are required):
cp .env.example .env
# Once you've added your API keys:
source .env
```

### 3. Run the Experiments

Steps (i)-(vi) below illustrate how to reproduce our results once the environment and API keys have been configured. Some of the steps are marked as "optional" because you can still conduct a complete, meaningful reproduction of the core results without running these steps. Technically speaking, however, *all* steps are optional: we include all of the files necessary to start from an arbitrary step and complete the reproduction from there. Just beware that each step will overwrite certain of those files.

### i. Construct evaluation prompts (Optional)

We include the prompts we used for obtaining initial benchmark results under the baseline and hinted conditions, but you can optionally (re-)build them yourself by running the follow scripts:

```bash
# GPQA experiments
bash experiments/gpqa/scripts/build_gpqa_evaluation_prompts.sh

# MMLU-Pro experiments
bash experiments/mmlu_pro/scripts/build_mmlu_pro_test200_evaluation_prompts.sh
```

### ii. Obtain benchmark results

Once the prompts have been created, you can obtain results on GPQA and MMLU-Pro using the scripts in the `experiments/gpqa/scripts/` and `experiments/mmlu_pro/scripts` directories. A separate `evaluate_*` script is included for each combination of model, hint type, and hint correctness. For example, to evaluate Claude 4.5 Haiku with incorrect `grader hacking` hints on GPQA, run:

```bash
bash experiments/gpqa/scripts/grader_hacking/evaluate_grader_hacking_incorrect_claude.sh
```

Or to evaluate Kimi K2 Thinking on MMLU-Pro in the baseline setting, run:

```bash
bash experiments/mmlu_pro/scripts/baseline/test200_evaluate_baseline_kimi.sh
```

All of these scripts call the `experiments/prompt.py` Python script and you may find it helpful or even necessary to adjust some of the command line options provided to the Python script by the bash script. See `experiments/prompt.py` for details on these command line options.

(Note also that all of these scripts will consume credits from the relevant model provider.)

### iii. Identify examples with changed responses

With results in both the baseline and hinted settings, you can identify examples where a model changed its answer between the baseline and a given hinted condition using the `experiments/{gpqa,mmlu_pro}/scripts/get_all_changed_responses.sh` scripts:

```bash
# GPQA
bash experiments/gpqa/scripts/get_all_changed_responses.sh

# MMLU-Pro
bash experiments/mmlu_pro/scripts/get_all_changed_responses.sh
```

JSONL files containing the subset of examples with changed responses (i.e. *any* change &mdash; whether to the hinted answer or not) will be written to `experiments/{gpqa,mmlu_pro}/results/changed_responses`, along with JSON files containing relevant statistics.

### iv. Construct verbalization evaluation prompts

Now we need to construct the prompts used to assess hint verbalization:

```bash
# GPQA
bash experiments/gpqa/scripts/build_gpqa_check_verbalization_prompts.sh

# MMLU-Pro
bash experiments/mmlu_pro/scripts/build_mmlu_pro_test200_evaluation_prompts.sh
```

### v. Evaluate verbalization

In the final step, we assess hint verbalization in CoTs from the subset of examples where a model changed its answer from a non-hint answer (in the baseline setting) *to* the hinted answer (in a given hinted setting). We check both whether a model acknowledges the presence of the hint in the input *and* whether it acknowledges relying on the hint. Both of these evaluations are done in a single LLM judge call. Just as in step (ii), there are separate evaluation scripts for each combination of model, hint type, and hint correctness and these scripts are in the same directory as those used to obtain the initial multiple-choice question answering results. To obtain verbalization results for Claude 4.5 Haiku with incorrect `grader hacking` hints on GPQA, run:

```bash
bash experiments/gpqa/scripts/grader_hacking/check_verbalization_grader_hacking_incorrect_claude.sh
```

To do the same for Kimi K2 Thinking on MMLU-Pro with correct `sycophancy` hints, run:

```bash
bash experiments/mmlu_pro/scripts/sycophancy_v1/test200_check_verbalization_sycophancy_v1_correct_kimi.sh
```

(Note: there are no verbalization evaluation scripts for the baseline setting, since verbalization checks only make sense in hinted settings.) The verbalization results will be written to `experiments/{gpqa,mmlu_pro}/results/check_verbalization/`.


## vi. Regenerate plots (Optional)

Optionally, once you have completed steps (i)-(v), you can regenerate plots from the paper.

To regenerate **Figure 2**:

```bash
# Aggregate data on hint usage into a CSV
python analysis/scripts/get_hint_usage_csv.py

# Boot up R for plotting (assuming R is installed)
R

# (From within the R terminal:) Generate Figure 2 plot (saved as analysis/plots/hint_usage_barplot.png)
source analysis/scripts/plot_hint_usage.r
```

To regenerate **Figure 3**:
```bash
# Aggregate data on hint verbalization into a CSV
python analysis/scripts/get_verbalization_csv.py

# Boot up R
R

# (From within the R terminal:) Generate Figure 3 plot (saved as analysis/plots/verbalization_barplot.png)
source analysis/scripts/plot_verbalization.r
```

To regenerate **Figure 4**:
```bash
# Aggregate task accuracy into a CSV
python analysis/scripts/get_accuracy_csv.py

# Boot up R
R

# (From within the R terminal:) Generate Figure 4 plot (saved as analysis/plots/accuracy_barplot.png)
source analysis/scripts/plot_accuracy.r
```

## Supported Models

Below are the models for which inference is supported. Models with a '*' next to them are models for which experimental results are reported in the paper.

### Anthropic Claude
- Claude 3.7 Sonnet
- Claude 4.5 Haiku*
- Claude 4.5 Sonnet

### DeepSeek
- DeepSeek R1

### Qwen
- Qwen3-Next-80B-A3B-Thinking*

### Others
- Kimi K2 Thinking*
- GPT-OSS 120b

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{walden2025reasoning,
  title={Reasoning Models Will Blatantly Lie About Their Reasoning},
  author={Walden, William},
  year={2025}
}
```

## Contributing

This codebase is intended for reproduction of experiments and not for extensive reuse. However, if you discover a problem, please submit it as a GitHub Issue.