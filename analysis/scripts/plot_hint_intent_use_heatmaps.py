"""Generate intent-vs-use heatmaps from per-example check-verbalization results.

This script reads per-example JSONL files from:
  - experiments/gpqa/results/check_verbalization
  - experiments/mmlu_pro/results/check_verbalization

It aggregates counts over hint type for each (dataset, model) pair and produces
heat maps with:
  - x-axis: hint_intent_option (a-e)
  - y-axis: hint_use_option (a-d)

Files containing any excluded budget token in the filename are ignored.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_MAP = {
    "claude_4.5_haiku": "Claude 4.5 Haiku",
    "qwen3_next": "Qwen 3 Next",
    "kimi_k2_5": "Kimi K2.5",
}

DATASET_DIRS = {
    "GPQA": Path("experiments/gpqa/results/check_verbalization"),
    "MMLU": Path("experiments/mmlu_pro/results/check_verbalization"),
}

EXCLUDED_TOKENS = ("1024", "2000", "4000", "8000", "16000", "32000")
INTENT_OPTIONS = ["a", "b", "c", "d", "e"]
USE_OPTIONS = ["a", "b", "c", "d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create hint_intent vs hint_use heatmaps aggregated over hint type "
            "for each model and dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/plots/hint_intent_use_heatmaps"),
        help="Directory to save heatmap images and counts CSV.",
    )
    return parser.parse_args()


def file_is_excluded(file_path: Path) -> bool:
    return any(token in file_path.name for token in EXCLUDED_TOKENS)


def parse_model_name(filename: str) -> str | None:
    for token, display_name in MODEL_MAP.items():
        if token in filename:
            return display_name
    return None


def extract_response_payload(example: dict) -> dict | None:
    result_block = example.get("result", {})
    response = result_block.get("response")

    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        response = response.strip()
        if not response:
            return None
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def collect_counts() -> tuple[dict[tuple[str, str, str, str], int], int, int]:
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    files_used = 0
    examples_used = 0

    for dataset_name, dataset_dir in DATASET_DIRS.items():
        jsonl_files = sorted(dataset_dir.glob("*_check_verbalization.jsonl"))
        for file_path in jsonl_files:
            if file_is_excluded(file_path):
                continue

            model_name = parse_model_name(file_path.name)
            if model_name is None:
                continue

            files_used += 1
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        example = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    payload = extract_response_payload(example)
                    if payload is None:
                        continue

                    hint_intent = str(payload.get("hint_intent", "")).lower()
                    hint_use = str(payload.get("hint_use", "")).lower()

                    if hint_intent not in INTENT_OPTIONS or hint_use not in USE_OPTIONS:
                        continue

                    counts[(dataset_name, model_name, hint_intent, hint_use)] += 1
                    examples_used += 1

    return counts, files_used, examples_used


def build_count_dataframe(
    counts: dict[tuple[str, str, str, str], int],
) -> pd.DataFrame:
    rows = []
    for (dataset, model, hint_intent, hint_use), count in sorted(counts.items()):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "hint_intent_option": hint_intent,
                "hint_use_option": hint_use,
                "count": count,
            }
        )

    return pd.DataFrame(rows)


def save_combined_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "font.weight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    datasets = list(DATASET_DIRS.keys())
    models = list(MODEL_MAP.values())

    # Build count and percentage matrices first.
    count_matrices: dict[tuple[str, str], pd.DataFrame] = {}
    pct_matrices: dict[tuple[str, str], pd.DataFrame] = {}
    for dataset_name in datasets:
        for model_name in models:
            sub = df[(df["dataset"] == dataset_name) & (df["model"] == model_name)]
            matrix = pd.DataFrame(
                0,
                index=USE_OPTIONS,
                columns=INTENT_OPTIONS,
                dtype=int,
            )
            for _, row in sub.iterrows():
                matrix.loc[row["hint_use_option"], row["hint_intent_option"]] = int(
                    row["count"]
                )
            count_matrices[(dataset_name, model_name)] = matrix

            total = int(matrix.values.sum())
            if total > 0:
                pct_matrix = matrix.astype(float) / total * 100.0
            else:
                pct_matrix = matrix.astype(float)
            pct_matrices[(dataset_name, model_name)] = pct_matrix

    n_rows = len(datasets)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    color_ref = None
    for i, dataset_name in enumerate(datasets):
        for j, model_name in enumerate(models):
            ax = axes[i][j]
            count_matrix = count_matrices[(dataset_name, model_name)]
            pct_matrix = pct_matrices[(dataset_name, model_name)]
            total = int(count_matrix.values.sum())
            color_ref = ax.imshow(
                pct_matrix.values,
                cmap="Blues",
                aspect="auto",
                vmin=0,
                vmax=100,
            )

            ax.set_xticks(range(len(INTENT_OPTIONS)))
            ax.set_yticks(range(len(USE_OPTIONS)))
            ax.set_title(
                f"{dataset_name} | {model_name} | n={total}", fontweight="bold"
            )

            for y_idx in range(len(USE_OPTIONS)):
                for x_idx in range(len(INTENT_OPTIONS)):
                    ax.text(
                        x_idx,
                        y_idx,
                        f"{pct_matrix.iat[y_idx, x_idx]:.1f}",
                        ha="center",
                        va="center",
                        color="black" if pct_matrix.iat[y_idx, x_idx] < 50 else "white",
                        fontweight="bold",
                    )

            # Suppress x-axis labels for top row subplots.
            if i == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(INTENT_OPTIONS)

            # Suppress y-axis labels for subplots in 2nd and 3rd columns.
            if j == 0:
                ax.set_yticklabels(USE_OPTIONS)
            else:
                ax.set_yticklabels([])

            # Use shared figure-level axis labels instead of per-subplot labels.
            ax.set_xlabel("")
            ax.set_ylabel("")

            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")

    fig.suptitle(
        "Hint Intent vs Hint Use Percentages (across all hint types)",
        fontsize=20,
        fontweight="bold",
        y=1.04,
    )
    fig.supxlabel("Hint Intent Type", fontsize=16, fontweight="bold")
    fig.supylabel("Hint Use Type", fontsize=16, fontweight="bold")
    cbar = fig.colorbar(color_ref, ax=axes, label="Percentage (%)", pad=0.02)
    cbar.set_label("Percentage (%)", fontsize=16, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    out_file = output_dir / "all_models_datasets_intent_use_heatmaps.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def main() -> None:
    args = parse_args()
    counts, files_used, examples_used = collect_counts()
    df = build_count_dataframe(counts)

    if df.empty:
        raise RuntimeError(
            "No valid per-example hint_intent/hint_use pairs were found."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts_csv = args.output_dir / "hint_intent_use_counts.csv"
    df.to_csv(counts_csv, index=False)

    combined_plot = save_combined_heatmap(df, args.output_dir)

    print(f"Processed files: {files_used}")
    print(f"Processed examples: {examples_used}")
    print(f"Saved counts CSV: {counts_csv}")
    print(f"Saved combined heatmap: {combined_plot}")


if __name__ == "__main__":
    main()
