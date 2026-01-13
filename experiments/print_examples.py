"""Interactive JSONL Example Browser.

This script loads a JSONL results file (such as those under
``experiments/gpqa/results``) and lets the user page forward and backward through
individual examples. For each example we display the fields needed by the user:

1. ``instance_id``
2. ``problem`` (from the ``meta`` block)
3. ``answer`` (from ``meta``)
4. ``reasoning_trace`` (from ``meta`` if available, otherwise from top level)

Navigation commands:
  n / <Enter>  next example
  p            previous example
  j <index>    jump to a specific example number (1-indexed)
  q            quit the viewer
  ?            show this help within the loop
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

from reasoning_models_lie.evaluation.multiple_choice import parse_multiple_choice_answer


WRAP_WIDTH = 100


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Interactively browse examples from a JSONL results file.",
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to the JSONL results file (e.g. experiments/gpqa/results/...).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-indexed example number to show first (default: 1).",
    )
    return parser.parse_args(argv)


def load_examples(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load and parse the JSONL file.

    Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If any line fails to parse as JSON.
    """

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    examples: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {jsonl_path}: {exc}"
                ) from exc
            if not isinstance(example, dict):
                raise ValueError(
                    f"Expected an object on line {line_number} of {jsonl_path}, got {type(example)}"
                )
            examples.append(example)
    if not examples:
        raise ValueError(f"No examples found in {jsonl_path}")
    return examples


def extract_fields(example: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """Extract the relevant fields from the example, handling missing data."""

    instance_id = str(example.get("instance_id", "<missing>"))

    meta = example.get("meta") or {}
    if not isinstance(meta, dict):  # Guard against unexpected types
        meta = {}

    problem = example.get("user_message")
    # problem = str(meta.get("problem") or meta.get("original_question") or "<missing>")
    gold_answer = str(meta.get("answer", "<missing>"))

    result = example.get("result") or {}
    reasoning_trace = result.get("reasoning_trace")
    if reasoning_trace is None:
        reasoning_trace = example.get("reasoning_trace")
    reasoning = str(reasoning_trace or "<missing>")

    predicted_answer = parse_multiple_choice_answer(result.get("response", ""))
    return instance_id, problem, gold_answer, reasoning, predicted_answer


def print_example(index: int, total: int, example: Dict[str, Any]) -> None:
    """Pretty-print a single example."""

    instance_id, problem, gold_answer, reasoning, predicted_answer = extract_fields(
        example
    )

    header = f"Example {index + 1} of {total}"
    divider = "=" * len(header)
    print(divider)
    print(header)
    print(divider)

    print(f"Instance ID: {instance_id}\n")

    print("Problem:")
    print(textwrap.fill(problem, width=WRAP_WIDTH, replace_whitespace=False))
    print()

    print(f"Gold Answer: {gold_answer}\n")
    print(f"Predicted Answer: {predicted_answer}\n")

    print("Reasoning Trace:")
    print(textwrap.fill(reasoning, width=WRAP_WIDTH, replace_whitespace=False))
    print()


def show_help() -> None:
    """Print available commands."""

    print(
        "Commands:"
        "\n  n / <Enter>  next example"
        "\n  p            previous example"
        "\n  j <index>    jump to example (1-indexed)"
        "\n  q            quit"
        "\n  ?            show this help"
    )


def interactive_loop(examples: List[Dict[str, Any]], start_index: int) -> None:
    """Run the interactive navigation loop."""

    total = len(examples)
    # Clamp the starting position to a valid index.
    position = max(0, min(total - 1, start_index - 1))

    while True:
        print_example(position, total, examples[position])

        command = input("Command [n/p/j/q/?]: ").strip()

        if command == "" or command.lower() == "n":
            position = min(total - 1, position + 1)
        elif command.lower() == "p":
            position = max(0, position - 1)
        elif command.lower().startswith("j"):
            parts = command.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("Invalid jump command. Usage: j <index>")
                continue
            target = int(parts[1]) - 1
            if 0 <= target < total:
                position = target
            else:
                print(f"Index out of range. Enter a value between 1 and {total}.")
        elif command.lower() == "q":
            print("Exiting.")
            return
        elif command == "?":
            show_help()
        else:
            print("Unrecognized command. Type '?' for help.")


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        examples = load_examples(args.jsonl_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        interactive_loop(examples, args.start)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
