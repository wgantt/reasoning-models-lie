"""Script for running asynchronous prompting experiments using ReasoningModelClient."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, Any, Optional

# Add parent directory to path to import reasoning_models_lie module
sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoning_models_lie.prompting.prompting_langchain import (
    ReasoningModelClientLangChain,
    LangChainModelType,
    REASONING_EFFORT_VALUES,
    REASONING_SUMMARY_VALUES,
)
from reasoning_models_lie.prompting.prompting_together import (
    ReasoningModelClientTogether,
    TogetherModelType,
)
from reasoning_models_lie.prompting.prompting_utils import ClientType

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


async def process_prompts_async(
    input_jsonl: str,
    output_jsonl: str,
    model_type: LangChainModelType | TogetherModelType,
    client_type: ClientType = ClientType.LANGCHAIN,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_thinking_tokens: int = 4096,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "auto",
    max_retries: int = 3,
    max_concurrent_requests: int = 10,
    examples_per_write: int = 10,
    model_kwargs: Dict[str, Any] = {},
) -> None:
    """
    Process prompts from a JSONL file asynchronously using ReasoningModelClient.

    Reads an input JSONL file where each line contains:
        - instance_id: Unique identifier for the instance
        - system_message: System message for the model
        - user_message: User prompt/question
        - meta: Metadata dictionary (optional)

    Writes an output JSONL file with the same format plus:
        - result: Full result dictionary from the prompting output

    Args:
        input_jsonl: Path to input JSONL file with prompts
        output_jsonl: Path to output JSONL file for results
        model_type: The type of model to use (ModelType enum)
        api_key: API key for the model provider (optional, uses env vars if not set)
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in the response
        max_thinking_tokens: Number of tokens allocated for model "thinking"
        reasoning_effort: Reasoning effort for OpenAI models
        reasoning_summary: Reasoning summary length for OpenAI models
        max_retries: Maximum number of retries for rate limiting
        max_concurrent_requests: Maximum concurrent requests for async operations
        examples_per_write: Number of examples to process before incrementally writing to file
        model_kwargs: Additional provider-specific parameters
    """
    # Read input JSONL file
    LOGGER.info(f"Reading prompts from {input_jsonl}")
    prompts = []
    try:
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Validate required fields
                    if "instance_id" not in data:
                        LOGGER.warning(
                            f"Line {line_num}: Missing 'instance_id', skipping"
                        )
                        continue
                    if "user_message" not in data:
                        LOGGER.warning(
                            f"Line {line_num}: Missing 'user_message', skipping"
                        )
                        continue
                    prompts.append(data)
                except json.JSONDecodeError as e:
                    LOGGER.error(f"Line {line_num}: Failed to parse JSON: {e}")
                    continue
    except FileNotFoundError:
        LOGGER.error(f"Input file not found: {input_jsonl}")
        raise
    except Exception as e:
        LOGGER.error(f"Error reading input file: {e}")
        raise

    if not prompts:
        LOGGER.warning("No valid prompts found in input file")
        return

    LOGGER.info(f"Loaded {len(prompts)} prompts")

    # Initialize ReasoningModelClient
    LOGGER.info(f"Initializing ReasoningModelClient with model {model_type.value}")
    try:
        if client_type == ClientType.TOGETHER:
            assert isinstance(
                model_type, TogetherModelType
            ), "Model type must be TogetherModelType for TOGETHER client"
            client = ReasoningModelClientTogether(
                model_type=model_type,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                max_thinking_tokens=max_thinking_tokens,
                max_retries=max_retries,
                max_concurrent_requests=max_concurrent_requests,
                model_kwargs=model_kwargs,
            )
        else:
            assert isinstance(
                model_type, LangChainModelType
            ), "Model type must be LangChainModelType for LANGCHAIN client"
            client = ReasoningModelClientLangChain(
                model_type=model_type,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                max_thinking_tokens=max_thinking_tokens,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                max_retries=max_retries,
                max_concurrent_requests=max_concurrent_requests,
                model_kwargs=model_kwargs,
            )
    except Exception as e:
        LOGGER.error(f"Failed to initialize ReasoningModelClient: {e}")
        raise

    # Process prompts asynchronously with incremental writing
    LOGGER.info(
        f"Processing {len(prompts)} prompts with up to {max_concurrent_requests} concurrent requests"
    )
    LOGGER.info(f"Writing results incrementally every {examples_per_write} examples")

    # Prepare output file (create directory and initialize empty file)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize empty output file
    with open(output_jsonl, "w", encoding="utf-8") as f:
        pass  # Just create/truncate the file

    async def process_single_prompt(prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single prompt and return the result with original data."""
        try:
            result = await client.aprompt(
                user_message=prompt_data["user_message"],
                system_message=prompt_data.get("system_message"),
                conversation_history=None,  # Not included in the input format
            )

            # Combine original data with result
            output_data = prompt_data.copy()
            output_data["result"] = result

            LOGGER.info(
                f"Successfully processed instance_id: {prompt_data['instance_id']}"
            )
            return output_data

        except Exception as e:
            LOGGER.error(
                f"Error processing instance_id {prompt_data['instance_id']}: {e}"
            )
            # Return original data with error information
            output_data = prompt_data.copy()
            output_data["result"] = {"error": str(e), "error_type": type(e).__name__}
            return output_data

    # Process prompts in batches and write incrementally
    total_processed = 0
    for i in range(0, len(prompts), examples_per_write):
        batch = prompts[i : i + examples_per_write]

        # Create tasks for this batch
        tasks = [process_single_prompt(prompt) for prompt in batch]

        # Execute batch tasks and gather results with progress bar
        batch_results = await tqdm_asyncio.gather(
            *tasks, desc=f"Batch {i//examples_per_write + 1}"
        )

        # Write batch results to output file (append mode)
        try:
            with open(output_jsonl, "a", encoding="utf-8") as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")

            total_processed += len(batch_results)
            LOGGER.info(
                f"Wrote {len(batch_results)} results to {output_jsonl} ({total_processed}/{len(prompts)} total)"
            )

        except Exception as e:
            LOGGER.error(f"Error writing batch results to output file: {e}")
            raise

    LOGGER.info(
        f"Successfully processed and wrote all {total_processed} results to {output_jsonl}"
    )


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run asynchronous prompting experiments using ReasoningModelClient"
    )

    # Required arguments
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file with prompts",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file for results",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=[
            "gpt-5-nano-2025-08-07",
            "gpt-5-mini-2025-08-07",
            "gpt-5-2025-08-07",
            "gpt-5.2-2025-12-11",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-3-7-sonnet-20250219",
            "deepseek-ai/DeepSeek-R1",
            "moonshotai/Kimi-K2-Thinking",
            "openai/gpt-oss-120b",
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
        ],
        help="Model type to use for prompting",
    )
    parser.add_argument(
        "--client-type",
        type=str,
        default="langchain",
        choices=["langchain", "together"],
        help="Client type to use (langchain or together). Default: langchain",
    )
    # Optional ReasoningModelClient arguments
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the model provider (optional, uses environment variables if not set)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 to 1.0). Default: 0.7",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens in the response. Default: 4096",
    )
    parser.add_argument(
        "--max-thinking-tokens",
        type=int,
        default=2048,
        help="Number of tokens allocated for model 'thinking'. Default: 2048",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=REASONING_EFFORT_VALUES,
        default=None,
        help="Reasoning effort levels for OpenAI models",
    )
    parser.add_argument(
        "--reasoning-summary",
        type=str,
        choices=REASONING_SUMMARY_VALUES,
        default="auto",
        help="Reasoning summary length for OpenAI models",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for rate limiting. Default: 3",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=10,
        help="Maximum concurrent requests for async operations. Default: 5",
    )
    parser.add_argument(
        "--examples-per-write",
        type=int,
        default=20,
        help="Number of examples to process before incrementally writing to output file. Default: 10",
    )

    # Logging level
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Default: INFO",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        help="Additional provider-specific parameters as a JSON string. Default: {}",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Convert model type string to ModelType enum
    model_type_map = {
        "gpt-5-nano-2025-08-07": LangChainModelType.GPT_5_NANO,
        "gpt-5-mini-2025-08-07": LangChainModelType.GPT_5_MINI,
        "gpt-5-2025-08-07": LangChainModelType.GPT_5,
        "gpt-5.2-2025-12-11": LangChainModelType.GPT_5_2,
        "claude-sonnet-4-5-20250929": LangChainModelType.CLAUDE_4_5_SONNET,
        "claude-haiku-4-5-20251001": LangChainModelType.CLAUDE_4_5_HAIKU,
        "claude-3-7-sonnet-20250219": LangChainModelType.CLAUDE_3_7_SONNET,
        "deepseek-ai/DeepSeek-R1": TogetherModelType.DEEPSEEK_R1,
        "moonshotai/Kimi-K2-Thinking": TogetherModelType.KIMI_K2_THINKING,
        "openai/gpt-oss-120b": TogetherModelType.GPT_OSS,
        "Qwen/Qwen3-Next-80B-A3B-Thinking": TogetherModelType.QWEN3_NEXT,
    }
    model_type = model_type_map[args.model_type]
    if args.kwargs:
        try:
            model_kwargs = json.loads(args.kwargs)
        except json.JSONDecodeError as e:
            LOGGER.error(f"Failed to parse --kwargs JSON string: {e}")
            sys.exit(1)
        LOGGER.info(f"Using additional model_kwargs: {model_kwargs}")
    else:
        model_kwargs = {}

    client_type = ClientType(args.client_type)

    # Run the async function
    try:
        asyncio.run(
            process_prompts_async(
                input_jsonl=args.input_jsonl,
                output_jsonl=args.output_jsonl,
                model_type=model_type,
                client_type=client_type,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_thinking_tokens=args.max_thinking_tokens,
                reasoning_effort=args.reasoning_effort,
                reasoning_summary=args.reasoning_summary,
                max_retries=args.max_retries,
                max_concurrent_requests=args.max_concurrent_requests,
                examples_per_write=args.examples_per_write,
                model_kwargs=model_kwargs,
            )
        )
        LOGGER.info("Processing completed successfully")
    except KeyboardInterrupt:
        LOGGER.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
