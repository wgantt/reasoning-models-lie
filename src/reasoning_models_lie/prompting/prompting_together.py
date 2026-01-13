"""Unified API for prompting reasoning models using LangChain."""

import asyncio
import logging
import os
import re

from enum import Enum
from together import AsyncTogether
from typing import Any, Dict, List, Optional

from reasoning_models_lie.prompting.prompting_utils import (
    RESPONSE_REGEX,
    THINKING_TAG_REGEX,
    APIError,
    PromptingError,
    ResponseParsingError,
    ValidationError,
)

LOGGER = logging.getLogger(__name__)


class TogetherModelType(Enum):
    """Supported reasoning model types."""

    KIMI_K2_THINKING = "moonshotai/Kimi-K2-Thinking"
    GPT_OSS = "openai/gpt-oss-120b"
    DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"
    QWEN3_NEXT = "Qwen/Qwen3-Next-80B-A3B-Thinking"


class ReasoningModelClientTogether:
    """
    Unified client for interacting with reasoning models via Together AI.

    Supports some reasoning models through a consistent interface.
    """

    def __init__(
        self,
        model_type: TogetherModelType,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_thinking_tokens: int = 2048,
        max_retries: int = 3,
        max_concurrent_requests: int = 10,
        model_kwargs: Dict[str, Any] = {},
    ):
        """
        Initialize the reasoning model client.

        Args:
            model_type: The type of model to use
            api_key: API key for the model provider (if not set, uses environment variables)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            max_thinking_tokens: Number of tokens allocated for model "thinking"
            max_retries: Maximum number of retries for rate limiting
            max_concurrent_requests: Maximum concurrent requests for async operations
            model_kwargs: Additional provider-specific parameters

        Raises:
            ValidationError: If invalid parameters are provided
            ValueError: If model type is not supported or API key is missing
        """
        # Validate parameters
        if not isinstance(model_type, TogetherModelType):
            raise ValidationError("model_type must be a ModelType enum value")

        if not 0.0 <= temperature <= 1.0:
            raise ValidationError("temperature must be between 0.0 and 1.0")

        if max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")

        if max_retries < 0:
            raise ValidationError("max_retries must be non-negative")

        if max_concurrent_requests <= 0:
            raise ValidationError("max_concurrent_requests must be positive")

        self.model_type = model_type
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_thinking_tokens = max_thinking_tokens
        self.model_kwargs = model_kwargs

        # Retry configuration
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests

        # Setup logger for retry operations
        self.logger = logging.getLogger(__name__)

        # Create semaphore for rate limiting concurrent requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Initialize the appropriate LangChain model
        try:
            self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "TOGETHER_API_KEY must be set in environment or passed as api_key parameter"
                )
            self.client = AsyncTogether(
                api_key=self.api_key,
                max_retries=self.max_retries,
            )
        except ValueError:
            # Re-raise ValueError for unsupported model types or missing API keys
            raise
        except Exception as e:
            LOGGER.error(f"Failed to initialize model client: {e}")
            raise PromptingError(f"Failed to initialize model client: {e}") from e

    async def aprompt(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronous version of the `prompt` method.

        Args:
            user_message: The user's prompt/question
            system_message: Optional system message to guide the model's behavior
            conversation_history: Optional list of previous messages in the conversation
                                 Format: [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            Dictionary containing:
                - content: The model's response text
                - reasoning_trace: Extracted reasoning trace (if available)
                - metadata: Additional metadata from the response

        Raises:
            ValidationError: If input validation fails
            APIError: If the API request fails
            ResponseParsingError: If response parsing fails
        """
        # Validate inputs
        if not user_message or not user_message.strip():
            raise ValidationError("user_message cannot be empty")

        if conversation_history is not None and not isinstance(
            conversation_history, list
        ):
            raise ValidationError("conversation_history must be a list")

        try:
            messages = []

            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})

            # Add conversation history if provided
            if conversation_history:
                for i, msg in enumerate(conversation_history):
                    if not isinstance(msg, dict):
                        raise ValidationError(
                            f"conversation_history[{i}] must be a dictionary"
                        )
                    if "role" not in msg or "content" not in msg:
                        raise ValidationError(
                            f"conversation_history[{i}] must have 'role' and 'content' keys"
                        )

                    if msg["role"] == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        messages.append(
                            {"role": "assistant", "content": msg["content"]}
                        )
                    else:
                        raise ValidationError(
                            f"Invalid role '{msg['role']}' in conversation_history[{i}]. Must be 'user' or 'assistant'"
                        )

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Get response from model with rate limiting
            try:
                async with self._semaphore:
                    response = await self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_type.value,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        **self.model_kwargs,
                    )
            except Exception as e:
                LOGGER.error(f"API request failed: {e}")
                raise APIError(f"Failed to get response from model: {e}") from e

            # Validate response
            if not response:
                raise ResponseParsingError("Invalid response received from model")

            # Extract response content and metadata
            try:
                result = {
                    "response": self._extract_response(response),
                    "reasoning_trace": self._extract_reasoning(response),
                    "content": response.choices[0].message.content,
                    "metadata": {
                        "model": self.model_type.value,
                        "response_metadata": getattr(response, "response_metadata", {}),
                    },
                }
            except Exception as e:
                LOGGER.error(f"Failed to extract response data: {e}")
                raise ResponseParsingError(f"Failed to parse response: {e}") from e

            return result

        except (ValidationError, APIError, ResponseParsingError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Catch any unexpected errors
            LOGGER.error(f"Unexpected error in aprompt method: {e}")
            raise PromptingError(f"Unexpected error: {e}") from e

    def _extract_reasoning(self, response) -> Optional[str]:
        """
        Extract reasoning trace from model response if available.

        Args:
            response: The model response object

        Returns:
            The extracted reasoning trace, or None if not available

        Raises:
            ResponseParsingError: If extraction fails unexpectedly
        """
        try:
            # for DeepSeek-R1, content and reasoning trace are combined
            if self.model_type in {
                TogetherModelType.DEEPSEEK_R1,
                TogetherModelType.QWEN3_NEXT,
            }:
                content = response.choices[0].message.content
                if not isinstance(content, str):
                    LOGGER.warning("Response content is not a string")
                    return None
                traces = re.findall(THINKING_TAG_REGEX, content.strip(), re.DOTALL)
                if not traces:
                    LOGGER.warning("No reasoning trace found in the response.")
                    return None
                else:
                    if len(traces) > 1:
                        LOGGER.warning(
                            "Multiple reasoning traces found; using the first one."
                        )
                    elif traces[0].startswith("<think>"):
                        # remove leading tag if present
                        traces[0] = traces[0][7:]
                    return traces[0].strip()
            # for other models, content and reasoning trace are separate
            else:
                return response.choices[0].message.reasoning.strip()
        except (KeyError, IndexError, AttributeError) as e:
            LOGGER.error(f"Error extracting reasoning trace: {e}")
            raise ResponseParsingError(f"Failed to extract reasoning trace: {e}") from e
        except Exception as e:
            LOGGER.error(f"Unexpected error extracting reasoning trace: {e}")
            return None

    def _extract_response(self, response) -> str:
        """
        Extract the main response content from the model response.

        Args:
            response: The model response object

        Returns:
            The extracted response text

        Raises:
            ResponseParsingError: If extraction fails
        """
        try:
            content = response.choices[0].message.content
            # for DeepSeek-R1, content and reasoning trace are combined
            if self.model_type in {
                TogetherModelType.DEEPSEEK_R1,
                TogetherModelType.QWEN3_NEXT,
            }:
                if not isinstance(content, str):
                    raise ResponseParsingError("Response content is not a string")
                extracted = re.findall(RESPONSE_REGEX, content.strip(), re.DOTALL)
                if not extracted:
                    LOGGER.warning("No response content found after reasoning trace.")
                    return ""
                else:
                    return extracted[0].strip()
            # for other models, content and reasoning trace are separate
            else:
                return content.strip()
        except (KeyError, IndexError, AttributeError) as e:
            LOGGER.error(f"Error extracting response: {e}")
            raise ResponseParsingError(f"Failed to extract response: {e}") from e
        except ResponseParsingError:
            # Re-raise our custom exception
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error extracting response: {e}")
            raise ResponseParsingError(
                f"Unexpected error during response extraction: {e}"
            ) from e
