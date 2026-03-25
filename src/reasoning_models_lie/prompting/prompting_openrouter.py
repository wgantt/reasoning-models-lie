"""Unified API for prompting reasoning models using OpenRouter."""

import asyncio
import logging
import os
import re

from enum import Enum
from openai import AsyncOpenAI
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModelType(Enum):
    """Supported reasoning model types on OpenRouter."""

    DEEPSEEK_R1 = "deepseek/deepseek-r1"
    DEEPSEEK_R1_DISTILL_QWEN_32B = "deepseek/deepseek-r1-distill-qwen-32b"
    QWEN_QWQ_32B = "qwen/qwq-32b"
    QWEN_3_NEXT_80B_A3B_THINKING = "qwen/qwen3-next-80b-a3b-thinking"
    QWEN_3_5_397B_A17B = "qwen/qwen3.5-397b-a17b"
    KIMI_K2_5 = "moonshotai/kimi-k2.5"
    GEMINI_2_5_PRO = "google/gemini-2.5-pro-preview"
    GPT_OSS_120B = "openai/gpt-oss-120b"


# Models that embed reasoning in <think> tags within content rather than
# returning it as a separate field
_THINK_TAG_MODELS = {
    OpenRouterModelType.QWEN_QWQ_32B,
}


class ReasoningModelClientOpenRouter:
    """
    Unified client for interacting with reasoning models via OpenRouter.

    Supports reasoning models through a consistent interface using the
    OpenAI-compatible OpenRouter API.
    """

    def __init__(
        self,
        model_type: OpenRouterModelType,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        max_concurrent_requests: int = 10,
        http_referer: Optional[str] = None,
        site_name: Optional[str] = None,
        model_kwargs: Dict[str, Any] = {},
    ):
        """
        Initialize the reasoning model client.

        Args:
            model_type: The type of model to use
            api_key: API key for OpenRouter (if not set, uses OPENROUTER_API_KEY env var)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            max_retries: Maximum number of retries for rate limiting
            max_concurrent_requests: Maximum concurrent requests for async operations
            http_referer: Optional site URL for OpenRouter attribution
            site_name: Optional site name for OpenRouter attribution
            model_kwargs: Additional provider-specific parameters

        Raises:
            ValidationError: If invalid parameters are provided
            ValueError: If model type is not supported or API key is missing
        """
        if not isinstance(model_type, OpenRouterModelType):
            raise ValidationError(
                "model_type must be an OpenRouterModelType enum value"
            )

        if not 0.0 <= temperature <= 1.0:
            raise ValidationError("temperature must be between 0.0 and 1.0")

        if max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")

        if max_retries < 0:
            raise ValidationError("max_retries must be non-negative")

        if max_concurrent_requests <= 0:
            raise ValidationError("max_concurrent_requests must be positive")

        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_kwargs = model_kwargs

        # Retry configuration
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests

        # Optional OpenRouter attribution headers
        self.extra_headers: Dict[str, str] = {}
        if http_referer:
            self.extra_headers["HTTP-Referer"] = http_referer
        if site_name:
            self.extra_headers["X-Title"] = site_name

        self.logger = logging.getLogger(__name__)

        # Create semaphore for rate limiting concurrent requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        try:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY must be set in environment or passed as api_key parameter"
                )
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=OPENROUTER_BASE_URL,
                max_retries=self.max_retries,
            )
        except ValueError:
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
                - response: The extracted response text (without reasoning trace)
                - reasoning_trace: Extracted reasoning trace (if available)
                - content: The raw message content from the model
                - metadata: Additional metadata from the response

        Raises:
            ValidationError: If input validation fails
            APIError: If the API request fails
            ResponseParsingError: If response parsing fails
        """
        if not user_message or not user_message.strip():
            raise ValidationError("user_message cannot be empty")

        if conversation_history is not None and not isinstance(
            conversation_history, list
        ):
            raise ValidationError("conversation_history must be a list")

        try:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

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

            messages.append({"role": "user", "content": user_message})

            try:
                async with self._semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model_type.value,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_headers=self.extra_headers or None,
                        **self.model_kwargs,
                    )
            except Exception as e:
                LOGGER.error(f"API request failed: {e}")
                raise APIError(f"Failed to get response from model: {e}") from e

            if not response:
                raise ResponseParsingError("Invalid response received from model")

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
            raise
        except Exception as e:
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
            if self.model_type in _THINK_TAG_MODELS:
                # Models that embed reasoning in <think> tags within content
                content = response.choices[0].message.content
                if content is None:
                    LOGGER.warning("Response content is None")
                    return None
                if not isinstance(content, str):
                    LOGGER.warning("Response content is not a string")
                    return None
                traces = re.findall(THINKING_TAG_REGEX, content.strip(), re.DOTALL)
                if not traces:
                    LOGGER.warning("No reasoning trace found in the response.")
                    return None
                if len(traces) > 1:
                    LOGGER.warning(
                        "Multiple reasoning traces found; using the first one."
                    )
                trace = traces[0]
                if trace.startswith("<think>"):
                    trace = trace[7:]
                return trace.strip()
            else:
                # Most OpenRouter reasoning models return reasoning as a separate field
                reasoning = response.choices[0].message.reasoning
                if reasoning is None:
                    LOGGER.warning("Reasoning field is None")
                    return None
                return reasoning.strip()
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
            The extracted response text (without reasoning trace)

        Raises:
            ResponseParsingError: If extraction fails
        """
        try:
            content = response.choices[0].message.content
            if self.model_type in _THINK_TAG_MODELS:
                if content is None:
                    LOGGER.warning("Response content is None")
                    return ""
                if not isinstance(content, str):
                    raise ResponseParsingError("Response content is not a string")
                extracted = re.findall(RESPONSE_REGEX, content.strip(), re.DOTALL)
                if not extracted:
                    LOGGER.warning("No response content found after reasoning trace.")
                    return ""
                return extracted[0].strip()
            elif content is None:
                LOGGER.warning("Response content is None")
                return ""
            else:
                return content.strip()
        except (KeyError, IndexError, AttributeError) as e:
            LOGGER.error(f"Error extracting response: {e}")
            raise ResponseParsingError(f"Failed to extract response: {e}") from e
        except ResponseParsingError:
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error extracting response: {e}")
            raise ResponseParsingError(
                f"Unexpected error during response extraction: {e}"
            ) from e
