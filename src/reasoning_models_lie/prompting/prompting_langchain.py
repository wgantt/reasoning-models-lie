"""Unified API for prompting reasoning models using LangChain."""

import asyncio
import logging
import os
import re

from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from reasoning_models_lie.prompting.prompting_utils import (
    RESPONSE_REGEX,
    THINKING_TAG_REGEX,
    APIError,
    PromptingError,
    ResponseParsingError,
    ValidationError,
)

LOGGER = logging.getLogger(__name__)


class LangChainModelType(Enum):
    """Supported reasoning model types."""

    CLAUDE_4_5_SONNET = "claude-sonnet-4-5-20250929"
    CLAUDE_4_5_HAIKU = "claude-haiku-4-5-20251001"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"


class ReasoningModelClientLangChain:
    """
    Unified client for interacting with reasoning models via LangChain.

    Supports Claude (Anthropic) and DeepSeek models through a consistent interface.
    """

    def __init__(
        self,
        model_type: LangChainModelType,
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
        if not isinstance(model_type, LangChainModelType):
            raise ValidationError("model_type must be a ModelType enum value")

        if not 0.0 <= temperature <= 1.0:
            raise ValidationError("temperature must be between 0.0 and 1.0")

        if max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")

        if model_type != LangChainModelType.DEEPSEEK_R1:
            # Note: DeepSeek-R1 does not use the thinking tokens parameter
            if max_thinking_tokens < 0:
                raise ValidationError("max_thinking_tokens must be non-negative")

            if max_tokens <= max_thinking_tokens:
                raise ValidationError(
                    "max_tokens must be greater than max_thinking_tokens"
                )

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
            if self.model_type in [
                LangChainModelType.CLAUDE_3_7_SONNET,
                LangChainModelType.CLAUDE_4_5_SONNET,
                LangChainModelType.CLAUDE_4_5_HAIKU,
            ]:
                api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY must be set in environment or passed as api_key parameter"
                    )
                if self.max_thinking_tokens == 0:
                    thinking_dict = {"type": "disabled"}
                else:
                    thinking_dict = {
                        "type": "enabled",
                        "budget_tokens": self.max_thinking_tokens,
                    }
                self.client = ChatAnthropic(
                    model=self.model_type.value,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    thinking=thinking_dict,
                    **self.model_kwargs,
                ).with_retry(
                    stop_after_attempt=self.max_retries,
                    wait_exponential_jitter=True,
                )
            elif model_type in [LangChainModelType.DEEPSEEK_R1]:
                # We access DeepSeek via Together API
                api_key = api_key or os.getenv("TOGETHER_API_KEY")
                if not api_key:
                    raise ValueError(
                        "TOGETHER_API_KEY must be set in environment or passed as api_key parameter"
                    )
                self.client = ChatTogether(
                    model=model_type.value,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **model_kwargs,
                ).with_retry(
                    stop_after_attempt=self.max_retries,
                    wait_exponential_jitter=True,
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except ValueError:
            # Re-raise ValueError for unsupported model types or missing API keys
            raise
        except Exception as e:
            LOGGER.error(f"Failed to initialize model: {e}")
            raise PromptingError(f"Failed to initialize model: {e}") from e

    def prompt(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Send a prompt to the reasoning model and get a response.

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
                messages.append(SystemMessage(content=system_message))

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
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                    else:
                        raise ValidationError(
                            f"Invalid role '{msg['role']}' in conversation_history[{i}]. Must be 'user' or 'assistant'"
                        )

            # Add current user message
            messages.append(HumanMessage(content=user_message))

            # Get response from model
            try:
                response = self.client.invoke(messages)
            except Exception as e:
                LOGGER.error(f"API request failed: {e}")
                raise APIError(f"Failed to get response from model: {e}") from e

            # Validate response
            if not response or not hasattr(response, "content"):
                raise ResponseParsingError("Invalid response received from model")

            # Extract response content and metadata
            try:
                result = {
                    "response": self._extract_response(response),
                    "reasoning_trace": self._extract_reasoning(response),
                    "content": response.content,
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
            LOGGER.error(f"Unexpected error in prompt method: {e}")
            raise PromptingError(f"Unexpected error: {e}") from e

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
                messages.append(SystemMessage(content=system_message))

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
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                    else:
                        raise ValidationError(
                            f"Invalid role '{msg['role']}' in conversation_history[{i}]. Must be 'user' or 'assistant'"
                        )

            # Add current user message
            messages.append(HumanMessage(content=user_message))

            # Get response from model with rate limiting
            try:
                async with self._semaphore:
                    response = await self.client.ainvoke(messages)
            except Exception as e:
                LOGGER.error(f"API request failed: {e}")
                raise APIError(f"Failed to get response from model: {e}") from e

            # Validate response
            if not response or not hasattr(response, "content"):
                raise ResponseParsingError("Invalid response received from model")

            # Extract response content and metadata
            if self.max_thinking_tokens == 0:
                reasoning_trace = None
            else:
                reasoning_trace = self._extract_reasoning(response)
            try:
                result = {
                    "response": self._extract_response(response),
                    "reasoning_trace": reasoning_trace,
                    "content": response.content,
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
            if self.model_type in {
                LangChainModelType.CLAUDE_3_7_SONNET,
                LangChainModelType.CLAUDE_4_5_SONNET,
                LangChainModelType.CLAUDE_4_5_HAIKU,
            }:
                if not response.content or not isinstance(response.content, list):
                    LOGGER.warning("Response content is not in expected list format")
                    return None
                if len(response.content) < 1:
                    LOGGER.warning("Response content list is empty")
                    return None
                if (
                    not isinstance(response.content[0], dict)
                    or "thinking" not in response.content[0]
                ):
                    LOGGER.warning("No 'thinking' key found in response content")
                    return None
                return response.content[0]["thinking"]
            elif self.model_type in {LangChainModelType.DEEPSEEK_R1}:
                if not isinstance(response.content, str):
                    LOGGER.warning("Response content is not a string")
                    return None
                traces = re.findall(THINKING_TAG_REGEX, response.content, re.DOTALL)
                if not traces:
                    LOGGER.warning("No reasoning trace found in the response.")
                    return None
                else:
                    if len(traces) > 1:
                        LOGGER.warning(
                            "Multiple reasoning traces found; using the first one."
                        )
                    return traces[0].strip()
            else:
                raise NotImplementedError(
                    "Reasoning trace extraction not implemented for this model type."
                )
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
            if self.model_type in {
                LangChainModelType.CLAUDE_3_7_SONNET,
                LangChainModelType.CLAUDE_4_5_SONNET,
                LangChainModelType.CLAUDE_4_5_HAIKU,
            }:
                if self.max_thinking_tokens == 0:
                    # When thinking is disabled, response is a string
                    if not isinstance(response.content, str):
                        raise ResponseParsingError("Response content is not a string")
                    return response.content
                elif not response.content or not isinstance(response.content, list):
                    raise ResponseParsingError(
                        "Response content is not in expected list format"
                    )
                if len(response.content) < 2:
                    raise ResponseParsingError(
                        "Response content list has fewer than 2 elements"
                    )
                if (
                    not isinstance(response.content[1], dict)
                    or "text" not in response.content[1]
                ):
                    raise ResponseParsingError(
                        "No 'text' key found in response content[1]"
                    )
                return response.content[1]["text"]
            elif self.model_type in {LangChainModelType.DEEPSEEK_R1}:
                if not isinstance(response.content, str):
                    raise ResponseParsingError("Response content is not a string")
                extracted = re.findall(RESPONSE_REGEX, response.content, re.DOTALL)
                if not extracted:
                    LOGGER.warning("No response content found after reasoning trace.")
                    return ""
                else:
                    return extracted[0].strip()
            else:
                raise NotImplementedError(
                    "Response extraction not implemented for this model type."
                )
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
