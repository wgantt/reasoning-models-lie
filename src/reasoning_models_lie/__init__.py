"""Reasoning Models Lie: Utilities for prompting reasoning models and analyzing reasoning traces."""

from reasoning_models_lie.prompting.prompting_langchain import (
    LangChainModelType,
    ReasoningModelClientLangChain,
)
from reasoning_models_lie.prompting.prompting_utils import (
    APIError,
    PromptingError,
    ResponseParsingError,
    ValidationError,
)


__version__ = "0.1.0"

__all__ = [
    "ReasoningModelClientLangChain",
    "LangChainModelType",
    "PromptingError",
    "APIError",
    "ResponseParsingError",
    "ValidationError",
]
