"""Utilities and exceptions for prompting language models.

This module provides common utilities, custom exceptions, and constants
used across different prompting implementations.
"""

from enum import Enum


class PromptingError(Exception):
    """Base exception for prompting errors."""

    pass


class APIError(PromptingError):
    """Exception raised for API-related errors."""

    pass


class ResponseParsingError(PromptingError):
    """Exception raised when response parsing fails."""

    pass


class ValidationError(PromptingError):
    """Exception raised for input validation errors."""

    pass


class ClientType(Enum):
    LANGCHAIN = "langchain"
    TOGETHER = "together"


THINKING_TAG_REGEX = r"(.*?)</think>"
RESPONSE_REGEX = r"</think>(.*)"
