"""Custom exceptions for text processing."""


class OpenAIQuotaExceededError(Exception):
    """Raised when OpenAI API quota is exceeded."""
    pass
