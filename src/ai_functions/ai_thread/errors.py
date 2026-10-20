"""Error types raised by AI function execution."""

from __future__ import annotations


class AIFunctionError(Exception):
    """Base error for AI function execution failures.

    Args:
        message: Human-readable explanation.
        function_name: Name of the ``AIFunction`` that raised.
    """

    def __init__(self, message: str, function_name: str = "") -> None:
        self.function_name = function_name
        super().__init__(message)


class ValidationError(AIFunctionError):
    """Post-condition validation failed for a cycle's result.

    Args:
        function_name: Name of the ``AIFunction`` whose result failed.
        errors: Post-condition name → failure message.

    Ensures:
        The formatted message joins ``errors`` as ``"k: v; ..."``.
    """

    def __init__(self, function_name: str, errors: dict[str, str]) -> None:
        self.validation_errors = errors
        msg = "; ".join(f"{k}: {v}" for k, v in errors.items())
        super().__init__(msg, function_name=function_name)


class PostConditionError(AIFunctionError):
    """A post-condition could not evaluate the result.

    Raised from a post-condition, it ends the cycle with this error instead of
    becoming validation feedback for the model's next attempt.
    """
