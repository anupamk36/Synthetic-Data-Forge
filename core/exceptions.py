"""
Custom exception hierarchy for Synthetic-Data-Forge.

Provides structured error types for clear error propagation
and user-friendly error messaging.
"""


class ForgeError(Exception):
    """Base exception for all Synthetic-Data-Forge errors."""


class ValidationError(ForgeError):
    """Invalid input data, schema, or configuration."""


class LLMError(ForgeError):
    """LLM communication or response parsing failure."""


class SinkError(ForgeError):
    """Data output / storage failure (local or S3)."""


class PrivacyError(ForgeError):
    """Privacy scorecard computation failure."""


class RelationalError(ForgeError):
    """Multi-table relationship or DAG errors."""


class ConfigError(ForgeError):
    """Missing or invalid configuration."""
