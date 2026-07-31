"""
Centralized configuration for Synthetic-Data-Forge.

All settings are loaded from environment variables with sensible defaults.
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

from core.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_URL: str = os.environ.get("FORGE_OLLAMA_URL", "http://localhost:11434")
"""Base URL for the Ollama server (no trailing slash)."""

DEFAULT_LLM_MODEL: str = os.environ.get("FORGE_LLM_MODEL", "qwen2.5:1.5b")
LLM_BATCH_SIZE: int = int(os.environ.get("FORGE_LLM_BATCH_SIZE", "50"))
LLM_TIMEOUT_SECONDS: int = int(os.environ.get("FORGE_LLM_TIMEOUT", "300"))
LLM_TEMPERATURE: float = float(os.environ.get("FORGE_LLM_TEMPERATURE", "0.3"))
LLM_MAX_RETRIES: int = int(os.environ.get("FORGE_LLM_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Cloud LLM Provider API Keys
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.environ.get("FORGE_ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("FORGE_OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.environ.get("FORGE_GEMINI_API_KEY", "")
ALCHEMY_API_KEY: str = os.environ.get("FORGE_ALCHEMY_API_KEY", "")
ALCHEMY_BASE_URL: str = os.environ.get(
    "FORGE_ALCHEMY_BASE_URL", "https://enrichment-dev-nightly.usw2.ds.platform.navify.com"
)
ALCHEMY_USER: str = os.environ.get("FORGE_ALCHEMY_USER", "ForgeFlow_AI")

LANGFUSE_PUBLIC_KEY: str = os.environ.get("FORGE_LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.environ.get("FORGE_LANGFUSE_SECRET_KEY", "")

DEFAULT_LLM_PROVIDER: str = os.environ.get("FORGE_LLM_PROVIDER", "ollama")
"""Default LLM provider: ollama, claude, openai, or gemini."""

# ---------------------------------------------------------------------------
# Chat Agent
# ---------------------------------------------------------------------------
CHAT_PROVIDER: str = os.environ.get("FORGE_CHAT_PROVIDER", "alchemy")
"""LLM provider for the AI chat assistant."""

CHAT_MODEL: str = os.environ.get("FORGE_CHAT_MODEL", "gemini-2.5-flash")
"""Model for chat conversations."""

CHAT_MAX_SESSIONS: int = int(os.environ.get("FORGE_CHAT_MAX_SESSIONS", "100"))
CHAT_SESSION_TTL: int = int(os.environ.get("FORGE_CHAT_SESSION_TTL", "3600"))
CHAT_MAX_TURNS: int = int(os.environ.get("FORGE_CHAT_MAX_TURNS", "50"))
CHAT_TEMPERATURE: float = float(os.environ.get("FORGE_CHAT_TEMPERATURE", "0.4"))

LLM_TOKEN_BUDGET_USD: float = float(os.environ.get("FORGE_LLM_TOKEN_BUDGET", "1.0"))
"""Max spend per generation run in USD. Set 0 for unlimited."""

LLM_VALIDATION_SAMPLE_RATE: float = float(os.environ.get("FORGE_LLM_VALIDATION_SAMPLE_RATE", "1.0"))
"""Fraction of rows to send through LLM semantic validation (0.0-1.0)."""

LLM_VALIDATION_BATCH_SIZE: int = int(os.environ.get("FORGE_LLM_VALIDATION_BATCH_SIZE", "50"))

# ---------------------------------------------------------------------------
# Output / Sinks
# ---------------------------------------------------------------------------
OUTPUT_ROOT: str = os.path.abspath(os.environ.get("FORGE_OUTPUT_ROOT", os.getcwd()))
"""All local output paths must resolve under this directory."""

MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("FORGE_MAX_UPLOAD_SIZE_MB", "200"))

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
PHARMA_SAFE_MODE: bool = os.environ.get("FORGE_PHARMA_SAFE_MODE", "true").lower() in ("1", "true", "yes")
"""When True, disables generation of credit card numbers, SSNs, and other regulated PII."""

# ---------------------------------------------------------------------------
# Privacy scorecard
# ---------------------------------------------------------------------------
DCR_EXACT_THRESHOLD: float = float(os.environ.get("FORGE_DCR_EXACT_THRESHOLD", "0.01"))
DCR_HIGH_RISK_PCT: float = float(os.environ.get("FORGE_DCR_HIGH_RISK_PCT", "5.0"))
DCR_HIGH_RISK_MIN: float = float(os.environ.get("FORGE_DCR_HIGH_RISK_MIN", "0.005"))
DCR_MEDIUM_RISK_PCT: float = float(os.environ.get("FORGE_DCR_MEDIUM_RISK_PCT", "1.0"))
DCR_MEDIUM_RISK_MIN: float = float(os.environ.get("FORGE_DCR_MEDIUM_RISK_MIN", "0.02"))
DCR_MAX_ROWS: int = int(os.environ.get("FORGE_DCR_MAX_ROWS", "5000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("FORGE_LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ .\-]{0,127}$")


def validate_column_name(name: str) -> str:
    """Return *name* unchanged if it matches the safe-name pattern, else raise."""
    if not _SAFE_NAME_RE.match(name):
        raise ConfigError(
            f"Invalid column name '{name}'. "
            "Names must start with a letter or underscore, contain only "
            "alphanumerics/underscores/hyphens/dots/spaces, and be ≤128 chars."
        )
    return name


def validate_output_path(path: str) -> str:
    """Resolve *path* and ensure it lives under OUTPUT_ROOT."""
    resolved = os.path.abspath(os.path.expanduser(path))
    if not resolved.startswith(OUTPUT_ROOT):
        raise ConfigError(
            f"Output path '{resolved}' is outside the allowed root '{OUTPUT_ROOT}'. "
            "Set FORGE_OUTPUT_ROOT to widen the allowed output area."
        )
    return resolved
