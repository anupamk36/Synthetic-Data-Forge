"""
Multi-provider LLM abstraction for Synthetic-Data-Forge.

Supports Ollama (local), Claude (Anthropic), OpenAI, and Gemini
through a unified interface for data generation and semantic validation.
"""

import json
import logging
import re
from abc import ABC, abstractmethod

import requests

from core import config
from core.exceptions import LLMError

logger = logging.getLogger(__name__)

DATA_GEN_SYSTEM_PROMPT = """Return a JSON array of objects. STRICT RULES:
1. Output ONLY valid JSON. No markdown, no text before or after.
2. Use compact format (no extra whitespace).
3. Generate EXACTLY the requested number of objects, then STOP.
4. Follow field constraints exactly.
5. Ensure cross-column consistency (e.g., age should match job seniority, zip should match city/state)."""

VALIDATION_SYSTEM_PROMPT = """You are a data quality validator. You will receive rows of synthetic data as a JSON array.
Review each row for cross-column logical consistency. Fix ONLY rows with inconsistencies.
Return the corrected JSON array. Do NOT add or remove rows. Output ONLY valid JSON."""


def _build_generation_prompt(
    schema: dict, field_hints: dict, num_records: int, profile_summary: dict | None = None
) -> str:
    field_info = []
    for col, dtype in schema.items():
        hint = field_hints.get(col, "") if field_hints else ""
        if hint:
            field_info.append(f"- {col}: {hint}")
        else:
            field_info.append(f"- {col} ({dtype})")

    prompt = "Fields:\n" + "\n".join(field_info)

    if profile_summary:
        correlations = profile_summary.get("key_correlations", [])
        if correlations:
            prompt += "\n\nKey correlations to preserve:\n"
            for corr in correlations[:10]:
                prompt += f"- {corr}\n"

        constraints = profile_summary.get("constraints", [])
        if constraints:
            prompt += "\nConstraints:\n"
            for c in constraints[:10]:
                prompt += f"- {c}\n"

    prompt += (
        f"\n\nGenerate EXACTLY {num_records} records as a compact JSON array. "
        f"Output ONLY the JSON, stop immediately after the closing bracket ]:\n"
    )
    return prompt


def _parse_json_lenient(text: str) -> list[dict] | None:
    """Parse a JSON array, salvaging partial results if truncated."""
    text = re.sub(r"```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
        return None
    except json.JSONDecodeError:
        pass

    last_brace = text.rfind("}")
    if last_brace != -1:
        candidate = text[: last_brace + 1].rstrip().rstrip(",") + "\n]"
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and data:
                logger.warning("Salvaged %d records from truncated response", len(data))
                return data
        except json.JSONDecodeError:
            pass

    return None


class LLMProvider(ABC):
    """Base class for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'claude', 'openai')."""

    @abstractmethod
    def generate_batch(
        self, schema: dict, field_hints: dict, num_records: int, profile_summary: dict | None = None
    ) -> list[dict]:
        """Generate a batch of records matching the schema."""

    @abstractmethod
    def validate_rows(self, rows: list[dict], schema: dict, profile_summary: dict | None = None) -> list[dict]:
        """Semantically validate/correct rows for cross-column consistency."""

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the provider is reachable and configured."""

    @abstractmethod
    def estimate_cost(self, schema: dict, num_records: int) -> float:
        """Estimate API cost in USD for the given generation task."""

    def _estimate_tokens(self, schema: dict, num_records: int) -> tuple[int, int]:
        """Rough estimate of input/output tokens for a generation call."""
        input_tokens = 200 + len(schema) * 30
        output_tokens = num_records * (50 + len(schema) * 20)
        return input_tokens, output_tokens


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider — free, runs on-device."""

    def __init__(self, model: str | None = None, ollama_url: str | None = None):
        self.ollama_url = (ollama_url or config.OLLAMA_URL).rstrip("/")
        self._available: bool | None = None
        if model:
            self.model = model
        else:
            available = self.get_available_models()
            self.model = available[0] if available else config.DEFAULT_LLM_MODEL

    @property
    def name(self) -> str:
        return "ollama"

    def health_check(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            self._available = resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            self._available = False
        return self._available

    def get_available_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except (requests.ConnectionError, requests.Timeout):
            pass
        return []

    def estimate_cost(self, schema: dict, num_records: int) -> float:
        return 0.0

    def _warm_model(self):
        try:
            requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": "10m"},
                timeout=60,
            )
        except Exception as e:
            logger.warning("Ollama keep-alive ping failed: %s", e)

    def generate_batch(
        self, schema: dict, field_hints: dict, num_records: int, profile_summary: dict | None = None
    ) -> list[dict]:
        if not self.health_check():
            return []

        prompt = (
            DATA_GEN_SYSTEM_PROMPT
            + "\n\n"
            + _build_generation_prompt(schema, field_hints, num_records, profile_summary)
        )

        tokens_per_record = 200 + len(schema) * 50
        num_predict = int(max(4096, num_records * tokens_per_record * 1.5))

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": config.LLM_TEMPERATURE,
                        "num_predict": num_predict,
                        "num_thread": 8,
                    },
                },
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                logger.error("Ollama returned status %d: %s", resp.status_code, resp.text[:500])
                return []

            raw = resp.json().get("response", "").strip()
            if not raw:
                return []

            data = _parse_json_lenient(raw)
            return data if data is not None else []
        except requests.Timeout:
            logger.error("Ollama request timed out after %ds", config.LLM_TIMEOUT_SECONDS)
        except Exception as e:
            logger.error("Ollama request failed: %s", e)
        return []

    def validate_rows(self, rows: list[dict], schema: dict, profile_summary: dict | None = None) -> list[dict]:
        if not self.health_check() or not rows:
            return rows

        prompt = (
            VALIDATION_SYSTEM_PROMPT + "\n\n"
            f"Schema: {json.dumps(schema)}\n\n"
            f"Data to validate:\n{json.dumps(rows)}\n\n"
            "Return the corrected JSON array:"
        )

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {"temperature": 0.1, "num_predict": len(json.dumps(rows)) * 2},
                },
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            if resp.status_code == 200:
                raw = resp.json().get("response", "").strip()
                data = _parse_json_lenient(raw)
                if data and len(data) == len(rows):
                    return data
        except Exception as e:
            logger.warning("Ollama validation failed: %s", e)
        return rows


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider."""

    MODELS = {
        "claude-haiku-4-5-20251001": {"input_cost": 0.80, "output_cost": 4.00},
        "claude-sonnet-4-6-20250514": {"input_cost": 3.00, "output_cost": 15.00},
        "claude-opus-4-6-20250514": {"input_cost": 15.00, "output_cost": 75.00},
    }
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.model = model or self.DEFAULT_MODEL
        self._client = None

    @property
    def name(self) -> str:
        return "claude"

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise LLMError("anthropic package not installed. Run: pip install anthropic") from None
        return self._client

    def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            client.models.list(limit=1)
            return True
        except Exception:
            return False

    def estimate_cost(self, schema: dict, num_records: int) -> float:
        input_tok, output_tok = self._estimate_tokens(schema, num_records)
        pricing = self.MODELS.get(self.model, self.MODELS[self.DEFAULT_MODEL])
        return input_tok / 1_000_000 * pricing["input_cost"] + output_tok / 1_000_000 * pricing["output_cost"]

    def generate_batch(
        self, schema: dict, field_hints: dict, num_records: int, profile_summary: dict | None = None
    ) -> list[dict]:
        client = self._get_client()
        prompt = _build_generation_prompt(schema, field_hints, num_records, profile_summary)

        _, output_tok = self._estimate_tokens(schema, num_records)

        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=min(output_tok * 2, 128000),
                temperature=config.LLM_TEMPERATURE,
                system=DATA_GEN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            data = _parse_json_lenient(raw)
            return data if data is not None else []
        except Exception as e:
            logger.error("Claude generation failed: %s", e)
            return []

    def validate_rows(self, rows: list[dict], schema: dict, profile_summary: dict | None = None) -> list[dict]:
        if not rows:
            return rows
        client = self._get_client()

        prompt = (
            f"Schema: {json.dumps(schema)}\n\n"
            f"Data to validate:\n{json.dumps(rows)}\n\n"
            "Return the corrected JSON array:"
        )

        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=len(json.dumps(rows)) * 3,
                temperature=0.1,
                system=VALIDATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            data = _parse_json_lenient(raw)
            if data and len(data) == len(rows):
                return data
        except Exception as e:
            logger.warning("Claude validation failed: %s", e)
        return rows


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    MODELS = {
        "gpt-4o-mini": {"input_cost": 0.15, "output_cost": 0.60},
        "gpt-4o": {"input_cost": 2.50, "output_cost": 10.00},
        "gpt-4.1-mini": {"input_cost": 0.40, "output_cost": 1.60},
        "gpt-4.1": {"input_cost": 2.00, "output_cost": 8.00},
    }
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or self.DEFAULT_MODEL
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai") from None
        return self._client

    def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False

    def estimate_cost(self, schema: dict, num_records: int) -> float:
        input_tok, output_tok = self._estimate_tokens(schema, num_records)
        pricing = self.MODELS.get(self.model, self.MODELS[self.DEFAULT_MODEL])
        return input_tok / 1_000_000 * pricing["input_cost"] + output_tok / 1_000_000 * pricing["output_cost"]

    def generate_batch(
        self, schema: dict, field_hints: dict, num_records: int, profile_summary: dict | None = None
    ) -> list[dict]:
        client = self._get_client()
        prompt = _build_generation_prompt(schema, field_hints, num_records, profile_summary)
        _, output_tok = self._estimate_tokens(schema, num_records)

        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=min(output_tok * 2, 128000),
                temperature=config.LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": DATA_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            data = _parse_json_lenient(raw)
            return data if data is not None else []
        except Exception as e:
            logger.error("OpenAI generation failed: %s", e)
            return []

    def validate_rows(self, rows: list[dict], schema: dict, profile_summary: dict | None = None) -> list[dict]:
        if not rows:
            return rows
        client = self._get_client()

        prompt = (
            f"Schema: {json.dumps(schema)}\n\n"
            f"Data to validate:\n{json.dumps(rows)}\n\n"
            "Return the corrected JSON array:"
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=len(json.dumps(rows)) * 3,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            data = _parse_json_lenient(raw)
            if data and len(data) == len(rows):
                return data
        except Exception as e:
            logger.warning("OpenAI validation failed: %s", e)
        return rows


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    MODELS = {
        "gemini-2.0-flash": {"input_cost": 0.10, "output_cost": 0.40},
        "gemini-2.5-flash-preview-04-17": {"input_cost": 0.15, "output_cost": 0.60},
        "gemini-2.5-pro-preview-03-25": {"input_cost": 1.25, "output_cost": 10.00},
    }
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or self.DEFAULT_MODEL
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise LLMError("google-genai package not installed. Run: pip install google-genai") from None
        return self._client

    def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            client.models.list(config={"page_size": 1})
            return True
        except Exception:
            return False

    def estimate_cost(self, schema: dict, num_records: int) -> float:
        input_tok, output_tok = self._estimate_tokens(schema, num_records)
        pricing = self.MODELS.get(self.model, self.MODELS[self.DEFAULT_MODEL])
        return input_tok / 1_000_000 * pricing["input_cost"] + output_tok / 1_000_000 * pricing["output_cost"]

    def generate_batch(
        self, schema: dict, field_hints: dict, num_records: int, profile_summary: dict | None = None
    ) -> list[dict]:
        client = self._get_client()
        prompt = (
            DATA_GEN_SYSTEM_PROMPT
            + "\n\n"
            + _build_generation_prompt(schema, field_hints, num_records, profile_summary)
        )

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=min(self._estimate_tokens(schema, num_records)[1] * 2, 65536),
                ),
            )
            raw = response.text.strip()
            data = _parse_json_lenient(raw)
            return data if data is not None else []
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            return []

    def validate_rows(self, rows: list[dict], schema: dict, profile_summary: dict | None = None) -> list[dict]:
        if not rows:
            return rows
        client = self._get_client()

        prompt = (
            VALIDATION_SYSTEM_PROMPT + "\n\n"
            f"Schema: {json.dumps(schema)}\n\n"
            f"Data to validate:\n{json.dumps(rows)}\n\n"
            "Return the corrected JSON array:"
        )

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=len(json.dumps(rows)) * 3,
                ),
            )
            raw = response.text.strip()
            data = _parse_json_lenient(raw)
            if data and len(data) == len(rows):
                return data
        except Exception as e:
            logger.warning("Gemini validation failed: %s", e)
        return rows


class AlchemyProvider(LLMProvider):
    """Alchemy AI Gateway provider — OpenAI-compatible enterprise endpoint."""

    MODELS = {
        "gemini-2.5-flash": {"input_cost": 0.30, "output_cost": 2.50},
        "gemini-2.5-pro": {"input_cost": 1.25, "output_cost": 10.00},
        "gpt-4.1": {"input_cost": 2.00, "output_cost": 8.00},
        "gpt-4.1-mini": {"input_cost": 0.40, "output_cost": 1.60},
        "claude-3.5-sonnet": {"input_cost": 6.00, "output_cost": 30.00},
        "claude-3.7-sonnet": {"input_cost": 3.00, "output_cost": 15.00},
        "claude-4.0-opus": {"input_cost": 15.00, "output_cost": 75.00},
        "claude-4.1-opus": {"input_cost": 15.00, "output_cost": 75.00},
        "claude-4.0-sonnet": {"input_cost": 3.00, "output_cost": 15.00},
        "claude-4.5-sonnet": {"input_cost": 3.30, "output_cost": 16.50},
        "claude-4.5-opus": {"input_cost": 5.50, "output_cost": 27.50},
    }
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or config.ALCHEMY_API_KEY
        self.base_url = (base_url or config.ALCHEMY_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self._client = None
        self._available: bool | None = None

    @property
    def name(self) -> str:
        return "alchemy"

    def _get_client(self):
        if self._client is None:
            try:
                import base64

                import openai

                headers = {}
                if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
                    creds = f"{config.LANGFUSE_PUBLIC_KEY}:{config.LANGFUSE_SECRET_KEY}"
                    encoded = base64.b64encode(creds.encode()).decode()
                    headers["x-langfuse-auth"] = f"Basic {encoded}"
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=f"{self.base_url}/v1",
                    default_headers=headers or None,
                )
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai") from None
        return self._client

    def health_check(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.api_key or not self.base_url:
            self._available = False
            return False
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "user": config.ALCHEMY_USER,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 50,
                },
                timeout=10,
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def get_available_models(self) -> list[str]:
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                return [m["id"] for m in models if "embed" not in m["id"] and "titan" not in m["id"]]
        except Exception as e:
            logger.warning("Failed to fetch available models: %s", e)
        return [self.DEFAULT_MODEL]

    def estimate_cost(self, schema: dict, num_records: int) -> float:
        input_tok, output_tok = self._estimate_tokens(schema, num_records)
        pricing = self.MODELS.get(self.model, self.MODELS[self.DEFAULT_MODEL])
        return input_tok / 1_000_000 * pricing["input_cost"] + output_tok / 1_000_000 * pricing["output_cost"]

    def generate_batch(
        self, schema: dict, field_hints: dict, num_records: int, profile_summary: dict | None = None
    ) -> list[dict]:
        client = self._get_client()
        prompt = _build_generation_prompt(schema, field_hints, num_records, profile_summary)
        _, output_tok = self._estimate_tokens(schema, num_records)

        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=min(output_tok * 2, 128000),
                temperature=config.LLM_TEMPERATURE,
                user=config.ALCHEMY_USER,
                messages=[
                    {"role": "system", "content": DATA_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            data = _parse_json_lenient(raw)
            return data if data is not None else []
        except Exception as e:
            logger.error("Alchemy generation failed: %s", e)
            return []

    def validate_rows(self, rows: list[dict], schema: dict, profile_summary: dict | None = None) -> list[dict]:
        if not rows:
            return rows
        client = self._get_client()

        prompt = (
            f"Schema: {json.dumps(schema)}\n\n"
            f"Data to validate:\n{json.dumps(rows)}\n\n"
            "Return the corrected JSON array:"
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=len(json.dumps(rows)) * 3,
                temperature=0.1,
                user=config.ALCHEMY_USER,
                messages=[
                    {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            data = _parse_json_lenient(raw)
            if data and len(data) == len(rows):
                return data
        except Exception as e:
            logger.warning("Alchemy validation failed: %s", e)
        return rows

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None, temperature: float | None = None):
        """Yield streaming chat completion chunks with tool calling support."""
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature or config.CHAT_TEMPERATURE,
            "user": config.ALCHEMY_USER,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return client.chat.completions.create(**kwargs)

    def chat_complete(self, messages: list[dict], tools: list[dict] | None = None, temperature: float | None = None):
        """Non-streaming chat completion with tool calling support."""
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or config.CHAT_TEMPERATURE,
            "user": config.ALCHEMY_USER,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return client.chat.completions.create(**kwargs)


# Provider registry
_PROVIDERS = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "alchemy": AlchemyProvider,
}

AVAILABLE_PROVIDERS = list(_PROVIDERS.keys())


def get_provider(name: str, api_key: str | None = None, model: str | None = None, **kwargs) -> LLMProvider:
    """Factory function to create a provider by name."""
    cls = _PROVIDERS.get(name.lower())
    if cls is None:
        raise LLMError(f"Unknown provider '{name}'. Available: {AVAILABLE_PROVIDERS}")

    if name.lower() == "ollama":
        return cls(model=model, ollama_url=kwargs.get("ollama_url"))
    if name.lower() == "alchemy":
        return cls(api_key=api_key, model=model, base_url=kwargs.get("base_url"))
    return cls(api_key=api_key, model=model)


def get_provider_models(name: str, api_key: str | None = None) -> list[str]:
    """Return available model names for a provider."""
    cls = _PROVIDERS.get(name.lower())
    if cls is None:
        return []
    if name.lower() == "ollama":
        return OllamaProvider().get_available_models()
    if name.lower() == "alchemy":
        p = AlchemyProvider(api_key=api_key)
        return p.get_available_models()
    return list(cls.MODELS.keys())
