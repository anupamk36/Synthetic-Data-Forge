"""Tests for the multi-provider LLM abstraction layer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.llm_providers import (
    AVAILABLE_PROVIDERS,
    AlchemyProvider,
    ClaudeProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
    get_provider_models,
)


class TestProviderRegistry:
    def test_available_providers(self):
        assert "ollama" in AVAILABLE_PROVIDERS
        assert "claude" in AVAILABLE_PROVIDERS
        assert "openai" in AVAILABLE_PROVIDERS
        assert "gemini" in AVAILABLE_PROVIDERS
        assert "alchemy" in AVAILABLE_PROVIDERS

    def test_get_provider_ollama(self):
        p = get_provider("ollama")
        assert isinstance(p, OllamaProvider)
        assert p.name == "ollama"

    def test_get_provider_claude(self):
        p = get_provider("claude", api_key="test-key")
        assert isinstance(p, ClaudeProvider)
        assert p.name == "claude"

    def test_get_provider_openai(self):
        p = get_provider("openai", api_key="test-key")
        assert isinstance(p, OpenAIProvider)
        assert p.name == "openai"

    def test_get_provider_gemini(self):
        p = get_provider("gemini", api_key="test-key")
        assert isinstance(p, GeminiProvider)
        assert p.name == "gemini"

    def test_get_provider_alchemy(self):
        p = get_provider("alchemy", api_key="test-key", base_url="http://test:8000")
        assert isinstance(p, AlchemyProvider)
        assert p.name == "alchemy"

    def test_get_provider_unknown(self):
        from core.exceptions import LLMError
        with pytest.raises(LLMError, match="Unknown provider"):
            get_provider("unknown_provider")

    def test_get_provider_models_cloud(self):
        models = get_provider_models("claude")
        assert len(models) > 0
        assert any("haiku" in m for m in models)

    def test_get_provider_models_openai(self):
        models = get_provider_models("openai")
        assert len(models) > 0
        assert any("gpt" in m for m in models)


class TestOllamaProvider:
    def test_estimate_cost_is_free(self):
        p = OllamaProvider()
        assert p.estimate_cost({"col": "String"}, 1000) < 0.001

    def test_health_check_offline(self):
        p = OllamaProvider(ollama_url="http://localhost:19999")
        assert p.health_check() is False

    def test_generate_returns_empty_when_offline(self):
        p = OllamaProvider(ollama_url="http://localhost:19999")
        result = p.generate_batch({"name": "String"}, {}, 10)
        assert result == []

    def test_validate_returns_original_when_offline(self):
        p = OllamaProvider(ollama_url="http://localhost:19999")
        rows = [{"name": "Alice"}, {"name": "Bob"}]
        result = p.validate_rows(rows, {"name": "String"})
        assert result == rows


class TestClaudeProvider:
    def test_no_key_health_check(self):
        p = ClaudeProvider(api_key="")
        assert p.health_check() is False

    def test_estimate_cost(self):
        p = ClaudeProvider(api_key="test")
        cost = p.estimate_cost({"name": "String", "age": "Int64"}, 100)
        assert cost > 0

    @patch("core.llm_providers.ClaudeProvider._get_client")
    def test_generate_batch_success(self, mock_client):
        records = [{"name": "Alice", "age": 30}]
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(records))]
        mock_client.return_value.messages.create.return_value = mock_msg

        p = ClaudeProvider(api_key="test-key")
        result = p.generate_batch({"name": "String", "age": "Int64"}, {}, 1)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    @patch("core.llm_providers.ClaudeProvider._get_client")
    def test_generate_batch_failure(self, mock_client):
        mock_client.return_value.messages.create.side_effect = Exception("API error")

        p = ClaudeProvider(api_key="test-key")
        result = p.generate_batch({"name": "String"}, {}, 1)
        assert result == []


class TestOpenAIProvider:
    def test_no_key_health_check(self):
        p = OpenAIProvider(api_key="")
        assert p.health_check() is False

    def test_estimate_cost(self):
        p = OpenAIProvider(api_key="test")
        cost = p.estimate_cost({"name": "String", "age": "Int64"}, 100)
        assert cost > 0

    @patch("core.llm_providers.OpenAIProvider._get_client")
    def test_generate_batch_success(self, mock_client):
        records = [{"name": "Bob", "age": 25}]
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(records)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.return_value.chat.completions.create.return_value = mock_response

        p = OpenAIProvider(api_key="test-key")
        result = p.generate_batch({"name": "String", "age": "Int64"}, {}, 1)
        assert len(result) == 1
        assert result[0]["name"] == "Bob"


class TestGeminiProvider:
    def test_no_key_health_check(self):
        p = GeminiProvider(api_key="")
        assert p.health_check() is False

    def test_estimate_cost(self):
        p = GeminiProvider(api_key="test")
        cost = p.estimate_cost({"name": "String", "age": "Int64"}, 100)
        assert cost > 0


class TestJsonParsing:
    def test_clean_json(self):
        from core.llm_providers import _parse_json_lenient
        data = _parse_json_lenient('[{"a": 1}, {"a": 2}]')
        assert len(data) == 2

    def test_markdown_wrapped(self):
        from core.llm_providers import _parse_json_lenient
        data = _parse_json_lenient('```json\n[{"a": 1}]\n```')
        assert len(data) == 1

    def test_truncated_array(self):
        from core.llm_providers import _parse_json_lenient
        data = _parse_json_lenient('[{"a": 1}, {"a": 2}, {"a":')
        assert data is not None
        assert len(data) == 2

    def test_single_object(self):
        from core.llm_providers import _parse_json_lenient
        data = _parse_json_lenient('{"a": 1}')
        assert len(data) == 1

    def test_garbage(self):
        from core.llm_providers import _parse_json_lenient
        data = _parse_json_lenient('not json at all')
        assert data is None


class TestAlchemyProvider:
    def test_no_key_health_check(self):
        p = AlchemyProvider(api_key="", base_url="http://test:8000")
        assert p.health_check() is False

    def test_estimate_cost(self):
        p = AlchemyProvider(api_key="test", base_url="http://test:8000")
        cost = p.estimate_cost({"name": "String", "age": "Int64"}, 100)
        assert cost > 0
        assert cost < 1.0

    def test_default_model(self):
        p = AlchemyProvider(api_key="test", base_url="http://test:8000")
        assert p.model == "gemini-2.5-flash"

    @patch("core.llm_providers.AlchemyProvider._get_client")
    def test_generate_batch_success(self, mock_client):
        records = [{"name": "Alice", "age": 30}]
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(records)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.return_value.chat.completions.create.return_value = mock_response

        p = AlchemyProvider(api_key="test-key", base_url="http://test:8000")
        result = p.generate_batch({"name": "String", "age": "Int64"}, {}, 1)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

        call_kwargs = mock_client.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["user"] == "ForgeFlow_AI"

    @patch("core.llm_providers.AlchemyProvider._get_client")
    def test_generate_batch_failure(self, mock_client):
        mock_client.return_value.chat.completions.create.side_effect = Exception("API error")

        p = AlchemyProvider(api_key="test-key", base_url="http://test:8000")
        result = p.generate_batch({"name": "String"}, {}, 1)
        assert result == []

    @patch("core.config.LANGFUSE_PUBLIC_KEY", "pk-test")
    @patch("core.config.LANGFUSE_SECRET_KEY", "sk-test")
    def test_langfuse_header_added(self):
        import base64
        p = AlchemyProvider(api_key="test-key", base_url="http://test:8000")
        p._client = None
        with patch("openai.OpenAI") as mock_openai:
            p._get_client()
            call_kwargs = mock_openai.call_args[1]
            assert "x-langfuse-auth" in call_kwargs["default_headers"]
            expected = base64.b64encode(b"pk-test:sk-test").decode()
            assert call_kwargs["default_headers"]["x-langfuse-auth"] == f"Basic {expected}"

    @patch("core.config.LANGFUSE_PUBLIC_KEY", "")
    @patch("core.config.LANGFUSE_SECRET_KEY", "")
    def test_langfuse_header_skipped_without_keys(self):
        p = AlchemyProvider(api_key="test-key", base_url="http://test:8000")
        p._client = None
        with patch("openai.OpenAI") as mock_openai:
            p._get_client()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["default_headers"] is None
