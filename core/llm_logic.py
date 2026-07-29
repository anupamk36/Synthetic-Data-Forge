"""
LLM-powered synthetic data generation engine.

Delegates to the multi-provider abstraction in llm_providers.py.
This module preserves the original LLMLogicEngine API for backward compatibility
while routing through the provider layer.
"""

import logging
import time

from core import config
from core.llm_providers import OllamaProvider, get_provider, get_provider_models
from core.validation import sanitize_field_descriptions

logger = logging.getLogger(__name__)


class LLMLogicEngine:
    """Synthetic data generation engine using configurable LLM providers.

    Defaults to Ollama for backward compatibility. Pass provider_name to use
    a cloud provider (claude, openai, gemini).
    """

    def __init__(self, model: str = None, ollama_url: str = None,
                 provider_name: str = None, api_key: str = None):
        self.provider_name = provider_name or "ollama"

        if self.provider_name == "ollama":
            self._provider = OllamaProvider(model=model, ollama_url=ollama_url)
        else:
            self._provider = get_provider(self.provider_name, api_key=api_key, model=model)

    @property
    def model(self) -> str:
        return self._provider.model

    def is_available(self) -> bool:
        return self._provider.health_check()

    def get_available_models(self) -> list:
        if isinstance(self._provider, OllamaProvider):
            return self._provider.get_available_models()
        return get_provider_models(self.provider_name)

    def estimate_cost(self, schema: dict, count: int) -> float:
        return self._provider.estimate_cost(schema, count)

    def generate_data(self, schema: dict, count: int, field_descriptions: dict = None,
                      progress_callback=None, batch_callback=None,
                      stop_check=None, profile_summary: dict = None) -> list:
        """Generate *count* records using the LLM provider in batches."""
        if not self.is_available():
            return []

        if isinstance(self._provider, OllamaProvider):
            self._provider._warm_model()

        field_descriptions = sanitize_field_descriptions(field_descriptions)
        all_records = []

        # Scale batch size down for complex schemas to avoid token overflow
        base_batch = config.LLM_BATCH_SIZE
        num_cols = len(schema)
        if num_cols > 8:
            batch_size = max(2, base_batch // (num_cols // 4))
        elif num_cols > 5:
            batch_size = max(3, base_batch // 2)
        else:
            batch_size = base_batch

        logger.info("LLM batch size: %d (schema has %d columns)", batch_size, num_cols)
        batches_needed = (count + batch_size - 1) // batch_size

        for i in range(batches_needed):
            if stop_check and stop_check():
                logger.info("LLM generation stopped by user at %d/%d records", len(all_records), count)
                break

            current_batch_size = min(batch_size, count - len(all_records))
            if current_batch_size <= 0:
                break

            logger.info("LLM batch %d/%d (%d records) via %s",
                        i + 1, batches_needed, current_batch_size, self.provider_name)

            batch = self._generate_batch_with_retry(
                schema, current_batch_size, field_descriptions, profile_summary
            )
            if not batch:
                logger.warning("LLM batch %d failed. Stopping early.", i + 1)
                break
            all_records.extend(batch)
            if batch_callback:
                batch_callback(batch)
            if progress_callback:
                progress_callback(len(all_records), count)

        logger.info("LLM generation complete: %d records produced via %s",
                     len(all_records), self.provider_name)
        return all_records[:count]

    def validate_rows(self, rows: list[dict], schema: dict,
                      profile_summary: dict = None) -> list[dict]:
        """Semantically validate/correct rows using the LLM provider."""
        return self._provider.validate_rows(rows, schema, profile_summary)

    def _generate_batch_with_retry(self, schema, batch_size,
                                   field_descriptions, profile_summary) -> list:
        for attempt in range(1, config.LLM_MAX_RETRIES + 1):
            batch = self._provider.generate_batch(
                schema, field_descriptions or {}, batch_size, profile_summary
            )
            if batch:
                return batch
            if attempt < config.LLM_MAX_RETRIES:
                wait = 2 ** attempt
                logger.info("Retrying LLM batch in %ds (attempt %d/%d)",
                            wait, attempt, config.LLM_MAX_RETRIES)
                time.sleep(wait)

        # All retries failed at original size — try once more at half size
        if batch_size > 1:
            smaller = max(1, batch_size // 2)
            logger.info("Retrying with reduced batch size: %d → %d", batch_size, smaller)
            batch = self._provider.generate_batch(
                schema, field_descriptions or {}, smaller, profile_summary
            )
            if batch:
                return batch

        return []
