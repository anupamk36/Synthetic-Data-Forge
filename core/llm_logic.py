import logging
import requests
import json
import re
import time

from core import config
from core.exceptions import LLMError
from core.validation import sanitize_field_descriptions

logger = logging.getLogger(__name__)

DATA_GEN_SYSTEM_PROMPT = """Return a JSON array of objects. STRICT RULES:
1. Output ONLY valid JSON. No markdown, no text before or after.
2. Use compact format (no extra whitespace).
3. Generate EXACTLY the requested number of objects, then STOP.
4. Follow field constraints exactly.
5. Ensure cross-column consistency."""


class LLMLogicEngine:
    """Core synthetic data generation engine using Ollama."""

    def __init__(self, model: str = None, ollama_url: str = None):
        self.model = model or config.DEFAULT_LLM_MODEL
        self.ollama_url = (ollama_url or config.OLLAMA_URL).rstrip("/")
        self._available = None

    @property
    def _generate_url(self) -> str:
        return f"{self.ollama_url}/api/generate"

    @property
    def _tags_url(self) -> str:
        return f"{self.ollama_url}/api/tags"

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(self._tags_url, timeout=3)
            self._available = resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            self._available = False
        if not self._available:
            logger.warning("Ollama is not reachable at %s", self.ollama_url)
        return self._available

    def get_available_models(self) -> list:
        """Get list of models pulled in Ollama."""
        try:
            resp = requests.get(self._tags_url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except (requests.ConnectionError, requests.Timeout):
            pass
        return []

    def _warm_model(self):
        """Send a tiny no-op request to pre-load the model into memory."""
        try:
            requests.post(
                self._generate_url,
                json={
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": "10m",
                },
                timeout=60,
            )
        except Exception:
            pass  # best-effort; actual generation will fail with a clear error

    def generate_data(self, schema: dict, count: int, field_descriptions: dict = None,
                      progress_callback=None, stop_check=None) -> list:
        """Generate *count* records using the LLM in batches."""
        if not self.is_available():
            return []

        # Pre-warm: load model into GPU/CPU memory before the first real batch
        self._warm_model()

        field_descriptions = sanitize_field_descriptions(field_descriptions)
        all_records = []
        batch_size = config.LLM_BATCH_SIZE
        batches_needed = (count + batch_size - 1) // batch_size

        for i in range(batches_needed):
            if stop_check and stop_check():
                logger.info("LLM generation stopped by user at %d/%d records", len(all_records), count)
                break

            current_batch_size = min(batch_size, count - len(all_records))
            if current_batch_size <= 0:
                break

            logger.info("LLM batch %d/%d (%d records)", i + 1, batches_needed, current_batch_size)
            batch = self._generate_batch_with_retry(schema, current_batch_size, field_descriptions)
            if not batch:
                logger.warning("LLM batch %d failed. Stopping early.", i + 1)
                break
            all_records.extend(batch)
            if progress_callback:
                progress_callback(len(all_records), count)

        logger.info("LLM generation complete: %d records produced", len(all_records))
        return all_records[:count]

    def _generate_batch_with_retry(self, schema, batch_size, field_descriptions) -> list:
        """Retry wrapper around _generate_batch with exponential backoff."""
        for attempt in range(1, config.LLM_MAX_RETRIES + 1):
            batch = self._generate_batch(schema, batch_size, field_descriptions)
            if batch:
                return batch
            if attempt < config.LLM_MAX_RETRIES:
                wait = 2 ** attempt
                logger.info("Retrying LLM batch in %ds (attempt %d/%d)", wait, attempt, config.LLM_MAX_RETRIES)
                time.sleep(wait)
        return []

    @staticmethod
    def _parse_json_lenient(text: str):
        """Parse a JSON array, salvaging partial results if truncated."""
        # 1. Try clean parse first
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

        # 2. Truncated array — find the last complete object and close the array
        last_brace = text.rfind("}")
        if last_brace != -1:
            candidate = text[:last_brace + 1].rstrip().rstrip(",") + "\n]"
            try:
                data = json.loads(candidate)
                if isinstance(data, list) and data:
                    logger.warning("Salvaged %d records from truncated LLM response", len(data))
                    return data
            except json.JSONDecodeError:
                pass

        return None

    def _generate_batch(self, schema: dict, batch_size: int, field_descriptions: dict = None) -> list:
        """Request a batch of records from the LLM."""
        field_info = []
        for col, dtype in schema.items():
            desc = field_descriptions.get(col, "") if field_descriptions else ""
            if desc:
                field_info.append(f"- {col}: {desc}")
            else:
                field_info.append(f"- {col} ({dtype})")

        schema_prompt = "\n".join(field_info)

        prompt = (
            f"{DATA_GEN_SYSTEM_PROMPT}\n\n"
            f"Fields:\n{schema_prompt}\n\n"
            f"Generate EXACTLY {batch_size} records as a compact JSON array. "
            f"Output ONLY the JSON, stop immediately after the closing bracket ]:\n"
        )

        # Scale token budget generously: 200 base + 50 per field, × 1.5 safety margin
        tokens_per_record = 200 + len(schema) * 50
        num_predict = int(max(4096, batch_size * tokens_per_record * 1.5))

        try:
            resp = requests.post(
                self._generate_url,
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
                logger.error("Ollama returned an empty response string.")
                return []

            # Clean markdown backticks if any
            clean_json = re.sub(r"```(?:json)?\s*\n?", "", raw)
            clean_json = re.sub(r"\n?```", "", clean_json).strip()

            # Try to salvage truncated JSON arrays
            data = self._parse_json_lenient(clean_json)
            if data is not None:
                return data
            logger.error("LLM JSON decode error — could not parse response. Preview: %s", raw[:500])
        except requests.Timeout:
            logger.error("LLM request timed out after %ds", config.LLM_TIMEOUT_SECONDS)
        except Exception as e:
            logger.error("LLM request failed: %s", e)

        return []

