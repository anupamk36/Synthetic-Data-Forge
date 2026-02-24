import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"
BATCH_SIZE = 10  # LLM generation is slow, keep batches small

DATA_GEN_SYSTEM_PROMPT = """You are a synthetic data generator. Given a schema and semantic constraints, 
generate realistic, semantically coherent data records.

STRICT RULES:
1. Return ONLY a valid JSON list of objects. No explanation, markdown, or backticks.
2. Each object must match the provided schema.
3. CRITICAL: Follow all semantic descriptions/constraints strictly. 
   Examples of constraints you MUST follow:
   - "must end with .com" -> user@example.com, test@corp.com
   - "Sex: M or F" -> M, F, M, M, F
   - "Price: between 10 and 50" -> 12.5, 45.0, 31.9
4. Ensure cross-column consistency (e.g., 'San Francisco' always implies State 'CA').
5. Generate exactly the number of records requested.
"""


class LLMLogicEngine:
    """Core synthetic data generation engine using Ollama."""

    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            self._available = resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            self._available = False
        return self._available

    def get_available_models(self) -> list:
        """Get list of models pulled in Ollama."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except (requests.ConnectionError, requests.Timeout):
            pass
        return []

    def generate_data(self, schema: dict, count: int, field_descriptions: dict = None) -> list:
        """
        Generate `count` records using the LLM in batches.
        """
        if not self.is_available():
            return []

        all_records = []
        batches_needed = (count + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(batches_needed):
            current_batch_size = min(BATCH_SIZE, count - len(all_records))
            if current_batch_size <= 0:
                break
            
            print(f"[LLM] Generating batch {i+1}/{batches_needed} ({current_batch_size} records)...")
            batch = self._generate_batch(schema, current_batch_size, field_descriptions)
            if not batch:
                print(f"[LLM] Batch {i+1} failed or returned empty. Stopping.")
                break
            all_records.extend(batch)

        print(f"[LLM] Generation complete. Total records: {len(all_records)}")
        return all_records[:count]

    def _generate_batch(self, schema: dict, batch_size: int, field_descriptions: dict = None) -> list:
        """Request a batch of records from the LLM."""
        field_info = []
        for col, dtype in schema.items():
            desc = field_descriptions.get(col, "") if field_descriptions else ""
            if desc:
                # If description is present, prioritize it and remove dtype to avoid confusion
                field_info.append(f"- {col}: {desc}")
            else:
                field_info.append(f"- {col} ({dtype})")

        schema_prompt = "\n".join(field_info)
        
        prompt_parts = [
            DATA_GEN_SYSTEM_PROMPT,
            f"Schema and Field Descriptions:\n{schema_prompt}",
            f"Target Count: {batch_size}"
        ]
            
        prompt = "\n\n".join(prompt_parts) + "\n"

        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 2000},  # Higher temp for variety, larger buffer
                },
                timeout=180,
            )
            if resp.status_code != 200:
                print(f"[LLM] Error: Ollama returned status {resp.status_code}")
                print(f"[LLM] Response body: {resp.text[:500]}")
                return []

            raw = resp.json().get("response", "").strip()
            if not raw:
                print("[LLM] Error: Ollama returned an empty response string.")
                return []

            # Clean markdown backticks if any
            clean_json = re.sub(r"```(?:json)?\s*\n?", "", raw)
            clean_json = re.sub(r"\n?```", "", clean_json).strip()
            
            try:
                data = json.loads(clean_json)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Some LLMs might wrap it in a root object
                    for key in data:
                        if isinstance(data[key], list):
                            return data[key]
                    return [data]
                else:
                    print(f"[LLM] Error: Expected list or dict, got {type(data).__name__}")
            except json.JSONDecodeError as e:
                print(f"[LLM] JSON decode error: {str(e)}")
                print(f"[LLM] Raw response preview: {raw[:500]}...")
        except Exception as e:
            print(f"[LLM] Request failed: {str(e)}")

        return []

