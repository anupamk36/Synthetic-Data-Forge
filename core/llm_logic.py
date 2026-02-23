"""
LLM-Powered Business Logic Injection Engine.

Translates natural language rules into Python filter functions
using a local Ollama LLM instance (via Docker).

Includes a fallback regex-based rule parser for common patterns
so simple rules work even without the LLM.
"""

import requests
import json
import re
import polars as pl


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"

SYSTEM_PROMPT = """You are a data validation code generator. Given a natural language rule about data columns, 
return ONLY a Python lambda function that takes a dictionary (row) and returns True if the row satisfies the rule.

STRICT RULES:
- Return ONLY the raw lambda on a single line. No explanation, no markdown, no backticks.
- Column names are dictionary keys: row['column_name']
- For date comparisons, values are datetime.date objects.
- For string operations, use standard Python str methods (.endswith, .startswith, .lower, etc.)
- Start your response with: lambda row:

Examples:
Rule: "discount_price must be less than original_price"
lambda row: row['discount_price'] < row['original_price']

Rule: "email must end with .com"
lambda row: str(row['email']).endswith('.com')

Rule: "ship_date must be at least 2 days after order_date"
lambda row: (row['ship_date'] - row['order_date']).days >= 2

Rule: "age must be between 18 and 65"
lambda row: 18 <= row['age'] <= 65

Rule: "name must not be empty"
lambda row: len(str(row['name'])) > 0

Rule: "status must be either active or pending"
lambda row: row['status'] in ['active', 'pending']
"""


# --- Fallback rule patterns (no LLM needed) ---
FALLBACK_PATTERNS = [
    # "X must end with Y" / "X must end in Y"
    (r"(\w+).*?(?:must|should)\s+end\s+(?:with|in)\s+[\"']?([^\"']+)[\"']?",
     lambda col, val: f"lambda row: str(row['{col}']).endswith('{val.strip()}')"),

    # "X must start with Y"
    (r"(\w+).*?(?:must|should)\s+start\s+with\s+[\"']?([^\"']+)[\"']?",
     lambda col, val: f"lambda row: str(row['{col}']).startswith('{val.strip()}')"),

    # "X must contain Y"
    (r"(\w+).*?(?:must|should)\s+contain\s+[\"']?([^\"']+)[\"']?",
     lambda col, val: f"lambda row: '{val.strip()}' in str(row['{col}'])"),

    # "X must be greater than N"
    (r"(\w+).*?(?:must|should)\s+be\s+(?:greater|more)\s+than\s+(\d+(?:\.\d+)?)",
     lambda col, val: f"lambda row: row['{col}'] > {val}"),

    # "X must be less than N"
    (r"(\w+).*?(?:must|should)\s+be\s+(?:less|smaller)\s+than\s+(\d+(?:\.\d+)?)",
     lambda col, val: f"lambda row: row['{col}'] < {val}"),

    # "X must be between A and B"
    (r"(\w+).*?(?:must|should)\s+be\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)",
     lambda col, lo, hi: f"lambda row: {lo} <= row['{col}'] <= {hi}"),

    # "X must not be empty"
    (r"(\w+).*?(?:must|should)\s+not\s+be\s+empty",
     lambda col: f"lambda row: len(str(row['{col}'])) > 0"),

    # "X must be less than Y" (column comparison)
    (r"(\w+).*?(?:must|should)\s+be\s+(?:less|lower|smaller)\s+than\s+(\w+)",
     lambda col1, col2: f"lambda row: row['{col1}'] < row['{col2}']"),

    # "X must be greater than Y" (column comparison)
    (r"(\w+).*?(?:must|should)\s+be\s+(?:greater|higher|more)\s+than\s+(\w+)",
     lambda col1, col2: f"lambda row: row['{col1}'] > row['{col2}']"),

    # "X must equal Y" / "X must be Y"
    (r"(\w+).*?(?:must|should)\s+(?:equal|be)\s+[\"']([^\"']+)[\"']",
     lambda col, val: f"lambda row: str(row['{col}']) == '{val}'"),
]


class LLMLogicEngine:
    """Translates natural language rules into executable filters via Ollama."""

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

    def _try_fallback(self, rule_text: str, schema: dict) -> str:
        """Try to match the rule against known patterns without needing the LLM."""
        rule_lower = rule_text.lower().strip()

        for pattern_entry in FALLBACK_PATTERNS:
            pattern = pattern_entry[0]
            builder = pattern_entry[1]

            match = re.search(pattern, rule_lower, re.IGNORECASE)
            if match:
                groups = match.groups()
                # Validate that matched column names exist in schema
                col_name = groups[0]
                # Find actual column name (case-insensitive)
                actual_col = None
                for schema_col in schema:
                    if schema_col.lower() == col_name.lower():
                        actual_col = schema_col
                        break

                if actual_col:
                    # Rebuild groups with actual column name
                    adjusted = (actual_col,) + groups[1:]
                    try:
                        return builder(*adjusted)
                    except TypeError:
                        pass

        return None

    def translate_rule(self, rule_text: str, schema: dict) -> str:
        """
        Translate a natural language rule into a lambda string.
        
        Tries fallback pattern matching first, then falls back to LLM.
        """
        # 1. Try fallback patterns first (fast, no LLM needed)
        fallback = self._try_fallback(rule_text, schema)
        if fallback:
            return fallback

        # 2. Try LLM
        if not self.is_available():
            return None

        column_info = ", ".join([f"{col} ({dtype})" for col, dtype in schema.items()])
        prompt = f"{SYSTEM_PROMPT}\n\nAvailable columns: {column_info}\n\nRule: {rule_text}\n"

        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 150},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                raw = resp.json().get("response", "").strip()
                result = self._extract_lambda(raw)
                if result:
                    return result
        except (requests.ConnectionError, requests.Timeout):
            pass

        return None

    def _extract_lambda(self, raw_text: str) -> str:
        """Extract a clean lambda expression from LLM output."""
        if not raw_text:
            return None

        # Strip common markdown wrappers
        text = raw_text.strip()

        # Remove markdown code fences: ```python ... ``` or ``` ... ```
        text = re.sub(r"```(?:python)?\s*\n?", "", text)
        text = re.sub(r"\n?```", "", text)

        # Remove inline backticks
        text = text.strip("`").strip()

        # Remove "Output:" prefix if present
        text = re.sub(r"^(?:Output\s*:\s*)", "", text, flags=re.IGNORECASE).strip()

        # Try to find any lambda in the cleaned text
        match = re.search(r"(lambda\s+row\s*:.+?)(?:\n|$)", text)
        if match:
            result = match.group(1).strip()
            # Remove trailing backticks or quotes
            result = result.rstrip("`'\"")
            return result

        # Check if the entire cleaned text is a lambda
        if text.startswith("lambda row:"):
            return text.split("\n")[0].strip().rstrip("`'\"")

        return None

    def compile_rule(self, lambda_str: str):
        """Safely compile a lambda string into a callable function."""
        if not lambda_str or "lambda row:" not in lambda_str:
            return None
        try:
            # Allow access to built-in functions needed for string/date operations
            safe_builtins = {
                "str": str,
                "int": int,
                "float": float,
                "len": len,
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "bool": bool,
                "list": list,
                "set": set,
                "True": True,
                "False": False,
                "None": None,
            }
            fn = eval(lambda_str, {"__builtins__": safe_builtins}, {})
            return fn
        except Exception:
            return None

    def apply_rules(self, df: pl.DataFrame, rules: list, schema: dict) -> tuple:
        """
        Apply a list of natural language rules to a DataFrame.

        Instead of filtering out non-compliant rows (which can result in 0 records),
        this method keeps compliant rows and regenerates non-compliant ones
        up to a maximum number of retries.

        Returns: (result_df, results_log)
        """
        from core.generator import ForgeEngine

        results = []
        compiled_rules = []

        # Compile all rules first
        for rule_text in rules:
            rule_text = rule_text.strip()
            if not rule_text:
                continue

            lambda_str = self.translate_rule(rule_text, schema)
            if not lambda_str:
                results.append({
                    "rule": rule_text,
                    "lambda": None,
                    "success": False,
                    "error": "Could not translate rule (LLM unreachable and no fallback pattern matched)",
                    "rows_regenerated": 0,
                })
                continue

            fn = self.compile_rule(lambda_str)
            if not fn:
                results.append({
                    "rule": rule_text,
                    "lambda": lambda_str,
                    "success": False,
                    "error": f"Failed to compile: {lambda_str}",
                    "rows_regenerated": 0,
                })
                continue

            compiled_rules.append((rule_text, lambda_str, fn))

        if not compiled_rules:
            return df, results

        # Apply all rules: keep compliant rows, regenerate non-compliant ones
        engine = ForgeEngine()
        target_count = len(df)
        max_retries = 5

        rows = df.to_dicts()
        final_rows = []
        total_regenerated = 0

        for row in rows:
            if self._row_passes_all(row, compiled_rules):
                final_rows.append(row)
            else:
                # Try to regenerate a compliant row
                replaced = False
                for attempt in range(max_retries):
                    new_row_df = engine.generate_records(schema, 1)
                    new_row = new_row_df.to_dicts()[0]
                    if self._row_passes_all(new_row, compiled_rules):
                        final_rows.append(new_row)
                        total_regenerated += 1
                        replaced = True
                        break
                if not replaced:
                    # Keep original row if we can't find a compliant one
                    final_rows.append(row)

        # If we still need more rows (shouldn't happen, but safety net)
        result_df = pl.DataFrame(final_rows) if final_rows else df.clear()

        for rule_text, lambda_str, fn in compiled_rules:
            # Count how many rows in the final result pass this rule
            passing = sum(1 for r in final_rows if self._safe_check(r, fn))
            results.append({
                "rule": rule_text,
                "lambda": lambda_str,
                "success": True,
                "error": None,
                "rows_regenerated": total_regenerated,
                "compliance_rate": f"{round(100 * passing / len(final_rows), 1)}%" if final_rows else "N/A",
            })

        return result_df, results

    def _row_passes_all(self, row: dict, compiled_rules: list) -> bool:
        """Check if a row passes all compiled rules."""
        for _, _, fn in compiled_rules:
            if not self._safe_check(row, fn):
                return False
        return True

    def _safe_check(self, row: dict, fn) -> bool:
        """Safely check a row against a rule function."""
        try:
            return bool(fn(row))
        except Exception:
            return False

