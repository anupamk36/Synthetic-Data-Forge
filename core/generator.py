"""
Core Synthetic Data Generation Engine.

Uses smart Faker providers that detect column semantics from names
(e.g., 'email' → fake.email(), 'phone' → fake.phone_number()).
"""

import polars as pl
from faker import Faker
import re


# Map column name patterns to Faker providers
# Order matters — more specific patterns should come first
SMART_PROVIDERS = [
    (r"id|index|key", lambda fake: fake.random_int(0, 10000)),
    # Email
    (r"e[-_]?mail", lambda fake: fake.email()),
    # Phone
    (r"phone|mobile|cell|tel", lambda fake: fake.phone_number()),
    # Name variants
    (r"first[-_]?name", lambda fake: fake.first_name()),
    (r"last[-_]?name|surname", lambda fake: fake.last_name()),
    (r"full[-_]?name|(?:^name$)", lambda fake: fake.name()),
    (r"user[-_]?name", lambda fake: fake.user_name()),
    # Address
    (r"(?:^address$)|street", lambda fake: fake.street_address()),
    (r"city", lambda fake: fake.city()),
    (r"state|province", lambda fake: fake.state()),
    (r"country", lambda fake: fake.country()),
    (r"zip|postal", lambda fake: fake.zipcode()),
    # Internet
    (r"url|website|link", lambda fake: fake.url()),
    (r"ip[-_]?addr|ip$", lambda fake: fake.ipv4()),
    (r"domain", lambda fake: fake.domain_name()),
    # Business
    (r"company|org", lambda fake: fake.company()),
    (r"job|title|position", lambda fake: fake.job()),
    # Text
    (r"description|comment|note|bio|text", lambda fake: fake.sentence()),
    (r"paragraph", lambda fake: fake.paragraph()),
    # IDs
    (r"uuid|guid", lambda fake: fake.uuid4()),
    (r"ssn", lambda fake: fake.ssn()),
    # Financial
    (r"credit[-_]?card|card[-_]?num", lambda fake: fake.credit_card_number()),
    (r"iban", lambda fake: fake.iban()),
    (r"currency", lambda fake: fake.currency_code()),
    # Color
    (r"color|colour", lambda fake: fake.color_name()),
]


class ForgeEngine:
    """Core synthetic data generation engine with smart column detection."""

    def __init__(self):
        self.fake = Faker()
        self._provider_cache = {}

    def _get_provider(self, col_name: str, dtype: str):
        """
        Get the appropriate Faker provider for a column.

        First checks for smart name-based matching, then falls back to dtype.
        """
        cache_key = (col_name.lower(), dtype)
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        # Try smart name-based matching (only for String columns)
        if "String" in dtype or dtype in ("Utf8", "Categorical"):
            col_lower = col_name.lower()
            for pattern, provider in SMART_PROVIDERS:
                if re.search(pattern, col_lower):
                    self._provider_cache[cache_key] = provider
                    return provider

        # Fallback to dtype-based generation
        provider = self._dtype_provider(dtype)
        self._provider_cache[cache_key] = provider
        return provider

    def _dtype_provider(self, dtype: str):
        """Default provider based on data type."""
        if "Int" in dtype:
            return lambda fake: fake.random_int(0, 10000)
        elif "Float" in dtype:
            return lambda fake: fake.pyfloat(right_digits=2, positive=True)
        elif "Date" in dtype:
            return lambda fake: fake.date_this_decade()
        else:
            return lambda fake: fake.word()

    def generate_records(self, schema: dict, count: int, **kwargs) -> pl.DataFrame:
        """Generate a DataFrame with `count` rows using smart providers or LLM."""
        use_llm = kwargs.get("use_llm", False)
        llm_engine = kwargs.get("llm_engine")
        field_descriptions = kwargs.get("field_descriptions")

        if use_llm and llm_engine:
            records = llm_engine.generate_data(schema, count, field_descriptions=field_descriptions)
            if records:
                # Normalize records to match schema (hanld missing/extra keys)
                schema_cols = set(schema.keys())
                normalized = []
                for rec in records:
                    row = {}
                    for col in schema_cols:
                        row[col] = rec.get(col)
                    normalized.append(row)
                try:
                    return pl.DataFrame(normalized)
                except Exception as e:
                    print(f"[Forge] Failed to create DataFrame from LLM data: {e}")
                    print("[Forge] Falling back to Faker...")
        
        # Pre-resolve providers for each column
        providers = {
            col: self._get_provider(col, dtype)
            for col, dtype in schema.items()
        }

        data = []
        for _ in range(count):
            row = {col: provider(self.fake) for col, provider in providers.items()}
            data.append(row)

        return pl.DataFrame(data)