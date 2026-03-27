"""
Core Synthetic Data Generation Engine.

Uses smart Faker providers that detect column semantics from names
(e.g., 'email' → fake.email(), 'phone' → fake.phone_number()).
"""

import logging
import polars as pl
from faker import Faker
import re

from core import config
from core.exceptions import ForgeError

logger = logging.getLogger(__name__)

# Providers flagged as regulated-PII (disabled in PHARMA_SAFE_MODE)
_PII_PROVIDERS = [
    (r"ssn", lambda fake: fake.ssn()),
    (r"credit[-_]?card|card[-_]?num", lambda fake: fake.credit_card_number()),
    (r"iban", lambda fake: fake.iban()),
]

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
    # Financial (only active when NOT in pharma-safe mode)
    (r"currency", lambda fake: fake.currency_code()),
    # Color
    (r"color|colour", lambda fake: fake.color_name()),
]


def _build_providers() -> list:
    """Build the full provider list, respecting PHARMA_SAFE_MODE."""
    providers = list(SMART_PROVIDERS)
    if not config.PHARMA_SAFE_MODE:
        # Insert PII providers after IDs but before financial
        providers.extend(_PII_PROVIDERS)
    else:
        logger.info("PHARMA_SAFE_MODE active — SSN, credit card, IBAN providers disabled")
    return providers


class ForgeEngine:
    """Core synthetic data generation engine with smart column detection."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            import random
            random.seed(seed)
        self._provider_cache = {}
        self._providers = _build_providers()

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
            for pattern, provider in self._providers:
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
        """Generate a DataFrame with *count* rows using smart providers or LLM.

        Optional keyword args:
            progress_callback(done, total) — called after each batch
            stop_check() -> bool           — return True to abort early
        """
        use_llm = kwargs.get("use_llm", False)
        llm_engine = kwargs.get("llm_engine")
        field_descriptions = kwargs.get("field_descriptions")
        progress_callback = kwargs.get("progress_callback")
        stop_check = kwargs.get("stop_check")

        if use_llm and llm_engine:
            records = llm_engine.generate_data(
                schema, count,
                field_descriptions=field_descriptions,
                progress_callback=progress_callback,
                stop_check=stop_check,
            )
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
        
                schema_cols = set(schema.keys())
                normalized = []
                for rec in records:
                    row = {}
                    for col in schema_cols:
                        row[col] = rec.get(col)
                    normalized.append(row)
                try:
                    df = pl.DataFrame(normalized)
                    logger.info("LLM generated %d records successfully", len(df))
                    if progress_callback:
                        progress_callback(len(df), count)
                    return df
                except Exception as e:
                    logger.warning("LLM DataFrame creation failed: %s — falling back to Faker", e)

        # Pre-resolve providers for each column
        providers = {
            col: self._get_provider(col, dtype)
            for col, dtype in schema.items()
        }

        data: list[dict] = []
        # Generate in batches so we can report progress & honour stop requests
        batch_size = max(1, min(500, count // 20 or 1))
        for batch_start in range(0, count, batch_size):
            if stop_check and stop_check():
                logger.info("Generation stopped by user at %d/%d records", len(data), count)
                break
            current = min(batch_size, count - len(data))
            for _ in range(current):
                row = {col: provider(self.fake) for col, provider in providers.items()}
                data.append(row)
            if progress_callback:
                progress_callback(len(data), count)

        logger.info("Faker generated %d records for %d columns", len(data), len(schema))
        return pl.DataFrame(data) if data else pl.DataFrame(schema={col: pl.Utf8 for col in schema})