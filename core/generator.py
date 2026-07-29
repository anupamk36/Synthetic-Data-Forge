"""
Core Synthetic Data Generation Engine.

Three-stage pipeline:
  1. Statistical generation (Faker + Gaussian copula for correlations)
  2. LLM semantic validation (cloud or local LLM corrects inconsistencies)
  3. Post-validation quality check (distribution drift detection)
"""

import logging
import random
import re

import numpy as np
import polars as pl
from faker import Faker
from scipy import stats as scipy_stats

from core import config
from core.profiler import DataProfile

logger = logging.getLogger(__name__)

# Providers flagged as regulated-PII (disabled in PHARMA_SAFE_MODE)
_PII_PROVIDERS = [
    (r"ssn", lambda fake: fake.ssn()),
    (r"credit[-_]?card|card[-_]?num", lambda fake: fake.credit_card_number()),
    (r"iban", lambda fake: fake.iban()),
]

# Map column name patterns to Faker providers
SMART_PROVIDERS = [
    (r"id|index|key", lambda fake: fake.random_int(0, 10000)),
    (r"e[-_]?mail", lambda fake: fake.email()),
    (r"phone|mobile|cell|tel", lambda fake: fake.phone_number()),
    (r"first[-_]?name", lambda fake: fake.first_name()),
    (r"last[-_]?name|surname", lambda fake: fake.last_name()),
    (r"full[-_]?name|(?:^name$)", lambda fake: fake.name()),
    (r"user[-_]?name", lambda fake: fake.user_name()),
    (r"(?:^address$)|street", lambda fake: fake.street_address()),
    (r"city", lambda fake: fake.city()),
    (r"state|province", lambda fake: fake.state()),
    (r"country", lambda fake: fake.country()),
    (r"zip|postal", lambda fake: fake.zipcode()),
    (r"url|website|link", lambda fake: fake.url()),
    (r"ip[-_]?addr|ip$", lambda fake: fake.ipv4()),
    (r"domain", lambda fake: fake.domain_name()),
    (r"company|org", lambda fake: fake.company()),
    (r"job|title|position", lambda fake: fake.job()),
    (r"description|comment|note|bio|text", lambda fake: fake.sentence()),
    (r"paragraph", lambda fake: fake.paragraph()),
    (r"uuid|guid", lambda fake: fake.uuid4()),
    (r"currency", lambda fake: fake.currency_code()),
    (r"color|colour", lambda fake: fake.color_name()),
]


def _build_providers() -> list:
    providers = list(SMART_PROVIDERS)
    if not config.PHARMA_SAFE_MODE:
        providers.extend(_PII_PROVIDERS)
    else:
        logger.info("PHARMA_SAFE_MODE active — SSN, credit card, IBAN providers disabled")
    return providers


class ForgeEngine:
    """Core synthetic data generation engine with three-stage pipeline."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
        self._rng = np.random.default_rng(seed)
        self._provider_cache = {}
        self._providers = _build_providers()

    def _get_provider(self, col_name: str, dtype: str):
        cache_key = (col_name.lower(), dtype)
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        if "String" in dtype or dtype in ("Utf8", "Categorical"):
            col_lower = col_name.lower()
            for pattern, provider in self._providers:
                if re.search(pattern, col_lower):
                    self._provider_cache[cache_key] = provider
                    return provider

        provider = self._dtype_provider(dtype)
        self._provider_cache[cache_key] = provider
        return provider

    def _dtype_provider(self, dtype: str):
        if "Int" in dtype:
            return lambda fake: fake.random_int(0, 10000)
        elif "Float" in dtype:
            return lambda fake: fake.pyfloat(right_digits=2, positive=True)
        elif "Date" in dtype:
            return lambda fake: fake.date_this_decade()
        else:
            return lambda fake: fake.word()

    def generate_records(self, schema: dict, count: int, **kwargs) -> pl.DataFrame:
        """Generate a DataFrame with *count* rows using the three-stage pipeline.

        Keyword args:
            use_llm (bool): Use LLM for generation (legacy mode)
            llm_engine: LLMLogicEngine instance
            field_descriptions (dict): Column semantic hints
            progress_callback(done, total): Progress reporter
            stop_check() -> bool: Abort signal
            profile (DataProfile): Statistical profile for correlated generation
            enable_validation (bool): Run LLM semantic validation pass (default: True)
            validation_sample_rate (float): Fraction of rows to validate (0.0-1.0)
        """
        use_llm = kwargs.get("use_llm", False)
        llm_engine = kwargs.get("llm_engine")
        field_descriptions = kwargs.get("field_descriptions")
        progress_callback = kwargs.get("progress_callback")
        batch_callback = kwargs.get("batch_callback")
        stop_check = kwargs.get("stop_check")
        profile = kwargs.get("profile")
        enable_validation = kwargs.get("enable_validation", True)
        validation_sample_rate = kwargs.get(
            "validation_sample_rate", config.LLM_VALIDATION_SAMPLE_RATE
        )

        # ── Stage 1: Statistical Generation ──
        if use_llm and llm_engine:
            records = llm_engine.generate_data(
                schema, count,
                field_descriptions=field_descriptions,
                progress_callback=progress_callback,
                batch_callback=batch_callback,
                stop_check=stop_check,
                profile_summary=profile.summary_for_llm() if profile else None,
            )
            if records:
                schema_cols = set(schema.keys())
                normalized = [
                    {col: rec.get(col) for col in schema_cols}
                    for rec in records
                ]
                try:
                    df = pl.DataFrame(normalized)
                    logger.info("LLM generated %d records successfully", len(df))
                    if progress_callback:
                        progress_callback(len(df), count)

                    # Skip validation for LLM-generated data (already coherent)
                    return df
                except Exception as e:
                    logger.warning("LLM DataFrame creation failed: %s — falling back to Faker", e)

        # Faker-based generation, optionally with copula correlations
        if profile and self._has_numeric_correlations(profile):
            df = self._generate_with_copula(schema, count, profile, progress_callback)
        else:
            df = self._generate_faker(schema, count, stop_check, progress_callback, batch_callback)

        # ── Stage 2: LLM Semantic Validation ──
        if enable_validation and llm_engine and llm_engine.is_available() and len(df) > 0:
            df = self._validate_with_llm(
                df, schema, llm_engine, profile,
                sample_rate=validation_sample_rate,
            )

        return df

    def _generate_faker(self, schema: dict, count: int,
                        stop_check=None, progress_callback=None,
                        batch_callback=None) -> pl.DataFrame:
        """Original Faker-based generation (independent columns)."""
        providers = {
            col: self._get_provider(col, dtype)
            for col, dtype in schema.items()
        }

        data: list[dict] = []
        batch_size = max(1, min(500, count // 20 or 1))
        for _ in range(0, count, batch_size):
            if stop_check and stop_check():
                break
            current = min(batch_size, count - len(data))
            batch = []
            for _ in range(current):
                row = {col: provider(self.fake) for col, provider in providers.items()}
                batch.append(row)
            data.extend(batch)
            if batch_callback:
                batch_callback(batch)
            if progress_callback:
                progress_callback(len(data), count)

        logger.info("Faker generated %d records for %d columns", len(data), len(schema))
        return pl.DataFrame(data) if data else pl.DataFrame(schema={col: pl.Utf8 for col in schema})

    def _has_numeric_correlations(self, profile: DataProfile) -> bool:
        return any(
            c.method == "pearson" and c.significant
            for c in profile.correlations
        )

    def _generate_with_copula(self, schema: dict, count: int,
                              profile: DataProfile,
                              progress_callback=None) -> pl.DataFrame:
        """Generate correlated numeric data using a Gaussian copula,
        then fill non-numeric columns with Faker."""
        numeric_cols = [
            cs.name for cs in profile.column_stats
            if cs.is_numeric and cs.name in schema
        ]
        non_numeric_cols = [col for col in schema if col not in numeric_cols]

        data = {}

        if len(numeric_cols) >= 2:
            uniform_samples = self._copula_uniform_samples(profile, numeric_cols, count)
            for idx, col_name in enumerate(numeric_cols):
                cs = next(s for s in profile.column_stats if s.name == col_name)
                values = self._transform_uniform_to_target(uniform_samples[:, idx], cs)
                data[col_name] = self._cast_numeric(values, schema[col_name])
        else:
            for col_name in numeric_cols:
                cs = next(s for s in profile.column_stats if s.name == col_name)
                values = self._sample_single_numeric(cs, count)
                data[col_name] = self._cast_numeric(values, schema[col_name])

        self._fill_non_numeric(data, non_numeric_cols, schema, profile, count)

        if progress_callback:
            progress_callback(count, count)

        logger.info("Copula generated %d records for %d columns (%d correlated)",
                     count, len(schema), len(numeric_cols))
        return pl.DataFrame(data)

    def _copula_uniform_samples(self, profile: DataProfile,
                                numeric_cols: list[str], count: int) -> np.ndarray:
        corr_matrix = np.eye(len(numeric_cols))
        col_idx = {name: i for i, name in enumerate(numeric_cols)}

        for corr in profile.correlations:
            if corr.method == "pearson" and corr.col_a in col_idx and corr.col_b in col_idx:
                i, j = col_idx[corr.col_a], col_idx[corr.col_b]
                corr_matrix[i, j] = corr.value
                corr_matrix[j, i] = corr.value

        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(corr_matrix, 1.0)

        normal_samples = self._rng.multivariate_normal(
            mean=np.zeros(len(numeric_cols)), cov=corr_matrix, size=count,
        )
        return scipy_stats.norm.cdf(normal_samples)

    @staticmethod
    def _transform_uniform_to_target(uniform: np.ndarray, cs) -> np.ndarray:
        if cs.mean is not None and cs.std is not None and cs.std > 0:
            values = scipy_stats.norm.ppf(uniform, loc=cs.mean, scale=cs.std)
            if cs.min_val is not None:
                values = np.clip(values, cs.min_val, cs.max_val)
            return values
        lo = cs.min_val if cs.min_val is not None else 0
        hi = cs.max_val if cs.max_val is not None else 10000
        return uniform * (hi - lo) + lo

    def _sample_single_numeric(self, cs, count: int) -> np.ndarray:
        if cs.mean is not None and cs.std is not None and cs.std > 0:
            values = self._rng.normal(cs.mean, cs.std, count)
            if cs.min_val is not None:
                values = np.clip(values, cs.min_val, cs.max_val)
            return values
        lo = cs.min_val if cs.min_val is not None else 0
        hi = cs.max_val if cs.max_val is not None else 10000
        return self._rng.uniform(lo, hi, count)

    @staticmethod
    def _cast_numeric(values: np.ndarray, dtype: str) -> list:
        if "Int" in dtype:
            return values.astype(int).tolist()
        return np.round(values, 2).tolist()

    def _fill_non_numeric(self, data: dict, cols: list[str], schema: dict,
                          profile: DataProfile, count: int):
        for col_name in cols:
            cs = next((s for s in profile.column_stats if s.name == col_name), None)
            if cs and cs.is_categorical and cs.top_values:
                values = [v["value"] for v in cs.top_values]
                weights = [v["count"] for v in cs.top_values]
                data[col_name] = random.choices(values, weights=weights, k=count)
            else:
                provider = self._get_provider(col_name, schema[col_name])
                data[col_name] = [provider(self.fake) for _ in range(count)]

    def _validate_with_llm(self, df: pl.DataFrame, schema: dict,
                           llm_engine, profile: DataProfile | None,
                           sample_rate: float = 1.0) -> pl.DataFrame:
        """Stage 2: Send rows through LLM for semantic validation/correction."""
        if sample_rate <= 0:
            return df

        rows = df.to_dicts()
        n = len(rows)

        if sample_rate < 1.0:
            sample_size = max(1, int(n * sample_rate))
            sample_indices = sorted(random.sample(range(n), sample_size))
            sample_rows = [rows[i] for i in sample_indices]
        else:
            sample_rows = rows
            sample_indices = list(range(n))

        profile_summary = profile.summary_for_llm() if profile else None
        batch_size = config.LLM_VALIDATION_BATCH_SIZE
        corrected_rows = []

        for i in range(0, len(sample_rows), batch_size):
            batch = sample_rows[i:i + batch_size]
            validated = llm_engine.validate_rows(batch, schema, profile_summary)
            corrected_rows.extend(validated)

        # Merge corrections back
        if sample_rate < 1.0:
            for idx, corrected in zip(sample_indices, corrected_rows, strict=False):
                rows[idx] = corrected
        else:
            rows = corrected_rows

        try:
            result_df = pl.DataFrame(rows)
            logger.info("LLM validation corrected %d rows (%.0f%% sample rate)",
                        len(corrected_rows), sample_rate * 100)
            return result_df
        except Exception as e:
            logger.warning("Failed to create DataFrame from validated rows: %s", e)
            return df
