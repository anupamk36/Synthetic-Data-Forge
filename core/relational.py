"""
Hydra — Multi-Table Relational Integrity Engine.

Uses a DAG to determine generation order (parents before children)
and ensures foreign key consistency across synthetic tables.
Parent PK columns are guaranteed unique; child FK columns sample
exclusively from the parent's actual PK values so that every JOIN
on (parent.pk = child.fk) produces correct results.

When source DataFrames are provided, the engine learns value
distributions from the uploaded data and produces synthetic rows
that mirror real cardinality, ranges, and patterns.
"""

import logging
import random
from collections import defaultdict, deque
from datetime import date, datetime

import polars as pl
from faker import Faker

from core.exceptions import RelationalError
from core.generator import ForgeEngine
from core.validation import validate_relationship

logger = logging.getLogger(__name__)


class RelationalEngine:
    """
    Manages multi-table synthetic data generation with FK integrity.

    Usage:
        engine = RelationalEngine()
        engine.add_table("users", {"user_id": "Int64", "name": "String"})
        engine.add_table("orders", {"order_id": "Int64", "user_id": "Int64", "amount": "Float64"})
        engine.add_relationship("users", "user_id", "orders", "user_id")
        engine.set_source_data("users", users_df)
        engine.set_source_data("orders", orders_df)
        results = engine.generate_all({"users": 100, "orders": 500})
    """

    def __init__(self, seed: int | None = None):
        self.tables = {}          # name -> schema dict
        self.relationships = []   # list of (parent_table, parent_col, child_table, child_col)
        self.source_data = {}     # name -> pl.DataFrame (uploaded data)
        self.seed = seed
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
        self._forge = ForgeEngine(seed=seed)

    def add_table(self, name: str, schema: dict):
        """Register a table with its schema."""
        self.tables[name] = schema

    def set_source_data(self, name: str, df: pl.DataFrame):
        """Provide the original uploaded data for distribution-aware generation."""
        self.source_data[name] = df

    def add_relationship(self, parent_table: str, parent_col: str,
                         child_table: str, child_col: str):
        """Define a foreign key relationship with validation."""
        validate_relationship(self.tables, parent_table, parent_col, child_table, child_col)
        self.relationships.append((parent_table, parent_col, child_table, child_col))

    def build_dag(self) -> list:
        """
        Topological sort of tables based on FK relationships.
        Returns ordered list of table names (parents first).
        """
        graph = defaultdict(list)
        in_degree = {name: 0 for name in self.tables}

        for parent, _, child, _ in self.relationships:
            graph[parent].append(child)
            in_degree[child] = in_degree.get(child, 0) + 1

        # Kahn's algorithm
        queue = deque([t for t, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in graph[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.tables):
            remaining = set(self.tables.keys()) - set(order)
            raise RelationalError(
                f"Circular dependency detected involving tables: {remaining}. "
                "Cannot determine generation order."
            )

        return order

    def _pk_columns(self) -> dict[str, set[str]]:
        """Return {table_name: set of columns used as parent PK} across all relationships."""
        pk_cols: dict[str, set[str]] = defaultdict(set)
        for parent, pcol, _, _ in self.relationships:
            pk_cols[parent].add(pcol)
        return dict(pk_cols)

    # ------------------------------------------------------------------ #
    #  Distribution-aware value generation
    # ------------------------------------------------------------------ #

    def _build_column_profile(self, table_name: str, col: str, dtype: str) -> dict | None:
        """Extract a statistical profile from source data for a column."""
        src = self.source_data.get(table_name)
        if src is None or col not in src.columns:
            return None

        series = src[col].drop_nulls()
        if len(series) == 0:
            return None

        profile: dict = {"values": series.to_list()}

        if "Int" in dtype or "Float" in dtype:
            profile["min"] = series.min()
            profile["max"] = series.max()
            profile["mean"] = series.mean()
            profile["n_unique"] = series.n_unique()
        elif "Date" in dtype:
            profile["min"] = series.min()
            profile["max"] = series.max()
        else:
            profile["n_unique"] = series.n_unique()

        return profile

    def _generate_from_profile(self, profile: dict, dtype: str):
        """Generate a single value that mirrors the source distribution."""
        values = profile["values"]

        if "Int" in dtype:
            lo, hi = profile.get("min", 0), profile.get("max", 10000)
            return random.randint(int(lo), int(hi))
        elif "Float" in dtype:
            lo, hi = profile.get("min", 0.0), profile.get("max", 1000.0)
            return round(random.uniform(float(lo), float(hi)), 2)
        elif "Date" in dtype:
            lo, hi = profile.get("min"), profile.get("max")
            if lo and hi:
                if isinstance(lo, datetime):
                    lo = lo.date()
                if isinstance(hi, datetime):
                    hi = hi.date()
                if isinstance(lo, date) and isinstance(hi, date):
                    delta = (hi - lo).days
                    if delta > 0:
                        return lo + __import__("datetime").timedelta(days=random.randint(0, delta))
            return random.choice(values)
        else:
            # String / categorical — sample from observed values
            return random.choice(values)

    def _generate_unique_from_profile(self, profile: dict, col: str, dtype: str, count: int) -> list:
        """Generate *count* unique values using the source profile."""
        src_unique = list(set(profile["values"]))

        # If we need more unique values than source has, extend with synthetic
        if len(src_unique) >= count:
            random.shuffle(src_unique)
            return src_unique[:count]

        # Start with all real unique values, then generate more
        values = list(src_unique)
        seen = set(src_unique)

        if "Int" in dtype:
            lo, hi = int(profile.get("min", 1)), int(profile.get("max", 10000))
            # Extend range if needed
            hi = max(hi, lo + count * 2)
            attempts = 0
            while len(values) < count and attempts < count * 10:
                v = random.randint(lo, hi)
                if v not in seen:
                    seen.add(v)
                    values.append(v)
                attempts += 1
        elif "Float" in dtype:
            lo, hi = float(profile.get("min", 0.0)), float(profile.get("max", 10000.0))
            attempts = 0
            while len(values) < count and attempts < count * 10:
                v = round(random.uniform(lo, hi), 2)
                if v not in seen:
                    seen.add(v)
                    values.append(v)
                attempts += 1
        else:
            # String PKs — append sequential suffix
            base = src_unique[0] if src_unique else col
            idx = len(values) + 1
            while len(values) < count:
                v = f"{base}_{idx}"
                if v not in seen:
                    seen.add(v)
                    values.append(v)
                idx += 1

        # Final fallback: sequential ints
        while len(values) < count:
            fb = len(values) + 1
            values.append(fb if "Int" in dtype else f"{col}_{fb}")

        return values

    def _generate_unique_values(self, col: str, dtype: str, count: int,
                                table_name: str) -> list:
        """Generate *count* unique values for a PK column, using source data if available."""
        profile = self._build_column_profile(table_name, col, dtype)
        if profile:
            return self._generate_unique_from_profile(profile, col, dtype, count)

        # No source data — fall back to ForgeEngine smart providers
        provider = self._forge._get_provider(col, dtype)
        seen: set = set()
        values: list = []
        max_attempts = count * 10
        attempts = 0
        while len(values) < count and attempts < max_attempts:
            v = provider(self.fake)
            if v not in seen:
                seen.add(v)
                values.append(v)
            attempts += 1
        while len(values) < count:
            fallback = len(values) + 1
            if "Int" in dtype:
                values.append(fallback)
            else:
                values.append(f"{col}_{fallback}")
        return values

    def _generate_table(self, table_name: str, schema: dict, count: int,
                        fk_pools: dict, pk_cols: set[str]) -> pl.DataFrame:
        """Generate a single table's data with unique PKs and valid FK references."""
        # Identify FK sources for this table (child side)
        fk_sources: dict[str, tuple[str, str, str]] = {}
        for parent, pcol, child, ccol in self.relationships:
            if child == table_name:
                parent_dtype = self.tables[parent].get(pcol, "String")
                fk_sources[ccol] = (parent, pcol, parent_dtype)

        # Pre-generate unique values for PK columns
        pk_values: dict[str, list] = {}
        for col in pk_cols:
            if col in schema:
                pk_values[col] = self._generate_unique_values(
                    col, schema[col], count, table_name
                )

        # Build column profiles for non-PK, non-FK columns
        col_profiles: dict[str, dict | None] = {}
        for col, dtype in schema.items():
            if col not in pk_values and col not in fk_sources:
                col_profiles[col] = self._build_column_profile(table_name, col, dtype)

        data = []
        for row_idx in range(count):
            row = {}
            for col, dtype in schema.items():
                if col in pk_values:
                    row[col] = pk_values[col][row_idx]
                elif col in fk_sources:
                    parent_table, parent_col, parent_dtype = fk_sources[col]
                    pool = fk_pools.get((parent_table, parent_col), [])
                    if pool:
                        row[col] = random.choice(pool)
                    else:
                        logger.warning(
                            "FK pool empty for %s.%s -> %s.%s; using fallback",
                            table_name, col, parent_table, parent_col,
                        )
                        row[col] = self._forge._get_provider(col, dtype)(self.fake)
                else:
                    profile = col_profiles.get(col)
                    if profile:
                        row[col] = self._generate_from_profile(profile, dtype)
                    else:
                        provider = self._forge._get_provider(col, dtype)
                        row[col] = provider(self.fake)
            data.append(row)

        return pl.DataFrame(data)

    def generate_all(self, counts: dict) -> dict:
        """
        Generate all tables in DAG order with FK integrity.

        Args:
            counts: dict mapping table_name -> number of rows

        Returns:
            dict mapping table_name -> pl.DataFrame
        """
        order = self.build_dag()
        results = {}
        fk_pools = {}
        all_pk_cols = self._pk_columns()

        for table_name in order:
            schema = self.tables[table_name]
            count = counts.get(table_name, 100)
            pk_cols = all_pk_cols.get(table_name, set())

            df = self._generate_table(table_name, schema, count, fk_pools, pk_cols)
            results[table_name] = df

            # Populate FK pool from this table's PK columns
            for parent, pcol, _child, _ccol in self.relationships:
                if parent == table_name and pcol in df.columns:
                    fk_pools[(table_name, pcol)] = df[pcol].to_list()

            logger.info("Generated table '%s': %d rows", table_name, len(df))

        return results

    def verify_joins(self, results: dict) -> list[dict]:
        """
        Verify that every FK relationship produces a valid JOIN.

        Returns a list of dicts with verification results per relationship.
        Each dict: {parent, parent_col, child, child_col, status, orphan_count, orphan_pct}
        """
        report = []
        for parent, pcol, child, ccol in self.relationships:
            parent_df = results.get(parent)
            child_df = results.get(child)
            if parent_df is None or child_df is None:
                report.append({
                    "parent": parent, "parent_col": pcol,
                    "child": child, "child_col": ccol,
                    "status": "missing_table",
                    "orphan_count": -1, "orphan_pct": -1,
                })
                continue

            parent_keys = set(parent_df[pcol].to_list())
            child_keys = child_df[ccol].to_list()
            orphans = [v for v in child_keys if v not in parent_keys]

            total_child = len(child_keys)
            orphan_count = len(orphans)
            orphan_pct = round(orphan_count / total_child * 100, 2) if total_child else 0

            status = "pass" if orphan_count == 0 else "fail"
            report.append({
                "parent": parent, "parent_col": pcol,
                "child": child, "child_col": ccol,
                "status": status,
                "orphan_count": orphan_count,
                "orphan_pct": orphan_pct,
            })
            if orphan_count > 0:
                logger.warning(
                    "FK violation: %s.%s -> %s.%s has %d orphan rows (%.1f%%)",
                    child, ccol, parent, pcol, orphan_count, orphan_pct,
                )

        return report
