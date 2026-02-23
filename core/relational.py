"""
Hydra — Multi-Table Relational Integrity Engine.

Uses a DAG to determine generation order (parents before children)
and ensures foreign key consistency across synthetic tables.
"""

import polars as pl
from faker import Faker
from collections import defaultdict, deque


class RelationalEngine:
    """
    Manages multi-table synthetic data generation with FK integrity.

    Usage:
        engine = RelationalEngine()
        engine.add_table("users", {"user_id": "Int64", "name": "String"})
        engine.add_table("orders", {"order_id": "Int64", "user_id": "Int64", "amount": "Float64"})
        engine.add_relationship("users", "user_id", "orders", "user_id")
        results = engine.generate_all({"users": 100, "orders": 500})
    """

    def __init__(self):
        self.tables = {}        # name -> schema dict
        self.relationships = [] # list of (parent_table, parent_col, child_table, child_col)
        self.fake = Faker()

    def add_table(self, name: str, schema: dict):
        """Register a table with its schema."""
        self.tables[name] = schema

    def add_relationship(self, parent_table: str, parent_col: str,
                         child_table: str, child_col: str):
        """Define a foreign key relationship."""
        self.relationships.append((parent_table, parent_col, child_table, child_col))

    def build_dag(self) -> list:
        """
        Topological sort of tables based on FK relationships.
        Returns ordered list of table names (parents first).
        """
        # Build adjacency list
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

        # Check for cycles
        if len(order) != len(self.tables):
            remaining = set(self.tables.keys()) - set(order)
            raise ValueError(
                f"Circular dependency detected involving tables: {remaining}. "
                "Cannot determine generation order."
            )

        return order

    def _generate_table(self, table_name: str, schema: dict, count: int,
                        fk_pools: dict) -> pl.DataFrame:
        """
        Generate a single table's data.

        fk_pools: dict mapping (table_name, column_name) -> list of valid FK values
        """
        data = []
        # Identify which columns in this table are FK children
        fk_sources = {}
        for parent, pcol, child, ccol in self.relationships:
            if child == table_name:
                fk_sources[ccol] = (parent, pcol)

        for _ in range(count):
            row = {}
            for col, dtype in schema.items():
                if col in fk_sources:
                    # Use a value from the parent's generated pool
                    parent_table, parent_col = fk_sources[col]
                    pool = fk_pools.get((parent_table, parent_col), [])
                    if pool:
                        row[col] = self.fake.random_element(pool)
                    else:
                        row[col] = self._generate_value(dtype)
                else:
                    row[col] = self._generate_value(dtype)
            data.append(row)

        return pl.DataFrame(data)

    def _generate_value(self, dtype: str):
        """Generate a single fake value based on type string."""
        if "Int" in dtype:
            return self.fake.random_int(0, 10000)
        elif "Float" in dtype:
            return self.fake.pyfloat(right_digits=2, positive=True)
        elif "Date" in dtype:
            return self.fake.date_this_decade()
        else:
            return self.fake.word()

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

        for table_name in order:
            schema = self.tables[table_name]
            count = counts.get(table_name, 100)

            df = self._generate_table(table_name, schema, count, fk_pools)
            results[table_name] = df

            # Register this table's columns as potential FK pools for children
            for parent, pcol, child, ccol in self.relationships:
                if parent == table_name and pcol in df.columns:
                    fk_pools[(table_name, pcol)] = df[pcol].unique().to_list()

        return results
