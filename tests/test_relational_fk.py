"""Tests for RelationalEngine FK integrity and source-data-aware generation."""

import polars as pl

from core.relational import RelationalEngine


class TestFKIntegrity:
    """Verify that child FK values always exist in parent PK values."""

    def _simple_engine(self):
        engine = RelationalEngine(seed=42)
        engine.add_table("users", {"user_id": "Int64", "name": "String"})
        engine.add_table("orders", {"order_id": "Int64", "user_id": "Int64", "amount": "Float64"})
        engine.add_relationship("users", "user_id", "orders", "user_id")
        return engine

    def test_basic_join_integrity(self):
        engine = self._simple_engine()
        results = engine.generate_all({"users": 50, "orders": 200})

        parent_ids = set(results["users"]["user_id"].to_list())
        child_ids = set(results["orders"]["user_id"].to_list())

        # Every child FK must exist in parent PKs
        assert child_ids.issubset(parent_ids), (
            f"Orphan FK values: {child_ids - parent_ids}"
        )

    def test_parent_pk_uniqueness(self):
        engine = self._simple_engine()
        results = engine.generate_all({"users": 100, "orders": 50})

        user_ids = results["users"]["user_id"].to_list()
        assert len(user_ids) == len(set(user_ids)), "Parent PK column has duplicates"

    def test_verify_joins_all_pass(self):
        engine = self._simple_engine()
        results = engine.generate_all({"users": 50, "orders": 200})
        report = engine.verify_joins(results)

        assert len(report) == 1
        assert report[0]["status"] == "pass"
        assert report[0]["orphan_count"] == 0

    def test_three_table_chain(self):
        """users -> orders -> order_items: 3-level FK chain."""
        engine = RelationalEngine(seed=7)
        engine.add_table("users", {"user_id": "Int64", "email": "String"})
        engine.add_table("orders", {"order_id": "Int64", "user_id": "Int64", "total": "Float64"})
        engine.add_table("order_items", {"item_id": "Int64", "order_id": "Int64", "product": "String"})
        engine.add_relationship("users", "user_id", "orders", "user_id")
        engine.add_relationship("orders", "order_id", "order_items", "order_id")

        results = engine.generate_all({"users": 20, "orders": 80, "order_items": 300})
        report = engine.verify_joins(results)

        for r in report:
            assert r["status"] == "pass", (
                f"FK violation: {r['child']}.{r['child_col']} -> {r['parent']}.{r['parent_col']} "
                f"has {r['orphan_count']} orphans"
            )

    def test_many_to_many_children(self):
        """One parent, two child tables sharing the same FK."""
        engine = RelationalEngine(seed=99)
        engine.add_table("departments", {"dept_id": "Int64", "name": "String"})
        engine.add_table("employees", {"emp_id": "Int64", "dept_id": "Int64", "name": "String"})
        engine.add_table("budgets", {"budget_id": "Int64", "dept_id": "Int64", "amount": "Float64"})
        engine.add_relationship("departments", "dept_id", "employees", "dept_id")
        engine.add_relationship("departments", "dept_id", "budgets", "dept_id")

        results = engine.generate_all({"departments": 10, "employees": 100, "budgets": 30})
        report = engine.verify_joins(results)

        assert all(r["status"] == "pass" for r in report)


class TestSourceDataAware:
    """Verify that uploaded source data influences generation."""

    def test_source_data_int_range_respected(self):
        engine = RelationalEngine(seed=42)
        engine.add_table("products", {"product_id": "Int64", "price": "Float64"})
        engine.add_table("sales", {"sale_id": "Int64", "product_id": "Int64"})
        engine.add_relationship("products", "product_id", "sales", "product_id")

        # Source data has product_id in [100..105], price in [9.99..49.99]
        source = pl.DataFrame({
            "product_id": [100, 101, 102, 103, 104, 105],
            "price": [9.99, 19.99, 29.99, 39.99, 44.99, 49.99],
        })
        engine.set_source_data("products", source)

        results = engine.generate_all({"products": 6, "sales": 30})

        # PK values should come from or near the source range
        product_ids = results["products"]["product_id"].to_list()
        assert len(product_ids) == len(set(product_ids)), "PKs must be unique"

        # Price should be within [9.99, 49.99] since source data defines that range
        prices = results["products"]["price"].to_list()
        assert all(9.0 <= p <= 50.0 for p in prices), f"Prices outside source range: {prices}"

        # FK integrity
        report = engine.verify_joins(results)
        assert report[0]["status"] == "pass"

    def test_source_data_string_values_sampled(self):
        engine = RelationalEngine(seed=42)
        engine.add_table("categories", {"cat_id": "Int64", "name": "String"})

        source = pl.DataFrame({
            "cat_id": [1, 2, 3],
            "name": ["Electronics", "Clothing", "Food"],
        })
        engine.set_source_data("categories", source)

        results = engine.generate_all({"categories": 10})
        names = set(results["categories"]["name"].to_list())

        # All generated names should come from the source values
        assert names.issubset({"Electronics", "Clothing", "Food"}), (
            f"Generated names not from source: {names}"
        )

    def test_no_source_data_uses_faker(self):
        """Without source data, engine should still work via Faker."""
        engine = RelationalEngine(seed=42)
        engine.add_table("items", {"item_id": "Int64", "description": "String"})

        results = engine.generate_all({"items": 20})
        assert len(results["items"]) == 20

    def test_polars_join_actually_works(self):
        """End-to-end: generate data and perform an actual Polars JOIN."""
        engine = RelationalEngine(seed=42)
        engine.add_table("customers", {"cust_id": "Int64", "name": "String"})
        engine.add_table("orders", {"order_id": "Int64", "cust_id": "Int64", "total": "Float64"})
        engine.add_relationship("customers", "cust_id", "orders", "cust_id")

        source_custs = pl.DataFrame({
            "cust_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        })
        engine.set_source_data("customers", source_custs)

        results = engine.generate_all({"customers": 5, "orders": 20})

        # Perform the actual JOIN
        joined = results["orders"].join(
            results["customers"],
            on="cust_id",
            how="inner",
        )

        # Inner join should retain ALL order rows (no orphans)
        assert len(joined) == len(results["orders"]), (
            f"JOIN lost rows: {len(joined)} vs {len(results['orders'])} orders"
        )
        # Joined result should have the customer name column
        assert "name" in joined.columns
