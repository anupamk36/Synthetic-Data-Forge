"""Tests for core.relational — RelationalEngine."""

import pytest

from core.exceptions import RelationalError, ValidationError
from core.relational import RelationalEngine


class TestRelationalEngine:

    def test_single_table(self):
        engine = RelationalEngine()
        engine.add_table("users", {"id": "Int64", "name": "String"})
        results = engine.generate_all({"users": 50})
        assert "users" in results
        assert len(results["users"]) == 50

    def test_parent_child_fk(self):
        engine = RelationalEngine()
        engine.add_table("users", {"user_id": "Int64", "name": "String"})
        engine.add_table("orders", {"order_id": "Int64", "user_id": "Int64", "amount": "Float64"})
        engine.add_relationship("users", "user_id", "orders", "user_id")

        results = engine.generate_all({"users": 10, "orders": 50})
        user_ids = set(results["users"]["user_id"].to_list())
        order_user_ids = set(results["orders"]["user_id"].to_list())

        # All FK values must exist in parent
        assert order_user_ids.issubset(user_ids)

    def test_three_table_dag(self):
        engine = RelationalEngine()
        engine.add_table("a", {"id": "Int64"})
        engine.add_table("b", {"id": "Int64", "a_id": "Int64"})
        engine.add_table("c", {"id": "Int64", "b_id": "Int64"})
        engine.add_relationship("a", "id", "b", "a_id")
        engine.add_relationship("b", "id", "c", "b_id")

        results = engine.generate_all({"a": 5, "b": 20, "c": 100})
        assert len(results) == 3
        b_a_ids = set(results["b"]["a_id"].to_list())
        a_ids = set(results["a"]["id"].to_list())
        assert b_a_ids.issubset(a_ids)

    def test_cycle_detection(self):
        engine = RelationalEngine()
        engine.add_table("x", {"id": "Int64", "y_id": "Int64"})
        engine.add_table("y", {"id": "Int64", "x_id": "Int64"})
        engine.add_relationship("x", "id", "y", "x_id")
        engine.add_relationship("y", "id", "x", "y_id")

        with pytest.raises(RelationalError, match="Circular dependency"):
            engine.generate_all({"x": 10, "y": 10})

    def test_dag_order(self):
        engine = RelationalEngine()
        engine.add_table("child", {"id": "Int64", "parent_id": "Int64"})
        engine.add_table("parent", {"id": "Int64"})
        engine.add_relationship("parent", "id", "child", "parent_id")
        order = engine.build_dag()
        assert order.index("parent") < order.index("child")

    def test_invalid_relationship_table(self):
        engine = RelationalEngine()
        engine.add_table("users", {"id": "Int64"})
        with pytest.raises(ValidationError):
            engine.add_relationship("users", "id", "nonexistent", "id")

    def test_invalid_relationship_column(self):
        engine = RelationalEngine()
        engine.add_table("users", {"id": "Int64"})
        engine.add_table("orders", {"oid": "Int64"})
        with pytest.raises(ValidationError):
            engine.add_relationship("users", "nonexistent_col", "orders", "oid")
