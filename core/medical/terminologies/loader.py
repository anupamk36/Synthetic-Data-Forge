"""Loads and caches bundled medical terminology code sets."""

import json
import random
from pathlib import Path
from functools import lru_cache

CODESET_DIR = Path(__file__).parent.parent / "codeset_data"


@lru_cache(maxsize=None)
def _load_codeset(filename: str) -> dict:
    path = CODESET_DIR / filename
    with open(path) as f:
        return json.load(f)


def get_codes(filename: str) -> list[dict]:
    data = _load_codeset(filename)
    return data.get("codes", [])


def get_system_uri(filename: str) -> str:
    data = _load_codeset(filename)
    return data["system"]


def weighted_random_code(
    codes: list[dict],
    category: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Select a code using weighted random selection with optional filters."""
    rng = rng or random.Random()
    filtered = codes

    if category:
        filtered = [c for c in filtered if c.get("category") == category]

    if age is not None:
        filtered = [
            c for c in filtered
            if c.get("age_range") is None
            or (c["age_range"][0] <= age <= c["age_range"][1])
        ]

    if gender:
        filtered = [
            c for c in filtered
            if c.get("gender_bias") is None or c["gender_bias"] == gender
        ]

    if not filtered:
        filtered = codes

    weights = [c.get("weight", 1.0) for c in filtered]
    return rng.choices(filtered, weights=weights, k=1)[0]


def search_codes(filename: str, query: str, limit: int = 20) -> list[dict]:
    """Search bundled codes by display name (case-insensitive substring)."""
    codes = get_codes(filename)
    query_lower = query.lower()
    results = [c for c in codes if query_lower in c["display"].lower()]
    return results[:limit]
