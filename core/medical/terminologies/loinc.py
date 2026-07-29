"""LOINC lab observation code helpers."""

from core.medical.terminologies.loader import get_codes, get_system_uri, weighted_random_code

_FILENAME = "loinc_common.json"


def all_codes() -> list[dict]:
    return get_codes(_FILENAME)


def system_uri() -> str:
    return get_system_uri(_FILENAME)


def random_lab_code(category: str | None = None, rng=None) -> dict:
    return weighted_random_code(all_codes(), category=category, rng=rng)


def vital_sign_codes() -> list[dict]:
    return [c for c in all_codes() if c.get("category") == "vital_signs"]


def lab_codes() -> list[dict]:
    return [c for c in all_codes() if c.get("category") != "vital_signs"]
