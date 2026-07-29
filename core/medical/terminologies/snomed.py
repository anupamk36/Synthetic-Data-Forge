"""SNOMED CT clinical finding, procedure, and body structure helpers."""

from core.medical.terminologies.loader import get_codes, get_system_uri, weighted_random_code

_FILENAME = "snomed_common.json"


def all_codes() -> list[dict]:
    return get_codes(_FILENAME)


def system_uri() -> str:
    return get_system_uri(_FILENAME)


def random_finding(age: int | None = None, gender: str | None = None, rng=None) -> dict:
    findings = [c for c in all_codes() if c.get("type") == "finding"]
    return weighted_random_code(findings, age=age, gender=gender, rng=rng)


def random_procedure(rng=None) -> dict:
    procedures = [c for c in all_codes() if c.get("type") == "procedure"]
    return weighted_random_code(procedures, rng=rng)


def random_body_structure(rng=None) -> dict:
    structures = [c for c in all_codes() if c.get("type") == "body_structure"]
    return weighted_random_code(structures, rng=rng)


def random_substance(rng=None) -> dict:
    substances = [c for c in all_codes() if c.get("type") == "substance"]
    return weighted_random_code(substances, rng=rng)
