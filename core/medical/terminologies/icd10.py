"""ICD-10-CM diagnosis code helpers."""

from core.medical.terminologies.loader import get_codes, get_system_uri, weighted_random_code

_FILENAME = "icd10_common.json"


def all_codes() -> list[dict]:
    return get_codes(_FILENAME)


def system_uri() -> str:
    return get_system_uri(_FILENAME)


def random_diagnosis(
    category: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    rng=None,
) -> dict:
    return weighted_random_code(all_codes(), category=category, age=age, gender=gender, rng=rng)
