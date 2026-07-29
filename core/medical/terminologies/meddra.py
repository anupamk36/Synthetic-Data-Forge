"""MedDRA adverse event term helpers."""

from core.medical.terminologies.loader import get_codes, weighted_random_code

_FILENAME = "meddra_common.json"


def _load_terms() -> list[dict]:
    """Load MedDRA terms from the bundled file."""
    from core.medical.terminologies.loader import _load_codeset
    data = _load_codeset(_FILENAME)
    return data.get("terms", [])


def all_terms() -> list[dict]:
    return _load_terms()


def random_ae(
    category: str | None = None,
    therapeutic_area: str | None = None,
    rng=None,
) -> dict:
    """Select a random adverse event, optionally filtered by category or area."""
    terms = _load_terms()

    if category:
        filtered = [t for t in terms if t.get("category") == category]
        if filtered:
            terms = filtered

    if therapeutic_area:
        area_terms = [t for t in terms if therapeutic_area in t.get("relevant_areas", [])]
        if area_terms:
            terms = area_terms

    weights = [t.get("frequency_per_cycle", 0.1) for t in terms]
    rng = rng or __import__("random").Random()
    return rng.choices(terms, weights=weights, k=1)[0]


def ae_for_profile(profile_aes: list[str], rng=None) -> dict:
    """Select an AE from a profile's common AE list."""
    terms = _load_terms()
    matching = [t for t in terms if t.get("pt_name", "").lower() in [a.lower() for a in profile_aes]]
    if not matching:
        matching = terms
    rng = rng or __import__("random").Random()
    weights = [t.get("frequency_per_cycle", 0.1) for t in matching]
    return rng.choices(matching, weights=weights, k=1)[0]


def random_severity(term: dict, rng=None) -> str:
    """Select severity based on the term's severity distribution."""
    rng = rng or __import__("random").Random()
    dist = term.get("severity_distribution", {"mild": 0.5, "moderate": 0.35, "severe": 0.15})
    severities = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(severities, weights=weights, k=1)[0]
