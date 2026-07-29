"""Load and manage clinical trial profiles."""

from __future__ import annotations

from functools import lru_cache
from core.medical.terminologies.loader import _load_codeset

_PROFILES_FILE = "trial_profiles.json"


@lru_cache(maxsize=None)
def _load_profiles() -> dict:
    data = _load_codeset(_PROFILES_FILE)
    return data.get("profiles", {})


def list_profiles() -> list[dict]:
    """List available trial profiles with metadata."""
    profiles = _load_profiles()
    return [
        {
            "id": key,
            "display_name": profile["display_name"],
            "description": profile["description"],
            "therapeutic_area": profile["therapeutic_area"],
            "phase": profile["phase"],
            "target_enrollment": profile.get("target_enrollment", 100),
        }
        for key, profile in profiles.items()
    ]


def get_profile(profile_id: str) -> dict:
    """Get a trial profile by ID."""
    profiles = _load_profiles()
    if profile_id not in profiles:
        available = list(profiles.keys())
        raise ValueError(f"Unknown profile: {profile_id}. Available: {available}")
    return profiles[profile_id]


def get_visit_schedule(profile_id: str) -> list[dict]:
    """Get the visit schedule for a profile."""
    profile = get_profile(profile_id)
    return profile.get("visit_schedule", [])


def get_arms(profile_id: str) -> list[dict]:
    """Get arm definitions for a profile."""
    profile = get_profile(profile_id)
    return profile.get("arms", [])
