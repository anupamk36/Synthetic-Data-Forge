"""DICOM UID generation with proper format."""

from __future__ import annotations

import random

ORG_ROOT = "1.2.826.0.1.3680043.8.498"

_KNOWN_SOP_CLASSES = {
    "CT": "1.2.840.10008.5.1.4.1.1.2",
    "MR": "1.2.840.10008.5.1.4.1.1.4",
    "US": "1.2.840.10008.5.1.4.1.1.6.1",
    "DX": "1.2.840.10008.5.1.4.1.1.1.1",
    "CR": "1.2.840.10008.5.1.4.1.1.1",
    "MG": "1.2.840.10008.5.1.4.1.1.1.2",
    "PT": "1.2.840.10008.5.1.4.1.1.128",
    "NM": "1.2.840.10008.5.1.4.1.1.20",
}


def generate_uid(rng: random.Random | None = None) -> str:
    """Generate a DICOM-compliant UID (max 64 chars)."""
    rng = rng or random.Random()
    suffix = str(rng.randint(10**12, 10**15 - 1))
    uid = f"{ORG_ROOT}.{suffix}"
    return uid[:64]


def generate_study_uid(rng: random.Random | None = None) -> str:
    return generate_uid(rng)


def generate_series_uid(rng: random.Random | None = None) -> str:
    return generate_uid(rng)


def generate_instance_uid(rng: random.Random | None = None) -> str:
    return generate_uid(rng)


def get_sop_class_uid(modality: str) -> str:
    return _KNOWN_SOP_CLASSES.get(modality, "1.2.840.10008.5.1.4.1.1.7")


def generate_accession_number(rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    return f"ACC{rng.randint(100000, 999999)}"
