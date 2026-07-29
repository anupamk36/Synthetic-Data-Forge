"""FHIR Bundle assembly — collection and transaction bundles."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from core.medical.fhir.references import ReferenceRegistry


def build_bundle(
    registry: ReferenceRegistry,
    bundle_type: str = "collection",
    resource_types: list[str] | None = None,
) -> dict:
    """Assemble a FHIR Bundle from all registered resources."""
    if resource_types:
        resources = []
        for rt in resource_types:
            resources.extend(registry.resources_by_type(rt))
    else:
        resources = registry.all_resources()

    entries = []
    for resource in resources:
        entry = {"resource": resource}
        if bundle_type == "transaction":
            rt = resource.get("resourceType", "Unknown")
            rid = resource.get("id", "")
            entry["request"] = {
                "method": "PUT",
                "url": f"{rt}/{rid}",
            }
            entry["fullUrl"] = f"urn:uuid:{rid}"
        entries.append(entry)

    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": bundle_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(entries),
        "entry": entries,
    }


def bundle_to_ndjson(registry: ReferenceRegistry) -> dict[str, str]:
    """Export resources as NDJSON grouped by type. Returns {type: ndjson_string}."""
    result = {}
    for resource in registry.all_resources():
        rt = resource.get("resourceType", "Unknown")
        if rt not in result:
            result[rt] = ""
        result[rt] += json.dumps(resource, default=str) + "\n"
    return result


def bundle_stats(registry: ReferenceRegistry) -> dict:
    """Return resource count summary."""
    counts = {}
    for resource in registry.all_resources():
        rt = resource.get("resourceType", "Unknown")
        counts[rt] = counts.get(rt, 0) + 1
    return {
        "total": len(registry.all_resources()),
        "by_type": counts,
    }
