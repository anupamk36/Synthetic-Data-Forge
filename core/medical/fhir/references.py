"""Cross-resource FHIR reference resolution and integrity checking."""

from __future__ import annotations

from collections import defaultdict


class ReferenceRegistry:
    """Tracks generated resources and resolves FHIR references."""

    def __init__(self):
        self._resources: dict[str, dict] = {}
        self._by_type: dict[str, list[str]] = defaultdict(list)

    def register(self, resource_type: str, resource_id: str, resource: dict):
        full_ref = f"{resource_type}/{resource_id}"
        self._resources[full_ref] = resource
        self._by_type[resource_type].append(resource_id)

    def make_reference(self, resource_type: str, resource_id: str, display: str | None = None) -> dict:
        ref = {"reference": f"{resource_type}/{resource_id}"}
        if display:
            ref["display"] = display
        return ref

    def get_ids(self, resource_type: str) -> list[str]:
        return self._by_type.get(resource_type, [])

    def get_resource(self, resource_type: str, resource_id: str) -> dict | None:
        return self._resources.get(f"{resource_type}/{resource_id}")

    def all_resources(self) -> list[dict]:
        return list(self._resources.values())

    def resources_by_type(self, resource_type: str) -> list[dict]:
        return [
            self._resources[f"{resource_type}/{rid}"]
            for rid in self._by_type.get(resource_type, [])
        ]

    def verify_integrity(self) -> list[dict]:
        """Check that all Reference fields point to registered resources."""
        errors = []
        for ref_key, resource in self._resources.items():
            self._check_references(resource, ref_key, [], errors)
        return errors

    def _check_references(self, obj, source: str, path: list[str], errors: list[dict]):
        if isinstance(obj, dict):
            if "reference" in obj and isinstance(obj["reference"], str):
                ref_target = obj["reference"]
                if ref_target not in self._resources:
                    errors.append({
                        "source": source,
                        "path": ".".join(path + ["reference"]),
                        "target": ref_target,
                        "error": "dangling_reference",
                    })
            for key, value in obj.items():
                self._check_references(value, source, path + [key], errors)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._check_references(item, source, path + [str(i)], errors)
