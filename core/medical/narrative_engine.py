"""Clinical narrative engine — LLM-driven DocumentReference generation from FHIR data."""

from __future__ import annotations

import base64
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from core.medical.fhir.references import ReferenceRegistry
from core.medical.narrative_prompts import (
    DOC_TYPE_LOINC,
    DOC_TYPE_MAX_TOKENS,
    SYSTEM_PROMPTS,
    assemble_clinical_context,
    determine_doc_types,
    format_user_prompt,
)

logger = logging.getLogger(__name__)

_LOINC_SYSTEM = "http://loinc.org"


# ---------------------------------------------------------------------------
# FHIR DocumentReference builder
# ---------------------------------------------------------------------------

def build_document_reference(
    doc_id: str,
    doc_type: str,
    encounter_id: str,
    patient_id: str,
    narrative_text: str,
    authored_date: str,
) -> dict[str, Any]:
    """Construct a FHIR R4 DocumentReference from a narrative text.

    Args:
        doc_id:         Unique identifier for this DocumentReference.
        doc_type:       One of ALL_DOC_TYPES keys (e.g. "discharge_summary").
        encounter_id:   FHIR Encounter id this document belongs to.
        patient_id:     FHIR Patient id for the subject reference.
        narrative_text: Plain-text narrative produced by the LLM.
        authored_date:  ISO date string (YYYY-MM-DD or datetime).

    Returns:
        A valid FHIR R4 DocumentReference dict.
    """
    loinc_info = DOC_TYPE_LOINC.get(doc_type, {"code": "11488-4", "display": "Consult note"})

    encoded_data = base64.b64encode(narrative_text.encode("utf-8")).decode("ascii")

    return {
        "resourceType": "DocumentReference",
        "id": doc_id,
        "status": "current",
        "type": {
            "coding": [
                {
                    "system": _LOINC_SYSTEM,
                    "code": loinc_info["code"],
                    "display": loinc_info["display"],
                }
            ],
            "text": loinc_info["display"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": authored_date,
        "content": [
            {
                "attachment": {
                    "contentType": "text/plain",
                    "data": encoded_data,
                    "title": loinc_info["display"],
                    "creation": authored_date,
                }
            }
        ],
        "context": {
            "encounter": [{"reference": f"Encounter/{encounter_id}"}],
        },
    }


# ---------------------------------------------------------------------------
# LLM dispatch helper
# ---------------------------------------------------------------------------

def _call_llm(
    provider: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    """Dispatch an LLM call to whatever provider object was passed in.

    Strategy (tried in order):
    1. provider.chat_complete(messages, max_tokens) — Alchemy / generic protocol
    2. provider.messages.create(...)               — Anthropic SDK (Claude)
    3. provider.chat.completions.create(...)        — OpenAI SDK
    4. provider.models.generate_content(...)        — Google Gemini SDK
    5. HTTP POST to Ollama's /api/chat endpoint      — local Ollama

    Raises RuntimeError if all strategies fail.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Strategy 1: generic chat_complete (Alchemy, test doubles, etc.)
    if callable(getattr(provider, "chat_complete", None)):
        response = provider.chat_complete(messages=messages, temperature=0.7)
        return response.choices[0].message.content

    # Strategy 2: Anthropic SDK (provider IS the anthropic.Anthropic client)
    if hasattr(provider, "messages") and hasattr(provider.messages, "create"):
        try:
            response = provider.messages.create(
                model=getattr(provider, "_model", "claude-3-haiku-20240307"),
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            logger.debug("Provider strategy failed, falling through: %s", e)

    # Strategy 3: OpenAI SDK (provider IS the openai.OpenAI client)
    if hasattr(provider, "chat") and hasattr(provider.chat, "completions"):
        try:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            response = provider.chat.completions.create(
                model=getattr(provider, "_model", "gpt-4o-mini"),
                max_tokens=max_tokens,
                messages=full_messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.debug("Provider strategy failed, falling through: %s", e)

    # Strategy 4: Google Gemini SDK (provider IS genai.Client or similar)
    if hasattr(provider, "models") and hasattr(provider.models, "generate_content"):
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = provider.models.generate_content(
                model=getattr(provider, "_model", "gemini-2.0-flash"),
                contents=combined_prompt,
            )
            return response.text
        except Exception as e:
            logger.debug("Provider strategy failed, falling through: %s", e)

    # Strategy 5: Ollama HTTP API
    if hasattr(provider, "_ollama_url") or getattr(provider, "_provider_name", "") == "ollama":
        import json as _json
        import urllib.request

        base_url = getattr(provider, "_ollama_url", "http://localhost:11434")
        model = getattr(provider, "_model", "llama3.2:3b")
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream": False,
        }
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310 — base_url is server-side config, never user-supplied
            f"{base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            result = _json.loads(resp.read())
            return result["message"]["content"]

    raise RuntimeError(
        f"Cannot dispatch LLM call: provider {type(provider).__name__!r} "
        "does not implement a recognised interface."
    )


# ---------------------------------------------------------------------------
# ClinicalNarrativeEngine
# ---------------------------------------------------------------------------

class ClinicalNarrativeEngine:
    """Orchestrates LLM-based clinical narrative generation for FHIR encounters.

    Each encounter gets one or more DocumentReferences (discharge summary, radiology
    report, etc.) depending on the encounter class and available clinical data.
    """

    def __init__(
        self,
        provider: Any,
        max_workers: int = 4,
        allowed_types: list[str] | None = None,
    ):
        """
        Args:
            provider:       LLM provider object (any supported interface — see _call_llm).
            max_workers:    Thread pool size for concurrent LLM calls.
            allowed_types:  Whitelist of doc types to generate; None means all applicable.
        """
        self._provider = provider
        self._max_workers = max_workers
        self._allowed_types = allowed_types

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_for_encounter(
        self,
        registry: ReferenceRegistry,
        encounter_id: str,
        allowed_types: list[str] | None = None,
        register: bool = False,
    ) -> list[dict]:
        """Generate DocumentReferences for a single encounter.

        Args:
            registry:       ReferenceRegistry with all FHIR resources.
            encounter_id:   The encounter to generate narratives for.
            allowed_types:  Override the engine-level allowed_types for this call.
            register:       If True, register produced DocumentReferences in the registry.

        Returns:
            List of FHIR DocumentReference dicts.
        """
        effective_allowed = allowed_types if allowed_types is not None else self._allowed_types

        context = assemble_clinical_context(registry, encounter_id)
        doc_types = determine_doc_types(context, effective_allowed)

        if not doc_types:
            return []

        patient_id = context["patient"]["id"]
        authored_date = context["encounter"].get("end") or context["encounter"].get("start") or \
            datetime.now(timezone.utc).date().isoformat()

        docs: list[dict] = []

        def _generate_one(doc_type: str) -> dict | None:
            system_prompt = SYSTEM_PROMPTS[doc_type]
            user_prompt = format_user_prompt(doc_type, context)
            max_tokens = DOC_TYPE_MAX_TOKENS.get(doc_type, 800)
            try:
                narrative = _call_llm(
                    self._provider,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                logger.warning(
                    "LLM call failed for %s/%s: %s", encounter_id, doc_type, exc
                )
                return None

            doc_id = str(uuid.uuid4())
            return build_document_reference(
                doc_id=doc_id,
                doc_type=doc_type,
                encounter_id=encounter_id,
                patient_id=patient_id,
                narrative_text=narrative,
                authored_date=authored_date,
            )

        if len(doc_types) == 1 or self._max_workers <= 1:
            for dt in doc_types:
                doc = _generate_one(dt)
                if doc is not None:
                    docs.append(doc)
        else:
            with ThreadPoolExecutor(max_workers=min(self._max_workers, len(doc_types))) as pool:
                futures = {pool.submit(_generate_one, dt): dt for dt in doc_types}
                for future in as_completed(futures):
                    doc = future.result()
                    if doc is not None:
                        docs.append(doc)

        if register:
            for doc in docs:
                registry.register("DocumentReference", doc["id"], doc)

        return docs

    def generate_for_all_encounters(
        self,
        registry: ReferenceRegistry,
        allowed_types: list[str] | None = None,
        register: bool = False,
    ) -> list[dict]:
        """Generate DocumentReferences for every encounter in the registry.

        Args:
            registry:       ReferenceRegistry with all FHIR resources.
            allowed_types:  Override the engine-level allowed_types.
            register:       If True, register all produced DocumentReferences.

        Returns:
            Flat list of all generated FHIR DocumentReference dicts.
        """
        encounter_ids = registry.get_ids("Encounter")
        all_docs: list[dict] = []

        for enc_id in encounter_ids:
            try:
                docs = self.generate_for_encounter(
                    registry,
                    enc_id,
                    allowed_types=allowed_types,
                    register=register,
                )
                all_docs.extend(docs)
            except Exception as exc:
                logger.warning("Skipping encounter %s due to error: %s", enc_id, exc)

        return all_docs
