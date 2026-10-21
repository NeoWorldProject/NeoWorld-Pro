"""Response identity and cache provenance for the durable request ledger."""
from __future__ import annotations


class RecordedText(str):
    def __new__(cls, text, metadata):
        result = super().__new__(cls, text)
        result.api_metadata = receipt_metadata(metadata)
        return result


def metadata(response):
    return {"response_id": getattr(response, "id", None),
            "request_id": getattr(response, "_request_id", None),
            "model": getattr(response, "model", None),
            "provider_status": getattr(response, "status", None)}


def receipt_metadata(value):
    """Persist identity and cache settings, including when importing old receipts."""
    return {key: value[key] for key in (
        "response_id", "request_id", "model", "provider_status", "prompt_cache_key", "prompt_cache")
        if key in value}
