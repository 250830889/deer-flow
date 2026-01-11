from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage


def serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            serialized.append(
                {
                    "type": getattr(msg, "type", msg.__class__.__name__),
                    "name": getattr(msg, "name", None),
                    "content": msg.content,
                    "additional_kwargs": getattr(msg, "additional_kwargs", {}),
                }
            )
        elif isinstance(msg, dict):
            serialized.append(msg)
        else:
            serialized.append(
                {
                    "type": msg.__class__.__name__,
                    "content": str(msg),
                }
            )
    return serialized


def normalize_usage_metadata(usage: Any) -> Any:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, dict):
        return usage
    return {"value": str(usage)}


def extract_response_usage(response: Any) -> Any:
    if response is None:
        return None
    usage = getattr(response, "usage_metadata", None)
    normalized = normalize_usage_metadata(usage)
    if normalized:
        return normalized
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get(
            "usage"
        )
        return normalize_usage_metadata(token_usage)
    return None
