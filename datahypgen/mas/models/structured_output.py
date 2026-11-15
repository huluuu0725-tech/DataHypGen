"""Shared structured-output parsing, validation, and corrective retry helpers."""

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from json_repair import repair_json

try:
    from jsonschema.validators import validator_for as _jsonschema_validator_for
except Exception:  # pragma: no cover - syntax-only fallback when dependency is absent
    _jsonschema_validator_for = None


logger = logging.getLogger(__name__)
_MAX_REPORTED_SCHEMA_VIOLATIONS = 8


def parse_json_object(text: str, provider_name: str) -> Dict[str, Any]:
    """Parse a JSON object, applying the existing json-repair fallback."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text)
        payload = json.loads(repaired) if repaired else None
    if not isinstance(payload, dict):
        raise ValueError(
            f"{provider_name} returned JSON type {type(payload).__name__}, expected object"
        )
    return payload


def normalize_payload(payload: Any, schema: Dict[str, Any]) -> Tuple[Any, int, int, int]:
    """Drop schema-forbidden fields, flatten object items in string arrays, truncate bounded strings."""
    removed = 0
    truncated = 0
    flattened = 0

    def visit(value: Any, node: Dict[str, Any]) -> Any:
        nonlocal removed, truncated, flattened
        if not isinstance(node, dict):
            return value
        node_type = node.get("type")
        if node_type == "object" and isinstance(value, dict):
            properties = node.get("properties", {})
            if node.get("additionalProperties") is False and isinstance(properties, dict):
                for key in list(value):
                    if key not in properties:
                        del value[key]
                        removed += 1
            if isinstance(properties, dict):
                for key, child_schema in properties.items():
                    if key in value and isinstance(child_schema, dict):
                        value[key] = visit(value[key], child_schema)
            return value
        if node_type == "array" and isinstance(value, list):
            item_schema = node.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    if item_schema.get("type") == "string" and isinstance(item, dict):
                        parts = [part for part in item.values() if isinstance(part, str) and part.strip()]
                        if parts:
                            item = " — ".join(parts)
                            flattened += 1
                    value[index] = visit(item, item_schema)
            return value
        if node_type == "string" and isinstance(value, str):
            limit = node.get("maxLength")
            if isinstance(limit, int) and limit >= 0 and len(value) > limit:
                truncated += 1
                return value[:limit]
        return value

    return visit(payload, schema), removed, truncated, flattened


def schema_violations(payload: Any, schema: Dict[str, Any]) -> List[str]:
    """Return concise JSON Schema violations; fail open only if validation is unavailable."""
    if _jsonschema_validator_for is None or not isinstance(schema, dict):
        return []
    try:
        validator_cls = _jsonschema_validator_for(schema)
        validator_cls.check_schema(schema)
        errors = list(validator_cls(schema).iter_errors(payload))
    except Exception as exc:  # schema/dependency problems must not block generation
        logger.debug("Skipping structured-output schema validation: %s", exc)
        return []

    violations: List[str] = []
    for error in errors[:_MAX_REPORTED_SCHEMA_VIOLATIONS]:
        location = "$" + "".join(
            f"[{part!r}]" if isinstance(part, str) else f"[{part}]"
            for part in error.absolute_path
        )
        violations.append(f"{location}: {error.message}")
    return violations


async def enforce_schema_with_retry(
    *,
    payload: Dict[str, Any],
    schema: Dict[str, Any],
    retry: Callable[[List[str]], Awaitable[Dict[str, Any]]],
    provider_name: str,
    max_correction_retries: int = 1,
) -> Dict[str, Any]:
    """Normalize a payload and perform bounded provider-specific corrective retries."""
    payload, removed, truncated, flattened = normalize_payload(payload, schema)
    if removed or truncated or flattened:
        logger.info(
            "Normalized %s structured output locally "
            "(removed_extra_fields=%d, truncated_strings=%d, flattened_object_items=%d)",
            provider_name,
            removed,
            truncated,
            flattened,
        )
    violations = schema_violations(payload, schema)
    if not violations:
        return payload

    attempts = max(1, int(max_correction_retries))
    for attempt in range(1, attempts + 1):
        logger.warning(
            "%s output violates the requested schema (%d issue(s)); corrective retry %d/%d: %s",
            provider_name,
            len(violations),
            attempt,
            attempts,
            "; ".join(violations),
        )
        retry_payload = await retry(violations)
        retry_payload, removed, truncated, flattened = normalize_payload(retry_payload, schema)
        if removed or truncated or flattened:
            logger.info(
                "Normalized corrective %s output locally "
                "(removed_extra_fields=%d, truncated_strings=%d, flattened_object_items=%d)",
                provider_name,
                removed,
                truncated,
                flattened,
            )
        violations = schema_violations(retry_payload, schema)
        if not violations:
            logger.info("%s schema-corrective retry produced a conforming output", provider_name)
            return retry_payload

    raise ValueError(
        f"{provider_name} output still violates the requested schema after corrective retry: "
        + "; ".join(violations)
    )
