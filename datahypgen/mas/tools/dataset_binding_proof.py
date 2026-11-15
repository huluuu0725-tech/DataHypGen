"""Stable proof contract for pre-plan dataset bindings."""

import copy
import hashlib
import json
from typing import Any, Dict, Tuple


BINDING_PROOF_VERSION = "dataset_binding_proof_v1"

_RUNTIME_REQUIREMENT_FIELDS = {
    "dataset_binding_proof",
    "dataset_resolution_failure",
    "resolved_dataset",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def requirement_core(requirement: Dict[str, Any]) -> Dict[str, Any]:
    """Return the model-owned requirement fields, excluding runtime resolution state."""
    if not isinstance(requirement, dict):
        raise TypeError("dataset requirement must be an object")
    return {
        key: copy.deepcopy(value)
        for key, value in requirement.items()
        if key not in _RUNTIME_REQUIREMENT_FIELDS
        and not str(key).startswith("resolution_")
    }


def requirement_fingerprint(requirement: Dict[str, Any]) -> str:
    payload = _canonical_json(requirement_core(requirement)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def proof_digest(proof: Dict[str, Any]) -> str:
    body = {
        key: copy.deepcopy(value)
        for key, value in proof.items()
        if key != "proof_digest"
    }
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def build_dataset_binding_proof(
    requirement: Dict[str, Any],
    resolved_dataset: Dict[str, Any],
    resolution_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a proof only from an accepted ResourceCheck strong binding."""
    if str(resolution_report.get("status") or "") != "resolved_strong":
        raise ValueError("dataset binding proof requires resolved_strong status")
    if not isinstance(resolved_dataset, dict) or not resolved_dataset:
        raise ValueError("dataset binding proof requires a resolved dataset object")
    if str(resolved_dataset.get("binding_status") or "") != "resolved_strong":
        raise ValueError("resolved dataset must carry resolved_strong binding_status")
    blocking_reasons = list(resolution_report.get("blocking_reasons") or [])
    if blocking_reasons:
        raise ValueError("resolved_strong proof cannot contain blocking reasons")

    proof = {
        "version": BINDING_PROOF_VERSION,
        "requirement_fingerprint": requirement_fingerprint(requirement),
        "status": "resolved_strong",
        "resolved_dataset": copy.deepcopy(resolved_dataset),
        "resolution_source": str(
            resolution_report.get("resolution_source")
            or resolved_dataset.get("resolution_source")
            or ""
        ),
        "constraint_report": copy.deepcopy(
            resolution_report.get("constraint_report") or {}
        ),
        "blocking_reasons": [],
        "frozen_constraints_satisfied": bool(
            resolution_report.get("frozen_constraints_satisfied", False)
        ),
        "identity_check": {
            "status": "accepted",
            "source": "resource_check_strict_resolution",
        },
    }
    proof["proof_digest"] = proof_digest(proof)
    return proof


def validate_dataset_binding_proof(
    requirement: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate proof integrity without re-inferring identity from a cropped title."""
    proof = requirement.get("dataset_binding_proof")
    if not isinstance(proof, dict):
        return False, "dataset_binding_proof_missing", {}
    if proof.get("version") != BINDING_PROOF_VERSION:
        return False, "dataset_binding_proof_version_invalid", {}
    if proof.get("requirement_fingerprint") != requirement_fingerprint(requirement):
        return False, "dataset_binding_proof_requirement_drift", {}
    if proof.get("proof_digest") != proof_digest(proof):
        return False, "dataset_binding_proof_tampered", {}
    if proof.get("status") != "resolved_strong":
        return False, "dataset_binding_proof_not_strong", {}
    if proof.get("blocking_reasons"):
        return False, "dataset_binding_proof_has_blocking_reasons", {}
    identity_check = proof.get("identity_check")
    if not isinstance(identity_check, dict) or identity_check.get("status") != "accepted":
        return False, "dataset_binding_proof_identity_not_accepted", {}

    resolved_dataset = proof.get("resolved_dataset")
    if not isinstance(resolved_dataset, dict) or not resolved_dataset:
        return False, "dataset_binding_proof_resolved_dataset_invalid", {}
    if resolved_dataset.get("binding_status") != "resolved_strong":
        return False, "dataset_binding_proof_resolved_dataset_not_strong", {}
    if requirement.get("resolution_status") != "resolved_strong":
        return False, "dataset_binding_proof_requirement_status_invalid", {}
    if requirement.get("resolved_dataset") != resolved_dataset:
        return False, "dataset_binding_proof_resolved_dataset_drift", {}
    return True, "", copy.deepcopy(resolved_dataset)
