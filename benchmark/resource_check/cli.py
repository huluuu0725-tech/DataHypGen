"""Shared, resumable native resource evaluation command."""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

from benchmark.resource_check.semantics import VERSION, ResourceSemantics, canonical_json, validate_extraction


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "benchmark/resource_check/config.yaml"
SYSTEMS = {"deepscientist", "denario", "ideagent"}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def merge(base, overlay):
    result = dict(base)
    for key, value in overlay.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    paths = list(dict.fromkeys([ROOT / settings["base_config"], ROOT / settings["resource_config"]]))
    config = {}
    for source in paths:
        config = merge(config, yaml.safe_load(source.read_text(encoding="utf-8")))
    options = settings["semantics"]
    for name in ("max_tokens", "tool_candidate_limit"):
        if type(options[name]) is not int or options[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if options["model_provider"] not in config["models"]:
        raise ValueError("Semantic model provider must be configured explicitly")
    evidence_index = ROOT / config["tools"]["capability_index"]["path"]
    sources = [path, *paths, evidence_index, *sorted(Path(__file__).parent.glob("*.py")),
               *sorted((evidence_index.parent / "geo_capability_details").glob("*.json"))]
    fingerprint = hashlib.sha256(canonical_json([file_hash(item) for item in sources]).encode()).hexdigest()
    return config, options, fingerprint


def load_bundle(path, suite):
    bundle = read_json(path)
    request_path = path.parent / "request.json"
    request = read_json(request_path)
    if bundle["schema_version"] != "native_research_proposal_v1" or bundle["boundary"] != "planned":
        raise ValueError("Unsupported native proposal contract")
    if bundle["system"] not in SYSTEMS:
        raise ValueError("Unsupported native system")
    if bundle["request_sha256"] != file_hash(request_path):
        raise ValueError("Request checksum mismatch")
    for key in ("system", "task"):
        if bundle[key] != request[key]:
            raise ValueError(f"Bundle/request {key} mismatch")
    metadata = request["benchmark"]
    if "task_suite" in metadata:
        actual_suite = metadata["task_suite"]
    elif re.fullmatch(r"(single_dataset|dual_dataset|three_or_more_datasets)/task_\d{3}", request["task"]):
        # Early benchmark bundles predate the explicit task_suite field.
        actual_suite = "benchmark"
    else:
        raise ValueError("Request lacks an identifiable task suite")
    if actual_suite != suite:
        raise ValueError(f"Task suite does not match {suite} entry point")
    if not bundle["proposals"] or len(bundle["proposals"]) != bundle["collected_proposal_count"]:
        raise ValueError("Empty or inconsistent proposal bundle")
    if (bundle["requested_proposal_count"] < bundle["collected_proposal_count"]
            or bundle["missing_proposal_count"] != bundle["requested_proposal_count"] - bundle["collected_proposal_count"]):
        raise ValueError("Invalid requested/collected/missing proposal counts")
    artifacts = {item["path"]: item["sha256"] for item in bundle["artifacts"]}
    sources = [item["source_artifact"] for item in bundle["proposals"]]
    if len(set(sources)) != len(sources):
        raise ValueError("Duplicate proposal source artifacts")
    for item in bundle["proposals"]:
        artifact = (path.parent / item["source_artifact"]).resolve()
        if not artifact.is_relative_to(path.parent.resolve()):
            raise ValueError("Artifact path escapes the bundle")
        if file_hash(artifact) != artifacts[item["source_artifact"]]:
            raise ValueError("Proposal artifact checksum mismatch")
        if artifact.read_text(encoding="utf-8-sig") != item["native_proposal"]:
            raise ValueError("Proposal text differs from source artifact")
    return bundle, request


def save_new_or_replace(path, payload, overwrite):
    # Never truncate an existing historical artifact without --overwrite.
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


async def evaluate_path(path, *, suite, config, options, fingerprint, extract_only=False,
                        dry_run=False, overwrite=False, semantics=None, checker=None):
    bundle, request = load_bundle(path, suite)
    protocol = {"version": VERSION, "config_sha256": fingerprint, "bundle_sha256": file_hash(path)}
    extraction_path = path.parent / "resource_requirements.json"
    result_path = path.parent / "resource_check_evaluation.json"
    if dry_run:
        return {"path": str(path), "status": "dry_run", "proposals": len(bundle["proposals"])}
    if not extract_only and result_path.exists() and not overwrite:
        existing = read_json(result_path)
        if (existing.get("protocol") != protocol or not extraction_path.is_file()
                or existing.get("extraction_sha256") != file_hash(extraction_path)):
            raise ValueError(f"Stale result; select --overwrite explicitly: {result_path}")
        return {"path": str(result_path), "status": "skipped"}
    cached = None
    if extraction_path.exists() and not overwrite:
        cached = read_json(extraction_path)
        if cached.get("protocol") != protocol:
            raise ValueError(f"Stale extraction; select --overwrite explicitly: {extraction_path}")
        if len(cached["proposals"]) != len(bundle["proposals"]):
            raise ValueError("Cached extraction proposal count mismatch")
        for row, proposal in zip(cached["proposals"], bundle["proposals"]):
            if row["source_artifact"] != proposal["source_artifact"]:
                raise ValueError("Cached extraction source mismatch")
            validate_extraction(row["extraction"], proposal["native_proposal"], request["task_prompt"])
    if extract_only and cached is not None:
        return {"path": str(extraction_path), "status": "skipped"}
    if semantics is None:
        from dotenv import load_dotenv
        from datahypgen.mas.models.model_factory import ModelFactory
        load_dotenv(ROOT / ".env", override=True)
        model_config = {key: options[key] for key in ("model_provider", "temperature", "max_tokens")}
        model_config["_global_config"] = config
        model = ModelFactory.create_model_for_agent("resource_semantics", model_config)
        semantics = ResourceSemantics(model, temperature=options["temperature"], max_tokens=options["max_tokens"])
    from benchmark.resource_check.native import extract_bundle, evaluate_native
    if cached is None:
        rows = await extract_bundle(bundle, request, semantics)
        cached = {"protocol": protocol, "system": bundle["system"], "task": bundle["task"], "proposals": rows}
        save_new_or_replace(extraction_path, cached, overwrite)
    if extract_only:
        return {"path": str(extraction_path), "status": "extracted"}
    if checker is None:
        from benchmark.resource_check.checker import ExperimentResourceChecker
        checker = ExperimentResourceChecker(config)
    applications = await evaluate_native(
        cached["proposals"], request["task_prompt"], checker=checker, semantics=semantics,
        tool_candidate_limit=options["tool_candidate_limit"],
    )
    passed = sum(app["resource_check"]["execution_contract"]["resource_check_passed"] for app in applications)
    result = {"protocol": protocol, "extraction_sha256": file_hash(extraction_path),
              "evaluation_scope": "resource_check", "task_suite": suite,
              "system": bundle["system"], "task": bundle["task"], "applications": applications,
              "metrics": {"evaluated_proposals": len(applications), "passed_proposals": passed,
                          "pass_rate": passed / len(applications),
                          "requested_proposals": bundle["requested_proposal_count"],
                          "missing_proposals": bundle["missing_proposal_count"]}}
    save_new_or_replace(result_path, result, overwrite)
    return {"path": str(result_path), "status": "evaluated"}


def main(suite, *, default_input=None):
    parser = argparse.ArgumentParser(description="Shared native-proposal resource evaluation")
    default_root = ROOT / "benchmark/system_outputs"
    parser.add_argument("--input", type=Path, default=default_root if default_input is None else default_input,
                        help="One proposals.json or a directory recursively containing bundles")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--system", choices=sorted(SYSTEMS))
    parser.add_argument("--task")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if suite == "project" and default_input is None and not any(arg == "--input" or arg.startswith("--input=") for arg in sys.argv[1:]):
        parser.error("project suite requires an explicit --input path")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob("proposals.json"))
    if not paths:
        parser.error("No native proposal bundles found")
    config, options, fingerprint = load_config(args.config)

    async def run():
        selected = 0
        failures = 0
        for path in paths:
            try:
                header = read_json(path)
                if header.get("system") not in SYSTEMS:
                    continue
                if args.system and header["system"] != args.system:
                    continue
                if args.task and header.get("task") != args.task:
                    continue
                selected += 1
                print(f"Checking {path}", flush=True)
                result = await evaluate_path(
                    path, suite=suite, config=config, options=options, fingerprint=fingerprint,
                    extract_only=args.extract_only, dry_run=args.dry_run, overwrite=args.overwrite,
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as error:
                failures += 1
                print(json.dumps({"path": str(path), "status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
            if args.limit is not None and selected >= args.limit:
                break
        return 1 if failures or not selected else 0

    return asyncio.run(run())
