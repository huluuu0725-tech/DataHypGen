"""Scoped research storage shared by workflow memory and prompt context consumers."""
import copy
import hashlib
import json
import re
from pathlib import Path


def validate_scope(scope):
    if not isinstance(scope, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", scope) is None:
        raise ValueError("Research memory requires a single safe session/case scope")
    return scope


class ResearchMemory:
    """Own persistence and recovery; never rank evidence or decide scientific acceptance."""

    def __init__(self, scope, directory=None):
        self.scope = validate_scope(scope)
        self.directory = Path(directory) if directory is not None else None
        self.cards = {}
        self.documents = {}
        self.rounds = []
        self.versions = {}
        self.episodes = []
        self.events = []
        self.checkpoint = None
        self.base_document_id = None
        self.initialized = False
        if self.directory is not None:
            state = self.directory / "memory_state.json"
            if state.exists():
                value = json.loads(state.read_text(encoding="utf-8"))
                if value["schema_version"] != "research_memory_v2" or value["scope"] != scope:
                    raise ValueError("Research memory schema or scope mismatch")
                for key in ("cards", "rounds", "versions", "episodes", "events", "checkpoint", "base_document_id", "initialized"):
                    setattr(self, key, value[key])
                for identifier in value["document_ids"]:
                    self.documents[identifier] = self.read_source(identifier)
            elif (self.directory / "index.json").exists():
                raise ValueError("Legacy context directory requires explicit migration; refusing to overwrite it")

    def _path(self, identifier):
        if re.fullmatch(r"(?:dataset|literature):[a-f0-9]{24}", identifier) is None:
            raise ValueError("Invalid research source ID")
        return self.directory / "sources" / (identifier.replace(":", "_") + ".json")

    def read_source(self, identifier):
        if identifier in self.documents:
            return copy.deepcopy(self.documents[identifier])
        if self.directory is None:
            raise KeyError(identifier)
        value = json.loads(self._path(identifier).read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(value["record"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
        if identifier != value["kind"] + ":" + digest:
            raise ValueError("Stored research source content does not match its ID")
        return value

    def write_json(self, name, value):
        if self.directory is None:
            return
        target = self.directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".writing")
        staging.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(target)

    def persist(self):
        self.write_json("memory_state.json", {"schema_version": "research_memory_v2", "scope": self.scope,
            "cards": self.cards, "document_ids": list(self.documents), "rounds": self.rounds, "events": self.events,
            "versions": self.versions, "episodes": self.episodes,
            "checkpoint": self.checkpoint, "base_document_id": self.base_document_id, "initialized": self.initialized})

    def register(self, cards, documents, *, base_document_id=None):
        self.cards.update(copy.deepcopy(cards))
        for identifier, document in documents.items():
            if self.directory is not None:
                self._path(identifier)
            if identifier not in self.documents:
                self.documents[identifier] = copy.deepcopy(document)
                self.write_json("sources/" + identifier.replace(":", "_") + ".json", document)
        if base_document_id is not None:
            self.base_document_id = base_document_id
        self.persist()
        self.write_json("index.json", [{"id": key, "kind": doc["kind"], "title": doc["record"].get("title"),
            "doi": doc["record"].get("doi"), "path": "sources/" + key.replace(":", "_") + ".json"}
            for key, doc in self.documents.items()])

    def append_event(self, event):
        self.events.append(copy.deepcopy(event))
        self.persist()
        self.write_json("lookups.json", self.events)

    def save_summary(self, summary):
        found = next((r for r in self.rounds if r["round"] == summary["round"]), None)
        if found is not None:
            if found != summary:
                raise ValueError("A saved round summary cannot be silently overwritten")
            return
        self.rounds.append(copy.deepcopy(summary))
        self.persist()
        self.write_json("round_summaries.json", self.rounds)

    def save_checkpoint(self, value):
        self.checkpoint = copy.deepcopy(value)
        self.persist()

    def save_version(self, key, value):
        records = self.versions.setdefault(key, [])
        if records and records[-1] == value:
            return
        records.append(copy.deepcopy(value))
        self.persist()

    def save_episode(self, value):
        self.episodes.append(copy.deepcopy(value))
        self.persist()
        self.write_json("issue_episodes.json", self.episodes)

    def evidence_context(self):
        result = {"candidate_datasets": [], "literature": []}
        if self.base_document_id is not None:
            result["base_dataset"] = copy.deepcopy(self.documents[self.base_document_id]["record"])
        for identifier, doc in self.documents.items():
            if identifier != self.base_document_id:
                result["literature" if doc["kind"] == "literature" else "candidate_datasets"].append(copy.deepcopy(doc["record"]))
        return result
