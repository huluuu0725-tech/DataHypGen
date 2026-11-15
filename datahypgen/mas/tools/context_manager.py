"""Research context: content identity, source selection and memory integration."""

import copy
import hashlib
import json
import re

from datahypgen.mas.memory.memory_manager import InMemoryMemoryManager


def research_content_fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def tokens(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text.lower())) - {
        "the", "and", "for", "with", "from", "that", "this", "what", "does", "how",
        "research", "question", "evidence", "data", "source", "issue", "field", "status",
    }


def brief(value, limit=320):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…[摘要节选，原文保存在审计轨迹]"


def excerpt(text, query, limit=1400):
    if len(text) <= limit:
        return text
    matches = [(text.lower().find(term), term) for term in tokens(query)]
    positions = [pos for pos, term in matches if pos >= 0]
    start = max(0, min(positions) - 160) if positions else 0
    return text[start:start + limit]


class ContextReadUnavailable(RuntimeError):
    """No final decision was produced; this is not a scientific rejection."""

    def __init__(self, reason, details):
        super().__init__(reason)
        self.details = copy.deepcopy(details)


class ResearchContextManager:
    """Full source records stay outside prompts; selected excerpts are not sufficiency judgments."""

    def __init__(self, memory):
        self.memory = memory
        self._lookup_decisions = {}

    @staticmethod
    def idea_key(idea):
        if not isinstance(idea, dict):
            raise ValueError("Working context requires an explicit idea object")
        return research_content_fingerprint({key: node["hash"] for key, node in field_views(idea).items()})

    @staticmethod
    def issue_ids(value):
        if isinstance(value, dict):
            return ({value["issue_id"]} if isinstance(value.get("issue_id"), str) else set()).union(
                *(ResearchContextManager.issue_ids(v) for v in value.values()))
        if isinstance(value, (tuple, list)):
            return set().union(*(ResearchContextManager.issue_ids(v) for v in value))
        return set()

    def record_review(self, *, before, candidate, findings, audit, issues, round_number=0):
        """Store accepted context as a projection of actual review/writeback, never of advice."""
        key = self.idea_key(candidate)
        episode = {"kind": "revision_review", "round": round_number, "before_key": self.idea_key(before), "candidate_key": key,
            "issue_ids": sorted(self.issue_ids(findings)), "findings": copy.deepcopy(findings),
            "before": copy.deepcopy(before), "candidate": copy.deepcopy(candidate),
            "accepted": audit["can_commit"], "review": copy.deepcopy(audit)}
        self.memory.save_episode(episode)
        if audit["can_commit"]:
            self.memory.save_version(key, {"original_goal": copy.deepcopy(audit["goal_anchor"]),
                "goal_adjustment": copy.deepcopy(audit["goal_alignment"]),
                "issues": copy.deepcopy(issues), "obligations": copy.deepcopy(audit["obligations"]),
                "closure_outcomes": copy.deepcopy(audit["revision_acceptance"].get("finding_outcomes", []))
                    if audit["revision_acceptance"] is not None else [],
                "review_reason": audit.get("reason", ""), "source": "accepted_review_projection"})

    def working_view(self, idea, query, *, stage):
        key = self.idea_key(idea)
        versions = self.memory.versions.get(key, [])
        ids = self.issue_ids(query)
        result = {"idea_key": key, "context_available": bool(versions),
                  "policy": "Internal decisions are history, not external evidence. Missing history is not proof of no previous work."}
        if versions:
            current = copy.deepcopy(versions[-1])
            current["issues"] = [i for i in current["issues"] if not ids or i["issue_id"] in ids]
            current["issue_snapshot_policy"] = "Accepted-version snapshot only; live issue status in the current task payload takes precedence."
            relevant_fields = {field for finding in query.get("findings", []) for field in finding.get("dependent_fields", [])} if isinstance(query, dict) else set()
            if isinstance(query, dict) and isinstance(query.get("issue"), dict):
                relevant_fields.update(query["issue"].get("dependent_fields", []))
            if relevant_fields:
                current["obligations"] = [o for o in current["obligations"]
                    if any(path.split('.')[0].split('[')[0] in relevant_fields for path in o["field_ids"])]
            current["closure_outcomes"] = [o for o in current["closure_outcomes"]
                if not ids or any(f.get("finding_id") == o["finding_id"] and f.get("issue_id") in ids
                    for e in self.memory.episodes for f in e.get("findings", []))]
            if ids:
                current["review_reason"] = "\n".join(o["reason"] for o in current["closure_outcomes"])
            current["issues"] = [{k: i[k] for k in ("issue_id", "question", "closure_condition", "status") if k in i}
                                 for i in current["issues"]]
            current["closure_outcomes"] = [{k: brief(v, 500) if k == "reason" else v
                for k, v in outcome.items() if k in ("finding_id", "closure_met", "reason")}
                for outcome in current["closure_outcomes"]]
            result["current_understanding"] = current
        # Exact identity/version links first. Never treat lexical retrieval failure as no history.
        if stage in ("question", "answer", "adjudication", "revision", "review") and ids:
            result["history_index"] = [{"episode_id": n, "round": e["round"], "accepted": e["accepted"],
                "candidate_is_current": e["candidate_key"] == key}
                for n, e in enumerate(self.memory.episodes)
                if e["kind"] == "revision_review" and ids.intersection(e["issue_ids"])]
            related = [(n, e) for n, e in enumerate(self.memory.episodes)
                       if ids.intersection(e["issue_ids"]) and key in (e.get("before_key"), e.get("candidate_key"))]
            chosen = sorted({n for identifier in ids for n, e in related if identifier in e["issue_ids"]} )
            # At most the latest two relevant episodes per issue; full originals stay retrievable.
            selected = set()
            for identifier in ids:
                selected.update([n for n in chosen if identifier in self.memory.episodes[n]["issue_ids"]][-2:])
            result["related_history"] = [{"episode_id": n, "accepted": self.memory.episodes[n]["accepted"],
                "reason": self.memory.episodes[n]["review"].get("reason", ""),
                "outcome_summary": brief(self.memory.episodes[n]["review"].get("reason", ""), 500),
                "candidate_is_current": self.memory.episodes[n]["candidate_key"] == key,
                "kind": self.memory.episodes[n]["kind"]} for n in sorted(selected)]
        if stage != "score":
            result["previous_round_summaries"] = self.history()
            if not ids:
                result["history_index"] = [{"episode_id": n, "round": e["round"], "accepted": e["accepted"]}
                    for n, e in enumerate(self.memory.episodes) if e["kind"] == "revision_review"][-2:]
        return result

    def read_history(self, episode_id, part):
        if type(episode_id) is not int or not 0 <= episode_id < len(self.memory.episodes):
            raise ValueError("Unknown current-case episode")
        episode = self.memory.episodes[episode_id]
        if episode["kind"] != "revision_review":
            raise ValueError("History access must start from an indexed review episode")
        if part == "decision":
            return {"episode_id": episode_id, "accepted": episode["accepted"],
                    "findings": copy.deepcopy(episode["findings"]), "review": copy.deepcopy(episode["review"])}
        if part == "changes":
            return {"episode_id": episode_id, "accepted": episode["accepted"],
                    "before": copy.deepcopy(episode["before"]), "candidate": copy.deepcopy(episode["candidate"])}
        if part == "exchanges":
            ids = set(episode["issue_ids"])
            traces = [e for e in self.memory.episodes if e["kind"] == "round_trace" and e["round"] == episode["round"]]
            nodes = [n for e in traces for n in e["trace"].get("socratic_dual_tree", {}).get("new_nodes", [])]
            node_ids = {identifier for finding in episode["findings"]
                        for identifier in finding.get("source_node_ids", [])}
            # Native dual-tree reviews reference node IDs; issue-tree reviews reference stable issues.
            selected = {n["node_id"] for n in nodes if "node_id" in n and
                        (n.get("issue_id") in ids or n.get("node_id") in node_ids)}
            while True:
                parents = {n["parent_node_id"] for n in nodes if n.get("node_id") in selected and n.get("parent_node_id")}
                expanded = selected | parents
                if expanded == selected:
                    break
                selected = expanded
            return {"episode_id": episode_id, "record_available": bool(traces),
                "policy": "An incomplete round may not have a saved trace; missing trace is not proof that no exchange occurred.",
                "exchanges": [copy.deepcopy(n) for n in nodes if n.get("node_id") in selected or n.get("issue_id") in ids]}
        raise ValueError("Unknown history part")

    def ingest(self, evidence):
        context = evidence.get("context", {})
        documents = {}
        base_document_id = None
        groups = [("dataset", [context["base_dataset"]])] if context.get("base_dataset") else []
        groups.extend((kind, context.get(key, [])) for kind, key in (
            ("dataset", "candidate_datasets"), ("literature", "literature")))
        for record in evidence.get("records", []):
            groups.extend((kind, record.get(key, [])) for kind, key in (
                ("dataset", "dataset_results"), ("literature", "literature_results")))
        for kind, records in groups:
            for record in records:
                if not isinstance(record, dict):
                    continue
                identifier = kind + ":" + research_content_fingerprint(record)[:24]
                documents[identifier] = {"kind": kind, "record": copy.deepcopy(record)}
                if record == context.get("base_dataset"):
                    base_document_id = identifier
        cards = evidence_cards(evidence)
        source_documents = {"source:" + research_content_fingerprint({"kind": "literature" if d["kind"] == "literature" else "dataset_metadata",
            "record": d["record"]})[:24]: (identifier, d["record"]) for identifier, d in documents.items()}
        for card in cards.values():
            if card["source_id"] in source_documents:
                identifier, record = source_documents[card["source_id"]]
                card["source_document_id"] = identifier
                card["source_metadata"] = {key: copy.deepcopy(record[key]) for key in ("title", "doi", "url", "metadata_url") if key in record}
        self.memory.register(cards, documents, base_document_id=base_document_id)

    def select_cards(self, query, *, pinned=(), limit=18, char_budget=14000, stage=None):
        wanted = tokens(query)
        ranked = sorted(((len(tokens(c["text"]) & wanted), key, c) for key, c in self.memory.cards.items() if c["external"]),
                        key=lambda row: (-row[0], row[1]))
        selected = {}
        for key in pinned:
            if key in self.memory.cards and self.memory.cards[key]["external"]:
                selected[key] = copy.deepcopy(self.memory.cards[key])
        used = sum(len(c["text"]) for c in selected.values())
        for score, key, card in ranked:
            if score == 0 or key in selected or len(selected) >= limit:
                continue
            item = copy.deepcopy(card)
            item["text"] = excerpt(card["text"], query)
            item["excerpt_of_stored_source"] = True
            item["source_locator"] = {"card_id": key, "start": card["text"].find(item["text"]),
                                      "total_characters": len(card["text"])}
            if used + len(item["text"]) <= char_budget:
                selected[key] = item
                used += len(item["text"])
        self.memory.append_event({"stage": stage, "query": brief(query, 1000), "selected_ids": list(selected),
                            "characters": used, "status": "candidates_only_not_proof"})
        return selected

    def for_stage(self, stage, query, *, evidence=None, pinned=()):
        if stage not in ("grounding", "scan", "question", "answer", "adjudication", "revision", "review", "score"):
            raise ValueError("Unknown research context stage")
        if evidence is not None:
            self.ingest(evidence)
        search_query = query
        if isinstance(query, dict):
            if query.get("issue") is not None:
                search_query = query["issue"]
            elif "findings" in query:
                search_query = query["findings"]
        result = {"stage": stage, "selected_evidence": self.select_cards(search_query, pinned=pinned, stage=stage),
                "lookup_policy": "Locally retrieved excerpts only. Absence from this selection is not absence from the source or proof of unavailability."}
        if stage != "score":
            current = query.get("idea", {}) if isinstance(query, dict) else {}
            result["working_context"] = self.working_view(current, query, stage=stage)
        return result

    def read_source_span(self, identifier, start, end):
        """Exact external text only; offsets make large sources incrementally recoverable."""
        card = self.memory.cards[identifier]
        if not card["external"] or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(card["text"]) or end - start > 6000:
            raise ValueError("Invalid external source span")
        return {**copy.deepcopy(card), "text": card["text"][start:end],
                "source_locator": {"card_id": identifier, "start": start, "end": end,
                                   "total_characters": len(card["text"])}}

    @staticmethod
    def _missing_ranges(start, end, provided):
        cursor, missing = start, []
        for left, right in sorted(provided):
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                missing.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            missing.append((cursor, end))
        return missing

    @staticmethod
    def _append_context(prompt, extra):
        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError:
            return prompt + "\nContext read results:\n" + json.dumps(extra, ensure_ascii=False)
        parsed.setdefault("context_read_results", []).append(extra)
        return json.dumps(parsed, ensure_ascii=False)

    async def decide_with_sources(self, call_model, *, cards, history_ids=(), **kwargs):
        """Read only unseen spans; reserve a bounded final call with reading disabled."""
        # The caller supplies the program-owned index; source text cannot grant history access.
        history_ids = list(history_ids)
        if any(type(n) is not int or not 0 <= n < len(self.memory.episodes)
               or self.memory.episodes[n]["kind"] != "revision_review" for n in history_ids):
            raise ValueError("Invalid context history index")
        if not cards and not history_ids:
            return await call_model(**kwargs)
        base_schema = copy.deepcopy(kwargs["schema"])
        read_schema = copy.deepcopy(base_schema)
        read_schema["properties"]["source_requests"] = {"type": "array", "maxItems": 2, "items": {
            "type": "object", "additionalProperties": False, "properties": {
                "card_id": {"type": "string", **({"enum": list(cards)} if cards else {})},
                "start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 1}},
            "required": ["card_id", "start", "end"]}}
        if not cards:
            read_schema["properties"]["source_requests"]["maxItems"] = 0
        read_schema["properties"]["history_requests"] = {"type": "array", "maxItems": 2 if history_ids else 0,
            "items": {"type": "object", "additionalProperties": False,
                "properties": {"episode_id": {"type": "integer", **({"enum": history_ids} if history_ids else {})},
                    "part": {"type": "string", "enum": ["decision", "changes", "exchanges"]}},
                "required": ["episode_id", "part"]}}
        read_schema.setdefault("required", []).extend(["source_requests", "history_requests"])
        final_schema = copy.deepcopy(base_schema)
        final_schema["properties"].update({
            "context_completion": {"type": "string", "enum": ["complete", "insufficient_context"]},
            "missing_context": {"type": "string"}})
        final_schema.setdefault("required", []).extend(["context_completion", "missing_context"])
        provided = {}
        for identifier, card in cards.items():
            original = self.memory.cards[identifier]["text"]
            locator = card.get("source_locator")
            begin = locator["start"] if locator is not None else original.find(card["text"])
            if begin < 0 or original[begin:begin + len(card["text"])] != card["text"]:
                raise ValueError("Supplied source passage has no exact original location")
            provided[identifier] = [(begin, begin + len(card["text"]))]
        history_seen = set()
        prompt = self._append_context(kwargs["prompt"], {"source_inventory": [
            {"card_id": key, "provided_ranges": provided[key],
             "total_characters": len(self.memory.cards[key]["text"]),
             "unprovided_ranges": self._missing_ranges(0, len(self.memory.cards[key]["text"]), provided[key])}
            for key in cards], "policy": "Read only unprovided ranges. A fully supplied source needs no request."})
        finalize = False

        def unavailable(reason, attempt):
            details = {"kind": "context_read_unavailable", "reason": reason,
                       "calls": attempt + 1, "provisional_answer_consumed": False}
            self.memory.append_event(details)
            return ContextReadUnavailable(reason, details)

        for attempt in range(3):
            final_mode = finalize or attempt == 2
            instruction = (
                "Reading is complete and disabled. Use the supplied material to answer the task. "
                "Do not emit source_requests or history_requests. Return context_completion=complete "
                "and missing_context='' if you can complete this stage; otherwise use "
                "context_completion=insufficient_context and state the specific missing information. "
                "Do not invent evidence or claim an unperformed validation."
                if final_mode else
                "Context access: return source_requests=[] and history_requests=[] to answer now. "
                "Request at most two source spans of at most 6000 characters each or two indexed history parts. "
                "Already supplied ranges will not be read again. Other response fields are provisional "
                "while any request is present. An omitted excerpt is not proof of source absence."
            )
            raw = copy.deepcopy(await call_model(**{**kwargs,
                "schema": final_schema if final_mode else read_schema,
                "system_prompt": kwargs["system_prompt"] + "\n" + instruction, "prompt": prompt}))
            if not isinstance(raw, dict):
                raise unavailable("Model returned no usable context-stage response", attempt)
            requests = raw.pop("source_requests", [])
            history_requests = raw.pop("history_requests", [])
            if final_mode:
                if requests or history_requests:
                    raise unavailable("Model repeated a read request after reading was disabled", attempt)
                completion = raw.pop("context_completion", None)
                missing = raw.pop("missing_context", None)
                if completion != "complete" or not isinstance(missing, str) or missing.strip():
                    reason = missing if isinstance(missing, str) and missing.strip() else "Model did not complete the final context-stage answer"
                    raise unavailable(reason, attempt)
                self.memory.append_event({"kind": "context_finalized", "calls": attempt + 1})
                return raw
            if not requests and not history_requests:
                return raw
            if not isinstance(requests, list) or not isinstance(history_requests, list) or len(requests) > 2 or len(history_requests) > 2:
                raise unavailable("Invalid context read request", attempt)
            expanded, history, notices = {}, [], []
            try:
                for request in requests:
                    identifier = request["card_id"]
                    if identifier not in cards:
                        raise ValueError("Source request is outside the supplied inventory")
                    # Validate the original requested range even if it is already covered.
                    self.read_source_span(identifier, request["start"], request["end"])
                    missing_ranges = self._missing_ranges(request["start"], request["end"], provided[identifier])
                    if not missing_ranges:
                        notices.append({"source_request": request, "status": "already_provided"})
                    for left, right in missing_ranges:
                        previous = expanded.get(identifier, cards[identifier])
                        item = self.read_source_span(identifier, left, right)
                        item["delivered_spans"] = list(previous.get("delivered_spans", [previous["text"]]))
                        if item["text"] not in item["delivered_spans"]:
                            item["delivered_spans"].append(item["text"])
                        expanded[identifier] = item
                        provided[identifier].append((left, right))
                for request in history_requests:
                    pair = (request["episode_id"], request["part"])
                    if pair[0] not in history_ids:
                        raise ValueError("Unknown current-case review episode")
                    if pair in history_seen:
                        notices.append({"history_request": request, "status": "already_provided"})
                        continue
                    history.append(self.read_history(*pair))
                    history_seen.add(pair)
            except (KeyError, TypeError, ValueError) as exc:
                raise unavailable("Invalid context read: " + str(exc), attempt) from exc
            cards.update(expanded)
            finalize = not expanded and not history
            self.memory.append_event({"kind": "context_expansion", "source_requests": copy.deepcopy(requests),
                "history_requests": copy.deepcopy(history_requests), "provided_ranges": copy.deepcopy(provided),
                "notices": copy.deepcopy(notices), "made_progress": not finalize})
            prompt = self._append_context(prompt, {"expanded_external_sources": expanded,
                "historical_records_not_external_evidence": history, "read_notices": notices,
                "instruction": "Already-provided material remains in this prompt. Complete the task using it."})

    def evidence_view(self, evidence, query, *, stage="question"):
        return self.for_stage(stage, query, evidence={"context": evidence})

    def issue_view(self, issue):
        return {key: copy.deepcopy(value) for key, value in vars(issue).items() if key not in (
            "history", "dependency_fingerprint", "evidence_fingerprint", "last_served")}

    def complete_round(self, trace):
        self.memory.save_episode({"kind": "round_trace", "round": trace["round"],
            "issue_ids": sorted(self.issue_ids(trace.get("questioning_findings", []))), "trace": copy.deepcopy(trace)})
        nodes = trace.get("socratic_dual_tree", {}).get("new_nodes", [])
        questions = list(dict.fromkeys(n.get("question", "") for n in nodes))
        answers = list(dict.fromkeys(n.get("answer", "") for n in nodes))
        changes = [{"field": item["field"], "written": brief(item.get("after"), 180)}
                   for item in trace.get("applied_idea_field_updates", [])]
        acceptance = trace.get("acceptance", {})
        summary = (f"第{trace['round']}轮：主要质询：{brief(questions, 650)}；"
                   f"回答要点：{brief(answers, 950)}；实际写回：{brief(changes, 650)}；"
                   f"是否接受：{trace.get('accepted')}；审核意见：{brief(acceptance.get('reason', ''), 500)}。"
                   "回答中的建议不等于已实施，摘要不构成外部证据或问题关闭证明。")
        self.memory.save_summary({"round": trace["round"], "summary": summary})

    def history(self):
        return [r["summary"] for r in self.memory.rounds]

    async def lookup(self, channel, finding, call_model):
        """Check directory first; reuse only after a source-cited sufficiency decision."""
        query = finding.get("evidence_need", "")
        if not query:
            return None
        wanted = tokens(query)
        ranked = sorted(((len(tokens(doc["record"]) & wanted), key, doc) for key, doc in self.memory.documents.items()
                         if doc["kind"] == channel), key=lambda row: (-row[0], row[1]))
        selected = {key: doc for score, key, doc in ranked[:5] if score > 0}
        if not selected:
            return None
        field = "literature_results" if channel == "literature" else "dataset_results"
        snippets = {}
        for key, doc in selected.items():
            cards = evidence_cards({"records": [{field: [doc["record"]]}]})
            text = "\n".join(card["text"] for card in cards.values() if card["external"])
            snippets[key] = excerpt(text, query, 2200)
        cache_key = research_content_fingerprint({"channel": channel, "need": query, "scope": {k: finding[k] for k in (
            "question", "closure_condition", "affected_claim", "target_field", "dependent_fields",
            "required_contrast") if k in finding}, "sources": snippets})
        cached = self._lookup_decisions.get(cache_key)
        if cached is not None:
            self.memory.append_event({"kind": "local_sufficiency_cache", "channel": channel,
                                      "cache_key": cache_key, "reused": cached["outcome"] is not None})
            return copy.deepcopy(cached["outcome"])
        decision = await call_model(system_prompt="Check whether locally stored external sources satisfy the specific information need. Topic overlap alone is insufficient. Future empirical results cannot be replaced by plans.",
            prompt=json.dumps({"need": query, "sources": snippets}, ensure_ascii=False),
            schema={"type": "object", "additionalProperties": False, "properties": {
                "sufficient": {"type": "boolean"}, "reason": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "properties": {"source_id": {"type": "string", "enum": list(selected)}, "quote": {"type": "string"}},
                    "required": ["source_id", "quote"]}}}, "required": ["sufficient", "reason", "citations"]},
            temperature=0.0, max_tokens=1200)
        citations = decision["citations"]
        valid = bool(citations) and all(c["source_id"] in snippets and c["quote"].strip() and c["quote"] in snippets[c["source_id"]] for c in citations)
        reused = decision["sufficient"] is True and valid
        self.memory.append_event({"channel": channel, "need": query, "decision": decision, "reused": reused})
        if not reused:
            self._lookup_decisions[cache_key] = {"outcome": None}
            return None
        identifiers = list(dict.fromkeys(c["source_id"] for c in citations))
        outcome = {"query": query, "query_source": "local_evidence_directory", "results": [copy.deepcopy(selected[k]["record"]) for k in identifiers],
                "sources": identifiers, "errors": [], "local_reuse_decision": decision}
        self._lookup_decisions[cache_key] = {"outcome": copy.deepcopy(outcome)}
        return outcome


def ensure_research_context(agent):
    if hasattr(agent, "research_context"):
        return agent.research_context
    manager = getattr(agent, "memory_manager", None)
    if manager is None:
        manager = InMemoryMemoryManager()
        agent.memory_manager = manager
    scope = agent.research_memory_scope if hasattr(agent, "research_memory_scope") else "isolated-idea"
    directory = getattr(agent, "socratic_context_directory", None)
    agent.research_context = ResearchContextManager(manager.open_research_memory(scope, directory=directory))
    return agent.research_context


def preserve_content_schema(schema):
    """Keep shape validation; make string length guidance non-destructive."""
    value = copy.deepcopy(schema)
    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "string":
                limit = node.pop("maxLength", None)
                if limit is not None:
                    node["description"] = node.get("description", "") + (
                        f" Prefer about {limit} characters; complete the scientific meaning. Never cut a sentence.")
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(value)
    return value


def model_stage(schema):
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    for field, stage in (("revised_idea", "revision"), ("revision_target_resolved", "review"),
                         ("checks", "review"), ("coverage", "scan"), ("same_issue", "adjudication"),
                         ("verdict", "adjudication"), ("question", "question"),
                         ("answer_status", "answer"), ("sufficient", "local_evidence_check"),
                         ("idea_operational_coherence", "score")):
        if field in properties:
            return stage
    return "other_model_stage"


IDEA_FIELDS = (
    "knowledge_anchor_and_motivation", "data_synergy_and_gap", "research_question",
    "core_hypothesis", "theoretical_mechanism", "computability_and_falsifiability",
    "novelty_constraints_and_value", "component_usage",
)


RESPONSIBILITIES = {
    "research_question": "Research goal; not an empirical finding.",
    "core_hypothesis": "Hypothesis to test; affirmative wording alone is not a result.",
    "theoretical_mechanism": "Proposed explanation and its interpretation conditions.",
    "computability_and_falsifiability": "Observations, contrasts and falsification design; not observed results.",
    "novelty_constraints_and_value": "Expected contribution and scope; not demonstrated benefit.",
    "knowledge_anchor_and_motivation": "Background and motivation; factual assertions must match evidence.",
    "data_synergy_and_gap": "Data capabilities, limitations and necessity.",
    "component_usage": "Proposed use of each data component.",
}


def field_views(idea):
    nodes = {}

    def walk(value, path, field):
        if isinstance(value, str) and value.strip():
            nodes[path] = {"id": path, "field": field, "text": value,
                "responsibility": RESPONSIBILITIES[field], "hash": research_content_fingerprint(value)}
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + "." + key, field)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + f"[{index}]", field)

    for field in IDEA_FIELDS:
        if field in idea:
            walk(idea[field], field, field)
    return nodes


def evidence_cards(evidence):
    """Keep source provenance; review prose is never external evidence.

    Only known context/retrieval containers supply external records. Even there,
    classifier decisions and generated method summaries remain internal context.
    """
    cards = {}
    paper_fields = {"title", "year", "journal", "doi", "url", "authors", "abstract",
                    "full_text", "text", "excerpt"}
    paper_text_fields = {"abstract", "full_text", "text", "excerpt"}
    dataset_fields = {"id", "title", "doi", "type", "provider", "source", "dataset_source",
        "parameters", "data_url", "metadata_url", "docs", "is_collection", "coverage",
        "spatial_extent", "temporal_extent", "gsd", "resolution", "cadence", "bands",
        "description", "detail", "relations", "relation_count", "spatial_coverage"}
    internal_fields = {"modification_direction", "resolution_hint", "relevance", "relevance_filter",
                       "retrieval_errors", "evidence_summary", "queries", "required_channel_decision"}

    def walk(value, path, source_type="internal", source_id="internal", empirical=False):
        if isinstance(value, str) and value.strip():
            provenance = {"source_type": source_type, "source_id": source_id,
                          "external": source_type != "internal", "empirical_result_candidate": empirical}
            identifier = "e:" + research_content_fingerprint({"text": value, **provenance})[:24]
            if identifier in cards:
                cards[identifier]["paths"].append(path)
            else:
                cards[identifier] = {"text": value, "paths": [path], **provenance}
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in internal_fields:
                    walk(item, path + "." + key)
                else:
                    walk(item, path + "." + key, source_type, source_id, empirical)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + f"[{index}]", source_type, source_id, empirical)
        elif value is not None and not isinstance(value, str):
            walk(str(value), path, source_type, source_id, empirical)

    def source(record, path, kind):
        if not isinstance(record, dict):
            walk(record, path)
            return
        source_id = "source:" + research_content_fingerprint({"kind": kind, "record": record})[:24]
        fields = paper_fields if kind == "literature" else dataset_fields
        for key, value in record.items():
            if key in fields and not (kind == "literature" and key in paper_text_fields and not isinstance(value, str)):
                walk(value, path + "." + key, kind, source_id,
                     empirical=kind == "literature" and key in paper_text_fields)
            else:
                walk(value, path + "." + key)

    def container(value, path, kind):
        if isinstance(value, list):
            for index, record in enumerate(value):
                source(record, path + f"[{index}]", kind)
        else:
            source(value, path, kind)

    for key, value in evidence.items():
        if key == "context" and isinstance(value, dict):
            for name, content in value.items():
                path = "evidence.context." + name
                if name in ("base_dataset", "candidate_datasets", "literature"):
                    container(content, path, "literature" if name == "literature" else "dataset_metadata")
                else:
                    walk(content, path)
        elif key == "records" and isinstance(value, list):
            for index, record in enumerate(value):
                path = f"evidence.records[{index}]"
                if not isinstance(record, dict):
                    walk(record, path)
                    continue
                for name, content in record.items():
                    if name in ("literature_results", "dataset_results"):
                        container(content, path + "." + name,
                                  "literature" if name == "literature_results" else "dataset_metadata")
                    else:
                        walk(content, path + "." + name)
        else:
            walk(value, "evidence." + key)
    return cards
