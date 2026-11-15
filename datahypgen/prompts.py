# Legacy prompt - no longer used. Kept for reference only.
# All evaluation now uses skill-based reviewers from skills/ directory.
IDEA_QUALITY_EVALUATION_SYSTEM_PROMPT_LEGACY = (
    "You are a skeptical scientific-idea evaluator and professional geographer. "
    "Evaluate exactly one named dimension using only the supplied evidence pack. "
    "Choose anchored subcriterion scores and cite concrete source fields. "
    "Do not output a dimension total, confidence, overall score, ranking, gate, feasibility, "
    "promotion decision, or execution recommendation. "
    "\n\n"
    "CRITICAL: You MUST return all four required fields in your JSON response:\n"
    "1. subscores: object with subcriterion scores\n"
    "2. supporting_evidence: array of evidence items (required, cannot be omitted)\n"
    "3. counterevidence_or_gaps: array of counterevidence or gap descriptions (required, cannot be omitted - even if empty, return [])\n"
    "4. rationale: string with your overall assessment (required, cannot be omitted - provide a concise 1-2 sentence rationale)\n"
    "\n"
    "Omitting rationale, counterevidence_or_gaps, or any other required field will cause evaluation to fail. "
    "Always include rationale and counterevidence_or_gaps even if they are brief. "
    "Return strict JSON matching the schema."
)


# Legacy rubrics - no longer used. Kept for reference only.
# All evaluation now uses skill-based reviewers from skills/ directory.
IDEA_QUALITY_DIMENSION_RUBRICS_LEGACY = {
    "realizability": {
        "question": "Does the idea avoid fatal conceptual blockers and have a foreseeable technical path to validation?",
        "subcriteria": {
            "conceptual_consistency": "The research question, hypothesis, mechanism, and claim do not contain a fatal internal contradiction.",
            "observable_in_principle": "The core variables are observable or constructible in principle from foreseeable geospatial, remote-sensing, statistical, or public auxiliary data.",
            "technical_path_plausibility": "Existing GIS, remote-sensing, statistical, or machine-learning methods provide a plausible path to turn the idea into an empirical study.",
            "fatal_blocker_absence": "No irreversible blocker is evident, such as impossible measurement, temporal reversal, incompatible data semantics, or unrecoverable scale mismatch.",
        },
    },
    "novelty": {
        "question": "What substantive scientific content is new relative to the closest retrieved work or historical idea?",
        "subcriteria": {
            "closest_work_difference": "The idea states a concrete difference from its closest evidence-backed neighbor.",
            "non_incremental_contribution": "The difference changes the question, explanation, or knowledge gain rather than only data, place, or model choice.",
            "non_obviousness": "The contribution requires a non-trivial inference or data relationship not directly implied by prior work.",
        },
    },
    "data_grounding": {
        "question": "Can the supplied data support the variables, scope, and strength of the idea's claims?",
        "subcriteria": {
            "target_observability": "The target variables, proxies, and labels are observable or credibly constructible.",
            "spatiotemporal_scope_fit": "Coverage, resolution, temporal semantics, and geographic scope match the research question.",
            "source_role_adequacy": "The declared data sources are sufficient for the core claim and have clear roles: a single source has no hidden dependency, while multiple sources contribute necessary, non-redundant information.",
            "claim_calibration": "The claim strength and applicability boundaries respect the data limitations.",
        },
    },
    "logical_coherence": {
        "question": "Does the idea's motivation, question, hypothesis, mechanism, and observable expectation form a connected, contradiction-free scientific chain?",
        "subcriteria": {
            "motivation_question_link": "The research question directly addresses the stated motivation or knowledge gap.",
            "question_hypothesis_link": "The directional hypothesis answers the research question without changing the scientific object.",
            "mechanism_observable_expectation_link": "The mechanism entails the observable expectation without requiring a complete validation or falsification design.",
            "internal_consistency": "The variables, causal strength, scope, and assumptions contain no unsupported logical jump or contradiction.",
        },
    },
    "significance": {
        "question": "If answered reliably, what meaningful scientific or practical change would follow?",
        "subcriteria": {
            "problem_consequence": "The question concerns a scientifically or practically consequential phenomenon.",
            "knowledge_gain": "The result would materially change explanation, prediction, identification, or intervention capability.",
            "field_or_practice_impact": "The result could influence subsequent research, decisions, policy, or practice.",
        },
    },
    "validation_design_quality": {
        "question": "Does the StepPlan define a validation design that can genuinely test the idea without changing its scientific object?",
        "subcriteria": {
            "hypothesis_alignment": "The validation target remains aligned with the idea's research question and core hypothesis.",
            "diagnostic_contrast": "The planned contrast, baseline, control, or null model can distinguish the proposed mechanism from relevant alternatives.",
            "metric_claim_fit": "The planned metrics and success/failure criteria can support or contradict the claim at the claimed strength.",
            "falsification_design": "The planned falsification outcome would genuinely challenge the core hypothesis rather than only expose an execution detail.",
        },
    },
    "reproducibility_readiness": {
        "question": "Can another researcher reconstruct a comparable validation from the StepPlan and ResourceCheck evidence?",
        "subcriteria": {
            "validation_workflow_completeness": "Steps, inputs, outputs, dependencies, and termination conditions form a closed validation workflow.",
            "data_reproducibility": "Data sources, versions, scopes, access paths, and preprocessing boundaries are explicit and checked.",
            "method_environment_reproducibility": "Tools, dependencies, parameters, randomness, and environment requirements are explicit.",
            "evaluation_protocol_reproducibility": "Baselines, metrics, references, and success or failure criteria are explicit.",
        },
    },
}


IDEA_QUALITY_SUBCRITERION_ANCHOR_OVERRIDES = {
    "realizability": {
        "observable_in_principle": (
            "These anchors override the generic anchors for `observable_in_principle`. "
            "Judge whether the core variables could be observed or constructed from any "
            "foreseeable public geospatial, remote-sensing, statistical, or modeled source, "
            "not whether that source is shipped as a record inside this evidence pack. An "
            "auxiliary signal named by the idea or listed in `declared_dataset_requirements` "
            "but not accompanied by its own dataset record is not by itself a reason to score "
            "below 75, provided the named source is a recognizable public product with a "
            "foreseeable acquisition route. Reserve 50 or lower for a variable whose "
            "measurement is genuinely unestablished, for example a quantity that no known "
            "instrument or model product reports at the required semantics, scale, or period."
        ),
        "technical_path_plausibility": (
            "These anchors override the generic anchors for `technical_path_plausibility`. "
            "Judge whether existing GIS, remote-sensing, statistical, or machine-learning "
            "methods could turn the idea into an empirical study. Routine operations such as "
            "spatial joins, zonal statistics, polygon overlays, resampling, regression, or "
            "supervised classification count as an established path even when the evidence "
            "pack does not ship the auxiliary layers those operations would consume. Reserve "
            "50 or lower for a concrete methodological obstacle, such as a required "
            "identification strategy with no known analogue, or a scale or semantics mismatch "
            "with no foreseeable reconciliation."
        ),
        "fatal_blocker_absence": (
            "These anchors override the generic anchors for `fatal_blocker_absence`. "
            "Score 100 when the evidence explicitly rules out any plausible fatal blocker; "
            "75 when no fatal blocker is evidenced, even if material but non-fatal gaps or "
            "uncertainties remain; 50 only when a concrete unresolved condition could make "
            "the core claim fundamentally impossible and no foreseeable workaround is "
            "established; 25 when the evidence strongly indicates such a fundamental blocker; "
            "and 0 when the evidence establishes that core validation is impossible. Missing "
            "execution-level bindings, an auxiliary dataset not being included in the supplied "
            "evidence pack, or other planning detail is not by itself a fatal blocker when a "
            "foreseeable public data source or technical route can resolve it. If the rationale "
            "describes a gap as material but not fatal, `fatal_blocker_absence` must be at least 75."
        ),
    },
}


IDEA_MATURATION_QUESTIONING_SYSTEM_PROMPT = (
    "You are a skeptical research-idea reviewer. Identify only material gaps in the "
    "current idea structure. Do not design an executable workflow or introduce fields "
    "outside the supplied schema. Output strict JSON."
)


IDEA_MATURATION_REVISION_SYSTEM_PROMPT = (
    "You are a research-idea editor. Revise only the explicitly authorized idea fields. "
    "Use only evidence records referenced by the revision package; never invent a paper, "
    "dataset, variable, or data capability. "
    "Keep the revised observable inside the temporal, spatial, and label affordances of the "
    "supplied datasets; a revision that needs an unavailable measurement is not a valid revision. "
    "Preserve the scientific object and keep the idea at hypothesis and falsifiability level; "
    "do not write an execution plan. "
    "Output strict JSON."
)


IDEA_MATURATION_ACCEPTANCE_SYSTEM_PROMPT = (
    "You are a semantic acceptance reviewer for a revised research idea. Judge whether "
    "the requested gap was actually resolved while the scientific object, dataset relevance, "
    "cross-field meaning, revision-package scope, and evidence dependencies remain coherent. "
    "Output strict JSON."
)


IDEA_DUAL_TREE_SYSTEM_PROMPT = (
    "You are one side of an idea-native Socratic dual tree. Challenge the supplied "
    "research idea, state the strongest answer available from that idea alone, then judge "
    "the answer. Stay at idea level and output strict JSON."
)

# 三类 Socratic 调用共用同一个 system prompt。角色差异改由 user prompt 的第一行任务句
# 承担（"Run the critic turn" / "Defend the current idea" / "Independently adjudicate"）。
# 原因是 system 消息排在 user 之前：三个不同的 system prompt 会让前缀在第一条消息就分叉，
# 后面 21KB 的共享 idea+evidence 再怎么前置也无法命中前缀缓存。
IDEA_SOCRATIC_SYSTEM_PROMPT = (
    "You take one role at a time in a persistent Socratic idea-tree exchange: critic, "
    "defender, or independent adjudicator. The task line in the message names your role for "
    "this turn; act only in that role. Use only the supplied idea and evidence, and when a "
    "challenge rests on a premise contradicted by dataset features, rebut that premise "
    "explicitly. Stay at idea level and output strict JSON."
)
# 兼容旧引用；三者现在指向同一个字符串，以保持前缀连续。
IDEA_SOCRATIC_BRANCH_SYSTEM_PROMPT = IDEA_SOCRATIC_SYSTEM_PROMPT
IDEA_SOCRATIC_RESPONSE_SYSTEM_PROMPT = IDEA_SOCRATIC_SYSTEM_PROMPT
IDEA_SOCRATIC_ADJUDICATOR_SYSTEM_PROMPT = IDEA_SOCRATIC_SYSTEM_PROMPT
IDEA_SELECTION_FV_SYSTEM_PROMPT = (
    "You independently classify one complete research idea against fixed evidence using the "
    "supplied axis-specific basis codes. Do not infer its system source, revision round, or prior "
    "score. Output strict JSON."
)


IDEA_GENERATION_SYSTEM_PROMPT = (
    "You generate concise, testable geospatial research ideas. Use only the supplied "
    "idea schema. Do not create an execution design, task plan, dataset requirement list, or "
    "method contract. Output strict JSON."
)


def build_idea_generation_prompt(
    *,
    idea_type: str,
    type_question: str,
    applicability: str,
    ideas_per_type: int,
    dataset_context_json: str,
    literature_context_json: str,
    candidate_datasets_json: str = "[]",
    cross_domain_axes_json: str = "[]",
    composite_component_roles_json: str = "[]",
) -> str:
    cross_domain_block = ""
    if cross_domain_axes_json and cross_domain_axes_json.strip() not in {"", "[]"}:
        cross_domain_block = f"""
## Cross-Domain Combination Directions
{cross_domain_axes_json}

These directions were derived from the task dataset's physical semantics and named Earth
system domains where its core variable plays a meaningful role, with `role` stating whether
it acts as driver, response, covariate, stratifier, or constraint. They are exploration
directions, not dataset requirements. Use them to diversify the batch: prefer that the ideas
in this type do not all stay inside the task dataset's own research field, and where a
direction fits this idea type, build the mechanism around the stated role. The task dataset or
composite anchor must be able to supply the core variable, and any auxiliary data must come from the
verified candidate list. Ignore any direction that cannot be grounded that way rather than
inventing data for it.
"""

    composite_block = ""
    if composite_component_roles_json and composite_component_roles_json.strip() not in {"", "[]"}:
        composite_block = f"""
## Required Composite Dataset Contract
{composite_component_roles_json}

The task input is a composite observation, not one base dataset plus optional auxiliaries.
Every generated idea must be centered on the declared combination. Use every listed component
role as a required part of the scientific mechanism, and explain the distinct contribution of
each role. Do not generate a single-component idea and do not replace a required component with
an invented dataset. The idea's `component_usage` must contain exactly one required entry for
each listed role.
"""

    return f"""Generate exactly {ideas_per_type} distinct research ideas for one fixed idea type.

idea_type: {idea_type}
type question: {type_question}
applicability: {applicability}

## Task Dataset Context
{dataset_context_json}
{composite_block}
{cross_domain_block}

## Literature Context
{literature_context_json}

The literature context is selected separately for this idea type under a fixed character
budget. `dataset_anchor` papers establish source capability or measurement reliability;
`dominant_scope` papers represent common uses in the retrieved corpus;
`underrepresented_scope` papers are relevant but locally sparse directions;
`type_bridge` papers connect the dataset to this idea type; and `diversity_fill` papers add
non-redundant coverage. Treat underrepresented as relative to this retrieved corpus, not as a
claim that the topic is globally understudied. Use the contrast between dominant and
underrepresented scopes to seek concise research gaps rather than repeating the dominant
method family.

## Candidate Auxiliary Datasets (verified available)
{candidate_datasets_json}

Candidate auxiliary datasets above are the type-specific subset of the batch inventory and
all have status `resolved_strong`. When present, the `detail` field carries a compact view of
authoritative description, bands, resolution, and temporal coverage. When an idea
semantically needs auxiliary data beyond the base dataset, ground it only in these candidates;
do not invent external datasets absent from both this list and the task dataset context. An
empty list means no strong auxiliary candidates were found for this idea type.

For every idea, return `idea_dataset_grounding` as a separate list of candidate references.
Each reference must contain an exact `candidate_id` from the supplied type-specific list and a
`source_role` describing what that source contributes. Use an empty list for a valid base-only
idea. Across the ideas in this type, vary candidate use where scientifically appropriate so the
batch explores the supplied candidate range. This grounding is provenance, not part of the
semantic idea and not an execution-level dataset requirement.

Each idea must use exactly the supplied idea structure. `idea_id` and `idea_type` must
be at most 50 tokens. Every other text value must be at most 100 tokens. Explain data
combination semantically through distinct single-source limits and a concrete synergy
opportunity; do not enumerate executable dataset requirements. The research question
must be answerable in principle, the hypothesis directional, and falsification explicit.
`computability_and_falsifiability` must state an observable variable, a
mechanism-relevant contrast, and an outcome that would falsify the hypothesis. Keep these
as idea-level semantics, not an implementation recipe.

Treat `dataset_features` and dataset-description facts as authoritative dataset
features. Do not reject or weaken an idea that uses a profiler-emitted dataset feature.

Do not emit a task plan, tool calls, data requirement list, validation protocol, or
execution-level method contract.
"""


def build_idea_quality_stage_guidance(dimension: str, evaluation_stage: str) -> str:
    """为实际 skill 与 legacy 入口提供同一阶段边界。"""
    stage_guidance = ""
    if dimension == "realizability" and evaluation_stage == "idea_only":
        stage_guidance = """
# Stage-Specific Realizability Rule
This is an `idea_only` evaluation. Judge whether the idea has a fatal
conceptual or technical blocker, not whether it already contains a complete
validation design. Do not penalize the idea merely because it has not specified
all controls, baselines, metrics, model choices, or execution parameters. Missing
planning detail belongs to planned-stage validation or reproducibility
assessment unless the absence exposes a fundamental impossibility.
No structured auxiliary-dataset contract exists at this stage, so judge auxiliary
signals named by the idea on whether a foreseeable public source could supply
them, not on whether a dataset record for them appears in this pack.
"""
    elif dimension == "realizability" and evaluation_stage == "planned":
        stage_guidance = """
# Stage-Specific Realizability Rule
This is a `planned` evaluation. `declared_dataset_requirements` lists the auxiliary
signals the StepPlan declared as scientifically necessary, with the reason for each
signal and its requested modality and spatiotemporal properties. Treat a listed
requirement as evidence that the corresponding variable has a declared acquisition
route and is therefore observable in principle; the absence of a full dataset record
for it in this pack is a packaging boundary, not a realizability defect.
Resolution status, resolved dataset identity, binding failures, and access paths are
deliberately withheld from this dimension and belong to `reproducibility_readiness`.
Never infer an execution-level binding outcome from this list, and never restate an
availability verdict as conceptual impossibility. A requirement that is scientifically
incoherent with the idea, or that demands a measurement no known product provides,
is still a legitimate realizability concern.
When citing anything from `declared_dataset_requirements`, use `source_type` `step_plan`.
Cite `task_dataset_description` or `dataset_portrait` as `dataset_evidence`.
"""
    elif dimension == "data_grounding":
        if evaluation_stage == "idea_only":
            stage_guidance = """
# Stage-Specific Data-Grounding Rule
This is an `idea_only` evaluation. Judge the semantic fit between the idea's own
problem description and the supplied task-dataset context. Do not penalize the
idea merely because StepPlan, ResourceCheck, access manifests, or structured
`required_datasets` are absent. If the idea mentions auxiliary signals, assess
whether the idea-level claim explains why those signals are needed and whether
the base task dataset can plausibly anchor the research question; treat missing
execution-level dataset bindings as a planning-stage issue, not an automatic
data-grounding failure.
Judge the adequacy of declared data-source roles, not the number of sources: a
genuinely sufficient single-source idea can receive the highest
`source_role_adequacy` anchor. Lower this subcriterion only when the core claim
has a hidden undeclared dependency, a declared source has no clear role, or
multiple sources are redundant rather than complementary.
"""
        elif evaluation_stage == "planned":
            stage_guidance = """
# Stage-Specific Data-Grounding Rule
This is a `planned` evaluation. `declared_dataset_requirements` describes which auxiliary
signals the StepPlan says are scientifically necessary, why each signal is needed, and its
requested modality and spatiotemporal properties. Judge whether those declared data roles
are necessary, non-redundant, and capable in principle of supporting the idea's claim.
Do not infer availability from this list and do not reward or penalize ResourceCheck
resolution or access status under `data_grounding`; concrete resolvability and accessibility
belong to `reproducibility_readiness`. An empty requirements list means the plan declared
no auxiliary dataset; judge whether the idea is genuinely base-data-only rather than
assuming that the empty list proves sufficiency.
When citing anything from `declared_dataset_requirements`, use `source_type` `step_plan`,
because those requirements are StepPlan output. Cite `task_dataset_description` or
`dataset_portrait` as `dataset_evidence`.
"""
    elif dimension == "reproducibility_readiness" and evaluation_stage == "planned":
        stage_guidance = """
# Stage-Specific Reproducibility Rule
Use `resource_check.dataset_bindings` to judge whether every declared
auxiliary requirement was mapped to a concrete, reconstructable source. Treat
`resolved_strong` as confirmed resolution; treat `resolved_weak`, `unresolved_semantic`,
`unresolved_real_missing`, `pending`, or `missing` as incomplete reproducibility evidence.
Also use the access manifest, guaranteed or missing datasets, and tool validation evidence
to judge actual accessibility. These availability verdicts must not be re-scored as
scientific data-role adequacy or validation-design quality.
"""
    elif dimension == "logical_coherence" and evaluation_stage == "idea_only":
        stage_guidance = """
# Stage-Specific Logical-Coherence Rule
This is an `idea_only` evaluation. Judge only whether the idea's scientific
argument chain is internally connected: motivation -> question -> hypothesis ->
mechanism -> observable expectation. Do not penalize the idea merely because it
has not specified a complete falsification design, control design, baseline,
metric, or execution workflow. Those belong to planned-stage validation design.
"""
    if dimension == "data_grounding":
        stage_guidance += """
# Scientific Evidence Boundary
Unknown coverage is a verification gap, not a confirmed mismatch. Do not equate
an absent dataset record with an undeclared scientific dependency. Penalize real
semantic incompatibility, an unjustified proxy, or an overextended claim, but not
unresolved binding or access. A declared acquisition route is not proof of access.
"""
    return stage_guidance


def build_idea_quality_dimension_prompt_legacy(
    *,
    dimension: str,
    idea_snapshot_json: str,
    dimension_evidence_json: str,
    evaluation_stage: str = "idea_only",
) -> str:
    rubric = IDEA_QUALITY_DIMENSION_RUBRICS_LEGACY.get(dimension)
    if rubric is None:
        raise ValueError(f"Unsupported idea quality dimension: {dimension}")
    subcriteria = "\n".join(
        f"- `{key}`: {description}"
        for key, description in rubric["subcriteria"].items()
    )
    anchor_overrides = IDEA_QUALITY_SUBCRITERION_ANCHOR_OVERRIDES.get(dimension, {})
    anchor_override_guidance = ""
    if anchor_overrides:
        anchor_override_guidance = "\n# Subcriterion-Specific Anchor Overrides\n" + "\n".join(
            f"- `{key}`: {description}"
            for key, description in anchor_overrides.items()
        )
    stage_guidance = build_idea_quality_stage_guidance(dimension, evaluation_stage)
    return f"""# Task
Evaluate only the `{dimension}` dimension for one scientific idea.

# Core Question
{rubric['question']}

# Subcriteria
{subcriteria}

# Idea Snapshot
{idea_snapshot_json}

# Allowed Evidence For This Dimension
{dimension_evidence_json}
{stage_guidance}

# Anchored Scoring
Choose exactly one anchor for every subcriterion:
- 0: missing, contradicted by evidence, or not satisfied
- 25: mentioned but not operational or missing a critical link
- 50: basically plausible but contains a material gap
- 75: well supported with only a minor ambiguity or limitation
- 100: explicit, traceable, and no material gap is evident
{anchor_override_guidance}

# Evidence Rules
- Every supporting evidence item must identify the subcriterion, source type, source field, and concrete quote or fact.
- `source_type` must be exactly one of these six literal values; never invent another value:
  - `idea_field`: any field of the idea snapshot itself.
  - `dataset_evidence`: `task_dataset_description`, `dataset_portrait`, or other supplied dataset records.
  - `literature_record`: a retrieved paper in `literature_neighbors` or `literature_context`.
  - `historical_idea`: a record in `historical_idea_neighbors`.
  - `step_plan`: `step_plan` steps or `declared_dataset_requirements` entries.
  - `resource_check`: `resource_check`, its `dataset_bindings`, access manifest, or tool validations.
- If the evidence you want to cite does not fit any of the six source types, do not invent a
  new type: put that observation in `counterevidence_or_gaps` instead.
- Put negative evidence, contradictions, and material uncertainty in `counterevidence_or_gaps`.
- Do not infer facts absent from the supplied evidence.
- Do not calculate a dimension total, confidence, or any aggregate score.
- Output budget is strict: write each `evidence` value as one short paraphrased sentence
  (at most about 200 characters) instead of pasting long verbatim quotes, and keep
  `rationale` and every `counterevidence_or_gaps` item to at most two short sentences.
"""


IDEA_EVIDENCE_CHANNEL_SYSTEM_PROMPT = (
    "You classify what kind of external evidence one research finding is asking for. "
    "Output strict JSON."
)


def build_idea_evidence_channel_prompt(*, evidence_need: str) -> str:
    return f"""Classify what kind of external evidence this stated need is asking for.

## Stated Evidence Need
{evidence_need}

Answer two independent yes/no questions about what would actually satisfy this need.

`needs_dataset`: true when satisfying the need requires an actual data product the
analysis could ingest — a raster or vector layer, a time series, a gridded or tabular
variable, a mask, a class or label layer, an event/incident record set, an inventory, a
catalog of dates, a calendar, an index, an anomaly series, or ground-truth measurements.
The phrasing does not have to contain the word "dataset": "time series of vapor pressure
deficit", "land-cover masks", "policy implementation dates", "burned-area timing data",
and "agricultural calendar" all require a dataset. Set it true whenever the need names a
concrete measured or recorded quantity that must be obtained from somewhere.

`needs_literature`: true when satisfying the need requires published findings, an
established method, a validation study, a reported empirical relationship, or a
methodological precedent — that is, something a paper can settle rather than something a
data file supplies.

Both can be true when the need asks for a method *and* the data to apply it to. Both are
false only when the need can be settled from the idea's own reasoning with no external
source at all — for example a purely definitional or internal-consistency fix.

When uncertain about `needs_dataset`, answer true: treating a real data requirement as
satisfied by a topically-related paper lets an unverifiable claim through, which is worse
than one extra retrieval.

`dataset_search_terms`: when `needs_dataset` is true, give the phrase you would type into
a geospatial data catalog to find this data product. Otherwise return an empty string.

Write it the way the dataset itself would be named, not the way the research need is
phrased. Two to four words. Only the data product type. No research verbs (distinguish,
validate, control, isolate), no qualifiers about resolution or period
(high-resolution, time-series, regional, annual, external), no words about what the data
is for.

Catalogs match on title and description tokens, so every extra word dilutes the signal:
"time-series vegetation indices ndvi" returns nothing while "vegetation index ndvi"
returns the AVHRR NDVI record. Some examples of the transformation:

- "Time-series vegetation indices (NDVI/EVI) to link spectral gain to biomass" ->
  "vegetation index ndvi"
- "Ecological vulnerability indices or species range maps to assess habitat loss" ->
  "species distribution"
- "Global or regional irrigation infrastructure maps to control for water use" ->
  "irrigation areas"
- "A historical land cover dataset compatible with GEE covering 1986-2016" ->
  "land cover classification"
- "Exposure data (population density, infrastructure) to validate risk clusters" ->
  "population density"
- "High-resolution DEM capable of resolving sub-meter micro-topography" ->
  "digital elevation model"

If the need names several distinct data products, pick the one the claim depends on most.

`literature_search_query`: the phrase you would type into a scholarly search engine
(OpenAlex, arXiv) to find published work that settles this need. Always provide it.

Build it from the substantive research gap only: a discipline anchor plus noun phrases
naming the object of study (e.g. "limnology lake stratification onset timing"). Six to
ten words. Do not name a standard composite index as the query — an index name is a
retrieval attractor that pulls back that index's own literature instead of the work that
settles this need; name the measured quantities instead. Never include critique
meta-language — words like hypothesis, contradiction, claim, defender, parent answer,
revision describe the debate about the idea, not the subject matter, and they pull
retrieval into unrelated fields (a query containing "convergence ... contradiction"
returns pure-mathematics papers). No research verbs, no quality qualifiers.
"""


IDEA_EVIDENCE_RELEVANCE_SYSTEM_PROMPT = (
    "You judge whether retrieved evidence candidates actually address one stated "
    "research evidence need. Output strict JSON."
)


def build_idea_evidence_relevance_prompt(
    *,
    evidence_need: str,
    modification_direction: str,
    literature_json: str,
    dataset_json: str,
) -> str:
    return f"""Judge each retrieved candidate against one research finding's evidence need.

## Stated Evidence Need
{evidence_need}

## What the revision is trying to fix
{modification_direction}

## Literature Candidates (JSON array, order matters)
{literature_json}

## Dataset Candidates (JSON array, order matters)
{dataset_json}

For each candidate answer one question: would this record actually help settle the stated
need — the right discipline, the right object of study, and the right kind of content?

Keyword overlap is not relevance. A pure-mathematics paper about "scalar curvature
convergence" shares words with a geomorphology need about flow convergence but settles
nothing; reject it. Conversely a paper or dataset phrased differently from the need but
about the same object of study is relevant; keep it.

Return `literature` and `dataset` as boolean arrays with exactly one entry per candidate,
in the same order as given. An empty candidate array maps to an empty boolean array.
"""


def build_idea_maturation_questioning_prompt(
    *,
    idea_json: str,
    dataset_context_json: str,
) -> str:
    return f"""Review this research idea before detailed task planning.

## Idea
{idea_json}

## Task Dataset Context
{dataset_context_json}

Return at most three findings. A finding must identify a real ambiguity, unsupported
inference, missing mechanism link, missing falsification condition, or unclear data
synergy. Treat dataset features in the Task Dataset Context as available dataset traits;
do not cite an execution-view omission as a reason to reject them. Do not ask for a task
plan, evaluation protocol, or implementation detail.
Use only the seven existing idea fields as target_field.
"""


def build_idea_dual_tree_side_prompt(
    *,
    side: str,
    depth: int,
    branch_definitions_json: str,
    idea_json: str,
    dataset_context_json: str,
) -> str:
    side_instruction = (
        "Find whether the idea is unsupported, internally inconsistent, or not truly necessary."
        if side == "weakness"
        else "Find a concrete scientific value or discovery opportunity that is present but not yet explicit."
    )
    return f"""Run depth {depth} of the {side} side of a Socratic idea tree.

{side_instruction}

## Fixed Branches
{branch_definitions_json}

## Current Idea
{idea_json}

## Task Dataset Context
{dataset_context_json}

For every branch, produce one concise challenge, the strongest defender reply grounded only
in the current idea, and a judgment. `needs_revision` identifies an unresolved weakness;
`value_opportunity` identifies value worth making explicit. Never ask for a task plan or
output an execution-level field. Do not recommend discarding the idea or shrinking it into
a trivial descriptive task; dataset features are available dataset traits.
"""


# Socratic 三类调用中长文本字段的字符预算。这些数字必须与 socratic_idea_dual_tree.py
# 里同名字段的 schema maxLength 及写回切片保持一致：schema 只是软提示，模型无法边写
# 边数字符，因此预算必须在 prompt 正文里明说，否则模型会按"写详细"的指令超限，
# 让模型层白花一次纠错重试。留 20 字符余量以吸收模型的估算误差。
SOCRATIC_QUESTION_CHAR_BUDGET = 400
SOCRATIC_ANSWER_CHAR_BUDGET = 460
SOCRATIC_ADJUDICATION_CHAR_BUDGET = 400


def _socratic_shared_prefix(*, idea_json: str, evidence_json: str) -> str:
    """三类 Socratic 调用（提问/辩护/裁决）共享的前缀。

    这两个字段合计约 21KB，且在同一分支深度对的五次调用之间逐字节相同。放在 prompt 最前面
    是为了让服务端的自动前缀缓存能命中：一旦任何可变内容（question/answer/side）排在
    它们之前，前缀就在那里分叉，21KB 只能每次全额重算。顺序即成本，不要重排。

    evidence 排在 idea 之前是有意的：evidence 在一轮内对所有分支和节点都相同，而 idea
    按分支裁剪后各不相同。先 evidence 后 idea 让前缀在跨分支时也能命中 evidence 那一段；
    反过来排则每个分支都在第一个字段就分叉。
    """
    return f"""Available evidence: {evidence_json}
Current idea: {idea_json}
"""


def build_idea_socratic_branch_prompt(*, idea_json: str, evidence_json: str, dimension: str, target_field: str, branch_instruction: str, depth: int, pair_layer: int, parent_summary: str) -> str:
    if pair_layer == 1:
        layer_guidance = (
            "Expose the strongest concrete weakness or value question on this branch. "
            "Do not propose a repair yet; establish the exact premise that must withstand pressure."
        )
    elif pair_layer == 2:
        layer_guidance = (
            "Follow up only on the first-layer question and answer in Parent-node state. "
            "If that answer concedes, partially supports, or cannot support a defect, ask for the "
            "minimum resolution specification: the affected idea meaning, the smallest semantic "
            "change needed, any missing evidence, and the condition that would close the defect. "
            "If the answer is supported, apply the strongest counterexample or boundary test. "
            "Do not switch dimension, open an unrelated issue, or rewrite the idea."
        )
    else:
        raise ValueError(f"pair_layer must be 1 or 2, got {pair_layer}")
    return f"""{_socratic_shared_prefix(idea_json=idea_json, evidence_json=evidence_json)}
Run the critic turn for one persistent Socratic idea-tree node.
Dimension: {dimension}
Target field: {target_field}
Depth: {depth}
Layer within depth pair: {pair_layer}
Branch-specific interrogation: {branch_instruction}
Parent-node state: {parent_summary}
{layer_guidance}
Ask one depth-specific falsification or pressure-test question. Do not request a task plan or execution contract.
Keep `question` and `failure_hypothesis` each under {SOCRATIC_QUESTION_CHAR_BUDGET} characters; state the point directly instead of restating the idea or the evidence.
"""


def build_idea_socratic_response_prompt(*, idea_json: str, evidence_json: str, dimension: str, target_field: str, branch_instruction: str, depth: int, pair_layer: int, parent_summary: str, question: str, failure_hypothesis: str) -> str:
    if pair_layer not in {1, 2}:
        raise ValueError(f"pair_layer must be 1 or 2, got {pair_layer}")
    second_layer_guidance = ""
    if pair_layer == 2:
        second_layer_guidance = """
If the first-layer answer in Parent-node state admitted a defect, identify the defect location,
minimum resolution condition, required evidence if any, and a concrete closure criterion.
Do not rewrite the idea; provide a constrained repair specification for later adjudication.
"""
    return f"""{_socratic_shared_prefix(idea_json=idea_json, evidence_json=evidence_json)}
Defend the current idea against one Socratic challenge.
Dimension: {dimension}
Target field: {target_field}
Depth: {depth}
Layer within depth pair: {pair_layer}
Interrogation rule: {branch_instruction}
Parent-node state: {parent_summary}
Challenge: {question}
Failure hypothesis: {failure_hypothesis}
Answer only from the current idea and evidence. If the challenge assumes a dataset trait is
missing but `dataset_features` or the dataset description contains that trait, explicitly
state that the challenge premise is false and defend the idea on that basis. Explicitly
mark unsupported reasoning instead of inventing a method, dataset, task plan, or result.
Keep `answer` under {SOCRATIC_ANSWER_CHAR_BUDGET} characters; give the reasoning that settles the challenge, not a recap of it.
{second_layer_guidance}
"""


def build_idea_socratic_adjudication_prompt(*, idea_json: str, evidence_json: str, dimension: str, target_field: str, branch_instruction: str, depth: int, parent_summary: str, first_question: str, first_failure_hypothesis: str, first_answer: str, first_answer_status: str, second_question: str, second_failure_hypothesis: str, second_answer: str, second_answer_status: str, side: str, axis_basis_menu: str) -> str:
    return f"""{_socratic_shared_prefix(idea_json=idea_json, evidence_json=evidence_json)}
Independently adjudicate one complete two-layer Socratic idea-tree exchange.
Side: {side}
Dimension: {dimension}
Target field: {target_field}
Depth pair: {depth - 1} -> {depth}
Branch-specific standard: {branch_instruction}
Previous depth-pair state: {parent_summary}
First-layer challenge: {first_question}
First-layer failure hypothesis: {first_failure_hypothesis}
First-layer defender answer: {first_answer}
First-layer defender status: {first_answer_status}
Second-layer challenge: {second_question}
Second-layer failure hypothesis: {second_failure_hypothesis}
Second-layer defender answer: {second_answer}
Second-layer defender status: {second_answer_status}
Judge the branch once from the complete two-layer chain. If the first answer admitted a defect,
use the second exchange to determine its location, minimum resolution condition, evidence need,
and closure criterion. Do not treat a proposed repair condition as if the idea were already
repaired; the actual writeback happens only after this adjudication.

Return `axis_basis_code`: exactly one code from this dimension's ordered ladder, judged from
the state of the branch AFTER the two-layer exchange.
Do not return a numeric score; code maps the basis code to its fixed score after this call.

{axis_basis_menu}

The lower three codes each assert that a material weakness survived the exchange, so picking one
commits you to naming that specific defect in `adjudication`. The top two mean this axis held, and
that is the expected outcome whenever the defender did its job — it is not a concession. Judge the
axis itself, not the effort spent probing it: a challenge the defender correctly rebutted is not
evidence of a defect, and an exchange that surfaced only refinements you cannot name as a concrete
error in the current idea text belongs in the top two.

Then select the verdict by applying these tests in order and stopping at the first one that holds.
The basis code you just chose constrains this choice; do not contradict it.

1. `resolved` — the basis code is one of the top two and this branch has no remaining concrete
   pressure point worth another depth.
2. `needs_revision` — the basis code is one of the lower three and rewriting an idea field would
   remove the defect you named. This is required whenever that defect is a mechanism failure, a
   semantic gap, an unsupported causal or scope claim, an uncontrolled confound, a definitional
   error, or a mismatch between the question and the hypothesis — but only when it is an actual
   error in the current idea text, not a direction you could keep probing. It applies even when the
   defender concedes the point and even when the needed fix looks obvious.
3. `evidence_needed` — the exchange cannot be settled without a specific piece of evidence
   absent from the available evidence. Name that evidence type in `evidence_need`. Before
   choosing this, check the `literature` and `candidate_datasets` entries in the available
   evidence: if a listed paper or candidate dataset already supplies what the exchange needs,
   the evidence is present, so pick another verdict and cite that entry in `adjudication`.
   Use this only when no idea-field rewrite could resolve the exchange.
4. `value_opportunity` — Value side only: a concrete latent value can be made explicit
   without lowering the idea's ambition.
5. `continue` — the branch was not defeated, but a different concrete pressure point remains on
   this branch for the next depth. Name that next pressure point in `next_focus`. Never use
   `continue` to avoid committing to a revision demand you already described in `adjudication`.

For `needs_revision` or `value_opportunity`, make `adjudication` a constrained repair
specification that names the affected meaning, minimum semantic change, and closure criterion;
put any missing external support only in `evidence_need`. Keep `adjudication` under
{SOCRATIC_ADJUDICATION_CHAR_BUDGET} characters: name the defect and the required change, and do
not replay the two-layer exchange or re-argue the mechanism at length. A defender status of
`unsupported` in either layer means the idea did not withstand this challenge, which is
incompatible with a top-two basis code and with `resolved`.

Treat dataset features as available dataset signals. If the defender correctly rebuts a
challenge premise using dataset features, treat that premise as resolved rather than turning
it into a revision demand. Do not recommend dropping the idea or shrinking it into a trivial
descriptive task; an answerability-preserving reframe is allowed only when the current
question cannot be answered from dataset features.

Use the existing `evidence_need` string for any external evidence missing from the current
evidence context, including evidence needed to support a `needs_revision` writeback. When the
gap is data, name the missing observable and say whether it requires a dataset, data layer,
inventory, records, labels, measurements, or an accepted proxy. Do not copy the critique,
current dataset name, or proposed method into `evidence_need`; return an empty string when no
external evidence is missing.

Set `requires_discriminating_contrast` to true when the defect you identified is that a
competing explanation would produce the same observation as the idea's own mechanism — a
confound, a conflation of two causes, an alternative account the stated comparison cannot rule
out. The test is concrete: given the idea's current `mechanism_relevant_contrast`, would the
rival explanation predict the same result? If yes, set it true. Such a defect cannot be fixed
by restating the claim more precisely; only a comparison that separates the two explanations
settles it. When true, use `required_contrast` to name the observable comparison, stratification,
or control that would separate them — describe the comparison itself, not the flaw. Leave
`required_contrast` empty and the flag false for every other kind of defect.

Set `undeclared_observable_dependency` to true only when the defect is that the observable
variable itself is an independently measured quantity — an outcome, event, label, or
gauge/instrument reading — that the base and declared predictor data cannot construct, and no
source in `data_synergy_and_gap` is declared to measure it. The test is concrete: name the
observable, then ask whether any declared source produces that measurement; a routine derivative
of the base data such as slope, ruggedness, or a zonal statistic is constructible and is not
this defect. This is an idea-internal declaration gap, not a question of whether the data exists
in the world — do not judge availability, only whether the measuring source is declared. Leave
the flag false for every other kind of defect, and never set it outside the
`idea_operational_coherence` axis.
When the flag is true, make `adjudication` a repair spec that names the ordered remedy, most
constructive first: prefer building a proxy-mechanism observable from the base and declared data
when one can land inside their temporal, spatial, and label affordances; only when an independent
measurement is genuinely unavoidable, declare that measuring source as an explicit data role; and
retreat the observable to a neighbouring answerable question only as the last resort. Do not
present retreat as the default, and never defer the observable's own data to retrieval.
"""


def build_idea_selection_fv_prompt(*, idea_json: str, evidence_context_json: str) -> str:
    return f"""Independently classify one complete research idea before detailed task planning.

## Research Idea
{idea_json}

## Fixed Evidence Context
{evidence_context_json}

Classify exactly these six axes. Judge only the supplied idea against the supplied evidence.
Do not infer an execution plan or require execution-level controls, metrics, dataset bindings,
or a method contract.
For each axis, return exactly one basis code from its ordered ladder below. Apply the most severe
applicable condition. A code from either of the top two levels is invalid when a material
idea-level weakness remains. Do not return numeric scores; code maps each basis code to its fixed
score after this single model call.

`idea_operational_coherence`:
- `absent_or_contradictory_design`: the question, observables, source roles, or falsification
  contrast are absent or mutually incompatible, leaving no recognizable answerable design.
- `major_answerability_defect`: a required observable or source is unavailable, or the design has
  a fundamental answerability failure without a clear idea-level repair. **When `idea_data_contract_status`
  shows `observable_source_declared: false` with non-empty `unresolved_dependency`, the observable requires
  a specialized data source (e.g., streamflow gauges for recession constants, event labels for susceptibility)
  that is not declared in `data_synergy_and_gap` — this is a major answerability defect at idea level.**
- `material_design_gap`: the design has a recognizable repair path, but an observable, source
  role, alignment relation, or falsification contrast remains materially underspecified.
- `answerable_with_minor_uncertainty`: the question, observables, source roles, and falsification
  contrast form an answerable design; only minor uncertainty remains.
- `explicit_answerable_design`: those elements are unusually explicit, mutually coherent, and
  supported by the fixed evidence with no material weakness.

`mechanism_soundness`:
- `absent_or_contradictory_mechanism`: the mechanism is absent or contradicts the directional
  hypothesis.
- `major_proxy_or_logic_failure`: a fatal proxy leap, reversed implication, or broken causal/logical
  link means the stated mechanism does not entail the claim.
- `material_mechanism_gap`: the intended mechanism is recognizable, but at least one material link
  or assumption is unsupported and repairable at idea level.
- `sound_with_minor_uncertainty`: the observable-to-process-to-hypothesis chain is sound, with only
  minor uncertainty.
- `explicit_evidence_supported_mechanism`: every material link is explicit and supported by the
  fixed evidence, with no material proxy or logic weakness.

`claim_defensibility`:
- `absent_or_indefensible_claim`: the directional claim is absent, self-contradictory, or
  fundamentally unsupported by the proposed observables.
- `major_scope_overreach`: the idea claims causation, attribution, or a hidden state far beyond
  what its observables and falsification condition could defend.
- `material_claim_support_gap`: the claim has a recognizable repair path, but its strength, scope,
  or target construct remains materially unsupported.
- `defensible_with_minor_uncertainty`: the claim is calibrated to the stated observables and
  falsification condition, with only minor uncertainty.
- `explicit_well_calibrated_claim`: claim strength, scope, observables, and falsification are
  unusually explicit and supported by the fixed evidence with no material weakness.

`novelty_frontier`:
- `absent_or_contradicted_novelty`: `novelty_constraints_and_value` is absent or empty, or fixed
  evidence directly shows that an explicitly claimed contrast duplicates the stated comparator.
- `generic_value_only`: the idea states usefulness, importance, or general novelty but names no
  existing-work comparator and no precise difference.
- `comparator_vague_or_unsupported`: a comparator or prior-work class is named, but the difference
  is vague, merely adjectival, or not defensible against the fixed evidence.
- `specific_defensible_contrast`: the idea states a specific comparator and exact non-trivial
  difference that is defensible against the fixed evidence, with only minor frontier uncertainty.
- `evidence_supported_frontier`: fixed literature evidence supports both the comparator and a
  precise non-trivial frontier that the idea explicitly states.

`discovery_leverage`:
- `absent_or_invalid_research_question`: the research question is absent, empty, or internally
  contradictory, so it defines no investigable outcome.
- `descriptive_restatement`: the primary result is only a map, count, distribution, correlation,
  or restatement of a known observable pattern, without revealing a latent target.
- `latent_target_not_discriminated`: a hidden state, mechanism, counterfactual, discrepancy, or
  phenomenon is named, but the proposed observables cannot distinguish it from a descriptive
  alternative.
- `discriminating_hidden_target`: the latent target and a contrast that can distinguish it from a
  descriptive alternative are explicit and coherent, with only minor uncertainty.
- `competing_explanations_resolved`: the idea names competing explanations or states and a
  falsifiable observation that would materially resolve which one holds.

`combination_essentiality`:
- First identify the declared source mode. Different views, bands, labels, or time slices of one
  dataset are single-source. Never lower a valid single-source idea merely for having one source or
  require it to add an auxiliary source.
- `contradictory_source_contract`: the declared source roles contradict the question or make the
  core observable impossible.
- `undeclared_or_unavailable_dependency`: the core claim relies on an undeclared auxiliary signal
  or on a declared external source with no corresponding available dataset in fixed evidence.
- `material_source_role_or_alignment_gap`: sources are identifiable, but their distinct roles,
  spatial/temporal/alignment relationship, sufficiency, or incremental information remains
  materially unclear; this also covers decorative multi-source context.
- `coherent_source_design`: a single source is sufficient with no evident hidden dependency, or
  each multi-source role and alignment is coherent and changes the observable mechanism, hidden
  state, or falsification signal, with only minor uncertainty.
- `explicit_sufficiency_or_essentiality`: for single-source mode, the source alone explicitly
  supplies every signal required by the core question and falsification condition with no hidden
  dependency; for multi-source mode, roles and alignment are explicit and removing an added source
  makes materially the same core question unanswerable.

Do not use Socratic leaves, verdicts, prior scores, version identifiers, or revision history; none
are part of this evaluation.
"""


def build_idea_maturation_revision_prompt(
    *,
    idea_json: str,
    findings_json: str,
    editable_fields_json: str,
    dataset_context_json: str,
    evidence_records_json: str = "[]",
    revision_package_json: str = "[]",
    claim_chain_derivation_context_json: str = "",
    rejected_attempts_json: str = "",
) -> str:
    claim_chain_section = (
        f"""
## Claim Chain Derivation Context
{claim_chain_derivation_context_json}

When revising `research_question`, `core_hypothesis`, `theoretical_mechanism`, or
`computability_and_falsifiability`, derive that whole claim chain from this context and the fixed
task dataset context. Treat the old claim-chain wording as stale evidence of what failed, not as a
constraint to preserve. Preserve the macro research object and motivation, but calibrate the new
question to what the supplied sources can measure: the temporal, spatial, label, and unit
affordances of the base dataset and the referenced candidate datasets bound the claim, and
`data_synergy_and_gap` only records the idea's own account of that boundary, so it cannot widen
it. Do not narrow the idea into mere
dataset description, QA, clustering, stratification, or artifact inspection unless that is already
the macro research object.
"""
        if claim_chain_derivation_context_json
        else ""
    )
    directional_section = (
        """
## Directional Claim Restatement (applies whenever `core_hypothesis` is editable)

An adjudication may falsify one half of a directional claim; the un-reviewed half must not be
carried over on inherited wording. Restate every direction surviving in the revised
`core_hypothesis.directional_statement` through the `directional_restatement` output field —
its schema states the required shape. For findings carrying `micro_empirical_check`, the
committed sign must match the computed `observed_sign` (or be `withdrawn`); a qualifier
("normalized for X", "controlling for Y") narrows scope and never verifies or flips a sign.
"""
        if "core_hypothesis" in (editable_fields_json or "")
        else ""
    )
    rejected_section = (
        f"""
## Rejected Revisions For These Same Findings
{rejected_attempts_json}

Each entry is a patch semantic acceptance already rejected, with the failure codes it triggered.
Write a different patch that clears every listed failure code. Do not resubmit a rejected wording
or a paraphrase of it. `no_substantive_change` means the patch only restated the current idea;
`target_unresolved` means it did not answer the cited finding; `research_object_lost` and
`base_dataset_irrelevant` mean it drifted off the original scientific object or its dataset;
`descriptive_retreat` means it replaced the claim with description; `falsifiable_signal_missing`
means `computability_and_falsifiability` still lacks the observable, the contrast, and the
falsifying pattern; `observable_outside_data_affordance` means the observable, contrast, or
falsification condition needs a time axis, resolution, label, reference measurement, or unit
convention that no supplied source has, so move the claim to a neighbouring question the
supplied sources can decide instead of restating the same unobservable target;
`claim_scope_invalid` means the claim over-reached; `idea_incoherent` means the resulting fields make incompatible scientific statements. Identify the conflicting field excerpts and explain the scientific incompatibility; whether one or both fields changed is not a coherence criterion;
`discriminating_contrast_unchanged` means a finding required a comparison that separates the
idea's mechanism from a competing explanation, but `mechanism_relevant_contrast` is unchanged —
rewording the claim will never clear this code, only a different contrast will;
`directional_sign_contradicts_stat` means your committed `directional_restatement.surviving_sign`
(or the surviving claim text) contradicts a computed statistic attached to the finding
(`micro_empirical_check`): set `surviving_sign` to the entry's `observed_sign` (or `withdrawn`)
and align the statement — per the Directional Claim Restatement Rule.
"""
        if rejected_attempts_json
        else ""
    )
    return f"""Revise a research idea before detailed task planning.

## Current Idea
{idea_json}

## Findings To Resolve
{findings_json}

A finding carrying `micro_empirical_check` brings a statistic computed on the task data for
that specific challenge (`contrast` reports groups, row_filter and the observed direction;
`directive.required_rule` states the outcomes its surviving claim may take, with an adoptable
`observed_fragment`). Treat it as settled data about that comparison, not as another opinion
in the exchange.

## Mandatory Retrieved Evidence Records
{evidence_records_json}

## Authorized Revision Package
{revision_package_json}

## Editable Fields
{editable_fields_json}

## Task Dataset Context
{dataset_context_json}
If `required_composite_components` is present, the idea is anchored to that complete
multi-dataset combination. Every listed role must remain scientifically necessary and must not
be silently dropped during revision. A composite task is not equivalent to using only the first
component.
{claim_chain_section}
{directional_section}
{rejected_section}
Return only a patch for the editable fields. Preserve the original scientific object.

## Revision Package Is Authoritative
Resolve only `idea_field_update` items, modify only their `authorized_fields`, and cite no
evidence outside their `evidence_record_ids`; a retrieved hit is a bounded source, not
permission to infer unlisted bands, labels, coverage, resolution, or causal support.
- `evidence_requirement: "none_required"` marks a definitional or internal-consistency fix the
  idea's own reasoning settles: resolve it from existing content even when its record is not
  `grounded` — but you still may not introduce a new named dataset, signal, variable, or
  capability.
- Add no dataset absent from the referenced `dataset_results`, and no signal, variable,
  material property, or terrain attribute that appears in no referenced result.
- Items routed to `step_plan_open_point` or `unresolved` stay open: never write them as settled
  facts. Keeping them open does not mean adding a caveat per item — a revision that piles up
  caveats is rejected as a descriptive retreat.
- `consistency_authorized_fields` are alignment-only: bring them in line with the NEW claim
  (question, hypothesis, mechanism pointing the same direction as each other) and remove
  wording the rewritten claim contradicts. Do not resolve the other finding with them, assert
  it settled, or smuggle in unsupported dependencies. A half-rewritten idea that still asserts
  the old direction is incoherent.
- Do not reframe a route supported by dataset features: when the signals cannot carry the
  original causal, temporal, or label-strength claim, rewrite it as an observable
  proxy-mechanism claim serving the same scientific object — never descriptive clustering,
  stratification, mask QA, or data-structure analysis.

## Observable Affordance Boundary
The rewrite must land inside what the supplied datasets can measure. Before writing the patch,
check the revised `observable_variable` and `mechanism_relevant_contrast` against the Task
Dataset Context and candidate datasets on four tests:
- temporal: no rate, trend, timing, onset, dynamic, or before/after observable when every
  source is a static layer, single snapshot, or multi-year climatology — a static source
  cannot acquire a time axis from another dataset;
- spatial: nothing finer than the coarsest source the contrast depends on; name the operative
  resolution when scale-sensitive;
- label: any class, severity level, category, ground-truth value, or validation reference must
  exist in the base or a referenced candidate dataset;
- unit: no absolute threshold on a quantity whose unit or convention the documentation flags
  as unconfirmed.
When the claim cannot be rescued inside these boundaries, move it to a neighbouring question
the sources can decide, keeping the same macro research object — that is not a
`descriptive_retreat`; retreating to clustering, stratification, mask QA, or data-structure
description still is.

## Findings Marked `requires_discriminating_contrast`
A rival explanation predicts the same observation as the idea's mechanism. Rewording cannot
resolve this; sharpening makes it worse — "a proxy for X rather than mere Y" asserts a
discrimination the current comparison does not perform. Either change
`mechanism_relevant_contrast` into a comparison whose outcome differs by which explanation is
true (start from the finding's `required_contrast`; stratify by a variable only one explanation
predicts, compare subgroups the rival treats identically, or add a control group where the
idea's mechanism is absent but the rival still applies — inside the Affordance Boundary above),
or, when no source supports any such comparison, weaken `proxy_or_inference_claim` until it no
longer asserts the discrimination at all. No middle path: keeping the assertion while renaming
what it discriminates fails `discriminating_contrast_unchanged`. Never state a claim more
precisely than the contrast can test.

## Aligned Claim Chain
When several claim-chain fields are authorized, revise them as one chain: `research_question`,
`core_hypothesis`, and `theoretical_mechanism` must assert the same direction — never leave one
field at the pre-revision direction while another points at the revised one.

## Resulting Idea
- Change only fields whose scientific meaning needs correction or alignment; an already
  compatible question or hypothesis stays unchanged, and synchronized edits are not required.
  Changing both question and hypothesis does not by itself establish coherence or preservation
  of the research target.
- `computability_and_falsifiability` must name the observable variable, the mechanism-relevant
  contrast, and the result pattern that would falsify the hypothesis, and must test the revised
  claim. Rewrite it when missing, incomplete, or still aimed at the pre-revision claim; leave it
  when already correct — do not restate it just to show the field was touched. A caveat alone
  is not a substantive revision.
- Do not specify data download, preprocessing, models, thresholds, task steps, output figures,
  validation protocols, or fields outside the current idea; create no dataset requirements;
  keep the patch compact and do not restate unchanged background.
"""


def build_idea_maturation_acceptance_prompt(
    *,
    idea_before_json: str,
    idea_after_json: str,
    findings_json: str,
    dataset_context_json: str,
    evidence_records_json: str = "[]",
    revision_package_json: str = "[]",
) -> str:
    return f"""Semantically check whether an idea revision should be applied.

## Idea Before Revision
{idea_before_json}

## Proposed Idea After Revision
{idea_after_json}

## Findings The Revision Was Required To Resolve
{findings_json}

## Mandatory Retrieved Evidence Records
{evidence_records_json}

## Authorized Revision Package
{revision_package_json}

## Task Dataset Context
{dataset_context_json}

If `required_composite_components` is present, pass only when the revised idea still uses every
listed component role as a necessary part of the composite mechanism; silently reducing the idea
to one component fails even if that component remains relevant.

## Pass Rule
Pass only when the revision resolves the cited finding, changes at least one idea field,
preserves the original scientific object, keeps the task dataset materially relevant, and leaves
all idea fields mutually coherent. Coherence requires `research_question`, `core_hypothesis`,
and `theoretical_mechanism` to assert the same direction: set `idea_fields_coherent` false when
they predict opposite directions for a condition, even if each field reads well alone. Evaluate
the meaning of the resulting fields, not which fields were edited — a question-only or
hypothesis-only edit can be coherent, and changing both can still contradict or replace the
original target. Compare research objects, variable meanings, population and temporal scope,
reference baseline, predicted relationships, and falsification conditions. For a semantic
rejection, quote the conflicting field excerpts in `reason` and explain how the incompatibility
affects the research design. Do not infer incompatibility merely from unchanged text or a diff,
and do not reject because a profiler-emitted dataset feature was not exposed in one local
execution view.

## Per-Flag Criteria
- `revision_package_consistent` false when a changed field is outside the item's
  `authorized_fields`, the writeback does not implement the stated `writeback_intent`, or an
  item routed to `step_plan_open_point`/`unresolved` is presented as solved.
- A change inside a `consistency_authorized_fields` entry is expected when it merely aligns the
  field with the rewritten claim: do not set `revision_package_consistent` or
  `deferred_findings_preserved` false for that alone — set them false only when the change goes
  further, presenting a deferred finding as solved or smuggling in an unsupported dependency.
- `evidence_dependencies_supported` false only when the revised idea introduces or relies on a
  named dataset, signal, variable, material property, terrain attribute, coverage, resolution,
  or data capability absent from the package-referenced evidence records. For an item with
  `evidence_requirement: "none_required"` the referenced record is expected not to be `grounded`
  (a definitional or internal-consistency fix): judge only whether a new named external entity
  was introduced, never the record being `insufficient` or empty. Search queries are not
  evidence; only concrete returned literature and dataset records are.
- `deferred_findings_preserved` false only when the revised idea affirmatively asserts a deferred
  or unresolved finding is settled (the confound is now controlled, the gap closed, the
  limitation no longer applies). Saying nothing is not a violation, and the idea need not carry a
  caveat per deferred finding.
- `discriminating_claim_withdrawn` true only when a finding carries
  `requires_discriminating_contrast: true`, `mechanism_relevant_contrast` is unchanged, and the
  revision resolved the confound by genuinely weakening `proxy_or_inference_claim` — the claim no
  longer asserts that the observation distinguishes the idea's mechanism from the rival
  explanation, and what remains is licensed by the existing contrast. False whenever the claim
  still asserts the discrimination however worded, or became more specific or assertive than
  before; false when no finding requires a discriminating contrast.
- When the available signals do not support a strong causal, temporal, or label-strength claim,
  require an observable proxy claim that still answers the original scientific target; reject
  caveat-only revisions and retreats to descriptive clustering, stratification, mask QA, or
  data-structure analysis.
- Judge `falsifiable_signal_added_or_clarified` on the revised idea's signal as it stands, not on
  whether this patch rewrote it: true when the observable variable, mechanism-relevant contrast,
  and falsifying result pattern are present, complete, and still test the revised hypothesis
  (including an already-correct signal left untouched); false only when missing, incomplete, or
  now testing a different claim. Pass only when the resulting question preserves non-trivial
  scientific value; do not require or reward task steps, pipelines, tools, parameters, data
  downloads, output artifacts, or an execution-level method contract.
- `observable_within_data_affordance` false when the revised `computability_and_falsifiability`
  needs what no supplied source can measure: a rate, trend, timing, onset, dynamic, or
  before/after change while every source is static; a feature finer than the coarsest source the
  contrast depends on; a class, severity level, ground-truth value, or validation reference that
  neither the base nor a referenced candidate dataset provides; an absolute threshold on a
  quantity whose unit or convention the documentation flags as unconfirmed. Judge against the
  datasets, not the idea's own `data_synergy_and_gap`. Moving the claim to a neighbouring
  question the sources can decide, keeping the same macro research object, satisfies this check
  and is not a `descriptive_retreat`.
"""
