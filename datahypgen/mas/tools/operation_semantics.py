"""Deterministic operation-level compatibility checks for capability anchors."""

import re
from typing import Any, Dict, Iterable, List, Set


_OPERATION_PATTERNS = {
    "data_access": (
        r"\b(?:read|load|open|fetch|download|retrieve|acquire|access|ingest)\b",
        r"\b(?:imagecollection|featurecollection)\b",
        r"\bquery\s+(?:gee|earth\s*engine|catalog|dataset)\b",
    ),
    "scaling": (
        r"\b(?:standardiz(?:e|es|ed|ing|ation)|normaliz(?:e|es|ed|ing|ation)|scal(?:e|es|ed|ing))\b",
        r"\bunit\s+variance\b",
    ),
    "correlation": (
        r"\b(?:correlat(?:e|es|ed|ing|ion)|pearson|spearman|kendall)\b",
        r"\bassociation\s+(?:coefficient|test)\b",
        # code-style 函数名：尾部 r/tau 等后缀会挡住 \b 词边界
        r"\b(?:pearsonr|spearmanr|kendalltau)",
    ),
    "group_comparison": (
        r"\b(?:t[ -]?test|anova|mann[ -]?whitney|wilcoxon|kruskal)\b",
        r"\b(?:compare|difference)\s+(?:between|across)\s+(?:groups?|classes|samples?)\b",
        r"\bgroup\s+(?:comparison|difference|test)\b",
        # code-style 函数名：ttest_ind / mannwhitneyu 的下划线/后缀会挡住 \b 尾边界
        r"\b(?:ttest|mannwhitney|f_oneway|kruskalwallis|chisquare|chi2)",
        r"\bhypothesis\s+test",
    ),
    "zonal_statistics": (
        r"\bzonal\b",
        r"\b(?:polygon|region|zone)\s+(?:aggregate|aggregation|statistics?|summary)\b",
    ),
    "point_sampling": (
        r"\bpoint\s+(?:query|sample|sampling|extraction)\b",
        r"\bextract\s+(?:raster\s+)?values?\s+at\s+points?\b",
    ),
    "regression": (r"\bregress(?:es|ed|ing|ion|or)?\b",),
    "classification": (r"\bclassif(?:y|ies|ied|ying|ication|ier)\b",),
    "clustering": (r"\bcluster(?:ing)?\b", r"\bk[ -]?means\b"),
    "interpolation": (r"\binterpolat(?:e|es|ed|ing|ion)\b", r"\bkriging\b"),
    "aggregation": (r"\baggregat(?:e|es|ed|ing|ion)\b", r"\bgroupby\b"),
    "data_join": (
        r"\b(?:merg(?:e|es|ed|ing)|join(?:s|ed|ing)?)\b",
        r"\b(?:table|dataframe)\s+(?:merge|join)\b",
    ),
    # 'map' 被有意排除：map/mapping 是地学步骤描述的日常动词（"map values into
    # classes"），作为 visualization 信号误报率过高，会从散文里制造伪操作族。
    # "produce a map" 类数据契约需求由 _TOKEN_SYNONYM_GROUPS 的可视化产物组承接。
    "visualization": (r"\b(?:plot|visualiz(?:e|es|ed|ing|ation)|chart)\b",),
    "reprojection": (r"\breproject(?:s|ed|ing|ion)?\b", r"\btransform\s+crs\b"),
    "resampling": (r"\bresampl(?:e|es|ed|ing)\b",),
    "rasterization": (r"\brasteriz(?:e|es|ed|ing|ation)\b",),
    "vectorization": (
        r"\bvectoriz(?:e|es|ed|ing|ation)\b",
        r"\bpolygoniz(?:e|es|ed|ing|ation)\b",
    ),
}

_GENERIC_OPERATION_TOKENS = {
    "analysis",
    "calculate",
    "compute",
    "data",
    "dataset",
    "execute",
    "function",
    "generate",
    "method",
    "operation",
    "process",
    "result",
    "step",
    "test",
    "using",
}

# 英语虚词封闭集：token 化时无条件过滤。closure 校验的 subset 规则曾把
# required_*_semantics 原文里的 with/and 等介词连词当成强制语义项（实测
# 'table with slope bins and mean water occurrence' 仅因证据缺 'with' 而判
# 契约失败），需求越长被强制的无关 token 越多。词袋模型本就不处理否定与
# 句法，滤掉虚词只减噪声、不损失判别信息。
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and",
    "or", "by", "from", "as", "into", "onto", "upon", "over", "under",
    "above", "below", "between", "across", "within", "without", "through",
    "during", "before", "after", "along", "around", "about", "against",
    "per", "each", "via", "versus", "vs", "than", "then", "when", "where",
    "which", "while", "that", "this", "these", "those", "there", "here",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "its", "their", "our", "your", "his", "her",
    "it", "they", "we", "you", "he", "she", "not", "no", "any", "all",
    "both", "some", "such", "other", "another", "if", "so", "but", "nor",
    "up", "down", "out", "off", "etc",
})

# 词形归一豁免：本身以 s/ies 结尾但并非复数变形，剥离会破坏语义；
# 'axes' 剥成 'ax' 更糟，保留原形（axes/axis 对齐已知不支持，见测试固化）。
_TOKEN_STEM_EXEMPT = frozenset({"series", "timeseries", "axes"})


def _normalize_token(token: str) -> str:
    """保守的确定性词形归一：复数与 -ing 归并到基本形。

    只处理高置信的规则形态（rasters→raster、classes→class、geometries→
    geometry、binning→bin），不做真正的词干化：-ed 等派生形态交给同义词组
    显式收录，避免过度合并产生碰撞。要求两侧证据走同一函数，归一后形态
    即使不是词典形（indices→indice、merging→merg）也不影响集合匹配的
    正确性——merging/scaling 类 -e 结尾动词的归一形与原形不一致，由操作族
    正则（data_join/scaling）兜底，测试中固化为已知限制。
    """
    if token in _TOKEN_STEM_EXEMPT:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "ches", "shes", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        # 双写辅音还原：binning→binn→bin、mapping→mapp→map
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    return token

# 领域同义词组：组内任一 token 出现即视为整组语义都已覆盖。
# 数据契约组按 refinement 实际产出的 required_*_semantics 词汇校准
# （categorical_raster / continuous_raster / binary_mask / table / vector_zones /
#   gee_image_collection 等）：这些抽象值域词几乎不会出现在函数签名或步骤
# inputs/outputs 里，必须映射到计划文本中的具体表述，否则闭合校验系统性失败。
_TOKEN_SYNONYM_GROUPS = (
    frozenset({"gee", "ee", "earth", "engine"}),
    # 分类值域 ↔ 土地覆盖/类别表述
    frozenset({
        "categorical", "category", "categories", "class", "classes",
        "classification", "classified", "discrete", "thematic",
        "landcover", "lulc", "nlcd",
    }),
    # 连续值域 ↔ 数值/指数/统计量表述
    frozenset({
        "continuous", "float", "floating", "numeric", "numerical",
        "value", "values", "index", "intensity", "gradient",
        "variance", "deviation", "ratio", "weight", "weights",
    }),
    # 掩膜/布尔值域
    frozenset({"mask", "masked", "masking", "binary", "boolean", "bool"}),
    # 表格数据 ↔ dataframe/统计函数证据
    frozenset({
        "table", "tabular", "dataframe", "df", "csv", "series",
        "record", "records", "statistics", "stats",
    }),
    # 栅格数据 ↔ 影像/像元/波段表述
    frozenset({"raster", "image", "grid", "pixel", "pixels", "band", "bands"}),
    # 矢量数据 ↔ 几何/要素/分区表述
    frozenset({
        "vector", "geometry", "geometries", "polygon", "polygons",
        "feature", "features", "shapefile", "geojson", "zones", "zonal",
    }),
    # 影像集合 ↔ 时序栈表述
    frozenset({"collection", "collections", "timeseries", "stack", "catalog"}),
    # 降水量 ↔ 降雨表述（refinement 与步骤散文各按学科习惯选词）
    frozenset({"precipitation", "rainfall", "rain", "precip"}),
    # 可视化产物：'map' 已从 visualization 操作族正则移除（多义噪声），
    # 但 required_*_semantics 里 "map" 作为数据契约仍需与 plot/chart 证据对齐。
    frozenset({
        "map", "plot", "chart", "figure", "visualization", "visualize",
        "visual", "choropleth",
    }),
)

# 同义词成员豁免最短 token 长度过滤（如 'ee' 长度为 2，但必须保留）。
# 组匹配在归一化空间进行：成员先过 _normalize_token，与证据 token 同构。
_NORMALIZED_SYNONYM_GROUPS = tuple(
    frozenset(_normalize_token(member) for member in group)
    for group in _TOKEN_SYNONYM_GROUPS
)
_SYNONYM_MEMBER_TOKENS = (
    set().union(*_TOKEN_SYNONYM_GROUPS) | set().union(*_NORMALIZED_SYNONYM_GROUPS)
)


def normalize_function_name(value: Any) -> str:
    return str(value or "").strip().casefold()


def function_name_matches(function_name: Any, expected_names: Iterable[Any]) -> bool:
    actual = normalize_function_name(function_name)
    if not actual:
        return False
    actual_leaf = actual.rsplit(".", 1)[-1]
    for expected in expected_names:
        normalized = normalize_function_name(expected)
        if not normalized:
            continue
        if actual == normalized or actual_leaf == normalized.rsplit(".", 1)[-1]:
            return True
    return False


def infer_operation_labels(*values: Any) -> Set[str]:
    raw_text = " ".join(str(value or "") for value in values)
    # 证据里混杂 code-style 标识符（rasterstats.zonal_stats、point_query、
    # ee.ImageCollection）。`_` `.` `/` `-` 都是 \w 或紧邻字符，会挡住 \b 词边界，
    # 使 \bzonal\b、\bpoint\s+query\b 这类 pattern 系统性漏判。把分隔符与
    # camelCase 边界规范成空格后与原文一并匹配：原有命中全部保留，只增不减。
    normalized_text = re.sub(r"[._/\-]+", " ", _split_camel_case(raw_text))
    text = f"{raw_text} {normalized_text}".casefold()
    return {
        label
        for label, patterns in _OPERATION_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }


def _split_camel_case(value: str) -> str:
    """在 camelCase/PascalCase 边界插入空格，使 ImageCollection 能拆成 image + collection。"""
    split_text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", split_text)


def _operation_tokens(value: Any) -> Set[str]:
    split_text = _split_camel_case(str(value or "")).casefold()
    tokens: Set[str] = set()
    for token in re.findall(r"[a-z0-9]+", split_text):
        if token in _STOPWORDS or token in _GENERIC_OPERATION_TOKENS:
            continue
        normalized = _normalize_token(token)
        if (
            len(normalized) > 2
            or normalized in _SYNONYM_MEMBER_TOKENS
            or token in _SYNONYM_MEMBER_TOKENS
        ):
            tokens.add(normalized)
    return tokens


def content_tokens(value: Any) -> Set[str]:
    """Public alias of the shared tokenizer for cross-module content comparison.

    步骤身份比对必须与语义判定用同一套停用词表和词形归一，否则两处分词规则会
    各自漂移，出现"语义校验认为措辞一致、身份匹配认为不同步骤"的自相矛盾。
    """
    return _operation_tokens(value)


def _expand_synonym_tokens(tokens: Set[str]) -> Set[str]:
    expanded = set(tokens)
    for group in _NORMALIZED_SYNONYM_GROUPS:
        if tokens & group:
            expanded.update(group)
    return expanded


def _token_covered(token: str, evidence_tokens: Set[str]) -> bool:
    """token 命中证据：字面出现，或其同义词组内任一成员出现。"""
    if token in evidence_tokens:
        return True
    return any(
        token in group and group & evidence_tokens
        for group in _NORMALIZED_SYNONYM_GROUPS
    )


def _is_type_token(token: str) -> bool:
    """同义词组成员即"类型/值域词"（raster/table/mask/collection...）。

    这些词描述数据是什么而非要做什么，是数据契约的形态骨架。"""
    return any(token in group for group in _NORMALIZED_SYNONYM_GROUPS)


# 多词需求里自由词（非类型词）的覆盖率边界：≤3 个自由 token 的需求仍要求
# 全覆盖，防止校验器退化成"任一词命中即过"；≥4 token 的自由短语需求
# （如 'table with slope bins and mean water occurrence' 的修饰部分）允许
# 25% 的措辞出入——subset 规则下强制项随需求长度线性增长，是随长度恶化
# 的系统性误杀。类型词不参与打折：形态错了（要 table 给 raster）就是错。
_LONG_REQUIREMENT_MIN_TOKENS = 4
_LONG_REQUIREMENT_COVERAGE = 0.75

SEMANTIC_SATISFIED = "satisfied"
SEMANTIC_CONTRADICTED = "contradicted"
SEMANTIC_UNPROVEN = "unproven"


def semantic_requirement_verdict(requirement: Any, evidence: Any) -> str:
    """判定语义需求相对证据的状态，区分"被证伪"与"未被证明"。

    闭合校验只应对正向矛盾硬失败。措辞覆盖不足属于"无法证明"：Refinement 的
    required_*_semantics 是自由文本，同一事物常有多种写法（land_use_class_codes
    与 class_codes、informal_residential_mask 与 binary_mask_or_image_of_
    informal_residential），且可用证据本身残缺（compact 候选不含 raw_description），
    在这种证据集上把"没匹配上"当成违约，会系统性误杀正确的计划。

    返回 SEMANTIC_SATISFIED / SEMANTIC_CONTRADICTED / SEMANTIC_UNPROVEN。
    """
    requirement_text = str(requirement or "").strip()
    if not requirement_text:
        return SEMANTIC_SATISFIED
    # 操作族标签与显式 token 两条证据路径任一成立即可：像 gee_image 这类需求
    # 会意外命中 data_access 操作族，但 ee.Image 函数签名不含 load/read 等词，
    # 单靠标签分支会漏判，此时应继续用 token 分支判定。
    required_labels = infer_operation_labels(requirement_text)
    if required_labels and required_labels.intersection(infer_operation_labels(evidence)):
        return SEMANTIC_SATISFIED
    required_tokens = _operation_tokens(requirement_text)
    if not required_tokens:
        # 过滤后无实质 token = 无可检验内容 = 不构成约束。
        return SEMANTIC_SATISFIED
    evidence_tokens = _operation_tokens(evidence)
    type_tokens = {token for token in required_tokens if _is_type_token(token)}
    covered_type_tokens = {
        token for token in type_tokens if _token_covered(token, evidence_tokens)
    }
    free_tokens = required_tokens - type_tokens
    free_covered = sum(
        1 for token in free_tokens if _token_covered(token, evidence_tokens)
    )
    all_free_covered = free_covered == len(free_tokens)
    long_free_covered = (
        len(free_tokens) >= _LONG_REQUIREMENT_MIN_TOKENS
        and free_covered / len(free_tokens) >= _LONG_REQUIREMENT_COVERAGE
    )

    if type_tokens and not covered_type_tokens:
        # 形态骨架一个都没命中。只有当证据明确声明了别的形态时，才构成
        # "要 table 却给 raster"式的正向矛盾；证据根本没提形态则只是未证明。
        # 部分命中不算矛盾：需求常写成 mask_or_image 这类可选形态。
        evidence_type_tokens = {
            token for token in evidence_tokens if _is_type_token(token)
        }
        if evidence_type_tokens and not (
            evidence_type_tokens & _expand_synonym_tokens(type_tokens)
        ):
            return SEMANTIC_CONTRADICTED
        return SEMANTIC_UNPROVEN

    if covered_type_tokens == type_tokens and (all_free_covered or long_free_covered):
        return SEMANTIC_SATISFIED
    return SEMANTIC_UNPROVEN


def semantic_requirement_satisfied(requirement: Any, evidence: Any) -> bool:
    """Require the semantic content terms, or a known operation-family match."""
    return semantic_requirement_verdict(requirement, evidence) == SEMANTIC_SATISFIED


def evaluate_operation_compatibility(
    step: Dict[str, Any],
    capability_entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Prove or refute that a selected capability implements the step's operation.

    证据权威分层（高到低），各通道依次尝试、互不短路：
      1. forbidden_function_names —— 显式禁止，唯一无条件硬否决；
      2. preferred_function_names 命中 —— 显式点名，最强通过证据；
      3. required_operation（refinement 显式下发）—— 硬边界：标签通道与
         归一化 token 通道先后尝试求证；仅当约束与候选**双方都有**操作族
         标签且完全不交（正向矛盾证据）才判硬 operation_mismatch；
      4. 步骤散文（step_name/description）推断的标签 —— 纯启发式线索：
         匹配可作为通过证据（operation_match, label_source=step_prose），
         不匹配只说明"无法证明"，落 advisory 的 operation_unproven，
         绝不产生硬失败——散文措辞无权否决候选。

    返回 dict 除既有键外新增：
      advisory: bool —— compatible=False 时 True 表示仅缺证据/启发式分歧，
        不构成正向矛盾，消费方应按 advisory 放行而非硬失败；
      label_source: 通过判定所依据标签的来源（required_operation/step_prose）。
    """
    constraints = step.get("operation_constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}

    function_name = str(
        capability_entry.get("function_name")
        or (step.get("implementation_hint") or {}).get("function_name")
        or ""
    ).strip()
    preferred = constraints.get("preferred_function_names") or []
    forbidden = constraints.get("forbidden_function_names") or []
    if isinstance(preferred, str):
        preferred = [preferred]
    if isinstance(forbidden, str):
        forbidden = [forbidden]

    if function_name_matches(function_name, forbidden):
        return {
            "compatible": False,
            "status": "forbidden_function",
            "advisory": False,
            "required_operations": [],
            "candidate_operations": [],
            "reason": f"selected function is forbidden by refinement: {function_name}",
        }

    required_operation = str(constraints.get("required_operation") or "").strip()
    explicit_labels = infer_operation_labels(required_operation)
    prose_labels = infer_operation_labels(
        step.get("step_name"),
        step.get("description"),
    )
    # 报告字段：显式约束存在时只报告它推出的标签，不再让散文顶替——
    # 显式 required_operation 推不出标签时以空集示人，而不是伪装成散文标签。
    required_labels = explicit_labels if required_operation else prose_labels
    candidate_labels = infer_operation_labels(
        function_name,
        capability_entry.get("raw_description"),
        capability_entry.get("signature"),
    )

    def _with_preferred_advisory(result: Dict[str, Any]) -> Dict[str, Any]:
        if preferred and not function_name_matches(function_name, preferred):
            result["preferred_function_matched"] = False
            result["preferred_function_names"] = preferred
            result["advisory_warnings"] = [
                "selected function does not match advisory preferred functions"
            ]
        elif preferred:
            result["preferred_function_matched"] = True
            result["preferred_function_names"] = preferred
        return result

    # refinement 点名的 preferred 函数是最强的确定性证据：选中即视为实现了
    # 所需操作。该检查必须先于操作族判定——code-style 函数名（如 ttest_ind）
    # 经常无法被操作族正则识别，否则会出现"按建议选了函数仍被判不兼容"。
    if preferred and function_name_matches(function_name, preferred):
        return {
            "compatible": True,
            "status": "preferred_function_match",
            "advisory": False,
            "required_operations": sorted(required_labels),
            "candidate_operations": sorted(candidate_labels),
            "reason": "selected function matches an explicit preferred function",
            "preferred_function_matched": True,
            "preferred_function_names": preferred,
        }

    if required_operation:
        # 通道一：显式约束的操作族标签证真。
        overlap = sorted(explicit_labels.intersection(candidate_labels))
        if overlap:
            return _with_preferred_advisory({
                "compatible": True,
                "status": "operation_match",
                "advisory": False,
                "label_source": "required_operation",
                "required_operations": sorted(explicit_labels),
                "candidate_operations": sorted(candidate_labels),
                "reason": f"matched operation families: {overlap}",
            })
        # 通道二：矛盾判定必须先于 token 通道——约束与候选双方都推得出
        # 操作族标签且完全不交是正向矛盾证据（required="correlation ..."
        # vs 纯 visualization 候选），不允许被弱 token 重叠（如双方都
        # 提到 'slope'）翻案。任一侧推不出标签（词表覆盖缺口）不是矛盾，
        # 继续尝试 token 证据——显式约束的硬边界语义是"不得被证伪"。
        if explicit_labels and candidate_labels:
            return {
                "compatible": False,
                "status": "operation_mismatch",
                "advisory": False,
                "required_operations": sorted(explicit_labels),
                "candidate_operations": sorted(candidate_labels),
                "reason": (
                    "explicit required_operation and candidate imply disjoint "
                    "operation families"
                ),
            }
        # 通道三：显式约束的归一化 token。'binning of slope values' 推不出
        # 操作族标签，但归一后 binning→bin 可与 numpy.digitize 描述里的
        # bins→bin 对齐。类型/值域词（raster/table/mask 等同义词组成员）
        # 描述数据形态而非操作，单独重叠不足以证真：真实历史记录中
        # required="raster binning" 曾仅凭 'raster' 重叠放行了栅格堆叠函数。
        # 因此需求含操作词时，重叠必须至少命中一个操作词；需求为纯类型名
        # （如 required_operation="raster"）时才退回类型词字面重叠判定。
        required_tokens = _operation_tokens(required_operation)
        candidate_tokens = _operation_tokens(
            " ".join(
                str(value or "")
                for value in (
                    function_name,
                    capability_entry.get("raw_description"),
                    capability_entry.get("signature"),
                )
            )
        )
        term_overlap = required_tokens.intersection(candidate_tokens)
        operation_terms = {
            token for token in required_tokens if not _is_type_token(token)
        }
        operation_term_hits = sorted(term_overlap.intersection(operation_terms))
        if operation_term_hits or (not operation_terms and term_overlap):
            return _with_preferred_advisory({
                "compatible": True,
                "status": "explicit_operation_term_match",
                "advisory": False,
                "label_source": "required_operation",
                "required_operations": sorted(explicit_labels),
                "candidate_operations": sorted(candidate_labels),
                "reason": (
                    "matched explicit operation terms: "
                    f"{operation_term_hits or sorted(term_overlap)}"
                ),
            })
        return {
            "compatible": False,
            "status": "operation_unproven",
            "advisory": True,
            "required_operations": sorted(explicit_labels),
            "candidate_operations": sorted(candidate_labels),
            "reason": (
                "no deterministic evidence links the selected function to the "
                "explicit required_operation (insufficient evidence, not a "
                "contradiction)"
            ),
        }

    # 无显式约束：散文标签只是启发式线索。
    overlap = sorted(prose_labels.intersection(candidate_labels))
    if overlap:
        return _with_preferred_advisory({
            "compatible": True,
            "status": "operation_match",
            "advisory": False,
            "label_source": "step_prose",
            "required_operations": sorted(prose_labels),
            "candidate_operations": sorted(candidate_labels),
            "reason": f"matched operation families: {overlap}",
        })
    if prose_labels and candidate_labels:
        return {
            "compatible": False,
            "status": "operation_unproven",
            "advisory": True,
            "required_operations": sorted(prose_labels),
            "candidate_operations": sorted(candidate_labels),
            "reason": (
                "prose-inferred operation labels disagree with candidate labels; "
                "prose is a heuristic hint and cannot refute the candidate"
            ),
        }
    return {
        "compatible": False,
        "status": "operation_unproven",
        "advisory": True,
        "required_operations": sorted(required_labels),
        "candidate_operations": sorted(candidate_labels),
        "reason": "no deterministic evidence links the selected function to the step operation",
    }


def merge_operation_constraints(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge issue-level constraints for one step without silently dropping conflicts."""
    issue_ids: List[str] = []
    required_operations: List[str] = []
    preferred: List[str] = []
    forbidden: List[str] = []
    input_semantics: List[str] = []
    output_semantics: List[str] = []
    removal_requested = False
    for item in items:
        issue_id = str(item.get("issue_id") or "").strip()
        if issue_id:
            issue_ids.append(issue_id)
        if item.get("removal_requested"):
            removal_requested = True
        operation = str(item.get("required_operation") or "").strip()
        if operation:
            required_operations.append(operation)
        for field, target in (
            ("preferred_function_names", preferred),
            ("forbidden_function_names", forbidden),
            ("required_input_semantics", input_semantics),
            ("required_output_semantics", output_semantics),
        ):
            values = item.get(field) or []
            if isinstance(values, str):
                values = [values]
            target.extend(str(value).strip() for value in values if str(value).strip())

    return {
        "issue_ids": list(dict.fromkeys(issue_ids)),
        "required_operation": "; ".join(dict.fromkeys(required_operations)),
        "preferred_function_names": list(dict.fromkeys(preferred)),
        "forbidden_function_names": list(dict.fromkeys(forbidden)),
        "required_input_semantics": list(dict.fromkeys(input_semantics)),
        "required_output_semantics": list(dict.fromkeys(output_semantics)),
        # Refinement 要求删除该步骤时，"步骤消失"是闭合成功的证据而非违约，
        # 闭合校验必须能区分这一点，否则执行指令反而导致 missing step 硬失败。
        "removal_requested": removal_requested,
    }


__all__ = [
    "SEMANTIC_CONTRADICTED",
    "SEMANTIC_SATISFIED",
    "SEMANTIC_UNPROVEN",
    "content_tokens",
    "evaluate_operation_compatibility",
    "function_name_matches",
    "infer_operation_labels",
    "merge_operation_constraints",
    "semantic_requirement_satisfied",
    "semantic_requirement_verdict",
]
