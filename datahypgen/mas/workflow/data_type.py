"""
Workflow Models for DataGenHyp

This module contains all data models and enums used throughout the workflow system,
including findings, Tasks, WorkflowSessions,
and WorkflowStates.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


# manifest source 里代表"本地可直接访问"或"尚未解析"的前缀，其余一律视为远程项。
# 用排除法而不是枚举远程前缀：每接入一个新数据源都去补白名单，漏一个就等于
# 让该源的探测结果静默失效。
_NON_REMOTE_SOURCE_PREFIXES = ("LOCAL:", "UNRESOLVED:")

# 探测结论中视为"可放行"的状态。advisory 表示该源改由别处求证（如 GEE
# 走 ResourceCheck 自己的 ee 探测），不是失败。
_PROBE_OK_STATUSES = frozenset({"pass", "advisory"})


def is_remote_manifest_source(source: Any) -> bool:
    """判断 manifest source 是否指向需要联网求证的远程资源。"""
    normalized = str(source or "").strip()
    if not normalized:
        return False
    return not normalized.upper().startswith(_NON_REMOTE_SOURCE_PREFIXES)


class WorkflowState(Enum):
    """
    工作流状态枚举
    
    定义了 DataGenHyp MAS 系统中所有可能的工作流状态,
    包括原有的研究想法生成流程和新增的地理领域发现流程。
    """
    INITIAL = "initial"
    ERROR = "error"
    LITERATURE_SEARCH = "literature_search"
    DATASET_PROFILING = "dataset_profiling"
    TASK_PLANNING = "task_planning"
    PLAN_REFINEMENT = "plan_refinement"
    RESOURCE_CHECK = "resource_check"
    IDEA_EVALUATION = "idea_evaluation"
    AWAITING_FEEDBACK = "awaiting_feedback"
    COMPLETED = "completed" 
    def __str__(self) -> str:
        """返回状态的字符串表示"""
        return self.value
    
    def __repr__(self) -> str:
        """返回状态的详细表示"""
        return f"WorkflowState.{self.name}"
    
    @classmethod
    def get_all_states(cls) -> List['WorkflowState']:
        """获取所有状态"""
        return list(cls)
    
    
    def can_transition_to(self, next_state: 'WorkflowState') -> bool:
        """
        判断是否可以转换到下一个状态
        
        Args:
            next_state: 目标状态
            
        Returns:
            是否允许转换
        """
        # 获取允许的转换关系
        allowed_transitions = self.get_allowed_transitions()
        return next_state in allowed_transitions.get(self, [])
    
    @classmethod
    def get_allowed_transitions(cls) -> dict['WorkflowState', set['WorkflowState']]:
        """Return the complete runtime transition contract for session phases."""
        return {
            cls.INITIAL: {
                cls.DATASET_PROFILING,
                cls.LITERATURE_SEARCH,
                cls.ERROR,
            },
            cls.DATASET_PROFILING: {cls.LITERATURE_SEARCH, cls.ERROR},
            cls.LITERATURE_SEARCH: {cls.TASK_PLANNING, cls.ERROR},
            cls.TASK_PLANNING: {
                cls.RESOURCE_CHECK,
                cls.IDEA_EVALUATION,
                cls.ERROR,
            },
            cls.RESOURCE_CHECK: {
                cls.PLAN_REFINEMENT,
                cls.IDEA_EVALUATION,
                # enable_plan_refinement_phase=False 时 RESOURCE_CHECK 是终态阶段，
                # 全部 application 完成后直接收口，不经 PLAN_REFINEMENT/IDEA_EVALUATION。
                cls.COMPLETED,
                cls.ERROR,
            },
            cls.PLAN_REFINEMENT: {
                cls.TASK_PLANNING,
                cls.RESOURCE_CHECK,
                cls.IDEA_EVALUATION,
                cls.ERROR,
            },
            cls.IDEA_EVALUATION: {cls.COMPLETED, cls.ERROR},
            cls.AWAITING_FEEDBACK: {cls.COMPLETED, cls.ERROR},
            cls.ERROR: set(),
            cls.COMPLETED: set(),
        }
    
    def get_next_states(self) -> set['WorkflowState']:
        """获取当前状态可以转换到的下一个状态集合"""
        return self.get_allowed_transitions().get(self, set())


@dataclass(frozen=True)
class IdeaTypeDefinition:
    """Fixed idea-type framing used before StepPlan generation."""

    idea_type: str
    type_question: str
    applicability: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_type": self.idea_type,
            "type_question": self.type_question,
            "applicability": self.applicability,
        }


@dataclass
class IdeaCandidate:
    """Structured pre-StepPlan idea candidate."""

    idea_id: str
    idea_type: str
    knowledge_anchor_and_motivation: Dict[str, Any]
    data_synergy_and_gap: Dict[str, Any]
    research_question: str
    core_hypothesis: Dict[str, Any]
    theoretical_mechanism: Dict[str, Any]
    computability_and_falsifiability: Dict[str, Any]
    novelty_constraints_and_value: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "idea_type": self.idea_type,
            "knowledge_anchor_and_motivation": self.knowledge_anchor_and_motivation,
            "data_synergy_and_gap": self.data_synergy_and_gap,
            "research_question": self.research_question,
            "core_hypothesis": self.core_hypothesis,
            "theoretical_mechanism": self.theoretical_mechanism,
            "computability_and_falsifiability": self.computability_and_falsifiability,
            "novelty_constraints_and_value": self.novelty_constraints_and_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaCandidate":
        return cls(
            idea_id=str(data.get("idea_id", "")),
            idea_type=str(data.get("idea_type", "")),
            knowledge_anchor_and_motivation=dict(data.get("knowledge_anchor_and_motivation", {}) or {}),
            data_synergy_and_gap=dict(data.get("data_synergy_and_gap", {}) or {}),
            research_question=str(data.get("research_question", "")),
            core_hypothesis=dict(data.get("core_hypothesis", {}) or {}),
            theoretical_mechanism=dict(data.get("theoretical_mechanism", {}) or {}),
            computability_and_falsifiability=dict(data.get("computability_and_falsifiability", {}) or {}),
            novelty_constraints_and_value=dict(data.get("novelty_constraints_and_value", {}) or {}),
        )
    
'''
@dataclass
class finding:
    """Data class for research findings (formerly hypotheses)."""
    id: str
    text: str
    score: float = 0.0
    rationale: str = ""
    critiques: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    refine_evidence: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    references: List[Dict[str, Any]] = field(default_factory=list)
    experimental_approach: str = ""
    detailed_findings: Dict[str, Any] = field(default_factory=dict)
    method_details: Dict[str, Any] = field(default_factory=dict)
    method_critiques: List[str] = field(default_factory=list)
    refined_method_details: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rationale": self.rationale,
            "critiques": self.critiques,
            "evidence": self.evidence,
            "refine_evidence": self.refine_evidence,
            "iteration": self.iteration,
            "scores": self.scores,
            "references": self.references,
            "experimental_approach": self.experimental_approach,
            "detailed_findings": self.detailed_findings,
            "method_details": self.method_details,
            "refined_method_details": self.refined_method_details,
            "method_critiques": self.method_critiques,
            "parent_id": self.parent_id,
            "generated_at": self.generated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'finding':
        """Create an finding from a dictionary."""
        if isinstance(data.get("generated_at"), str):
            data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        return cls(**data)

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
'''

@dataclass
class Finding:

    id: str  # 发现ID
    text: str  # 发现内容描述
    score: float = 0.0  # 发现总分
    spatial_extent: Optional[Dict[str, Any]] = None  # 空间范围
    """
    空间范围示例：
    {
        "type": "bbox",  # bbox, polygon, point
        "coordinates": {
            "min_lon": 120.0, "max_lon": 121.0,
            "min_lat": 30.0, "max_lat": 31.0
        },
        "crs": "EPSG:4326"  # 坐标参考系统
    }
    """
    temporal_extent: Optional[Dict[str, Any]] = None  # 时间范围
    """
    时间范围示例：
    {
        "start": "2020-01-01",
        "end": "2023-12-31",
        "granularity": "monthly"  # yearly, monthly, daily
    }
    """
    
    data_sources: List[Dict[str, Any]] = field(default_factory=list)  # 数据来源
    """
    数据来源示例：
    [
        {
            "source_type": "OpenStreetMap",
            "query": "amenity=school",
            "retrieved_at": "2024-01-01T10:00:00",
            "record_count": 150
        },
        {
            "source_type": "Satellite Imagery",
            "dataset": "Landsat-8",
            "date": "2023-06-15"
        }
    ]
    """
    
    discovery_type: str = ""  # 发现类型
    """
    发现类型：
    - "spatial_pattern": 空间分布模式
    - "temporal_trend": 时间演变趋势
    - "spatial_correlation": 空间关联关系
    - "anomaly": 异常现象
    - "clustering": 聚类现象
    """
    rationale: str = ""  # 发现的理由/解释
    scores: Dict[str, float] = field(default_factory=dict)  # 多维度评分
    """
    多维度评分示例：
    {
        "novelty": 8.5,           # 新颖性
        "scientific_value": 7.0,   # 科学性
        "practical_value": 9.0,    # 实用性
        "data_quality": 8.0,       # 数据质量
        "spatial_significance": 7.5 # 空间显著性 (新增维度)
    }
    """
    
    comments: List[Dict[str, Any]] = field(default_factory=list)  # 评估意见
    """
    评估意见示例：
    [
        {
            "evaluator": "agent",  # agent 或 human
            "timestamp": "2024-01-01T10:00:00",
            "dimension": "novelty",
            "score": 8.5,
            "comments": "该发现揭示了之前未被注意到的空间分布规律..."
        },
        {
            "evaluator": "human",
            "timestamp": "2024-01-01T11:00:00",
            "dimension": "practical_value",
            "score": 9.0,
            "comments": "具有很强的城市规划应用价值"
        }
    ]
    """
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # 初始证据
    refine_evidence: List[Dict[str, Any]] = field(default_factory=list)  # 优化后的证据
    references: List[Dict[str, Any]] = field(default_factory=list)  # 参考文献
    workflow_history: List[Dict[str, Any]] = field(default_factory=list)  # 工作流历史
    """
    工作流历史示例：
    [
        {
            "stage": "data_retrieval",
            "timestamp": "2024-01-01T09:00:00",
            "agent": "geo_data_search",
            "action": "Retrieved OSM data for bbox",
            "details": {"record_count": 150}
        },
        {
            "stage": "finding_generation",
            "timestamp": "2024-01-01T09:30:00",
            "agent": "finding_generation_agent",
            "action": "Generated initial finding",
            "details": {"pattern_type": "spatial_clustering"}
        },
        {
            "stage": "evaluation",
            "timestamp": "2024-01-01T10:00:00",
            "agent": "finding_evaluation_agent",
            "action": "Evaluated finding",
            "details": {"total_score": 8.2}
        },
        {
            "stage": "optimization",
            "timestamp": "2024-01-01T10:30:00",
            "agent": "iterative_optimization_agent",
            "action": "Refined finding based on feedback",
            "details": {"iteration": 1, "improvement": 0.5}
        }
    ]
    """
    experimental_approach: str = ""  # 实验方法（可用于记录分析方法）
    method_details: Dict[str, Any] = field(default_factory=dict)  # 方法细节
    refined_method_details: Dict[str, Any] = field(default_factory=dict)  # 优化后的方法
    critiques: List[str] = field(default_factory=list)  # 批评意见
    method_critiques: List[str] = field(default_factory=list)  # 方法批评
    iteration: int = 0  # 迭代次数
    parent_id: Optional[str] = None  # 父发现ID（如果是从其他发现演化而来）
    detailed_findings: Dict[str, Any] = field(default_factory=dict)  # 详细想法
    generated_at: datetime = field(default_factory=datetime.now)  # 生成时间
    updated_at: datetime = field(default_factory=datetime.now)  # 更新时间 (新增)
    status: str = "draft"  # 状态 (新增)
    """
    状态可以是：
    - "draft": 草稿
    - "under_evaluation": 评估中
    - "approved": 已批准
    - "rejected": 已拒绝
    - "archived": 已归档
    """
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "spatial_extent": self.spatial_extent,
            "temporal_extent": self.temporal_extent,
            "data_sources": self.data_sources,
            "discovery_type": self.discovery_type,
            "rationale": self.rationale,
            "scores": self.scores,
            "comments": self.comments,
            "evidence": self.evidence,
            "refine_evidence": self.refine_evidence,
            "references": self.references,
            "workflow_history": self.workflow_history,
            "experimental_approach": self.experimental_approach,
            "method_details": self.method_details,
            "refined_method_details": self.refined_method_details,
            "critiques": self.critiques,
            "method_critiques": self.method_critiques,
            "iteration": self.iteration,
            "parent_id": self.parent_id,
            "detailed_findings": self.detailed_findings,
            "generated_at": self.generated_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Finding':
        """Create a Finding from a dictionary."""
        # 处理时间字段
        if isinstance(data.get("generated_at"), str):
            data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        elif "updated_at" not in data:
            data["updated_at"] = data.get("generated_at", datetime.now())
            
        return cls(**data) 
    def add_workflow_step(self, stage: str, agent: str, action: str, 
                          details: Optional[Dict[str, Any]] = None) -> None:
        """添加工作流步骤记录"""
        self.workflow_history.append({
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "details": details or {}
        })
        self.updated_at = datetime.now()
    
    def add_evaluation(self, evaluator: str, dimension: str, 
                       score: float, comments: str) -> None:
        """添加评估意见"""
        self.comments.append({
            "evaluator": evaluator,
            "timestamp": datetime.now().isoformat(),
            "dimension": dimension,
            "score": score,
            "comments": comments
        })
        self.scores[dimension] = score
        self.updated_at = datetime.now()
    
    def calculate_weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        if not self.scores:
            return 0.0
        
        if weights is None:
            # 均等权重
            return sum(self.scores.values()) / len(self.scores)
        
        # 加权计算
        weighted_sum = 0.0
        total_weight = 0.0
        for dimension, score in self.scores.items():
            if dimension in weights:
                weighted_sum += score * weights[dimension]
                total_weight += weights[dimension]
        
        if total_weight > 0:
            self.score = weighted_sum / total_weight
        else:
            self.score = sum(self.scores.values()) / len(self.scores)
        
        return self.score
    
    def get_spatial_center(self) -> Optional[Dict[str, float]]:
        """获取空间范围的中心点"""
        if not self.spatial_extent or not self.spatial_extent.get("coordinates"):
            return None
        
        coords = self.spatial_extent["coordinates"]
        if self.spatial_extent["type"] == "bbox":
            return {
                "lon": (coords["min_lon"] + coords["max_lon"]) / 2,
                "lat": (coords["min_lat"] + coords["max_lat"]) / 2
            }
        elif self.spatial_extent["type"] == "point":
            return coords
        
        return None
    
    def update_status(self, new_status: str) -> None:
        """更新状态"""
        valid_statuses = ["draft", "under_evaluation", "approved", "rejected", "archived"]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status: {new_status}. Must be one of {valid_statuses}")
        
        self.status = new_status
        self.updated_at = datetime.now()
        self.add_workflow_step(
            stage="status_update",
            agent="system",
            action=f"Status changed to {new_status}",
            details={"previous_status": self.status, "new_status": new_status}
        )


# =============================================================================
# task_plan step 目标结构（schema 语义文档）
# =============================================================================
#
# task_plan 是一个 List[Dict]，每个 dict 代表一个分析步骤。
# 数据和步骤依赖只由 inputs / outputs 产物引用合同定义，不从数组顺序推断。
#
# 已有字段（保持不变）：
#   - id              : int       # 步骤编号
#   - step_name       : str       # 步骤名称
#   - description     : str       # 步骤描述
#   - tools           : List[str] # 需要的工具列表
#   - inputs          : List[Dict]# base_dataset / required_dataset / step_output 引用
#   - outputs         : List[Dict]# 恰好一个带全局唯一 ID 的主产物
#
# 新增字段（Commit 1 预留，后续 Commit 由 TASK_PLANNING / ResourceCheck 填充）：
#   - task_type              : Optional[str]
#       # 步骤的任务类型（如 "classification", "change_detection", "index_calculation"）
#       # 用于 capability index 匹配和工具可达性评估
#   - implementation_hint    : Optional[str]
#       # 实现提示（如 "use ee.Reducer.mean()", "apply NDVI formula"）
#       # 来自 capability index 的推荐实现方式
#   - implementation_candidates : Optional[List[Dict[str, Any]]]
#       # 候选实现方案列表，每项包含 {tool, api_pattern, confidence, source}
#       # 由 capability search 填充，ResourceCheck 据此评估可达性
#   - tool_reachability      : Optional[Dict[str, Any]]
#       # 工具可达性摘要（ResourceCheck 填充），结构：
#       # {
#       #     "status": "reachable" | "partial" | "unreachable",
#       #     "checked_tools": [{"tool": str, "reachable": bool, "reason": str}],
#       #     "best_candidate": Optional[str]
#       # }
# =============================================================================


#任务数据结构定义
@dataclass
class Task:
    """Data class for research tasks (formerly research goals)."""
    id: str
    spatial_extent: Dict[str, Any]
    data_requirements: List[str]
    analysis_methods: List[str]
    description: str
    background: str = ""
    ref_code_path: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "spatial_extent": self.spatial_extent,
            "data_requirements": self.data_requirements,
            "analysis_methods": self.analysis_methods,
            "description": self.description,
            "background": self.background,
            "ref_code_path": self.ref_code_path,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create a Task from a dictionary."""
        # Create a copy to avoid modifying the original
        task_data = data.copy()
        
        # Remove constraints field if present (for backward compatibility)
        task_data.pop("constraints", None)
        # Historical sessions duplicated per-application descriptions here.
        task_data.pop("application_scenario", None)
        
        # Handle created_at timestamp
        if isinstance(task_data.get("created_at"), str):
            task_data["created_at"] = datetime.fromisoformat(task_data["created_at"])

        return cls(**task_data)


@dataclass
class ApplicationRecord:
    """Per-application workflow state. This is the authoritative workflow unit."""

    idea_id: str
    application_name: str
    application_description: str
    application_context: str = ""
    approach_description: str = ""
    idea_candidate: Dict[str, Any] = field(default_factory=dict)
    idea_dataset_grounding: List[Dict[str, str]] = field(default_factory=list)
    idea_revision_audit: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    status: str = "pending"
    final_disposition: str = ""
    task_plan: List[Dict[str, Any]] = field(default_factory=list)
    required_datasets: List[Dict[str, Any]] = field(default_factory=list)
    capability_anchor_meta: Dict[str, Any] = field(default_factory=dict)
    step_plan_generation_audit: List[Dict[str, Any]] = field(default_factory=list)
    plan_revision_history: List[Dict[str, Any]] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    rationale: str = ""
    resource_check: Dict[str, Any] = field(default_factory=dict)
    resource_check_feedback: Dict[str, Any] = field(default_factory=dict)
    plan_refinement_result: Dict[str, Any] = field(default_factory=dict)
    plan_refinement_attempt_history: List[Dict[str, Any]] = field(default_factory=list)
    idea_evaluation: Dict[str, Any] = field(default_factory=dict)
    idea_evaluation_history: List[Dict[str, Any]] = field(default_factory=list)
    replan_round: int = 0
    evaluation_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "application_name": self.application_name,
            "application_description": self.application_description,
            "application_context": self.application_context,
            "approach_description": self.approach_description,
            "idea_candidate": self.idea_candidate,
            "idea_dataset_grounding": self.idea_dataset_grounding,
            "idea_revision_audit": self.idea_revision_audit,
            "priority": self.priority,
            "status": self.status,
            "final_disposition": self.final_disposition,
            "task_plan": self.task_plan,
            "required_datasets": self.required_datasets,
            "capability_anchor_meta": self.capability_anchor_meta,
            "step_plan_generation_audit": self.step_plan_generation_audit,
            "plan_revision_history": self.plan_revision_history,
            "datasets": self.datasets,
            "rationale": self.rationale,
            "resource_check": self.resource_check,
            "resource_check_feedback": self.resource_check_feedback,
            "plan_refinement_result": self.plan_refinement_result,
            "plan_refinement_attempt_history": self.plan_refinement_attempt_history,
            "idea_evaluation": self.idea_evaluation.to_dict() if hasattr(self.idea_evaluation, "to_dict") else self.idea_evaluation,
            "idea_evaluation_history": self.idea_evaluation_history,
            "replan_round": self.replan_round,
            "evaluation_result": self.evaluation_result,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicationRecord":
        return cls(
            idea_id=data.get("idea_id", ""),
            application_name=data.get("application_name", ""),
            application_description=data.get("application_description", ""),
            application_context=data.get("application_context", ""),
            approach_description=data.get("approach_description", data.get("proposed_method", "")),
            idea_candidate=data.get("idea_candidate", {}),
            idea_dataset_grounding=data.get("idea_dataset_grounding", []),
            idea_revision_audit=data.get("idea_revision_audit", {}),
            priority=data.get("priority", 0),
            status=data.get("status", "pending"),
            final_disposition=data.get("final_disposition", ""),
            task_plan=data.get("task_plan", []),
            required_datasets=data.get("required_datasets", []),
            capability_anchor_meta=data.get("capability_anchor_meta", {}),
            step_plan_generation_audit=data.get("step_plan_generation_audit", []),
            plan_revision_history=data.get("plan_revision_history", []),
            datasets=data.get("datasets", []),
            rationale=data.get("rationale", ""),
            resource_check=data.get("resource_check", {}),
            resource_check_feedback=data.get("resource_check_feedback", {}),
            plan_refinement_result=data.get("plan_refinement_result", {}),
            plan_refinement_attempt_history=data.get("plan_refinement_attempt_history", []),
            idea_evaluation=_normalize_idea_evaluation_payload(data.get("idea_evaluation", {})),
            idea_evaluation_history=data.get("idea_evaluation_history", []),
            replan_round=data.get("replan_round", 0),
            evaluation_result=data.get("evaluation_result"),
        )


CROSS_DOMAIN_AXIS_ROLES = (
    "driver",
    "response",
    "covariate",
    "stratifier",
    "constraint",
)
CROSS_DOMAIN_AXIS_DEFAULT_ROLE = "covariate"
CROSS_DOMAIN_AXIS_SOURCE = "cross_domain_expansion_skill"
CROSS_DOMAIN_AXIS_MAX_SEARCH_TERMS = 2


@dataclass
class CrossDomainAxis:
    """DATASET_PROFILING 阶段跨领域推演方向的契约。

    由 `CrossDomainExpander` 技能产出，写入 `dataset_portrait.cross_domain_axes`，
    被 Survey（检索词种子 / 锚定豁免 / 评分提示 / DomainFilter 判据）、
    CandidateDatasetScout（`cross_domain` 检索轴）与 IdeaGeneration（多元化方向）消费。

    该契约是推演方向而非数据需求：`dataset_search_terms` 和 `literature_query` 都是
    检索建议，不构成执行级依赖；`role` 描述数据集核心变量在目标领域中承担的语义角色。
    技能失败时上游写入空列表，因此所有消费方都必须容忍该字段缺失。
    """
    domain_name: str
    role: str = CROSS_DOMAIN_AXIS_DEFAULT_ROLE
    dataset_search_terms: List[str] = field(default_factory=list)
    literature_query: str = ""
    source: str = CROSS_DOMAIN_AXIS_SOURCE

    def validation_errors(self) -> List[str]:
        """Return field-level violations of the cross-domain axis contract."""
        errors: List[str] = []
        if not (isinstance(self.domain_name, str) and self.domain_name.strip()):
            errors.append("domain_name: must contain non-whitespace text")
        if self.role not in CROSS_DOMAIN_AXIS_ROLES:
            errors.append(
                f"role: must be one of {list(CROSS_DOMAIN_AXIS_ROLES)}"
            )
        if not isinstance(self.dataset_search_terms, list):
            errors.append("dataset_search_terms: must be a list")
        else:
            for index, term in enumerate(self.dataset_search_terms):
                if not (isinstance(term, str) and term.strip()):
                    errors.append(
                        f"dataset_search_terms[{index}]: must contain non-whitespace text"
                    )
        if not isinstance(self.literature_query, str):
            errors.append("literature_query: must be a string")
        # 检索词与文献 query 至少一侧可用，否则该方向对下游没有任何可执行内容。
        if not self.dataset_search_terms and not str(self.literature_query or "").strip():
            errors.append(
                "dataset_search_terms/literature_query: at least one must be non-empty"
            )
        return errors

    def is_valid(self) -> bool:
        return not self.validation_errors()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "role": self.role,
            "dataset_search_terms": list(self.dataset_search_terms),
            "literature_query": self.literature_query,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossDomainAxis":
        if not isinstance(data, dict):
            raise TypeError("CrossDomainAxis payload must be an object")
        raw_terms = data.get("dataset_search_terms")
        if isinstance(raw_terms, str):
            raw_terms = [raw_terms]
        elif not isinstance(raw_terms, list):
            raw_terms = []
        terms = [
            str(term).strip()
            for term in raw_terms
            if isinstance(term, (str, int, float))
            and not isinstance(term, bool)
            and str(term).strip()
        ][:CROSS_DOMAIN_AXIS_MAX_SEARCH_TERMS]
        role = str(data.get("role") or "").strip().lower()
        if role not in CROSS_DOMAIN_AXIS_ROLES:
            role = CROSS_DOMAIN_AXIS_DEFAULT_ROLE
        return cls(
            domain_name=str(data.get("domain_name") or "").strip(),
            role=role,
            dataset_search_terms=terms,
            literature_query=str(data.get("literature_query") or "").strip(),
            source=str(data.get("source") or CROSS_DOMAIN_AXIS_SOURCE).strip()
            or CROSS_DOMAIN_AXIS_SOURCE,
        )


def normalize_cross_domain_axes(raw: Any) -> List[Dict[str, Any]]:
    """Coerce a portrait payload into contract-valid cross-domain axis dicts.

    非法输入（整体非 list、元素非 dict、缺 domain_name、方向重名、两侧检索内容都为空）
    确定性丢弃而不抛异常：该字段是画像的可选增强，技能失败已返回空列表，此处继续
    保持 fail-open，避免让一条脏方向阻断整个画像阶段。
    """
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen_domains = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            axis = CrossDomainAxis.from_dict(item)
        except TypeError:
            continue
        if axis.validation_errors():
            continue
        key = axis.domain_name.casefold()
        if key in seen_domains:
            continue
        seen_domains.add(key)
        normalized.append(axis.to_dict())
    return normalized


@dataclass
class WorkflowSession:
    """Data class for workflow sessions."""
    id: str
    task: Task
    findings: List[Finding] = field(default_factory=list)  # 使用Finding存储发现
    iterations_completed: int = 0
    max_iterations: int = 4
    state: WorkflowState = WorkflowState.INITIAL
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    top_findings: List[str] = field(default_factory=list)  # 存储top发现的ID
    tool_usage: Dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    # 地理领域发现流程的额外字段
    literature_review: List[Dict[str, Any]] = field(default_factory=list)
    literature_review_raw: List[Dict[str, Any]] = field(default_factory=list)
    literature_filter_params: Dict[str, Any] = field(default_factory=dict)
    survey_results: Dict[str, Any] = field(default_factory=dict)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    # 画像负载保持开放字典（profiler 会持续增补语义字段），但其中
    # `cross_domain_axes` 由 CrossDomainAxis 约束，写入前经
    # normalize_cross_domain_axes() 归一，见 8.1 节。
    dataset_portrait: Dict[str, Any] = field(default_factory=dict)
    application_records: List[ApplicationRecord] = field(default_factory=list)
    active_application_id: Optional[str] = None
    planning_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "task": self.task.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
            "iterations_completed": self.iterations_completed,
            "max_iterations": self.max_iterations,
            "state": self.state.value,
            "feedback_history": self.feedback_history,
            "top_findings": self.top_findings,
            "tool_usage": self.tool_usage,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "literature_review": getattr(self, 'literature_review', []),
            "literature_review_raw": getattr(self, 'literature_review_raw', []),
            "literature_filter_params": getattr(self, 'literature_filter_params', {}),
            "survey_results": getattr(self, 'survey_results', {}),
            "domain_knowledge": getattr(self, 'domain_knowledge', {}),
            "dataset_portrait": getattr(self, 'dataset_portrait', {}),
            "application_records": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in getattr(self, "application_records", [])
            ],
            "active_application_id": getattr(self, "active_application_id", None),
            "planning_audit": getattr(self, "planning_audit", {}),
        }


@dataclass
class ExecutionContract:
    """
    Agent间传递的强类型执行契约。

    核心语义：resource_check_passed=True 表示 mandatory 资源已形成 strong bindable 合同，
    且任务步骤的工具可达性和类型链均达到执行阈值。
    对于经 ResourcesCheck 解析的辅助数据，这意味着：
    - 只有 `binding_status == "resolved_strong"` 的 requirement 才会写成 `bind_status == "bindable"`
    - `resolved_weak`、`unresolved_semantic`、`unresolved_real_missing` 等都必须保持 `bind_status == "unbound"`
    - GeoAnalyst 只能消费 manifest bindable 项、retrieved_datasets、resolved_dataset_bindings 中的 strong resolved 合同
    数据分析阶段不承担任何数据发现、替换或缺失处理职责。
    """
    # 资源检查结果（来自 ResourcesCheck_Agent）
    primary_data_verified: bool = False  # DEPRECATED: 仅作过渡观测，不参与决策
    primary_data_path: Optional[str] = None
    guaranteed_datasets: List[str] = field(default_factory=list)
    unavailable_tools: List[str] = field(default_factory=list)
    # True = 所有 mandatory 数据均 bindable 且远程探测通过。
    # mandatory 由 ResourcesCheck 的 execution_requirement_level 规则显式判定。
    resource_check_passed: bool = False

    # Commit 1: 数据访问清单（每项数据需求的绑定状态）
    data_access_manifest: List[Dict[str, Any]] = field(default_factory=list)
    """
    data_access_manifest 结构示例（强绑定可执行，弱绑定仅供修复诊断）：
    [
        {
            "requirement_id": "primary_dataset",
            "mandatory": True,
            "source": "LOCAL:/path/to/data",
            "access_mode": "direct_read",
            "bind_status": "bindable",
            "binding_status": "provided_primary",
            "bind_status_reason": "primary_dataset_verified",
            "resolution_source": "primary_dataset",
            "constraint_report": {},
            "local_stage_path": "/path/to/data",
            "spatial_coverage": {"type": "bbox", "coordinates": {...}},
            "connectivity_probe": None,  # 本地数据无需探测
            "note": ""
        },
        {
            "requirement_id": "rd_landsat",
            "mandatory": True,
            "dataset_title": "Landsat 8 Collection 2 Level 2",
            "source": "GEE:LANDSAT/LC08/C02/T1_L2",
            "access_mode": "direct_read",
            "bind_status": "bindable",
            "binding_status": "resolved_strong",
            "bind_status_reason": "strong_binding_accepted",
            "resolution_source": "local_strict_resolution",
            "constraint_report": {
                "region_match": "exact",
                "time_match": "exact",
                "role_match": "exact",
                "type_match": true,
                "theme_match": true,
                "family_match": true
            },
            "local_stage_path": None,
            "spatial_coverage": None,
            "id": "LANDSAT/LC08/C02/T1_L2",
            "type": "image_collection",
            "catalog_type": "image_collection",
            "ee_id": "LANDSAT/LC08/C02/T1_L2",
            "ee_object_type": "ImageCollection",
            "ee_ctor": "ee.ImageCollection",
            "docs": "https://developers.google.com/earth-engine/datasets/catalog/...",
            "sample_code": "https://code.earthengine.google.com/...",
            "provider": "USGS",
            "tags": ["landsat", "surface reflectance"],
            "thematic_group": "Land Imagery",
            "connectivity_probe": {
                "status": "pass",  # pass/fail/unknown
                "checked_at": "2024-01-01T10:00:00",
                "reason": "GEE authentication successful, asset accessible",
                "latency_ms": 150
            },
            "note": ""
        },
        {
            "requirement_id": "optional_dem",
            "mandatory": False,
            "source": "API:https://elevation-api.io/api/elevation",
            "access_mode": "stage_on_demand",
            "bind_status": "unbound",
            "binding_status": "resolved_weak",
            "bind_status_reason": "weak_binding_requires_step_plan_review",
            "resolution_source": "online_fallback_resolution",
            "constraint_report": {
                "region_match": "global",
                "time_match": "unknown",
                "role_match": "acceptable",
                "type_match": true,
                "theme_match": true,
                "family_match": false
            },
            "local_stage_path": None,
            "spatial_coverage": None,
            "connectivity_probe": {
                "status": "fail",
                "checked_at": "2024-01-01T10:00:05",
                "reason": "API endpoint returned 503 Service Unavailable",
                "latency_ms": 5000
            },
            "note": "optional, will skip if unavailable"
        }
    ]
    """

    # Commit 2: 远程探测缓存（会话级，避免重复探测）
    remote_probe_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """
    remote_probe_cache 结构：
    {
        "GEE:LANDSAT/LC08/C02/T1_L2": {
            "status": "pass",
            "checked_at": "2024-01-01T10:00:00",
            "reason": "...",
            "latency_ms": 150,
            "ttl_seconds": 3600
        }
    }
    """

    # 工具可达性与计划 DAG 校验维度
    step_tool_validations: List[Dict[str, Any]] = field(default_factory=list)
    """
    step_tool_validations 结构（Commit 4 填充）：
    每个步骤的实现验证结果列表。
    [
        {
            "step_id": 1,
            "step_name": "Calculate NDVI",
            "function_name": "ee.Image.normalizedDifference",
            "package": "ee",
            "task_type": "raster_band_math",
            "task_type_in_closed_set": true,
            "validation_level": "verified",   # verified / package_verified / llm_inference / no_hint
            "score": 1.0,                     # 1.0 / 0.6 / 0.3
            "gap_type": ""                    # "" / "coverage_gap" / "capability_gap"
        }
    ]
    """

    tool_reachability_score: float = 1.0
    """
    全局工具可达性评分 [0.0, 1.0]。
    1.0 = 所有步骤的所有工具均可达；0.0 = 无任何工具可达。
    计算方式（Commit 4）：min(step_scores)，取所有步骤得分的**最小值**。
      - 1.0 (verified)        : function_name 精确命中 capability index 且 pip_verified=True
      - 0.6 (package_verified): 函数未验证但包可 import
      - 0.3 (llm_inference)   : 仅 LLM 推断，不计入可执行通过
    默认 1.0 保持向后兼容（旧路径不做工具级可达性检查时视为全通过）。
    """

    implementation_compatibility_passed: bool = False
    """
    True 表示没有任何步骤存在操作语义上的正向矛盾证据
    （operation_mismatch / forbidden_function）。
    仅缺少确定性证据（operation_unproven，如 numpy/pandas/xarray 等通用库函数
    不在操作语义索引中）按 advisory 处理，不影响该标志。
    """

    weakest_step_id: Optional[int] = None
    """
    工具可达性最差的步骤 ID（Commit 4 由 ResourceCheck 填充）。
    若所有步骤均全部可达（score=1.0），则为 None。
    """

    plan_dag_valid: bool = False
    """
    显式产物引用 DAG 校验结果。缺少校验结果时默认 fail-closed。
    """

    plan_dag_issues: List[Dict[str, Any]] = field(default_factory=list)
    """
    plan_dag_issues 记录悬空引用、真实边类型冲突、循环和缺少核心产物等问题。
    [
        {
            "code": "edge_type_mismatch",
            "from_step_id": 1,
            "to_step_id": 2,
            "source_id": "aligned_raster",
            "producer_type": "raster",
            "consumer_type": "table"
        }
    ]
    """

    # 终止决策（来自 _evaluate_termination_condition）
    should_terminate: bool = False
    termination_reason: str = ""
    terminal_rule: str = ""
    recovery_hint: str = ""

    # 执行结果（来自 GeoAnalyst_Agent，resource_check_passed=True 后才填充）
    semantic_status: str = "not_executed"
    warning_count: int = 0
    success_count: int = 0
    has_saved_outputs: bool = False
    computed_reliability: str = "low"
    reliability_rule: str = ""

    def can_proceed_to_plan_review(self) -> tuple:
        """研究计划检查不以执行格式或目录覆盖分数判定能力失败。"""
        if self.should_terminate:
            return False, f"[{self.terminal_rule}] {self.termination_reason}"
        if self.unavailable_tools:
            return False, f"Required tools unavailable: {self.unavailable_tools}"
        if not self.implementation_compatibility_passed:
            return False, "Selected step implementations are not operation-compatible."
        if not self.resource_check_passed:
            return False, "Resource check did not pass — data availability not guaranteed."
        return True, "ok"

    def can_proceed_to_analysis(self) -> tuple:
        """
        分析阶段入口检查。
        数据绑定、工具可达性和计划 DAG 必须同时通过才能进入分析。
        """
        if self.should_terminate:
            return False, f"[{self.terminal_rule}] {self.termination_reason}"
        if self.unavailable_tools:
            return False, f"Required tools unavailable: {self.unavailable_tools}"
        if not self.plan_dag_valid:
            return False, f"Task-plan DAG is invalid: {self.plan_dag_issues}"
        if self.tool_reachability_score < 0.6:
            return False, (
                "Task-plan tool reachability is below the executable threshold: "
                f"{self.tool_reachability_score}"
            )
        if not self.implementation_compatibility_passed:
            return False, "Selected step implementations are not operation-compatible."
        if not self.resource_check_passed:
            return False, "Resource check did not pass — data availability not guaranteed."
        return True, "ok"

    def get_bindable_manifest_items(self) -> List[Dict[str, Any]]:
        """返回所有 bindable 的 manifest 项（供执行阶段使用）"""
        return [item for item in self.data_access_manifest if item.get("bind_status") == "bindable"]

    def get_mandatory_unbound_items(self) -> List[Dict[str, Any]]:
        """返回所有 mandatory 但 unbound 的项（用于诊断）"""
        return [
            item for item in self.data_access_manifest
            if item.get("mandatory", False) and item.get("bind_status") != "bindable"
        ]

    def get_mandatory_remote_probe_failed_items(self) -> List[Dict[str, Any]]:
        """Commit 2: 返回所有 mandatory 远程项且探测失败的项。

        判据是"探测过且没通过"，而不是匹配某几个 source 前缀。此前白名单只列
        `GEE:` 与 `API:`，datasets.json 接入的 STAC 源（NASA_CMR: 等）探测失败
        也进不了这个列表——门禁看不见的失败等于没有门禁。
        """
        failed_items = []
        for item in self.data_access_manifest:
            if not item.get("mandatory", False):
                continue
            if not is_remote_manifest_source(item.get("source", "")):
                continue
            probe = item.get("connectivity_probe")
            if probe and probe.get("status") not in _PROBE_OK_STATUSES:
                failed_items.append(item)
        return failed_items

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_data_verified": self.primary_data_verified,
            "primary_data_path": self.primary_data_path,
            "resource_check_passed": self.resource_check_passed,
            "guaranteed_datasets": self.guaranteed_datasets,
            "unavailable_tools": self.unavailable_tools,
            "data_access_manifest": self.data_access_manifest,
            "remote_probe_cache": self.remote_probe_cache,
            "step_tool_validations": self.step_tool_validations,
            "tool_reachability_score": self.tool_reachability_score,
            "implementation_compatibility_passed": self.implementation_compatibility_passed,
            "weakest_step_id": self.weakest_step_id,
            "plan_dag_valid": self.plan_dag_valid,
            "plan_dag_issues": self.plan_dag_issues,
            "should_terminate": self.should_terminate,
            "termination_reason": self.termination_reason,
            "terminal_rule": self.terminal_rule,
            "recovery_hint": self.recovery_hint,
            "semantic_status": self.semantic_status,
            "warning_count": self.warning_count,
            "success_count": self.success_count,
            "has_saved_outputs": self.has_saved_outputs,
            "computed_reliability": self.computed_reliability,
            "reliability_rule": self.reliability_rule,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionContract":
        return cls(
            primary_data_verified=data.get("primary_data_verified", False),
            primary_data_path=data.get("primary_data_path"),
            guaranteed_datasets=data.get("guaranteed_datasets", []),
            unavailable_tools=data.get("unavailable_tools", []),
            resource_check_passed=data.get("resource_check_passed", False),
            data_access_manifest=data.get("data_access_manifest", []),
            remote_probe_cache=data.get("remote_probe_cache", {}),
            step_tool_validations=data.get("step_tool_validations", []),
            tool_reachability_score=data.get("tool_reachability_score", 1.0),
            implementation_compatibility_passed=data.get(
                "implementation_compatibility_passed",
                False,
            ),
            weakest_step_id=data.get("weakest_step_id"),
            plan_dag_valid=data["plan_dag_valid"],
            plan_dag_issues=data["plan_dag_issues"],
            should_terminate=data.get("should_terminate", False),
            termination_reason=data.get("termination_reason", ""),
            terminal_rule=data.get("terminal_rule", ""),
            recovery_hint=data.get("recovery_hint", ""),
            semantic_status=data.get("semantic_status", "not_executed"),
            warning_count=data.get("warning_count", 0),
            success_count=data.get("success_count", 0),
            has_saved_outputs=data.get("has_saved_outputs", False),
            computed_reliability=data.get("computed_reliability", "low"),
            reliability_rule=data.get("reliability_rule", ""),
        )

@dataclass
class ResourceCheckFeedback:
    """
    RESOURCE_CHECK -> PLAN_REFINEMENT 传递的反馈契约。

    失败语义：terminal_rule 非空且 recoverable=false 时是终态失败；
              recoverable=true 时由 PLAN_REFINEMENT 生成 StepPlan 重规划反馈。

    必填字段（Commit 1-2 已有）：
      - terminal_rule    : 触发终止的规则名（空字符串表示"可重试"）
      - reason           : 检查失败的原因描述
      - recovery_hint    : 给 Refinement Agent 的恢复建议
      - missing_datasets : 缺失数据集路径列表
      - unavailable_tools: 不可用工具名列表
      - affected_steps   : 受影响的任务步骤编号或名称列表

    可选字段（Commit 3 新增，后续 ResourceCheck 逐步填充）：
      - step_failures         : 步骤级失败详情（每步的失败原因和失败类型）
      - weakest_step_id       : 可达性最差的步骤 ID
      - failure_class         : 失败归类（"data_missing" | "tool_unreachable" | "type_mismatch" | "mixed"）
      - dataset_resolution_failures : 数据集解析失败详情（含 binding_status / blocking_reasons / best_candidate 等）
      - suggested_task_type   : 建议的替代任务类型（来自 capability index）
      - suggested_conversion_task : 建议的类型转换步骤描述
    """
    # 已有必填字段
    terminal_rule: str
    reason: str
    recovery_hint: str
    missing_datasets: List[str]
    unavailable_tools: List[str]
    affected_steps: List[str]

    # Commit 3（本 Commit）新增：诊断增强字段，均有向后兼容默认值
    step_failures: List[Dict[str, Any]] = field(default_factory=list)
    """
    step_failures 结构：
    [
        {
            "step_id": 2,
            "step_name": "Download DEM",
            "failure_type": "tool_unreachable",  # data_missing | tool_unreachable | type_mismatch
            "detail": "ee module not importable",
            "tools_involved": ["ee"],
            "datasets_involved": []
        }
    ]
    """

    weakest_step_id: Optional[int] = None
    """可达性最差的步骤 ID（与 ExecutionContract.weakest_step_id 对应）。"""

    failure_class: str = ""
    """
    失败归类汇总：
      - "data_missing"      : 仅缺数据
      - "tool_unreachable"  : 仅工具不可达
      - "type_mismatch"     : 仅类型链不匹配
      - "mixed"             : 多种原因混合
      - ""                  : 未分类或旧路径（向后兼容）
    """

    suggested_task_type: str = ""
    """
    建议的替代任务类型（来自 capability index）。
    例如当前步骤无法做 change_detection，但可降级为 index_calculation。
    空字符串表示无建议。
    """

    suggested_conversion_task: str = ""
    """
    建议的类型转换步骤描述。
    例如 "Insert ee.Image.toFloat() between step 1 and step 2"。
    空字符串表示无建议。
    """

    dataset_resolution_failures: List[Dict[str, Any]] = field(default_factory=list)
    """
    dataset_resolution_failures 结构：
    [
        {
            "requirement_id": "rd_precip",
            "dataset_name": "Regional precipitation driver",
            "binding_status": "resolved_weak",
            "bind_status_reason": "weak_binding_requires_step_plan_review",
            "resolution_source": "online_fallback_resolution",
            "constraint_report": {...},
            "blocking_reasons": [],
            "best_candidate": {"name": "...", "id": "..."},
            "search_stage": "rewrite",
            "online_attempted": true,
            "failure_status": "semantic_rejection_unbound",
            "unbound_class": "semantic_rejection"
        }
    ]
    """

    recoverable: bool = False
    """
    资源检查失败是否可通过后续重规划恢复。
    true  表示应进入 PLAN_REFINEMENT 诊断并生成 StepPlan 重规划反馈；
    false 表示可直接视为终态失败。
    """

    replan_target: str = ""
    """
    回溯层级：
      - "step_plan" : 修正数据 requirement 及其显式步骤引用
      - ""          : 未指定 / 不可恢复
    """

    def is_terminal(self) -> bool:
        """terminal_rule 非空即为终止信号。"""
        return bool(self.terminal_rule)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "terminal_rule": self.terminal_rule,
            "reason": self.reason,
            "recovery_hint": self.recovery_hint,
            "missing_datasets": self.missing_datasets,
            "unavailable_tools": self.unavailable_tools,
            "affected_steps": self.affected_steps,
            "step_failures": self.step_failures,
            "weakest_step_id": self.weakest_step_id,
            "failure_class": self.failure_class,
            "dataset_resolution_failures": self.dataset_resolution_failures,
            "suggested_task_type": self.suggested_task_type,
            "suggested_conversion_task": self.suggested_conversion_task,
            "recoverable": self.recoverable,
            "replan_target": self.replan_target,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceCheckFeedback":
        return cls(
            terminal_rule=data.get("terminal_rule", ""),
            reason=data.get("reason", ""),
            recovery_hint=data.get("recovery_hint", ""),
            missing_datasets=data.get("missing_datasets", []),
            unavailable_tools=data.get("unavailable_tools", []),
            affected_steps=data.get("affected_steps", []),
            step_failures=data.get("step_failures", []),
            weakest_step_id=data.get("weakest_step_id"),
            failure_class=data.get("failure_class", ""),
            dataset_resolution_failures=data.get("dataset_resolution_failures", []),
            suggested_task_type=data.get("suggested_task_type", ""),
            suggested_conversion_task=data.get("suggested_conversion_task", ""),
            recoverable=bool(data.get("recoverable", False)),
            replan_target=data.get("replan_target", ""),
        )


@dataclass
class PlanRefinementResult:
    """
    PLAN_REFINEMENT 阶段输出契约（冻结）。

    该阶段不再直接改写任务计划，而是输出两类结构化结果：
      - final_review: 资源检查已通过，确认计划可执行且表达清晰
      - replan_feedback: 资源检查失败，为 TASK_PLANNING 提供重规划反馈

    必填字段：
      - mode                   : final_review 或 replan_feedback
      - should_replan          : 是否应返回 TASK_PLANNING 重规划
      - feedback_summary       : 对 TASK_PLANNING 或最终审查的摘要说明
      - step_level_adjustments : 步骤级修改建议
      - dataset_actions        : 数据需求调整建议
      - tool_actions           : 工具依赖调整建议
      - must_preserve          : 重规划时必须保留的目标/约束
      - replan_round      : 本次是第几轮 replan（从 1 开始计数，上限由编排器控制）
      - generated_at      : 生成时间（ISO 格式）
    """
    mode: str
    should_replan: bool
    feedback_summary: str
    step_level_adjustments: List[Dict[str, Any]]
    dataset_actions: List[Dict[str, Any]]
    tool_actions: List[Dict[str, Any]]
    must_preserve: List[str]
    replan_round: int
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # ── semantic_refinement 专用（可选，默认空）──
    semantic_alignment_issues: List[Dict[str, Any]] = field(default_factory=list)

    def validation_errors(self) -> List[str]:
        """Return field-level violations of the Refinement output contract."""
        errors: List[str] = []

        def is_non_empty_string(value: Any) -> bool:
            return isinstance(value, str) and bool(value.strip())

        def validate_action_items(items: Any, *, path: str) -> None:
            if not isinstance(items, list):
                errors.append(f"{path}: must be a list")
                return
            for index, item in enumerate(items):
                item_path = f"{path}[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{item_path}: must be an object")
                    continue
                if not is_non_empty_string(item.get("action")):
                    errors.append(f"{item_path}.action: must contain non-whitespace text")
                if not is_non_empty_string(item.get("reason")):
                    errors.append(f"{item_path}.reason: must contain non-whitespace text")

        def validate_repair_constraints(
            value: Any,
            *,
            path: str,
            allow_empty: bool,
        ) -> None:
            if not isinstance(value, dict):
                errors.append(f"{path}: must be an object")
                return
            required_keys = {
                "required_operation",
                "preferred_function_names",
                "forbidden_function_names",
                "required_input_semantics",
                "required_output_semantics",
            }
            missing_keys = sorted(required_keys - set(value))
            if missing_keys:
                errors.append(f"{path}: missing required fields {missing_keys}")
            if not isinstance(value.get("required_operation"), str):
                errors.append(f"{path}.required_operation: must be a string")
            for field_name in required_keys - {"required_operation"}:
                items = value.get(field_name)
                field_path = f"{path}.{field_name}"
                if not isinstance(items, list):
                    errors.append(f"{field_path}: must be a list")
                    continue
                for index, item in enumerate(items):
                    if not is_non_empty_string(item):
                        errors.append(
                            f"{field_path}[{index}]: must contain non-whitespace text"
                        )
            if missing_keys or errors_for_path(path):
                return
            if not allow_empty and not bool(
                value["required_operation"].strip()
                or value["preferred_function_names"]
                or value["forbidden_function_names"]
                or value["required_input_semantics"]
                or value["required_output_semantics"]
            ):
                errors.append(f"{path}: all constraint dimensions are empty")

        def errors_for_path(path: str) -> bool:
            return any(error.startswith(f"{path}.") or error.startswith(f"{path}:") for error in errors)

        valid_step_actions = {
            "replace_tool",
            "replace_function",
            "change_task_type",
            "add_conversion_step",
            "remove_step",
            "keep",
        }
        if not isinstance(self.should_replan, bool):
            errors.append("should_replan: must be a boolean")
        if isinstance(self.replan_round, bool) or not isinstance(self.replan_round, int):
            errors.append("replan_round: must be an integer")
        elif self.replan_round < 0:
            errors.append("replan_round: must be non-negative")
        if self.mode not in {"final_review", "replan_feedback"}:
            errors.append("mode: must be final_review or replan_feedback")
        if not is_non_empty_string(self.feedback_summary):
            errors.append("feedback_summary: must contain non-whitespace text")

        if not isinstance(self.step_level_adjustments, list):
            errors.append("step_level_adjustments: must be a list")
        else:
            for index, item in enumerate(self.step_level_adjustments):
                item_path = f"step_level_adjustments[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{item_path}: must be an object")
                    continue
                step_id = item.get("step_id")
                if isinstance(step_id, bool) or not isinstance(step_id, int):
                    errors.append(f"{item_path}.step_id: must be an integer")
                action = item.get("action")
                if action not in valid_step_actions:
                    errors.append(f"{item_path}.action: unsupported action {action!r}")
                if not is_non_empty_string(item.get("reason")):
                    errors.append(f"{item_path}.reason: must contain non-whitespace text")
                validate_repair_constraints(
                    item.get("repair_constraints"),
                    path=f"{item_path}.repair_constraints",
                    allow_empty=action == "keep",
                )
                if (
                    "suggested_replacement" in item
                    and not isinstance(item.get("suggested_replacement"), str)
                ):
                    errors.append(f"{item_path}.suggested_replacement: must be a string")

        validate_action_items(self.dataset_actions, path="dataset_actions")
        validate_action_items(self.tool_actions, path="tool_actions")
        if not isinstance(self.must_preserve, list):
            errors.append("must_preserve: must be a list")
        else:
            for index, item in enumerate(self.must_preserve):
                if not is_non_empty_string(item):
                    errors.append(
                        f"must_preserve[{index}]: must contain non-whitespace text"
                    )
        valid_issue_types = {
            "plan_mechanism_mismatch",
            "missing_idea_operationalization",
            "plan_scope_overreach",
        }
        valid_severities = {"high", "medium", "low"}
        if not isinstance(self.semantic_alignment_issues, list):
            errors.append("semantic_alignment_issues: must be a list")
        else:
            for index, item in enumerate(self.semantic_alignment_issues):
                item_path = f"semantic_alignment_issues[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{item_path}: must be an object")
                    continue
                if item.get("issue_type") not in valid_issue_types:
                    errors.append(f"{item_path}.issue_type: unsupported issue type")
                affected_step_ids = item.get("affected_step_ids")
                if not isinstance(affected_step_ids, list):
                    errors.append(f"{item_path}.affected_step_ids: must be a list")
                elif not affected_step_ids:
                    errors.append(f"{item_path}.affected_step_ids: must not be empty")
                else:
                    for step_index, step_id in enumerate(affected_step_ids):
                        if not (
                            (isinstance(step_id, int) and not isinstance(step_id, bool))
                            or is_non_empty_string(step_id)
                        ):
                            errors.append(
                                f"{item_path}.affected_step_ids[{step_index}]: "
                                "must be an integer or non-empty string"
                            )
                if item.get("severity") not in valid_severities:
                    errors.append(f"{item_path}.severity: unsupported severity")
                for field_name in ("reason", "evidence", "plan_fix"):
                    if not is_non_empty_string(item.get(field_name)):
                        errors.append(
                            f"{item_path}.{field_name}: must contain non-whitespace text"
                        )
                validate_repair_constraints(
                    item.get("repair_constraints"),
                    path=f"{item_path}.repair_constraints",
                    allow_empty=False,
                )
        if not is_non_empty_string(self.generated_at):
            errors.append("generated_at: must contain non-whitespace text")
        if self.semantic_alignment_issues and not self.should_replan:
            errors.append("should_replan: must be true when semantic alignment issues exist")
        if self.mode == "final_review":
            if self.should_replan is not False:
                errors.append("should_replan: must be false for final_review")
        elif self.mode == "replan_feedback":
            if self.should_replan is not True:
                errors.append("should_replan: must be true for replan_feedback")
            if not isinstance(self.replan_round, bool) and isinstance(self.replan_round, int):
                if self.replan_round < 1:
                    errors.append("replan_round: must be at least 1 for replan_feedback")
        return errors

    def is_valid(self) -> bool:
        """Validate the complete Refinement output contract deterministically."""
        return not self.validation_errors()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "should_replan": self.should_replan,
            "feedback_summary": self.feedback_summary,
            "step_level_adjustments": self.step_level_adjustments,
            "dataset_actions": self.dataset_actions,
            "tool_actions": self.tool_actions,
            "must_preserve": self.must_preserve,
            "replan_round": self.replan_round,
            "generated_at": self.generated_at,
            "semantic_alignment_issues": self.semantic_alignment_issues,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanRefinementResult":
        if not isinstance(data, dict):
            raise TypeError("PlanRefinementResult payload must be an object")
        return cls(
            mode=data.get("mode"),
            should_replan=data.get("should_replan"),
            feedback_summary=data.get("feedback_summary"),
            step_level_adjustments=data.get("step_level_adjustments"),
            dataset_actions=data.get("dataset_actions"),
            tool_actions=data.get("tool_actions"),
            must_preserve=data.get("must_preserve"),
            replan_round=data.get("replan_round"),
            generated_at=data.get("generated_at", datetime.now().isoformat()),
            semantic_alignment_issues=data.get("semantic_alignment_issues"),
        )




IDEA_QUALITY_SCORE_DECIMALS = 2
IDEA_QUALITY_EVALUATION_VERSION = "idea_quality_v3"
IDEA_QUALITY_DIMENSIONS = (
    "realizability",
    "novelty",
    "data_grounding",
    "logical_coherence",
    "significance",
    "validation_design_quality",
    "reproducibility_readiness",
)
IDEA_QUALITY_CORE_DIMENSIONS = IDEA_QUALITY_DIMENSIONS[:5]
IDEA_QUALITY_PLANNED_DIMENSIONS = IDEA_QUALITY_DIMENSIONS[5:]
IDEA_QUALITY_DIMENSION_STATUSES = {
    "evaluated",
    "insufficient_evidence",
    "not_applicable",
    "error",
}
IDEA_QUALITY_EVALUATION_STATUSES = {"complete", "partial", "error"}


def _round_idea_quality_score(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), IDEA_QUALITY_SCORE_DECIMALS)




def _normalize_idea_evaluation_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    if payload.get("evaluation_version") != IDEA_QUALITY_EVALUATION_VERSION:
        return dict(payload)
    return IdeaQualityEvaluationResult.from_dict(payload).to_dict()


@dataclass
class IdeaQualityDimensionResult:
    status: str
    score: Optional[float]
    assessment_confidence: Optional[float]
    subscores: Dict[str, Optional[float]] = field(default_factory=dict)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    counterevidence_or_gaps: List[str] = field(default_factory=list)
    rationale: str = ""

    def is_valid(self) -> bool:
        if self.status not in IDEA_QUALITY_DIMENSION_STATUSES:
            return False
        if self.status != "evaluated":
            return self.score is None and self.assessment_confidence is None
        if self.score is None or self.assessment_confidence is None or not self.subscores:
            return False
        return 0.0 <= float(self.score) <= 100.0 and 0.0 <= float(self.assessment_confidence) <= 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": _round_idea_quality_score(self.score),
            "assessment_confidence": (
                round(float(self.assessment_confidence), 4)
                if self.assessment_confidence is not None else None
            ),
            "subscores": {
                key: _round_idea_quality_score(value)
                for key, value in self.subscores.items()
            },
            "supporting_evidence": self.supporting_evidence,
            "counterevidence_or_gaps": self.counterevidence_or_gaps,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaQualityDimensionResult":
        return cls(
            status=str(data.get("status", "")),
            score=data.get("score"),
            assessment_confidence=data.get("assessment_confidence"),
            subscores=dict(data.get("subscores", {}) or {}),
            supporting_evidence=list(data.get("supporting_evidence", []) or []),
            counterevidence_or_gaps=[
                str(item) for item in (data.get("counterevidence_or_gaps", []) or [])
            ],
            rationale=str(data.get("rationale", "")),
        )


@dataclass
class IdeaQualityAggregates:
    idea_quality_score: Optional[float]
    planned_validation_score: Optional[float]
    extended_quality_score: Optional[float]
    minimum_core_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_quality_score": _round_idea_quality_score(self.idea_quality_score),
            "planned_validation_score": _round_idea_quality_score(self.planned_validation_score),
            "extended_quality_score": _round_idea_quality_score(self.extended_quality_score),
            "minimum_core_score": _round_idea_quality_score(self.minimum_core_score),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaQualityAggregates":
        return cls(
            idea_quality_score=data.get("idea_quality_score"),
            planned_validation_score=data.get("planned_validation_score"),
            extended_quality_score=data.get("extended_quality_score"),
            minimum_core_score=data.get("minimum_core_score"),
        )


@dataclass
class IdeaQualityEvaluationResult:
    idea_id: str
    evaluation_stage: str
    evaluation_status: str
    dimensions: Dict[str, IdeaQualityDimensionResult]
    aggregates: IdeaQualityAggregates
    summary: str = ""
    dominant_strengths: List[str] = field(default_factory=list)
    dominant_weaknesses: List[str] = field(default_factory=list)
    evaluation_version: str = IDEA_QUALITY_EVALUATION_VERSION
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_valid(self) -> bool:
        if self.evaluation_version != IDEA_QUALITY_EVALUATION_VERSION:
            return False
        if not self.idea_id.strip() or self.evaluation_stage not in {"idea_only", "planned"}:
            return False
        if self.evaluation_status not in IDEA_QUALITY_EVALUATION_STATUSES:
            return False
        if set(self.dimensions) != set(IDEA_QUALITY_DIMENSIONS):
            return False
        if not all(dimension.is_valid() for dimension in self.dimensions.values()):
            return False

        core_evaluated = all(
            self.dimensions[key].status == "evaluated"
            for key in IDEA_QUALITY_CORE_DIMENSIONS
        )
        if core_evaluated != (self.aggregates.idea_quality_score is not None):
            return False
        validation = self.dimensions["validation_design_quality"]
        reproducibility = self.dimensions["reproducibility_readiness"]
        if self.evaluation_stage == "idea_only":
            if (
                validation.status != "not_applicable"
                or reproducibility.status != "not_applicable"
                or self.aggregates.planned_validation_score is not None
                or self.aggregates.extended_quality_score is not None
            ):
                return False
        else:
            planned_evaluated = validation.status == "evaluated" and reproducibility.status == "evaluated"
            if planned_evaluated != (self.aggregates.extended_quality_score is not None):
                return False
            if (validation.status == "evaluated") != (self.aggregates.planned_validation_score is not None):
                return False
        if self.evaluation_status == "complete":
            return core_evaluated and (
                self.evaluation_stage == "idea_only"
                or (validation.status == "evaluated" and reproducibility.status == "evaluated")
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_version": self.evaluation_version,
            "idea_id": self.idea_id,
            "evaluation_stage": self.evaluation_stage,
            "evaluation_status": self.evaluation_status,
            "dimensions": {
                key: value.to_dict() if hasattr(value, "to_dict") else value
                for key, value in self.dimensions.items()
            },
            "aggregates": self.aggregates.to_dict(),
            "dominant_strengths": self.dominant_strengths,
            "dominant_weaknesses": self.dominant_weaknesses,
            "summary": self.summary,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaQualityEvaluationResult":
        raw_dimensions = data.get("dimensions", {}) or {}
        dimensions = {
            key: (
                value if isinstance(value, IdeaQualityDimensionResult)
                else IdeaQualityDimensionResult.from_dict(value)
            )
            for key, value in raw_dimensions.items()
            if isinstance(value, (IdeaQualityDimensionResult, dict))
        }
        raw_aggregates = data.get("aggregates", {}) or {}
        return cls(
            evaluation_version=str(data.get("evaluation_version", "")),
            idea_id=str(data.get("idea_id", "")),
            evaluation_stage=str(data.get("evaluation_stage", "")),
            evaluation_status=str(data.get("evaluation_status", "")),
            dimensions=dimensions,
            aggregates=(
                raw_aggregates if isinstance(raw_aggregates, IdeaQualityAggregates)
                else IdeaQualityAggregates.from_dict(raw_aggregates)
            ),
            dominant_strengths=[str(item) for item in (data.get("dominant_strengths", []) or [])],
            dominant_weaknesses=[str(item) for item in (data.get("dominant_weaknesses", []) or [])],
            summary=str(data.get("summary", "")),
            generated_at=str(data.get("generated_at", datetime.now().isoformat())),
        )
