"""
Orchestration Agent for DataGenHyp.

Implements central workflow orchestration for the current application-planning
pipeline and its downstream evaluation/execution phases.
"""

import copy
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from ..agents.base_agent import BaseAgent
from ..agents.plan_utils import (
    normalize_dataset_info_with_enriched_description,
    resolve_enriched_data_description,
)
from ..agents.step_plan_agent import StepPlanContractError
from ..memory.memory_manager import MemoryManager
from ..models.model_factory import ModelFactory
from ..skills.cross_domain_expander import CrossDomainExpander
from ..tools.candidate_dataset_scout import (
    DEFAULT_TOP_N as DEFAULT_CANDIDATE_INVENTORY_TOP_N,
    CandidateDatasetScout,
)
from ..tools.dataset_detail_enricher import DatasetDetailEnricher
from .data_type import (
    ApplicationRecord,
    Task,
    WorkflowSession,
    WorkflowState,
    ExecutionContract,
    normalize_cross_domain_axes,
)

logger = logging.getLogger(__name__)

class OrchestrationAgent:
    """
    Central coordinator for multi-agent research workflow.

    Orchestrates the current research pipeline centered on survey, planning,
    resource checking, plan refinement, idea evaluation, and downstream
    geo-analysis. Coordinates specialized agents,
    handles scientist feedback integration,and ensures proper sequencing of research phases. 
    Supports concurrent task execution,session persistence, state change callbacks, 
    and iterative refinement cycles.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        memory_manager (MemoryManager): Persistent storage manager
        model_factory (ModelFactory): Model instance creator
        agent_registry (Dict[str, BaseAgent]): Registered specialized agents
        max_iterations (int): Maximum refinement iterations
        active_sessions (Dict): Currently active workflow sessions
        session_callbacks (Dict): State change callback functions
    """

    def __init__(self,
                config: Dict[str, Any],
                memory_manager: MemoryManager,
                model_factory: ModelFactory = None,
                agent_registry: Dict[str, BaseAgent] = None):
        """
        Initialize orchestration agent with configuration and dependencies.

        Loads workflow configuration, initializes the agent registry and model factory,
        and prepares session tracking structures.

        Args:
            config (Dict[str, Any]): Configuration with workflow settings
            memory_manager (MemoryManager): Manager for session persistence
            model_factory (ModelFactory): Factory for model creation (optional)
            agent_registry (Dict[str, BaseAgent]): Specialized agents by type (optional)
        """
        self.config = config
        self.memory_manager = memory_manager
        self.model_factory = model_factory or ModelFactory()
        self.agent_registry = agent_registry or {}

        # Load workflow configuration
        workflow_config = config.get("workflow", {})
        self.max_iterations = workflow_config.get("max_iterations", 5)

        # Phase feature flags（A/B 灰度控制，读自 config.phases）
        phases_cfg = config.get("phases", {})
        self.enable_dataset_profiling_phase: bool = phases_cfg.get("enable_dataset_profiling_phase", True)
        self.enable_plan_refinement_phase: bool = phases_cfg.get("enable_plan_refinement_phase", True)
        self.enable_idea_maturation: bool = phases_cfg.get("enable_idea_maturation", True)

        # 候选数据集侦察（idea 生成前 scout + 详情富化，读自 config.planning）
        planning_cfg = config.get("planning", {}) or {}
        self.enable_candidate_dataset_scouting: bool = bool(
            planning_cfg.get("enable_candidate_dataset_scouting", True)
        )
        self.candidate_inventory_top_n: int = int(
            planning_cfg.get("candidate_inventory_top_n", DEFAULT_CANDIDATE_INVENTORY_TOP_N)
        )
        # 跨领域组合方向推演技能：画像丰富语义后推演数据集可结合的领域方向
        self.enable_cross_domain_expansion: bool = bool(
            planning_cfg.get("enable_cross_domain_expansion", True)
        )
        self._candidate_scout: Optional[CandidateDatasetScout] = None
        self._candidate_enricher: Optional[DatasetDetailEnricher] = None

        # Active session tracking
        self.active_sessions = {}
        self.session_callbacks = {}

        logger.info(
            f"OrchestrationAgent initialized: max_iterations={self.max_iterations}, "
            f"enable_dataset_profiling_phase={self.enable_dataset_profiling_phase}, "
            f"enable_plan_refinement_phase={self.enable_plan_refinement_phase}, "
            f"enable_idea_maturation={self.enable_idea_maturation}"
        )

    def _append_idea_evaluation_history(
        self,
        app: ApplicationRecord,
        *,
        stage: str,
        payload: Dict[str, Any],
    ) -> None:
        history = list(getattr(app, "idea_evaluation_history", []) or [])
        history.append(
            {
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
            }
        )
        app.idea_evaluation_history = history

    async def create_session(self,
                           task_description: str,
                           background: str = "",
                           spatial_extent: Dict[str, Any] = None,
                           data_requirements: List[str] = None,
                           analysis_methods: List[str] = None) -> str:
        """
        Create new workflow session for research task.

        Initializes workflow session with research goal .
        Creates task and session objects, stores in memory, and activates for execution.

        Args:
            task_description (str): Research objective description
            background (str): Additional context (optional)
            spatial_extent (Dict[str, Any]): Geographic extent information (for geospatial tasks)
            data_requirements (List[str]): Initial data requirements (can be empty, will be determined by agents)
            analysis_methods (List[str]): Initial analysis methods (can be empty, will be determined by agents)

        Returns:
            str: Unique session identifier for tracking
        """
        task_id = f"task_{int(time.time())}"
        
        # 构建Task对象，使用地理空间发现所需字段
        task = Task(
            id=task_id,
            description=task_description,
            spatial_extent=spatial_extent or {},
            data_requirements=data_requirements or [],
            analysis_methods=analysis_methods or [],
            background=background,
        )

        session_id = f"session_{int(time.time())}"
        session = WorkflowSession(
            id=session_id,
            task=task,
            max_iterations=self.max_iterations
        )

        await self.memory_manager.store_session(session)
        self.active_sessions[session_id] = session

        logger.info(f"Created new session {session_id} for task: {task_description}")
        return session_id

    async def run_session(self,
                        session_id: str,
                        on_state_change: Optional[Callable] = None) -> WorkflowSession:
        """
        Execute workflow session from current state to completion.

        Runs session through workflow phases using state machine, executing appropriate
        agent tasks at each phase. Continues until completion, error, or awaiting feedback
        state. Stores session state after each phase and triggers callbacks on transitions.

        Args:
            session_id (str): Session identifier to execute
            on_state_change (Optional[Callable]): Callback for state transitions (optional)

        Returns:
            WorkflowSession: Updated session with execution results

        Raises:
            Exception: If phase execution fails or session not found
        """
        session = await self._get_session(session_id)

        if on_state_change:
            self.session_callbacks[session_id] = on_state_change

        if session.state == WorkflowState.INITIAL:
            if self.enable_dataset_profiling_phase:
                await self._update_session_state(session, WorkflowState.DATASET_PROFILING)
            else:
                await self._update_session_state(session, WorkflowState.LITERATURE_SEARCH)

        try:
            while session.state not in [WorkflowState.COMPLETED, WorkflowState.ERROR]:
                await self._execute_current_phase(session)
                await self.memory_manager.store_session(session)

                if session.state == WorkflowState.AWAITING_FEEDBACK:
                    logger.info(f"Session {session_id} is awaiting feedback")
                    break

            if session.state == WorkflowState.COMPLETED:
                session.completed_at = datetime.now()
                await self.memory_manager.store_session(session)
                logger.info(f"System: Session {session_id} completed successfully after {session.iterations_completed} iterations")

            return session

        except Exception as e:
            logger.error(f"Error running session {session_id}: {str(e)}")
            if session.state != WorkflowState.COMPLETED:
                await self._update_session_state(session, WorkflowState.ERROR)
                await self.memory_manager.store_session(session)
            raise

    async def add_feedback(self,
                         session_id: str,
                         feedback: dict,
                         target_finding_ids: List[str] = None) -> str:
        """
        Add scientist feedback to workflow session.

        Incorporates external feedback into session history, optionally targeting specific
        records. If session is awaiting feedback, marks the session terminal. Stores
        updated session state with feedback entry including timestamp and iteration context.

        Args:
            session_id (str): Target session identifier
            feedback (dict): Feedback content and metadata
            target_finding_ids (List[str]): Specific findings to address (optional)

        Returns:
            str: Confirmation message with session ID
        """
        session = await self._get_session(session_id)

        feedback_entry = {
            "text": feedback,
            "timestamp": datetime.now().isoformat(),
            "target_findings": target_finding_ids,
            "iteration": session.iterations_completed
        }
        session.feedback_history.append(feedback_entry)

        if session.state == WorkflowState.AWAITING_FEEDBACK:
            await self._update_session_state(session, WorkflowState.COMPLETED)

        await self.memory_manager.store_session(session)
        return f"Feedback added to session {session_id}"

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve current session status and metadata.

        Returns comprehensive session information including current state,
        iteration progress, timestamps, and ranked application results.

        Args:
            session_id (str): Session identifier to query

        Returns:
            Dict[str, Any]: Status dictionary with keys: id, task, state,
                iterations_completed, max_iterations, finding_count, started_at,
                age_hours, top_findings (compat alias), application_ranking
        """
        session = await self._get_session(session_id)

        status = {
            "id": session.id,
            "task": session.task.description,
            "state": session.state.value,
            "iterations_completed": session.iterations_completed,
            "max_iterations": session.max_iterations,
            "finding_count": len(session.findings),
            "started_at": session.started_at.isoformat(),
            "age_hours": (datetime.now() - session.started_at).total_seconds() / 3600,
            "top_findings": [],
            "application_ranking": list(getattr(session, "application_ranking", []) or []),
        }

        status["top_findings"] = list(status["application_ranking"])

        return status

    async def get_top_findings(self, session_id: str, include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve ranked application results from session.

        Args:
            session_id (str): Session identifier to query
            include_all (bool): Retained for backward compatibility; ignored.

        Returns:
            List[Dict[str, Any]]: Ranked application entries
        """
        session = await self._get_session(session_id)
        return list(getattr(session, "application_ranking", []) or [])

    async def get_findings(self, session_id: str, limit: int = 10, include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve ranked application results from session with optional limit.

        Args:
            session_id (str): Session identifier to query
            limit (int): Maximum number of ranked entries to return (default: 10)
            include_all (bool): Retained for backward compatibility; ignored.

        Returns:
            List[Dict[str, Any]]: Ranked application entries
        """
        session = await self._get_session(session_id)
        ranking = list(getattr(session, "application_ranking", []) or [])
        return ranking[:limit]

    # Private methods
    async def _get_session(self, session_id: str) -> WorkflowSession:
        """
        Retrieve session from active cache or memory storage.

        Checks active sessions first for performance, falls back to memory manager
        if not cached. Loads and caches session if found in storage.

        Args:
            session_id (str): Session identifier to retrieve

        Returns:
            WorkflowSession: Session object with current state

        Raises:
            ValueError: If session not found in active sessions or memory
        """
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        session = await self.memory_manager.load_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        self.active_sessions[session_id] = session
        return session

    def _get_agent_info_for_state(self, state: WorkflowState) -> str:
        """
        Map workflow state to responsible agent name.

        Returns human-readable agent name for logging and display purposes.
        Maps workflow states to their corresponding agent handlers.

        Args:
            state (WorkflowState): Current workflow state

        Returns:
            str: Agent name or "Unknown Agent" if unmapped
        """
        agent_mapping = {
            # 地理空间发现流程
            WorkflowState.DATASET_PROFILING: "Dataset Profiler Agent",
            WorkflowState.LITERATURE_SEARCH: "Survey Agent",
            WorkflowState.TASK_PLANNING: "Plan Agent",
            WorkflowState.PLAN_REFINEMENT: "Plan Refinement Agent", 
            WorkflowState.RESOURCE_CHECK: "Resource Check Agent",
            WorkflowState.IDEA_EVALUATION: "Idea Evaluation Agent",
            WorkflowState.AWAITING_FEEDBACK: "System",
            WorkflowState.INITIAL: "System",
            WorkflowState.COMPLETED: "System",
            WorkflowState.ERROR: "System"
        }
        return agent_mapping.get(state, "Unknown Agent")

    @staticmethod
    def _validate_session_state_transition(
        old_state: WorkflowState,
        new_state: WorkflowState,
    ) -> bool:
        """Validate a session-phase transition and report whether state changes."""
        if not isinstance(old_state, WorkflowState):
            raise TypeError(f"Current session state must be WorkflowState, got {old_state!r}")
        if not isinstance(new_state, WorkflowState):
            raise TypeError(f"New session state must be WorkflowState, got {new_state!r}")
        if old_state == new_state:
            return False
        if not old_state.can_transition_to(new_state):
            raise ValueError(
                f"Invalid workflow state transition: {old_state.value} -> {new_state.value}"
            )
        return True

    async def _update_session_state(self, session: WorkflowSession, new_state: WorkflowState) -> None:
        """
        Transition session to new workflow state.

        Updates session state, logs state transition with agent context, and triggers registered callbacks.

        Args:
            session (WorkflowSession): Session to update
            new_state (WorkflowState): Target workflow state
        """
        old_state = session.state
        should_update = self._validate_session_state_transition(old_state, new_state)
        if not should_update:
            logger.debug(
                f"Session {session.id}: state remains {old_state.value}; no transition callback emitted"
            )
            return

        session.state = new_state

        # Log state transition with agent context
        old_agent_info = self._get_agent_info_for_state(old_state)
        new_agent_info = self._get_agent_info_for_state(new_state)
        logger.info(f"Session {session.id}: {old_agent_info} -> {new_agent_info}")

        if session.id in self.session_callbacks:
            try:
                callback = self.session_callbacks[session.id]
                await callback(session, old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {str(e)}")

    async def _execute_current_phase(self, session: WorkflowSession) -> None:
        """
        Route session to appropriate phase handler based on current state.

        Dispatches execution to specialized phase methods for geospatial discovery workflow.
        Transitions to error state if handler not found.

        Args:
            session (WorkflowSession): Session to execute
        """
        phase_handlers = {
            # 地理空间发现流程阶段
            WorkflowState.DATASET_PROFILING: self._run_dataset_profiling_phase,
            WorkflowState.LITERATURE_SEARCH: self._run_literature_search_phase,
            WorkflowState.TASK_PLANNING: self._run_task_planning_phase,
            WorkflowState.PLAN_REFINEMENT: self._run_plan_refinement_phase,
            WorkflowState.RESOURCE_CHECK: self._run_resource_check_phase,
            WorkflowState.IDEA_EVALUATION: self._run_idea_evaluation_phase,
            # 通用状态
            WorkflowState.AWAITING_FEEDBACK: self._run_awaiting_feedback_phase,
        }

        handler = phase_handlers.get(session.state)
        if handler:
            await handler(session)
        else:
            logger.error(f"Unknown state: {session.state}")
            await self._update_session_state(session, WorkflowState.ERROR)

    async def _run_awaiting_feedback_phase(self, session: WorkflowSession) -> None:
        """
        Handle awaiting feedback state.

        Passive phase where session waits for external scientist feedback via add_feedback().
        Does not execute agent tasks. External systems detect this state and provide input.
        Feedback is stored and the session is then closed by add_feedback().

        Args:
            session (WorkflowSession): Session in waiting state
        """
        logger.info(f"System: Session {session.id} is awaiting external feedback")

        # This phase is passive and only waits for external feedback.
        pass

    # Helper methods
    def _get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """
        Retrieve agent from registry by type.

        Looks up specialized agent instance for given type. Logs warning if not found.

        Args:
            agent_type (str): Agent type identifier

        Returns:
            Optional[BaseAgent]: Agent instance or None if not registered
        """
        if agent_type not in self.agent_registry:
            logger.warning(f"Agent type {agent_type} initialization failed")
            return None
        return self.agent_registry[agent_type]

    # ============================================================================
    # 地理空间发现流程阶段处理函数
    # ============================================================================

    def _generation_dataset_info(
        self,
        session: WorkflowSession,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source = session.task.spatial_extent if dataset_info is None else dataset_info
        portrait = getattr(session, "dataset_portrait", {})
        if not isinstance(portrait, dict) or portrait.get("enriched_data_description") in (None, ""):
            return copy.deepcopy(source)
        description = resolve_enriched_data_description(source, portrait)
        return copy.deepcopy(
            normalize_dataset_info_with_enriched_description(source, description)
        )

    async def _run_dataset_profiling_phase(self, session: WorkflowSession) -> None:
        """
        执行 DATASET_PROFILING 阶段：在文献检索前生成数据集画像。

        输出：
        - session.dataset_portrait
        - 保持 session.task.spatial_extent.base_dataset.data_description 为原始数据描述，不回写画像增量字段
        """
        logger.info(f"Dataset Profiler Agent: Starting dataset profiling for session {session.id}")

        dataset_profiler_agent = self._get_agent("dataset_profiler")
        if not dataset_profiler_agent:
            raise ValueError("Dataset profiler agent initialization failed")

        try:
            spatial_extent = getattr(session.task, "spatial_extent", {}) or {}
            base_dataset = spatial_extent.get("base_dataset", {}) if isinstance(spatial_extent, dict) else {}
            data_description = base_dataset.get("data_description", {}) if isinstance(base_dataset, dict) else {}

            context = {
                "dataset_info": spatial_extent,
                "spatial_extent": spatial_extent,
                "data_description": data_description,
            }
            portrait = await dataset_profiler_agent.execute(context, {})
            if not isinstance(portrait, dict):
                raise ValueError("Dataset profiler must return a dict portrait payload")

            # 画像丰富了数据集语义后，调用跨领域推演技能补充可结合方向
            if self.enable_cross_domain_expansion:
                cross_axes, expansion_audit = await self._run_cross_domain_expansion(
                    portrait, data_description
                )
                # 审计无条件写入：技能失败此前只留一行 warning，产物里零痕迹，
                # 事后无法区分"没产出轴"与"根本没跑过"。空轴必须可见。
                portrait["cross_domain_expansion_audit"] = expansion_audit
                # 写入 session 前按契约归一：技能已做过一次，但这里是画像负载进入
                # 会话状态的权威边界，恢复运行或其它写入路径同样要经过该门。
                cross_axes = normalize_cross_domain_axes(cross_axes)
                if cross_axes:
                    portrait["cross_domain_axes"] = cross_axes
                    # 去重追加：画像阶段重跑（resume / 重试）会重复调用本段，
                    # 直接 extend 会累积同名条目并挤占下游 hooks[:N] 的截断配额。
                    hooks = portrait.setdefault("suggested_research_hooks", [])
                    existing_hooks = {
                        str(hook).strip() for hook in hooks if str(hook).strip()
                    }
                    for axis in cross_axes:
                        hook = (
                            f"[cross-domain] {axis.get('domain_name', '')} "
                            f"({axis.get('role', '')})"
                        )
                        if hook not in existing_hooks:
                            existing_hooks.add(hook)
                            hooks.append(hook)

            session.dataset_portrait = portrait

            logger.info(
                f"Dataset Profiler Agent: session {session.id} completed, "
                f"hooks={len(portrait.get('suggested_research_hooks', []))}"
            )
            await self._update_session_state(session, WorkflowState.LITERATURE_SEARCH)

        except Exception as e:
            logger.error(f"Error in dataset profiling phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)

    async def _run_cross_domain_expansion(
        self,
        dataset_portrait: Dict[str, Any],
        data_description: Dict[str, Any],
    ) -> tuple:
        """调用跨领域推演技能；失败只记日志与审计，不阻断画像阶段。

        返回 (轴列表, 审计)。审计始终非空，因此"技能没跑""跑了报错""跑了但零
        产出"三种情况在产物里可区分——此前三者都表现为画像中没有该键。
        """
        try:
            agents_cfg = self.config.get("agents") or {}
            agent_cfg = dict(agents_cfg.get("cross_domain_expander") or {})
            agent_cfg["_global_config"] = self.config
            # 本技能是画像阶段的子步骤：未单独配置时跟随 dataset_profiler 的
            # provider，而不是回落到全局 default_provider——后者会让画像阶段中途
            # 静默切换到另一个后端，该后端不可用时只表现为零轴产出。
            profiler_provider = (agents_cfg.get("dataset_profiler") or {}).get(
                "model_provider", "default"
            )
            agent_cfg.setdefault("model_provider", profiler_provider)
            model = self.model_factory.create_model_for_agent(
                "cross_domain_expander", agent_cfg
            )
            expander = CrossDomainExpander(model)
            axes = await expander.expand(
                dataset_portrait=dataset_portrait,
                data_description=data_description,
            )
            return axes, copy.deepcopy(expander.last_run_audit)
        except Exception as exc:
            logger.warning("Cross-domain expansion skill failed: %s", exc)
            return [], {
                "attempted": True,
                "status": "skill_construction_failed",
                "axis_count": 0,
                "error": str(exc),
            }

    async def _run_literature_search_phase(self, session: WorkflowSession) -> None:
        """
        执行文献检索阶段。

        调用survey agent检索相关地理研究问题的文献，学习分析方法。检索的文献关键词主要来自于用户提出的任务描述与数据集描述，
        由于用户是否提供描述是可选的，当用户提出明确的描述时，优先goal进行检索；当用户没有提供明确描述时，
        根据数据集描述进行检索；当两者都没有提供时。

        Args:
            session (WorkflowSession): Session to process
        """
        logger.info(f"Survey Agent: Starting literature search for session {session.id}")

        survey_agent = self._get_agent("survey")
        if not survey_agent:
            logger.warning(
                "Survey Agent not available for session %s, skipping literature search and domain knowledge analysis",
                session.id,
            )
            session.survey_results = {"papers": [], "aggregated_views": {}}
            session.literature_review = []
            session.literature_review_raw = []
            session.literature_filter_params = {}
            session.domain_knowledge = {}
            await self._update_session_state(session, WorkflowState.TASK_PLANNING)
            return

        try:
            # 构建上下文，确保传递正确的任务描述
            spatial_extent = session.task.spatial_extent or {}
            dataset_portrait = getattr(session, "dataset_portrait", {}) or {}
            if isinstance(spatial_extent, dict):
                base_dataset = spatial_extent.get("base_dataset", {})
                if not isinstance(base_dataset, dict):
                    base_dataset = {}
                domain_name = base_dataset.get("domain", "")
                data_description = base_dataset.get("data_description", "")
            else:
                domain_name = ""
                data_description = ""
            enriched_data_description = (
                dataset_portrait.get("enriched_data_description")
                if isinstance(dataset_portrait, dict)
                else None
            )
            cli_task_name = self.config.get("task_name", "geo_casualdiscovery")
            context = {
                #表明当前任务用户是否提供任务描述，直接把 Task.description 作为描述传给 SurveyAgent
                "description": session.task.description,
                # 若 profiling 给出了增强描述，则优先下发给 SurveyAgent，但不回写原始 spatial_extent
                "data_description": enriched_data_description or data_description,
                "dataset_portrait": dataset_portrait,
                # region_name 是空间范围，不能作为文献检索的研究领域。
                "domain": domain_name,
                "task_name": cli_task_name,
                "session_id": session.id
            }
            
            literature_result = await survey_agent.execute(context, {})

            # 保留 SurveyAgent 的完整输出，区分原始全集与筛选集
            if isinstance(literature_result, dict):
                session.survey_results = literature_result
            else:
                session.survey_results = {
                    "papers": literature_result if isinstance(literature_result, list) else [],
                    "aggregated_views": {}
                }

            # 存储文献检索结果到 session
            if isinstance(literature_result, dict):
                session.literature_review = literature_result.get("papers", [])
                session.literature_review_raw = literature_result.get("papers_all", literature_result.get("papers", []))
                session.literature_filter_params = literature_result.get("filter_params", {})
            elif isinstance(literature_result, list):
                logger.warning("SurveyAgent returned list instead of dict, using directly as filtered/raw")
                session.literature_review = literature_result
                session.literature_review_raw = literature_result
                session.literature_filter_params = {}
            else:
                logger.error(f"Unexpected literature_result type: {type(literature_result)}")
                session.literature_review = []
                session.literature_review_raw = []
                session.literature_filter_params = {}

            # 领域知识分析：总结“应用主题/方法簇/数据集簇”
            domain_knowledge_agent = self._get_agent("domain_knowledge")
            if domain_knowledge_agent:
                logger.info(f"Domain Knowledge Agent: Starting domain knowledge analysis for session {session.id}")
                try:
                    dk_context = {
                        "survey_results": session.survey_results,
                        "domain": context.get("domain", ""),
                        "description": context.get("description", ""),
                        "data_description": context.get("data_description", "")
                    }
                    session.domain_knowledge = await domain_knowledge_agent.execute(dk_context, {})
                    logger.info(f"Domain Knowledge Agent: Completed domain knowledge analysis for session {session.id}")
                except Exception as e:
                    logger.warning(f"DomainKnowledgeAgent failed for session {session.id}: {e}")
            else:
                logger.warning(f"Domain Knowledge Agent not available for session {session.id}, skipping domain knowledge analysis")
            
            logger.info(f"Survey Agent: Retrieved {len(session.literature_review)} papers for session {session.id}")
            await self._update_session_state(session, WorkflowState.TASK_PLANNING)

        except Exception as e:
            logger.error(f"Error in literature search phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)

    # =====================================================================
    # Idea Evaluation & Selection
    # =====================================================================


    async def _filter_and_select_ideas(
        self,
        idea_result: Dict[str, Any],
        context: Dict[str, Any],
        audit_log: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """按 idea_type 选择 revised idea，每类最多保留 1 个。"""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for plan in idea_result.get("application_plans", []) or []:
            if not isinstance(plan, dict):
                continue
            idea = self._extract_idea_from_plan(plan)
            idea_type = str(idea.get("idea_type") or plan.get("idea_type") or "").strip()
            if not idea or not idea_type:
                continue
            grouped.setdefault(idea_type, []).append({"plan": plan, "idea": idea})

        selected: List[Dict[str, Any]] = []
        for idea_type, candidates in grouped.items():
            surviving = self._rule_filter_ideas(candidates)
            surviving_ids = {
                str(item["idea"].get("idea_id") or "") for item in surviving
            }
            # 落选候选的正文必须留档：只记 idea_id 的话，"被淘汰的那两个是不是更好"
            # 事后无法从产物复盘，任何对选择判据的调整也就无从验证效果。
            # 只留判据真正依赖的两个字段，15 个候选合计约 10KB，不影响产物体积。
            candidate_digest = [
                {
                    "idea_id": str(item["idea"].get("idea_id") or ""),
                    "rule_filter": (
                        "surviving"
                        if str(item["idea"].get("idea_id") or "") in surviving_ids
                        else "filtered_out"
                    ),
                    "selected": False,
                    "research_question": item["idea"].get("research_question", ""),
                    "theoretical_mechanism": copy.deepcopy(
                        item["idea"].get("theoretical_mechanism", {})
                    ),
                }
                for item in candidates
            ]
            audit_entry = {
                "idea_type": idea_type,
                "generated_idea_count": len(candidates),
                "candidate_idea_ids": [
                    str(item["idea"].get("idea_id") or "")
                    for item in candidates
                ],
                "surviving_idea_count": len(surviving),
                "selected_idea_id": "",
                "status": "dropped",
                "reason": "",
                "candidate_digest": candidate_digest,
            }
            if not surviving:
                audit_entry["reason"] = "type_all_failed_regeneration_required"
                if audit_log is not None:
                    audit_log.append(audit_entry)
                continue

            if len(surviving) == 1:
                best = surviving[0]
                decision = {"source": "single_candidate", "rationale": ""}
            else:
                best, decision = await self._llm_select_best_idea_within_type(
                    idea_type=idea_type,
                    candidates=surviving,
                    context=context,
                )

            best_id = str(best["idea"].get("idea_id") or "")
            audit_entry["status"] = "kept"
            audit_entry["reason"] = "selected"
            audit_entry["selected_idea_id"] = best_id
            for entry in candidate_digest:
                entry["selected"] = entry["idea_id"] == best_id
            # 记录判据来源与理由：selected_idea_id 单独看无法区分"模型真的选了它"
            # 与"调用失败静默回退到第 0 个"，两者在审计里长得一模一样。
            audit_entry["selection_source"] = decision.get("source", "")
            audit_entry["selection_rationale"] = decision.get("rationale", "")
            if audit_log is not None:
                audit_log.append(audit_entry)
            selected.append({
                "plan": best["plan"],
                "idea": best["idea"],
            })

        return selected

    @staticmethod
    def _extract_idea_from_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
        idea = plan.get("idea_candidate")
        if isinstance(idea, dict):
            return copy.deepcopy(idea)
        candidates = plan.get("idea_candidates")
        if isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, dict):
                    return copy.deepcopy(item)
        if plan.get("idea_type") and plan.get("core_hypothesis"):
            return copy.deepcopy(plan)
        return {}

    @staticmethod
    def _revision_audit_from_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
        audit = plan.get("idea_revision_audit") if isinstance(plan, dict) else None
        return copy.deepcopy(audit) if isinstance(audit, dict) else {}

    def _rule_filter_ideas(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        surviving: List[Dict[str, Any]] = []
        for item in candidates:
            idea = item.get("idea", {}) if isinstance(item, dict) else {}
            plan = item.get("plan", {}) if isinstance(item, dict) else {}
            if not isinstance(idea, dict) or not idea:
                continue
            final_status = str(plan.get("_iteration_final_status") or "").strip()
            if final_status == "rejected":
                continue
            if not str(idea.get("research_question") or "").strip():
                continue
            if not isinstance(idea.get("core_hypothesis"), dict):
                continue
            revision_audit = self._revision_audit_from_plan(plan)
            unresolved = revision_audit.get("unresolved_findings", [])
            unresolved_items = unresolved if isinstance(unresolved, list) else []
            if any(
                isinstance(finding, dict) and bool(finding.get("is_core_contradiction"))
                for finding in unresolved_items
            ):
                continue
            checks = revision_audit.get("semantic_writeback_checks", {})
            if isinstance(checks, dict) and checks.get("overall_passed") is False:
                continue
            surviving.append(item)
        return surviving

    async def _llm_select_best_idea_within_type(
        self,
        *,
        idea_type: str,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """在同一 idea_type 内选一个候选，返回 (选中项, 判据元信息)。

        两点设计要求，都是从真实失败反推的：

        其一，评判维度必须包含非显然性。原规则只要求"研究问题更清晰、假设可检验、
        可计算、未决问题更少"——全是安全性维度，于是三个候选里最四平八稳的那个
        必然胜出，最可能非显然的那个必然落选。Geomorpho90m_Slope 一轮里五个方向
        选出的 idea 全部落在地貌学正典上（坡度×水≈TWI、坡度×降水≈USLE 的 R×LS），
        评估阶段 non_obviousness 只有 40 分，正是这条筛选规则的下游表现。

        其二，机制字段必须进入 payload。原 payload 不含 `theoretical_mechanism`
        与 `data_synergy_and_gap`，而这两者恰恰是判断机制是否显然、跨源耦合是否
        实质的唯一依据；不传就等于要求模型凭研究问题的措辞猜机制。

        另注：本函数在 maturation 之前也会被调用一次，此时 `idea_revision_audit`
        尚未生成，`unresolved_findings` / `handoff_notes_for_step_plan` 必然为空。
        空字段照原样塞进 payload 会让"未决问题更少"成为恒真噪声，因此改为仅在确有
        信号时才带上这两项，并同步启用对应的评判规则。
        """

        def _audit_field(item: Dict[str, Any], key: str) -> List[Any]:
            value = self._revision_audit_from_plan(item["plan"]).get(key, [])
            return value if isinstance(value, list) else []

        has_audit_signal = any(
            _audit_field(item, "unresolved_findings")
            or _audit_field(item, "handoff_notes_for_step_plan")
            for item in candidates
        )

        payload = []
        for idx, item in enumerate(candidates):
            idea = item["idea"]
            entry = {
                "index": idx,
                "idea_id": idea.get("idea_id", ""),
                "research_question": idea.get("research_question", ""),
                "core_hypothesis": idea.get("core_hypothesis", {}),
                "theoretical_mechanism": idea.get("theoretical_mechanism", {}),
                "data_synergy_and_gap": idea.get("data_synergy_and_gap", {}),
                "novelty_constraints_and_value": idea.get("novelty_constraints_and_value", {}),
            }
            if has_audit_signal:
                entry["unresolved_findings"] = _audit_field(item, "unresolved_findings")
                entry["handoff_notes_for_step_plan"] = _audit_field(
                    item, "handoff_notes_for_step_plan"
                )
            payload.append(entry)

        audit_rule = (
            "\n5. Among candidates that remain tied after rule 4, prefer the one with "
            "fewer unresolved findings."
            if has_audit_signal
            else ""
        )
        prompt = f"""Select the best idea within one fixed idea_type.

idea_type: {idea_type}

Candidates:
{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}

Selection rules:
1. Pick at most one idea for this idea_type. Do not invent a new idea, and never
   select an idea whose hypothesis contradicts its own mechanism.
2. Usability bar: the research question must be clear, the hypothesis directional
   and falsifiable, and the observable computable from the sources named in
   `data_synergy_and_gap`. Candidates failing this bar are out of the running.
3. Among candidates that clear rule 2, decide on `theoretical_mechanism`: prefer
   the mechanism that is least obvious given what the field already knows, and the
   `data_synergy_and_gap` describing a real complementarity between sources rather
   than a convenient pairing. Do not pick the candidate that is merely the safest
   to state — the safest phrasing is usually the one that adds least.
4. A mechanism that reduces to a standard named index or a textbook pairing in its
   field is weakly novel, even when it is well written and easy to test. For terrain
   data, slope x water restates the topographic wetness index and slope x rainfall
   restates the LS/R factors of USLE/RUSLE; equivalent canonical pairings exist in
   other domains. Ask whether the idea would still teach us something if that index
   were already computed. If yes, favour it; if no, treat it as the weaker option.{audit_rule}

Return JSON with `best_index` and `rationale`. The rationale must name the concrete
mechanism difference that decided the choice, not restate the winner's abstract.
"""
        schema = {
            "type": "object",
            "properties": {
                "best_index": {"type": "integer"},
                "rationale": {"type": "string"},
            },
            "required": ["best_index", "rationale"],
        }
        rationale = ""
        try:
            idea_generation_agent = self._get_agent("idea_generation")
            if not idea_generation_agent:
                return candidates[0], {
                    "source": "fallback_no_agent",
                    "rationale": "",
                }
            result = await idea_generation_agent._call_model(
                prompt=prompt,
                schema=schema,
                system_prompt="You select one research idea within a fixed type.",
                temperature=0.0,
            )
            best_index = int(result.get("best_index", 0))
            rationale = str(result.get("rationale") or "")
        except Exception as exc:
            logger.warning("idea selection LLM failed for %s: %s", idea_type, exc)
            return candidates[0], {
                "source": "fallback_llm_error",
                "rationale": f"selection failed: {exc}",
            }
        if best_index < 0 or best_index >= len(candidates):
            logger.warning(
                "idea selection returned out-of-range index %s for %s (candidates=%d)",
                best_index,
                idea_type,
                len(candidates),
            )
            return candidates[0], {
                "source": "fallback_index_out_of_range",
                "rationale": rationale,
            }
        return candidates[best_index], {"source": "llm", "rationale": rationale}

    @staticmethod
    def _dataset_scout_keywords(dataset_info: Dict[str, Any]) -> str:
        """侦察用的任务关键词：只取结构化字段，不用 task.description。

        `session.task.description` 是系统提示词全文，拼进 catalog query 会稀释信号。
        这里用 dataset_name 与显式 domain —— 短且都是 catalog 会索引的词。
        """
        base_dataset = dataset_info.get("base_dataset") or {} if isinstance(dataset_info, dict) else {}
        data_description = base_dataset.get("data_description") or {}
        parts = [
            str(
                (data_description.get("dataset_name") if isinstance(data_description, dict) else "")
                or base_dataset.get("dataset_name")
                or ""
            ).strip(),
            str(base_dataset.get("domain") or dataset_info.get("domain") or "").strip(),
        ]
        return " ".join(part for part in parts if part)

    async def _scout_candidate_datasets(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the generation-time strong inventory and separate weak near misses."""
        empty = {
            "candidate_datasets": [],
            "near_miss_candidates": [],
            "query_audit": [],
            "axis_coverage": {},
        }
        if not self.enable_candidate_dataset_scouting:
            return empty
        try:
            if self._candidate_scout is None:
                self._candidate_scout = CandidateDatasetScout()
            if self._candidate_enricher is None:
                self._candidate_enricher = DatasetDetailEnricher()

            dataset_info = context.get("dataset_info") or {}
            base_dataset = copy.deepcopy(
                dataset_info.get("base_dataset", {}) if isinstance(dataset_info, dict) else {}
            )
            dataset_portrait = context.get("dataset_portrait") or {}
            enriched_description = (
                dataset_portrait.get("enriched_data_description")
                if isinstance(dataset_portrait, dict)
                else None
            )
            if isinstance(enriched_description, dict):
                base_dataset["data_description"] = copy.deepcopy(enriched_description)
            scout_result = await self._candidate_scout.scout(
                base_dataset,
                self._dataset_scout_keywords(dataset_info),
                top_n=self.candidate_inventory_top_n,
                cross_domain_axes=(
                    dataset_portrait.get("cross_domain_axes")
                    if isinstance(dataset_portrait, dict)
                    else None
                ),
            )
            enriched = self._candidate_enricher.enrich_batch(
                list(scout_result.get("candidate_datasets", []))
            )
            detailed = sum(1 for item in enriched if isinstance(item, dict) and "detail" in item)
            logger.info(
                "Candidate dataset scouting: %d strong candidates (%d with detail), %d near misses",
                len(enriched),
                detailed,
                len(scout_result.get("near_miss_candidates", [])),
            )
            return {
                "candidate_datasets": enriched,
                "near_miss_candidates": copy.deepcopy(
                    scout_result.get("near_miss_candidates", [])
                ),
                "query_audit": copy.deepcopy(scout_result.get("query_audit", [])),
                "axis_coverage": copy.deepcopy(scout_result.get("axis_coverage", {})),
            }
        except Exception as exc:
            logger.warning(
                "Candidate dataset scouting failed; continuing without candidates: %s", exc
            )
            return empty

    @staticmethod
    def _candidate_dataset_audit_entry(
        candidate: Dict[str, Any],
        *,
        stage: str = "",
    ) -> Dict[str, Any]:
        """Keep scout provenance readable in planning_audit without duplicating full detail.

        `stage` 在持久化边界确定性标注该候选是生成前 scout 得到的还是成熟化补充的。
        它不写回候选对象本身：成熟化追加的 grounded dataset 只带检索结果字段，
        没有六轴 scout 的 `retrieval_axes` / `product_family` / `observable`，
        而本函数对缺失字段是静默跳过的，不标注就只能靠与
        `initial_candidate_datasets` 做集合差分才能还原来源。
        """
        if not isinstance(candidate, dict):
            return {}
        audit = {
            field: copy.deepcopy(candidate[field])
            for field in (
                "query",
                "retrieval_axis",
                "retrieval_axes",
                "retrieval_queries",
                "source",
                "retrieval_channel",
                "dataset_source",
                "source_preference",
                "naming_note",
                "status",
                "id",
                "title",
                "type",
                "provider",
                "docs",
                "product_family",
                "observable",
            )
            if candidate.get(field) not in (None, "", [], {})
        }
        # 判定依据必须随候选留档。此前近失候选只留 status="resolved_weak"，
        # 而 weak 有两种成因——"约束未验证"（证据缺失）与"分数不足"（匹配确实弱），
        # 二者该不该放行完全不同。不留依据就无法回答"放宽门槛能挽回多少材料"，
        # 只能蒙着调阈值。这里只取判定所需的少数标量与短列表，
        # 不整块复制 binding_decision（constraint_report / score_breakdown 体积大且此处用不到）。
        decision = candidate.get("binding_decision")
        if isinstance(decision, dict) and decision:
            rationale = {
                field: copy.deepcopy(decision[field])
                for field in (
                    "binding_grade",
                    "accepted_for_binding",
                    "final_score",
                    "text_score",
                    "frozen_constraints_satisfied",
                    "verification_gaps",
                    "hard_rejections",
                    "soft_reasons",
                )
                if decision.get(field) not in (None, "", [], {})
            }
            if rationale:
                audit["binding_rationale"] = rationale
        if stage:
            audit["inventory_stage"] = stage
        detail = candidate.get("detail")
        if isinstance(detail, dict):
            audit_detail = {
                field: copy.deepcopy(detail[field])
                for field in (
                    "detail_source",
                    "bands",
                    "gsd",
                    "cadence",
                    "temporal_extent",
                    "spatial_extent",
                )
                if detail.get(field) not in (None, "", [], {})
            }
            if audit_detail:
                audit["detail"] = audit_detail
        return audit

    @staticmethod
    def _append_plan_revision_snapshot(
        application: Any,
        *,
        event: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist an application-level plan snapshot at a workflow boundary."""
        if isinstance(application, dict):
            history = application.setdefault("plan_revision_history", [])
            data = application
        else:
            history = application.plan_revision_history
            data = application.to_dict()
        if not isinstance(history, list):
            raise ValueError("plan_revision_history must be a list")
        refinement_result = data.get("plan_refinement_result", {}) or {}
        issue_ids = [
            str(item.get("issue_id") or "")
            for field in ("step_level_adjustments", "semantic_alignment_issues")
            for item in (refinement_result.get(field, []) or [])
            if isinstance(item, dict) and str(item.get("issue_id") or "")
        ]
        sequence = len(history) + 1
        history.append(
            {
                "snapshot_id": (
                    f"{data.get('idea_id', 'application')}:"
                    f"r{data.get('replan_round', 0)}:{sequence}:{event}"
                ),
                "event": event,
                "sequence": sequence,
                "replan_round": data.get("replan_round", 0),
                "recorded_at": datetime.now().isoformat(),
                "task_plan": copy.deepcopy(data.get("task_plan", [])),
                "required_datasets": copy.deepcopy(data.get("required_datasets", [])),
                "capability_anchor_meta": copy.deepcopy(
                    data.get("capability_anchor_meta", {})
                ),
                "step_plan_generation_audit": copy.deepcopy(
                    data.get("step_plan_generation_audit", [])
                ),
                "resource_check_feedback": copy.deepcopy(
                    data.get("resource_check_feedback", {})
                ),
                "plan_refinement_result": copy.deepcopy(refinement_result),
                "issue_ids": list(dict.fromkeys(issue_ids)),
                "metadata": copy.deepcopy(metadata or {}),
            }
        )

    def _build_application_plan_from_selected_idea(
        self,
        selection: Dict[str, Any],
        step_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        plan = selection["plan"]
        idea = selection["idea"]
        step_payload = step_result or {}
        if step_result is not None and not isinstance(
            step_result.get("required_datasets"), list
        ):
            raise ValueError(
                "StepPlan result must contain a top-level required_datasets list"
            )
        if step_result is not None and not isinstance(
            step_result.get("capability_anchor_meta"), dict
        ):
            raise ValueError(
                "StepPlan result must contain capability_anchor_meta"
            )
        revision_audit = self._revision_audit_from_plan(plan)
        application_plan = {
            "idea_id": idea.get("idea_id", plan.get("idea_id", "")),
            "idea_type": idea.get("idea_type", plan.get("idea_type", "")),
            "application_name": idea.get("research_question", "") or plan.get("application_name", ""),
            "application_description": self._summarize_idea_for_application_description(idea, plan),
            "application_context": f"idea_type={idea.get('idea_type', '')}",
            "approach_description": self._summarize_idea_for_approach(idea),
            "priority": plan.get("priority", 0),
            "datasets": [],
            "required_datasets": copy.deepcopy(
                step_payload["required_datasets"] if step_result is not None else []
            ),
            "task_plan": step_payload.get("task_plan", []),
            "rationale": self._summarize_idea_rationale(idea),
            "idea_candidate": copy.deepcopy(idea),
            "idea_dataset_grounding": copy.deepcopy(
                plan.get("idea_dataset_grounding", [])
            ),
            "idea_revision_audit": revision_audit,
            "capability_anchor_meta": copy.deepcopy(
                step_payload["capability_anchor_meta"]
                if step_result is not None else {}
            ),
            "step_plan_generation_audit": (
                [copy.deepcopy(step_payload["step_plan_generation_audit"])]
                if step_result is not None
                and isinstance(step_payload.get("step_plan_generation_audit"), dict)
                else []
            ),
        }
        if step_result is not None:
            self._append_plan_revision_snapshot(
                application_plan,
                event="initial_step_plan",
            )
        return application_plan

    def _build_failed_application_plan_from_step_error(
        self,
        selection: Dict[str, Any],
        error: StepPlanContractError,
    ) -> Dict[str, Any]:
        """Persist one invalid initial plan without failing the other applications."""
        terminal_rule = (
            "initial_step_plan_capability_coverage_gap"
            if getattr(error, "coverage_gap", False)
            else "initial_step_plan_contract_invalid"
        )
        issue_records = [
            {
                "code": terminal_rule,
                "message": str(issue),
            }
            for issue in error.issues
        ]
        capability_anchor_meta = copy.deepcopy(error.capability_anchor_meta)
        original_plan_dag_issues = capability_anchor_meta.get("plan_dag_issues")
        persisted_issue_records = (
            copy.deepcopy(original_plan_dag_issues)
            if isinstance(original_plan_dag_issues, list) and original_plan_dag_issues
            else copy.deepcopy(issue_records)
        )
        capability_anchor_meta["plan_dag_valid"] = False
        capability_anchor_meta["plan_dag_issues"] = copy.deepcopy(
            persisted_issue_records
        )
        step_result = {
            "required_datasets": copy.deepcopy(error.required_datasets),
            "task_plan": copy.deepcopy(error.task_plan),
            "capability_anchor_meta": capability_anchor_meta,
            "step_plan_generation_audit": copy.deepcopy(error.generation_audit),
        }
        app_plan = self._build_application_plan_from_selected_idea(selection, step_result)
        reason = str(error)
        execution_contract = ExecutionContract(
            resource_check_passed=False,
            plan_dag_valid=False,
            plan_dag_issues=copy.deepcopy(persisted_issue_records),
            should_terminate=True,
            termination_reason=reason,
            terminal_rule=terminal_rule,
        )
        app_plan.update(
            {
                "status": "failed_before_evaluation",
                "final_disposition": "failed_before_evaluation",
                "resource_check": {
                    "failure_stage": "initial_step_plan",
                    "execution_contract": execution_contract.to_dict(),
                },
                "resource_check_feedback": {
                    "terminal_rule": terminal_rule,
                    "reason": reason,
                    "recoverable": False,
                    "affected_steps": [],
                },
            }
        )
        return app_plan

    async def _execute_step_plan_with_verified_datasets(
        self,
        *,
        step_plan_agent: Any,
        resources_agent: Any,
        step_context: Dict[str, Any],
        step_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run requirement draft, strict resolution, then frozen final planning."""
        if resources_agent is None:
            raise ValueError(
                "Resource check agent initialization failed for plan-time dataset resolution"
            )

        def _resolution_audit(resolution_payload: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "status": resolution_payload.get("status"),
                "unresolvable_datasets": copy.deepcopy(
                    resolution_payload.get("unresolvable_datasets") or []
                ),
                "dataset_resolution_report": copy.deepcopy(
                    resolution_payload.get("dataset_resolution_report") or {}
                ),
                "resolution_log": copy.deepcopy(
                    resolution_payload.get("resolution_log") or []
                ),
                "omitted_optional_datasets": copy.deepcopy(
                    resolution_payload.get("omitted_optional_datasets") or []
                ),
            }

        def _resolution_issues(resolution_payload: Dict[str, Any]) -> List[str]:
            unresolved = list(resolution_payload.get("unresolvable_datasets") or [])
            reports = resolution_payload.get("dataset_resolution_report") or {}
            return [
                "dataset requirement "
                f"{requirement_id} did not reach resolved_strong: "
                f"{(reports.get(requirement_id) or {}).get('bind_status_reason') or (reports.get(requirement_id) or {}).get('status') or 'unresolved'}"
                for requirement_id in unresolved
            ] or ["plan-time dataset resolution did not reach resolved_strong"]

        draft_result = await step_plan_agent.draft_required_datasets(
            step_context,
            step_params,
        )
        draft_requirements = draft_result.get("required_datasets")
        if not isinstance(draft_requirements, list):
            raise StepPlanContractError(
                ["requirement draft did not return required_datasets as a list"],
                generation_audit=copy.deepcopy(
                    draft_result.get("step_plan_generation_audit") or {}
                ),
            )

        resolution = await resources_agent.resolve_plan_dataset_requirements(
            required_datasets=draft_requirements,
            dataset_info=step_context.get("dataset_info") or {},
        )
        verified_requirements = resolution.get("required_datasets")
        if not isinstance(verified_requirements, list):
            raise StepPlanContractError(
                ["plan-time dataset resolution returned an invalid requirement list"],
                required_datasets=draft_requirements,
                generation_audit=copy.deepcopy(
                    draft_result.get("step_plan_generation_audit") or {}
                ),
            )

        requirement_audit = copy.deepcopy(
            draft_result.get("step_plan_generation_audit") or {}
        )
        requirement_audit["dataset_requirement_resolution"] = _resolution_audit(
            resolution
        )
        unresolved_ids = list(resolution.get("unresolvable_datasets") or [])
        if resolution.get("status") != "resolved_strong" or unresolved_ids:
            repair_result = await step_plan_agent.repair_required_datasets(
                step_context,
                step_params,
                current_required_datasets=verified_requirements,
                dataset_resolution_report=resolution.get("dataset_resolution_report")
                or {},
                unresolvable_datasets=unresolved_ids,
                resolution_log=resolution.get("resolution_log") or [],
                base_audit=requirement_audit,
            )
            repaired_requirements = repair_result.get("required_datasets")
            if not isinstance(repaired_requirements, list):
                raise StepPlanContractError(
                    [
                        "requirement repair did not return required_datasets as a list"
                    ],
                    required_datasets=verified_requirements,
                    generation_audit=copy.deepcopy(
                        repair_result.get("step_plan_generation_audit") or {}
                    ),
                )
            resolution = await resources_agent.resolve_plan_dataset_requirements(
                required_datasets=repaired_requirements,
                dataset_info=step_context.get("dataset_info") or {},
            )
            verified_requirements = resolution.get("required_datasets")
            if not isinstance(verified_requirements, list):
                raise StepPlanContractError(
                    [
                        "plan-time dataset resolution returned an invalid requirement list after repair"
                    ],
                    required_datasets=repaired_requirements,
                    generation_audit=copy.deepcopy(
                        repair_result.get("step_plan_generation_audit") or {}
                    ),
                )
            requirement_audit = copy.deepcopy(
                repair_result.get("step_plan_generation_audit") or {}
            )
            requirement_audit["dataset_requirement_resolution_after_repair"] = (
                _resolution_audit(resolution)
            )
            unresolved_ids = list(resolution.get("unresolvable_datasets") or [])
            if resolution.get("status") != "resolved_strong" or unresolved_ids:
                raise StepPlanContractError(
                    _resolution_issues(resolution),
                    required_datasets=verified_requirements,
                    generation_audit=requirement_audit,
                )

        final_context = copy.deepcopy(step_context)
        final_context["verified_required_datasets"] = copy.deepcopy(
            verified_requirements
        )
        final_context["step_plan_requirement_audit"] = requirement_audit
        return await step_plan_agent.execute(final_context, step_params)

    def _build_step_plan_idea_payload(
        self,
        *,
        idea: Dict[str, Any],
        revision_audit: Dict[str, Any],
        dataset_info: Dict[str, Any],
        candidate_datasets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "payload_type": "idea",
            "idea": copy.deepcopy(idea),
            "planning_handoff": self._build_step_plan_handoff(revision_audit),
            "dataset_characteristics": copy.deepcopy(dataset_info),
            "candidate_datasets": copy.deepcopy(candidate_datasets),
        }

    @staticmethod
    def _trim_step_plan_literature_evidence(result: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = (
            "title",
            "abstract",
            "doi",
            "year",
            "journal",
            "methods",
            "applications",
            "datasets",
            "datasets_used",
            "source",
            "retrieval_source",
        )
        return {
            field: copy.deepcopy(result[field])
            for field in allowed_fields
            if result.get(field) not in (None, "", [], {})
        }

    @staticmethod
    def _trim_step_plan_dataset_evidence(result: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = (
            "id",
            "title",
            "name",
            "description",
            "data_description",
            "type",
            "provider",
            "source",
            "retrieval_channel",
            "dataset_source",
            "source_preference",
            "naming_note",
            "status",
            "modality",
            "coverage",
            "spatial_extent",
            "temporal_extent",
            "gsd",
            "resolution",
            "cadence",
            "bands",
        )
        trimmed = {
            field: copy.deepcopy(result[field])
            for field in allowed_fields
            if result.get(field) not in (None, "", [], {})
        }
        detail = result.get("detail") if isinstance(result.get("detail"), dict) else {}
        for field in (
            "description",
            "bands",
            "gsd",
            "cadence",
            "temporal_extent",
            "spatial_extent",
            "resolution",
        ):
            if field not in trimmed and detail.get(field) not in (None, "", [], {}):
                trimmed[field] = copy.deepcopy(detail[field])
        return trimmed

    @classmethod
    def _trim_step_plan_evidence_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = (
            "evidence_record_id",
            "finding_id",
            "target_field",
            "modification_direction",
            "required_channels",
            "status",
            "evidence_summary",
            "evidence_limits",
        )
        trimmed = {
            field: copy.deepcopy(record[field])
            for field in allowed_fields
            if record.get(field) not in (None, "", [], {})
        }
        if "literature_results" in record:
            trimmed["literature_results"] = [
                cls._trim_step_plan_literature_evidence(result)
                for result in (record.get("literature_results") or [])
                if isinstance(result, dict)
            ]
        if "dataset_results" in record:
            trimmed["dataset_results"] = [
                cls._trim_step_plan_dataset_evidence(result)
                for result in (record.get("dataset_results") or [])
                if isinstance(result, dict)
            ]
        return trimmed

    @classmethod
    def _build_step_plan_handoff(cls, revision_audit: Dict[str, Any]) -> Dict[str, Any]:
        """Derive the narrow, non-audit handoff consumed by StepPlan."""
        audit = revision_audit if isinstance(revision_audit, dict) else {}
        allowed_note_fields = (
            "finding_id",
            "target_field",
            "open_point",
            "step_plan_responsibility",
            "not_fixed_by_idea_stage",
            "note",
            "notes",
        )
        handoff_notes: List[Dict[str, Any]] = []
        linked_evidence_ids: set[str] = set()
        for item in audit.get("handoff_notes_for_step_plan", []) or []:
            if not isinstance(item, dict):
                continue
            linked_evidence_ids.update(
                str(evidence_id).strip()
                for evidence_id in (item.get("evidence_record_ids") or [])
                if str(evidence_id).strip()
            )
            note = {
                field: copy.deepcopy(item[field])
                for field in allowed_note_fields
                if field in item
            }
            if note:
                handoff_notes.append(note)

        evidence_by_finding = {
            str(item.get("finding_id") or ""): item
            for item in (audit.get("evidence_records", []) or [])
            if isinstance(item, dict) and str(item.get("finding_id") or "")
        }
        unresolved_dataset_requirements: List[Dict[str, Any]] = []
        unresolved_dataset_finding_ids: set[str] = set()
        seen_finding_ids: set[str] = set()
        for item in audit.get("unresolved_findings", []) or []:
            if not isinstance(item, dict):
                continue
            finding_id = str(item.get("finding_id") or "").strip()
            evidence = evidence_by_finding.get(finding_id, {})
            required_channels = {
                str(channel).strip().lower()
                for channel in (evidence.get("required_channels", []) or [])
                if str(channel).strip()
            }
            explicitly_dataset_unresolved = str(item.get("kind") or "").strip() in {
                "dataset_unresolved",
                "dataset_requirement_unresolved",
            }
            dataset_channel_unresolved = (
                "dataset" in required_channels
                and not list(evidence.get("dataset_results", []) or [])
            )
            if not explicitly_dataset_unresolved and not dataset_channel_unresolved:
                continue
            if finding_id and finding_id in seen_finding_ids:
                continue
            if finding_id:
                seen_finding_ids.add(finding_id)

            queries = evidence.get("queries") if isinstance(evidence.get("queries"), dict) else {}
            requirement = {
                "finding_id": finding_id,
                "target_field": str(
                    item.get("target_field") or evidence.get("target_field") or ""
                ).strip(),
                "dataset_query": str(
                    item.get("dataset_query") or queries.get("dataset") or ""
                ).strip(),
                "evidence_status": str(
                    item.get("evidence_status") or evidence.get("status") or ""
                ).strip(),
                "reason": str(item.get("reason") or "").strip(),
            }
            retrieval_errors = [
                str(error).strip()
                for error in (
                    item.get("retrieval_errors")
                    or evidence.get("retrieval_errors", [])
                    or []
                )
                if str(error).strip()
            ]
            if retrieval_errors:
                requirement["retrieval_errors"] = retrieval_errors
            unresolved_dataset_requirements.append(requirement)
            if finding_id:
                unresolved_dataset_finding_ids.add(finding_id)

        evidence_records = [
            cls._trim_step_plan_evidence_record(item)
            for item in (audit.get("evidence_records", []) or [])
            if isinstance(item, dict)
            and (
                str(item.get("evidence_record_id") or "").strip() in linked_evidence_ids
                or str(item.get("finding_id") or "").strip()
                in unresolved_dataset_finding_ids
            )
        ]

        return {
            "handoff_notes_for_step_plan": handoff_notes,
            "unresolved_dataset_requirements": unresolved_dataset_requirements,
            "evidence_records": evidence_records,
        }

    @staticmethod
    def _summarize_idea_for_application_description(idea: Dict[str, Any], plan: Dict[str, Any]) -> str:
        motivation = idea.get("knowledge_anchor_and_motivation", {})
        if isinstance(motivation, dict) and motivation.get("research_motivation"):
            return str(motivation.get("research_motivation"))
        return str(plan.get("application_description") or idea.get("research_question") or "")

    @staticmethod
    def _summarize_idea_for_approach(idea: Dict[str, Any]) -> str:
        core = idea.get("core_hypothesis", {})
        mechanism = idea.get("theoretical_mechanism", {})
        parts: List[str] = []
        if isinstance(core, dict):
            parts.extend([
                str(core.get("directional_statement") or "").strip(),
                str(core.get("proxy_or_inference_claim") or "").strip(),
            ])
        if isinstance(mechanism, dict):
            parts.append(str(mechanism.get("mechanism_chain") or "").strip())
        return " ".join(part for part in parts if part)

    @staticmethod
    def _summarize_idea_rationale(idea: Dict[str, Any]) -> str:
        value = idea.get("novelty_constraints_and_value", {})
        if isinstance(value, dict) and value.get("system_value"):
            return str(value.get("system_value"))
        core = idea.get("core_hypothesis", {})
        if isinstance(core, dict):
            return str(core.get("directional_statement") or "")
        return ""


    async def _run_task_planning_phase(self, session: WorkflowSession) -> None:
        """
        执行任务规划阶段（双 agent 分发）。

        根据 session 当前是否已有 application_records，区分：
        - initial_batch: 首次批量生成 application_plans
        - scoped_replan: 仅重规划 active application
        """
        logger.info(f"Plan Agent: Starting task planning for session {session.id}")

        idea_generation_agent = self._get_agent("idea_generation")
        step_plan_agent = self._get_agent("step_plan")
        resources_agent = self._get_agent("resources_check")
        if not idea_generation_agent:
            raise ValueError("Idea generation agent initialization failed")
        if not step_plan_agent:
            raise ValueError("Step plan agent initialization failed")

        cli_task_name = self.config.get("task_name", "geospatial_discovery")
        current_app: Optional[ApplicationRecord] = None
        planning_mode = ""

        try:
            application_records = list(getattr(session, "application_records", []) or [])
            if application_records and isinstance(application_records[0], dict):
                application_records = [ApplicationRecord.from_dict(item) for item in application_records]
                session.application_records = application_records

            plan_params: Dict[str, Any] = {}
            previous_task_plan: List[Dict[str, Any]] = []
            previous_required_datasets: List[Dict[str, Any]] = []
            plan_refinement_feedback: Dict[str, Any] = {}
            resource_check_feedback: Dict[str, Any] = {}
            replan_round = 0

            if not application_records:
                planning_mode = "initial_batch"
                plan_params = {"mode": "initial_batch"}
            else:
                active_application_id = getattr(session, "active_application_id", None)
                if not active_application_id:
                    raise ValueError(
                        "TASK_PLANNING cannot enter scoped_replan without active_application_id"
                    )
                current_app = next(
                    (app for app in application_records if app.idea_id == active_application_id),
                    None,
                )
                if current_app is None:
                    raise ValueError(
                        f"Active application {active_application_id} not found in application_records"
                    )
                if current_app.status != "replanning":
                    raise ValueError(
                        "TASK_PLANNING may only call scoped_replan when the active application is in replanning status"
                    )
                planning_mode = "scoped_replan"
                plan_params = {
                    "mode": "scoped_replan",
                    "scope": active_application_id,
                }
                previous_task_plan = current_app.task_plan
                previous_required_datasets = list(current_app.required_datasets or [])
                resource_check_feedback = current_app.resource_check_feedback or {}
                plan_refinement_feedback = current_app.plan_refinement_result or {}
                replan_round = current_app.replan_round

            dataset_portrait = getattr(session, "dataset_portrait", {}) or {}
            planning_audit = getattr(session, "planning_audit", {}) or {}
            persisted_candidate_datasets = (
                planning_audit.get("final_candidate_inventory", [])
                if isinstance(planning_audit, dict)
                else []
            )
            context = {
                "goal": session.task.description,
                "literature_review": session.literature_review if hasattr(session, 'literature_review') else [],
                "literature_filter_params": getattr(session, 'literature_filter_params', {}),
                "literature_review_raw_count": len(getattr(session, 'literature_review_raw', []) or []),
                "dataset_info": self._generation_dataset_info(session),
                "dataset_portrait": dataset_portrait,
                "available_tools": [],
                "domain_knowledge": getattr(session, "domain_knowledge", None),
                # For PlanAgent: use method-application-dataset triples as evidence for research question derivation trace
                "method_application_dataset_triples": (
                    (getattr(session, "survey_results", {}) or {}).get("aggregated_views", {}) or {}
                ).get("method_application_dataset_triples", []),
                "previous_task_plan": previous_task_plan,
                "previous_required_datasets": previous_required_datasets,
                "plan_refinement_feedback": plan_refinement_feedback,
                "resource_check_feedback": resource_check_feedback,
                "candidate_datasets": copy.deepcopy(persisted_candidate_datasets),
                "replan_round": replan_round,
                "session_id": session.id,
                "task_name": cli_task_name,
                "application_records": [
                    app.to_dict() if hasattr(app, 'to_dict') else app
                    for app in application_records
                ],
            }
            if current_app is not None:
                context["active_application"] = current_app.to_dict()

            if planning_mode == "initial_batch":
                # ── idea-first flow: generation → within-type triage → maturation → step plan ──
                scout_inventory = await self._scout_candidate_datasets(context)
                context["candidate_datasets"] = copy.deepcopy(
                    scout_inventory["candidate_datasets"]
                )
                context["initial_candidate_datasets"] = copy.deepcopy(
                    scout_inventory["candidate_datasets"]
                )
                context["near_miss_candidates"] = copy.deepcopy(
                    scout_inventory["near_miss_candidates"]
                )
                context["candidate_scout_query_audit"] = copy.deepcopy(
                    scout_inventory["query_audit"]
                )
                context["candidate_axis_coverage"] = copy.deepcopy(
                    scout_inventory["axis_coverage"]
                )
                idea_result = await idea_generation_agent.execute(context, {"mode": "initial_batch"})
                idea_selection_audit: List[Dict[str, Any]] = []
                triaged_ideas = await self._filter_and_select_ideas(
                    idea_result,
                    context,
                    audit_log=idea_selection_audit,
                )
                if not triaged_ideas:
                    raise ValueError("No ideas survived type-level triage")

                # Mature only one already-triaged idea from each fixed type. The
                # maturation agent never removes a candidate when a revision fails.
                idea_result = {
                    **idea_result,
                    "application_plans": [copy.deepcopy(item["plan"]) for item in triaged_ideas],
                    "idea_candidates": [copy.deepcopy(item["idea"]) for item in triaged_ideas],
                }

                # ── Idea maturation（feature flag 控制） ──
                if self.enable_idea_maturation:
                    idea_maturation_agent = self._get_agent("idea_maturation")
                    if not idea_maturation_agent:
                        raise ValueError(
                            "Idea maturation agent initialization failed while "
                            "enable_idea_maturation=true"
                        )
                    idea_maturation_agent.memory_manager = self.memory_manager
                    idea_maturation_agent.research_memory_scope = str(session.id)
                    maturation_result = await idea_maturation_agent.execute(
                        {**context, "idea_result": idea_result},
                        {},
                    )
                    idea_result = maturation_result.get("idea_result", idea_result)
                    maturation_traces = maturation_result.get(
                        "idea_maturation_traces",
                        [],
                    )
                    context["candidate_datasets"] = maturation_result.get(
                        "candidate_datasets",
                        context.get("candidate_datasets", []),
                    )
                else:
                    maturation_traces = []

                idea_generation_audit = idea_result.get("planning_audit", {}) or {}
                initial_candidates = [
                    candidate
                    for candidate in context.get("initial_candidate_datasets", []) or []
                    if isinstance(candidate, dict)
                ]
                initial_candidate_ids = {
                    str(candidate.get("id") or "").strip()
                    for candidate in initial_candidates
                    if str(candidate.get("id") or "").strip()
                }
                session.planning_audit = {
                    "planning_mode": planning_mode,
                    "generated_research_questions": [],
                    "initial_candidate_datasets_full": [
                        copy.deepcopy(candidate)
                        for candidate in initial_candidates
                    ],
                    "initial_candidate_datasets": [
                        self._candidate_dataset_audit_entry(
                            candidate,
                            stage="idea_generation_scout",
                        )
                        for candidate in initial_candidates
                    ],
                    "final_candidate_inventory_full": [
                        copy.deepcopy(candidate)
                        for candidate in context.get("candidate_datasets", []) or []
                        if isinstance(candidate, dict)
                    ],
                    "final_candidate_inventory": [
                        self._candidate_dataset_audit_entry(
                            candidate,
                            stage=(
                                "idea_generation_scout"
                                if str(candidate.get("id") or "").strip()
                                in initial_candidate_ids
                                else "idea_maturation"
                            ),
                        )
                        for candidate in context.get("candidate_datasets", []) or []
                        if isinstance(candidate, dict)
                    ],
                    "near_miss_candidates": [
                        self._candidate_dataset_audit_entry(
                            candidate,
                            stage="idea_generation_scout",
                        )
                        for candidate in context.get("near_miss_candidates", []) or []
                        if isinstance(candidate, dict)
                    ],
                    "candidate_scout_query_audit": copy.deepcopy(
                        context.get("candidate_scout_query_audit", [])
                    ),
                    "candidate_axis_coverage": copy.deepcopy(
                        context.get("candidate_axis_coverage", {})
                    ),
                    "idea_generation_audit": idea_generation_audit,
                    "idea_selection_audit": idea_selection_audit,
                    "generated_idea_count": len(idea_result.get("idea_candidates", []) or []),
                    "generated_idea_types": [
                        item.get("idea_type", "")
                        for item in idea_result.get("idea_candidates", []) or []
                        if isinstance(item, dict)
                    ],
                    "triaged_idea_count": len(idea_result.get("application_plans", []) or []),
                    "idea_maturation_traces": maturation_traces,
                }

                selected_ideas = await self._filter_and_select_ideas(
                    idea_result,
                    context,
                    audit_log=idea_selection_audit,
                )
                if not selected_ideas:
                    raise ValueError("No ideas survived evaluation")
                session.planning_audit["selected_idea_count"] = len(selected_ideas)
                session.planning_audit["selected_idea_types"] = [
                    item["idea"].get("idea_type", "")
                    for item in selected_ideas
                    if isinstance(item, dict) and isinstance(item.get("idea"), dict)
                ]

                application_plans = []
                idea_only_boundary = bool(getattr(session, "stop_after_idea_maturation", False))
                for item in selected_ideas:
                    if idea_only_boundary:
                        app_plan = self._build_application_plan_from_selected_idea(item)
                    else:
                        try:
                            step_result = await self._execute_step_plan_with_verified_datasets(
                                step_plan_agent=step_plan_agent,
                                resources_agent=resources_agent,
                                step_context=
                                {
                                    "idea": self._build_step_plan_idea_payload(
                                        idea=item["idea"],
                                        revision_audit=self._revision_audit_from_plan(item["plan"]),
                                        dataset_info=context.get("dataset_info", {}),
                                        candidate_datasets=context.get("candidate_datasets", []),
                                    ),
                                    "dataset_info": context.get("dataset_info", {}),
                                },
                                step_params={"mode": "plan_for_idea"},
                            )
                            app_plan = self._build_application_plan_from_selected_idea(
                                item,
                                step_result,
                            )
                        except StepPlanContractError as exc:
                            logger.error(
                                "Initial StepPlan contract failed for idea %s: %s",
                                item["idea"].get("idea_id", ""),
                                exc,
                            )
                            app_plan = self._build_failed_application_plan_from_step_error(
                                item,
                                exc,
                            )
                    application_plans.append(app_plan)

                session.application_records = [
                    ApplicationRecord.from_dict(app)
                    for app in application_plans
                    if isinstance(app, dict)
                ]
                if not session.application_records:
                    raise ValueError("initial_batch did not produce any valid ApplicationRecord payloads")
                first_pending_application_id = next(
                    (app.idea_id for app in session.application_records if app.status == "pending"),
                    None,
                )
                if first_pending_application_id:
                    session.active_application_id = first_pending_application_id
                    next_state = WorkflowState.RESOURCE_CHECK
                else:
                    first_failed_application_id = next(
                        (
                            app.idea_id
                            for app in session.application_records
                            if app.status == "failed_before_evaluation"
                        ),
                        None,
                    )
                    if not first_failed_application_id:
                        raise ValueError(
                            "initial_batch produced neither pending nor failed applications"
                        )
                    session.active_application_id = first_failed_application_id
                    next_state = WorkflowState.IDEA_EVALUATION
                session.planning_audit.update(
                    {
                        "generated_application_plan_count": len(idea_result.get("application_plans", []) or []),
                        "selected_application_count": len(session.application_records),
                        "selected_idea_ids": [
                            app.idea_id for app in session.application_records if getattr(app, "idea_id", "")
                        ],
                    }
                )
                logger.info(
                    f"Plan Agent: Generated {len(session.application_records)} application records "
                    f"({'idea_only' if idea_only_boundary else 'with plans'}) for session {session.id}"
                )


            else:
                idea = current_app.idea_candidate if isinstance(current_app.idea_candidate, dict) else {}
                if not idea:
                    raise ValueError("scoped_replan requires the original idea payload")
                step_result = await self._execute_step_plan_with_verified_datasets(
                    step_plan_agent=step_plan_agent,
                    resources_agent=resources_agent,
                    step_context=
                    {
                        "idea": self._build_step_plan_idea_payload(
                            idea=idea,
                            revision_audit=current_app.idea_revision_audit,
                            dataset_info=context.get("dataset_info", {}),
                            candidate_datasets=context.get("candidate_datasets", []),
                        ),
                        "dataset_info": context.get("dataset_info", {}),
                        "previous_task_plan": previous_task_plan,
                        "previous_required_datasets": previous_required_datasets,
                        "plan_refinement_feedback": plan_refinement_feedback,
                        "resource_check_feedback": resource_check_feedback,
                    },
                    step_params={"mode": "replan", "replan_round": replan_round},
                )
                if not isinstance(step_result.get("required_datasets"), list):
                    raise ValueError(
                        "StepPlan replan result must contain a top-level required_datasets list"
                    )
                current_app.task_plan = step_result.get("task_plan", [])
                current_app.required_datasets = copy.deepcopy(
                    step_result["required_datasets"]
                )
                capability_anchor_meta = step_result.get("capability_anchor_meta")
                if not isinstance(capability_anchor_meta, dict):
                    raise ValueError(
                        "StepPlan replan result must contain capability_anchor_meta"
                    )
                current_app.capability_anchor_meta = copy.deepcopy(
                    capability_anchor_meta
                )
                generation_audit = step_result.get("step_plan_generation_audit")
                if isinstance(generation_audit, dict):
                    current_app.step_plan_generation_audit.append(
                        copy.deepcopy(generation_audit)
                    )
                self._append_plan_revision_snapshot(
                    current_app,
                    event="step_plan_replan",
                    metadata={"routing": copy.deepcopy(generation_audit or {}).get("routing", {})},
                )
                logger.info(
                    f"Plan Agent: Replanned active application {current_app.idea_id} "
                    f"with {len(current_app.task_plan)} steps in replan round "
                    f"{current_app.replan_round} for session {session.id}"
                )

            if planning_mode == "scoped_replan":
                next_state = WorkflowState.RESOURCE_CHECK
            logger.info(f"task_planning ({planning_mode}) -> {next_state.value}")
            await self._update_session_state(session, next_state)

        except StepPlanContractError as e:
            coverage_gap = getattr(e, "coverage_gap", False)
            logger.error(
                "StepPlan contract failed in task planning phase "
                f"(coverage_gap={coverage_gap}): {str(e)}"
            )
            if planning_mode == "scoped_replan" and current_app is not None:
                if coverage_gap:
                    terminal_rule = "T8_capability_coverage_gap"
                    reason = (
                        "Refinement could not be closed: the capability index lacks a "
                        f"function implementing the required operation. {str(e)}"
                    )
                else:
                    terminal_rule = "T8_scoped_replan_failed"
                    reason = f"Scoped replanning failed for active application: {str(e)}"
                await self._mark_application_failed_before_evaluation(
                    session=session,
                    current_app=current_app,
                    terminal_rule=terminal_rule,
                    reason=reason,
                    recovery_hint=(current_app.plan_refinement_result or {}).get("feedback_summary", ""),
                )
                return
            await self._update_session_state(session, WorkflowState.ERROR)

        except Exception as e:
            logger.error(f"Error in task planning phase: {str(e)}")
            if planning_mode == "scoped_replan" and current_app is not None:
                await self._mark_application_failed_before_evaluation(
                    session=session,
                    current_app=current_app,
                    terminal_rule="T8_scoped_replan_failed",
                    reason=f"Scoped replanning failed for active application: {str(e)}",
                    recovery_hint=(current_app.plan_refinement_result or {}).get("feedback_summary", ""),
                )
                return
            await self._update_session_state(session, WorkflowState.ERROR)

    async def _run_resource_check_phase(self, session: WorkflowSession) -> None:
        """
        执行资源检查阶段（严格数据网关）。

        RESOURCE_CHECK 只检查当前 active application 的 task_plan 与上下文，
        并将绑定后的 task_plan、required_datasets、resource_check 和反馈写回该 application。
        """
        logger.info(f"Resource Check Agent: Starting strict data gateway for session {session.id}")

        resources_agent = self._get_agent("resources_check")
        if not resources_agent:
            raise ValueError("Resource check agent initialization failed")

        try:
            from ..workflow.data_type import ResourceCheckFeedback

            application_records = list(getattr(session, "application_records", []) or [])
            if application_records and isinstance(application_records[0], dict):
                application_records = [
                    ApplicationRecord.from_dict(item)
                    for item in application_records
                    if isinstance(item, dict)
                ]
                session.application_records = application_records

            active_application_id = getattr(session, "active_application_id", None)
            if not active_application_id:
                raise ValueError(
                    "RESOURCE_CHECK requires active_application_id to run application-scoped validation"
                )

            current_app = next(
                (app for app in application_records if app.idea_id == active_application_id),
                None,
            )
            if current_app is None:
                raise ValueError(
                    f"Active application {active_application_id} not found in application_records"
                )

            task_plan = current_app.task_plan
            if not isinstance(task_plan, list) or not task_plan:
                raise ValueError(
                    f"RESOURCE_CHECK requires a non-empty task_plan for active application {active_application_id}"
                )

            # DAG 标记仅保留给实际执行审计，不作为研究计划检查门禁。
            capability_anchor_meta = current_app.capability_anchor_meta
            plan_dag_valid = capability_anchor_meta.get("plan_dag_valid") is True
            plan_dag_issues = capability_anchor_meta.get("plan_dag_issues") or []

            logger.info(
                f"Resource Check Agent: using active application {current_app.idea_id} "
                f"task_plan ({len(task_plan)} steps) for session {session.id}"
            )

            application_dataset_info = self._generation_dataset_info(session)
            application_dataset_info["required_datasets"] = list(
                current_app.required_datasets or []
            )

            context = {
                "task_plan": task_plan,
                "dataset_info": application_dataset_info,
                "goal": current_app.application_description or session.task.description,
                "replan_round": current_app.replan_round,
                "active_application_id": current_app.idea_id,
                "idea": current_app.idea_candidate,
                "capability_anchor_meta": copy.deepcopy(capability_anchor_meta),
                "application_context": {
                    "task_id": session.task.id,
                    "session_id": session.id,
                    "idea_id": current_app.idea_id,
                    "application_name": current_app.application_name,
                    "application_description": current_app.application_description,
                    "application_context": current_app.application_context,
                    "priority": current_app.priority,
                },
            }

            result = await resources_agent.execute(context, {})

            termination = result.get("termination", {}) if isinstance(result, dict) else {}
            resource_dataset_info = result.get("dataset_info", {}) if isinstance(result, dict) else {}
            adjustments = result.get("adjustments", []) if isinstance(result, dict) else []
            manifest = result.get("data_access_manifest", []) if isinstance(result, dict) else []
            resource_task_plan = result.get("task_plan") if isinstance(result, dict) else None
            if not isinstance(resource_task_plan, list):
                raise ValueError(
                    "RESOURCE_CHECK result must contain an application-scoped task_plan list"
                )
            if not isinstance(resource_dataset_info.get("required_datasets"), list):
                raise ValueError(
                    "RESOURCE_CHECK dataset_info must contain a required_datasets list"
                )

            current_app.task_plan = copy.deepcopy(resource_task_plan)
            current_app.required_datasets = copy.deepcopy(
                resource_dataset_info["required_datasets"]
            )

            unavailable_tools: List[str] = []
            for adjustment in adjustments:
                if adjustment.get("type") not in {
                    "tool_not_found",
                    "tool_import_failed",
                }:
                    continue
                tool_name = str(adjustment.get("tool", "")).strip()
                if tool_name and tool_name not in unavailable_tools:
                    unavailable_tools.append(tool_name)

            contract = ExecutionContract(
                primary_data_verified=(
                    resource_dataset_info.get("primary_data_status") == "verified"
                ),
                primary_data_path=resource_dataset_info.get("data_path"),
                guaranteed_datasets=result.get("guaranteed_datasets", []) if isinstance(result, dict) else [],
                unavailable_tools=unavailable_tools,
                data_access_manifest=manifest,
                should_terminate=termination.get("should_terminate", False),
                termination_reason=termination.get("reason", ""),
                terminal_rule=termination.get("terminal_rule", ""),
                recovery_hint=termination.get("recovery_hint", ""),
                step_tool_validations=result.get("step_tool_validations", []) if isinstance(result, dict) else [],
                tool_reachability_score=result.get("tool_reachability_score", 1.0) if isinstance(result, dict) else 1.0,
                implementation_compatibility_passed=(
                    result.get("implementation_compatibility_passed", False)
                    if isinstance(result, dict)
                    else False
                ),
                weakest_step_id=result.get("weakest_step_id") if isinstance(result, dict) else None,
                plan_dag_valid=plan_dag_valid,
                plan_dag_issues=copy.deepcopy(plan_dag_issues),
            )

            mandatory_unbound = contract.get_mandatory_unbound_items()
            mandatory_probe_failed = contract.get_mandatory_remote_probe_failed_items()
            contract.resource_check_passed = (
                not contract.should_terminate
                and not contract.unavailable_tools
                and contract.implementation_compatibility_passed
                and len(mandatory_unbound) == 0
                and len(mandatory_probe_failed) == 0
            )

            if not contract.should_terminate and contract.terminal_rule:
                logger.error(
                    "Protocol error: RESOURCE_CHECK returned non-empty terminal_rule "
                    f"'{contract.terminal_rule}' while should_terminate=False for active application "
                    f"{current_app.idea_id} in session {session.id}"
                )
                await self._update_session_state(session, WorkflowState.ERROR)
                return

            current_app.resource_check = {
                "adjustments": adjustments,
                "missing_datasets": result.get("missing_datasets", []) if isinstance(result, dict) else [],
                "dataset_resolution_failures": result.get("dataset_resolution_failures", []) if isinstance(result, dict) else [],
                "dataset_info": resource_dataset_info,
                "guaranteed_datasets": result.get("guaranteed_datasets", []) if isinstance(result, dict) else [],
                "execution_contract": contract.to_dict(),
            }
            can_proceed, reason = contract.can_proceed_to_plan_review()
            affected_steps = []
            for adj in adjustments:
                affected_steps.extend(
                    [str(s) for s in adj.get("affected_steps", [])]
                )
                if adj.get("step_id") is not None:
                    affected_steps.append(str(adj["step_id"]))
            for step_failure in (
                result.get("step_failures", []) if isinstance(result, dict) else []
            ):
                if isinstance(step_failure, dict) and step_failure.get("step_id") is not None:
                    affected_steps.append(str(step_failure["step_id"]))
            affected_steps = list(dict.fromkeys(affected_steps))

            failure_class = result.get("failure_class", "") if isinstance(result, dict) else ""

            if contract.should_terminate:
                feedback = ResourceCheckFeedback(
                    terminal_rule=contract.terminal_rule,
                    reason=contract.termination_reason or reason,
                    recovery_hint=contract.recovery_hint,
                    missing_datasets=result.get("missing_datasets", []) if isinstance(result, dict) else [],
                    unavailable_tools=list(contract.unavailable_tools),
                    affected_steps=affected_steps,
                    step_failures=result.get("step_failures", []) if isinstance(result, dict) else [],
                    weakest_step_id=contract.weakest_step_id,
                    failure_class=failure_class,
                    dataset_resolution_failures=result.get("dataset_resolution_failures", []) if isinstance(result, dict) else [],
                    suggested_task_type=self._extract_suggested_task_type(result) if isinstance(result, dict) else "",
                    suggested_conversion_task=self._extract_suggested_conversion(result) if isinstance(result, dict) else "",
                    recoverable=bool(termination.get("recoverable", False)),
                    replan_target=str(termination.get("replan_target", "") or ""),
                )

                current_app.resource_check_feedback = feedback.to_dict()
                self._append_plan_revision_snapshot(
                    current_app,
                    event="resource_check",
                    metadata={
                        "resource_check_passed": contract.resource_check_passed,
                        "tool_reachability_score": contract.tool_reachability_score,
                        "implementation_compatibility_passed": contract.implementation_compatibility_passed,
                        "failure_class": failure_class,
                    },
                )
                await self.memory_manager.store_session(session)
                if not self.enable_plan_refinement_phase:
                    await self._advance_after_resource_check_without_refinement(
                        session=session,
                        current_app=current_app,
                        feedback=current_app.resource_check_feedback,
                    )
                    return
                await self._update_session_state(session, WorkflowState.PLAN_REFINEMENT)
                return

            if not can_proceed:
                feedback = ResourceCheckFeedback(
                    terminal_rule="",
                    reason=contract.termination_reason or reason,
                    recovery_hint=contract.recovery_hint,
                    missing_datasets=result.get("missing_datasets", []) if isinstance(result, dict) else [],
                    unavailable_tools=list(contract.unavailable_tools),
                    affected_steps=affected_steps,
                    step_failures=result.get("step_failures", []) if isinstance(result, dict) else [],
                    weakest_step_id=contract.weakest_step_id,
                    failure_class=failure_class,
                    dataset_resolution_failures=result.get("dataset_resolution_failures", []) if isinstance(result, dict) else [],
                    suggested_task_type=self._extract_suggested_task_type(result) if isinstance(result, dict) else "",
                    suggested_conversion_task=self._extract_suggested_conversion(result) if isinstance(result, dict) else "",
                    recoverable=True,
                    replan_target="step_plan",
                )

                current_app.resource_check_feedback = feedback.to_dict()
                self._append_plan_revision_snapshot(
                    current_app,
                    event="resource_check",
                    metadata={
                        "resource_check_passed": contract.resource_check_passed,
                        "tool_reachability_score": contract.tool_reachability_score,
                        "implementation_compatibility_passed": contract.implementation_compatibility_passed,
                        "failure_class": failure_class,
                    },
                )
                await self.memory_manager.store_session(session)
                if not self.enable_plan_refinement_phase:
                    await self._advance_after_resource_check_without_refinement(
                        session=session,
                        current_app=current_app,
                        feedback=current_app.resource_check_feedback,
                    )
                    return
                await self._update_session_state(session, WorkflowState.PLAN_REFINEMENT)
                return

            logger.info(
                f"Resource Check Agent: contract PASSED for active application {current_app.idea_id} "
                f"in session {session.id}. Guaranteed datasets: {contract.guaranteed_datasets}"
            )
            logger.info(
                f"[METRIC] resource_check_passed session={session.id} "
                f"application={current_app.idea_id} "
                f"replan_round={current_app.replan_round} "
                f"guaranteed_datasets={len(contract.guaranteed_datasets)}"
            )
            current_app.resource_check_feedback = {}
            self._append_plan_revision_snapshot(
                current_app,
                event="resource_check",
                metadata={
                    "resource_check_passed": contract.resource_check_passed,
                    "tool_reachability_score": contract.tool_reachability_score,
                    "implementation_compatibility_passed": contract.implementation_compatibility_passed,
                    "failure_class": failure_class,
                },
            )

            await self.memory_manager.store_session(session)
            if not self.enable_plan_refinement_phase:
                await self._advance_after_resource_check_without_refinement(
                    session=session,
                    current_app=current_app,
                    feedback={},
                )
                return
            await self._update_session_state(session, WorkflowState.PLAN_REFINEMENT)

        except Exception as e:
            logger.error(f"Error in resource check phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)


    # ------------------------------------------------------------------
    # Commit 4: helpers for extracting suggested_task_type / conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_suggested_task_type(result: Dict[str, Any]) -> str:
        """从 step_failures 中提取第一个 capability_gap 的 suggested_task_type。"""
        for f in result.get("step_failures", []):
            st = f.get("suggested_task_type", "")
            if st:
                return st
        return ""

    @staticmethod
    def _extract_suggested_conversion(result: Dict[str, Any]) -> str:
        """从 step_failures 的 capability gap 中提取显式转换建议（如有）。"""
        for f in result.get("step_failures", []):
            if f.get("gap_type") == "coverage_gap":
                return f"Step {f.get('step_id', '?')}: {f.get('detail', '')}"
        return ""

    async def _advance_after_resource_check_without_refinement(
        self,
        session: WorkflowSession,
        current_app: ApplicationRecord,
        feedback: Dict[str, Any],
    ) -> None:
        """在 no_refinement 配置下，直接将 ResourceCheck 结果固化为终态并推进下一个 application。"""
        if feedback:
            terminal_rule = feedback.get("terminal_rule") or "T9_resource_check_failed_without_refinement"
            await self._mark_application_failed_before_evaluation(
                session=session,
                current_app=current_app,
                terminal_rule=terminal_rule,
                reason=feedback.get("reason", "Resource check failed and plan refinement is disabled."),
                recovery_hint=feedback.get("recovery_hint", ""),
            )
            return

        current_app.status = "ready_for_evaluation"
        current_app.final_disposition = ""
        await self.memory_manager.store_session(session)
        await self._advance_after_application_terminal_state(session)

    async def _run_plan_refinement_phase(self, session: WorkflowSession) -> None:
        """
        执行 PLAN_REFINEMENT 阶段：仅针对当前 active application 做反思与终态写入。

        路由规则：
        - 若当前 application 的 resource_check_feedback.terminal_rule 命中，标记 failed_before_evaluation
        - 若 RefinementAgent 返回 replan_feedback，则将当前 application 标记为 replanning，
          跳回 TASK_PLANNING 触发 scoped_replan
        - 若返回 final_review，则标记 ready_for_evaluation

        本阶段不直接决定是否进入 IDEA_EVALUATION；终态后统一交给 advance hook。
        """
        logger.info(f"Refinement Agent: Starting plan refinement for session {session.id}")

        refinement_agent = self._get_agent("refinement")
        if not refinement_agent:
            logger.error("Refinement agent not available")
            await self._update_session_state(session, WorkflowState.ERROR)
            return

        try:
            from ..workflow.data_type import PlanRefinementResult

            application_records = list(getattr(session, "application_records", []) or [])
            if application_records and isinstance(application_records[0], dict):
                application_records = [
                    ApplicationRecord.from_dict(item)
                    for item in application_records
                    if isinstance(item, dict)
                ]
                session.application_records = application_records

            active_application_id = getattr(session, "active_application_id", None)
            if not active_application_id:
                raise ValueError(
                    "PLAN_REFINEMENT requires active_application_id for application-scoped routing"
                )

            current_app = next(
                (app for app in application_records if app.idea_id == active_application_id),
                None,
            )
            if current_app is None:
                raise ValueError(
                    f"Active application {active_application_id} not found in application_records"
                )

            resource_check_feedback = current_app.resource_check_feedback or {}
            resource_check_payload = current_app.resource_check or {}

            if resource_check_feedback.get("terminal_rule") and not resource_check_feedback.get("recoverable", False):
                logger.error(
                    "Refinement Agent: terminal resource check feedback received for active application "
                    f"{current_app.idea_id} ({resource_check_feedback.get('terminal_rule')}) in session {session.id}"
                )
                await self._mark_application_failed_before_evaluation(
                    session=session,
                    current_app=current_app,
                    terminal_rule=resource_check_feedback.get("terminal_rule", "failed_before_evaluation"),
                    reason=resource_check_feedback.get("reason", "Failed before evaluation."),
                    recovery_hint=resource_check_feedback.get("recovery_hint", ""),
                )
                return

            if resource_check_feedback.get("terminal_rule") and resource_check_feedback.get("recoverable", False):
                logger.info(
                    "Refinement Agent: recoverable terminal resource feedback will be routed through refinement "
                    f"for active application {current_app.idea_id} "
                    f"(rule={resource_check_feedback.get('terminal_rule')}, "
                    f"target={resource_check_feedback.get('replan_target', '')})"
                )

            has_resource_feedback = bool(resource_check_feedback)
            current_replan_round = current_app.replan_round
            next_replan_round = current_replan_round + 1
            replan_limit = self.config.get("workflow", {}).get("max_replan_rounds", 5)
            if has_resource_feedback:
                if next_replan_round > replan_limit:
                    logger.error(
                        f"Refinement Agent: replan_round {next_replan_round} > limit {replan_limit}, "
                        f"terminating session {session.id}"
                    )
                    await self._mark_application_failed_before_evaluation(
                        session=session,
                        current_app=current_app,
                        terminal_rule="T7_resource_replan_limit_exceeded",
                        reason=(
                            f"Resource-driven replanning exceeded max_replan_rounds={replan_limit} "
                            "while mandatory data issues remained unresolved."
                        ),
                        recovery_hint=resource_check_feedback.get("reason", ""),
                    )
                    return

            execution_contract = (
                resource_check_payload.get("execution_contract", {})
                if isinstance(resource_check_payload, dict) else {}
            )
            dataset_info = (
                resource_check_payload.get("dataset_info", {})
                if isinstance(resource_check_payload, dict) else {}
            )
            if not dataset_info:
                dataset_info = dict(getattr(session.task, "spatial_extent", {}) or {})
            dataset_info = self._generation_dataset_info(session, dataset_info)

            context = {
                "goal": current_app.application_description or session.task.description,
                "idea": copy.deepcopy(current_app.idea_candidate),
                "idea_revision_audit": copy.deepcopy(current_app.idea_revision_audit),
                "dataset_info": copy.deepcopy(dataset_info),
                "required_datasets": copy.deepcopy(current_app.required_datasets),
                "task_plan": copy.deepcopy(current_app.task_plan),
                "literature_review": getattr(session, "literature_review", []),
                "resource_check_feedback": resource_check_feedback or None,
                "execution_contract": execution_contract or None,
                "replan_round": (
                    next_replan_round
                    if has_resource_feedback
                    else current_replan_round
                ),
                "active_application_id": current_app.idea_id,
                "application_context": {
                    "idea_id": current_app.idea_id,
                    "application_name": current_app.application_name,
                    "application_description": current_app.application_description,
                    "application_context": current_app.application_context,
                },
            }

            result = await refinement_agent.execute(context, {})
            pr = PlanRefinementResult.from_dict(result)

            if pr.mode == "replan_feedback" and pr.should_replan is True:
                pr.replan_round = next_replan_round
            elif pr.mode == "final_review" and pr.should_replan is False:
                pr.replan_round = current_replan_round

            validation_errors = pr.validation_errors()
            current_app.plan_refinement_attempt_history.append({
                "replan_round": pr.replan_round,
                "recorded_at": datetime.now().isoformat(),
                "is_valid": not validation_errors,
                "validation_errors": list(validation_errors),
                "raw_result": copy.deepcopy(result),
            })

            if validation_errors:
                logger.error(
                    f"Refinement Agent: invalid refinement result "
                    f"(mode={pr.mode}, should_replan={pr.should_replan}, round={pr.replan_round}) "
                    f"for active application {current_app.idea_id} in session {session.id}; "
                    f"validation_errors={validation_errors}"
                )
                await self.memory_manager.store_session(session)
                await self._update_session_state(session, WorkflowState.ERROR)
                return

            if has_resource_feedback:
                if not pr.should_replan or pr.mode != "replan_feedback":
                    logger.error(
                        f"Refinement Agent: expected replan_feedback for active application {current_app.idea_id}, "
                        f"got mode={pr.mode}, should_replan={pr.should_replan}"
                    )
                    await self._update_session_state(session, WorkflowState.ERROR)
                    return

            current_app.plan_refinement_result = pr.to_dict()
            self._append_plan_revision_snapshot(
                current_app,
                event="plan_refinement",
                metadata={
                    "mode": pr.mode,
                    "should_replan": pr.should_replan,
                },
            )

            if pr.mode == "replan_feedback" and pr.should_replan:
                if not has_resource_feedback and next_replan_round > replan_limit:
                    logger.error(
                        f"Refinement Agent: quality-driven replan_round {next_replan_round} "
                        f"> limit {replan_limit}, "
                        f"terminating session {session.id}"
                    )
                    await self._mark_application_failed_before_evaluation(
                        session=session,
                        current_app=current_app,
                        terminal_rule="T7_quality_replan_limit_exceeded",
                        reason=(
                            f"Quality-driven replanning exceeded max_replan_rounds={replan_limit} "
                            "while semantic or discovery-quality issues remained unresolved."
                        ),
                        recovery_hint=pr.feedback_summary,
                    )
                    return

                # ── 歧义状态检查 ──
                # should_replan=true 但无 dataset/step/tool/semantic 任何问题 → ambiguous
                if not has_resource_feedback and not (
                    pr.semantic_alignment_issues
                    or pr.dataset_actions
                    or pr.step_level_adjustments
                    or pr.tool_actions
                ):
                    logger.error(
                        f"Refinement Agent: ambiguous refinement — should_replan=True but no "
                        f"dataset/step/tool/semantic issues found for app={current_app.idea_id}; "
                        f"feedback_summary={pr.feedback_summary!r:.120}"
                    )
                    await self._update_session_state(session, WorkflowState.ERROR)
                    return

                current_app.replan_round = next_replan_round
                current_app.status = "replanning"
                await self.memory_manager.store_session(session)

                if has_resource_feedback:
                    logger.info(
                        f"Refinement Agent: active application {current_app.idea_id} generated "
                        f"resource-driven replan feedback (round {pr.replan_round})"
                    )
                    logger.info(
                        f"[METRIC] replan_triggered session={session.id} "
                        f"application={current_app.idea_id} "
                        f"round={pr.replan_round} "
                        f"step_adjustments={len(pr.step_level_adjustments)} "
                        f"failure_class={resource_check_feedback.get('failure_class', '')}"
                    )
                else:
                    logger.info(
                        f"Refinement Agent: active application {current_app.idea_id} requested "
                        f"quality-driven replanning (round {pr.replan_round})"
                    )
                await self._update_session_state(session, WorkflowState.TASK_PLANNING)
                return

            if pr.mode != "final_review" or pr.should_replan:
                logger.error(
                    f"Refinement Agent: expected final_review or quality-driven replan_feedback for active application {current_app.idea_id}, "
                    f"got mode={pr.mode}, should_replan={pr.should_replan}"
                )
                await self._update_session_state(session, WorkflowState.ERROR)
                return

            current_app.status = "ready_for_evaluation"
            logger.info(
                f"Refinement Agent: active application {current_app.idea_id} final review completed "
                f"(round {pr.replan_round})"
            )
            await self.memory_manager.store_session(session)
            await self._advance_after_application_terminal_state(session)

        except Exception as e:
            logger.error(f"Error in plan refinement phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)

    def _normalize_application_records(self, session: WorkflowSession) -> List[ApplicationRecord]:
        application_records = list(getattr(session, "application_records", []) or [])
        if application_records and isinstance(application_records[0], dict):
            application_records = [
                ApplicationRecord.from_dict(item)
                for item in application_records
                if isinstance(item, dict)
            ]
            session.application_records = application_records
        return application_records

    async def _mark_application_failed_before_evaluation(
        self,
        session: WorkflowSession,
        current_app: ApplicationRecord,
        terminal_rule: str,
        reason: str,
        recovery_hint: str = "",
    ) -> None:
        feedback = dict(current_app.resource_check_feedback or {})
        feedback["terminal_rule"] = terminal_rule
        feedback["reason"] = reason
        if recovery_hint:
            feedback["recovery_hint"] = recovery_hint
        current_app.resource_check_feedback = feedback
        current_app.final_disposition = "failed_before_evaluation"
        current_app.status = "failed_before_evaluation"
        await self.memory_manager.store_session(session)
        await self._advance_after_application_terminal_state(session)

    def _get_active_application_record(
        self,
        session: WorkflowSession,
        application_records: Optional[List[ApplicationRecord]] = None,
    ) -> ApplicationRecord:
        records = application_records or self._normalize_application_records(session)
        active_application_id = getattr(session, "active_application_id", None)
        if not active_application_id:
            raise ValueError("active_application_id is required for application-scoped routing")
        current_app = next(
            (app for app in records if app.idea_id == active_application_id),
            None,
        )
        if current_app is None:
            raise ValueError(
                f"Active application {active_application_id} not found in application_records"
            )
        return current_app

    def _has_valid_idea_evaluation_result(self, app: ApplicationRecord) -> bool:
        from .data_type import IdeaQualityEvaluationResult

        payload = app.idea_evaluation or {}
        if not isinstance(payload, dict) or not payload:
            return False
        try:
            result = IdeaQualityEvaluationResult.from_dict(payload)
        except Exception:
            return False
        return result.is_valid()

    def _is_application_evaluation_complete(self, app: ApplicationRecord) -> bool:
        if app.status == "evaluated":
            return self._has_valid_idea_evaluation_result(app)
        if app.status == "failed_before_evaluation":
            return self._has_valid_idea_evaluation_result(app)
        return False

    def _find_next_pending_application(
        self,
        application_records: List[ApplicationRecord],
    ) -> Optional[ApplicationRecord]:
        return next((app for app in application_records if app.status == "pending"), None)

    def _find_next_application_pending_evaluation(
        self,
        application_records: List[ApplicationRecord],
    ) -> Optional[ApplicationRecord]:
        for app in application_records:
            if app.status not in {"ready_for_evaluation", "failed_before_evaluation", "evaluated"}:
                continue
            if not self._is_application_evaluation_complete(app):
                return app
        return None

    def _all_applications_finished_planning(
        self,
        application_records: List[ApplicationRecord],
    ) -> bool:
        return all(app.status not in {"pending", "replanning"} for app in application_records)

    def _all_applications_completed_evaluation(
        self,
        application_records: List[ApplicationRecord],
    ) -> bool:
        return all(self._is_application_evaluation_complete(app) for app in application_records)

    def _build_application_ranking(
        self,
        application_records: List[ApplicationRecord],
    ) -> List[Dict[str, Any]]:
        ranked_entries: List[Dict[str, Any]] = []
        for app in application_records:
            if not self._has_valid_idea_evaluation_result(app):
                continue
            idea_evaluation = app.idea_evaluation or {}
            aggregates = idea_evaluation.get("aggregates", {}) or {}
            idea_quality_score = aggregates.get("idea_quality_score")
            if idea_quality_score is None:
                continue
            ranked_entries.append(
                {
                    "idea_id": app.idea_id,
                    "application_name": app.application_name,
                    "priority": app.priority,
                    "idea_quality_score": float(idea_quality_score),
                    "extended_quality_score": aggregates.get("extended_quality_score"),
                    "minimum_core_score": aggregates.get("minimum_core_score"),
                    "evaluation_status": idea_evaluation.get("evaluation_status", ""),
                    "application_status": app.status,
                }
            )
        ranked_entries.sort(
            key=lambda item: (
                -float(item["idea_quality_score"]),
                int(item["priority"]),
                str(item["idea_id"]),
            )
        )
        return ranked_entries

    async def _advance_after_application_terminal_state(self, session: WorkflowSession) -> None:
        """统一承接 application 终态后的推进逻辑。"""
        application_records = self._normalize_application_records(session)
        if not application_records:
            raise ValueError("advance hook requires non-empty application_records")

        current_app = self._get_active_application_record(session, application_records)

        if self._is_application_evaluation_complete(current_app):
            next_eval_app = self._find_next_application_pending_evaluation(application_records)
            if next_eval_app is not None:
                should_transition = self._validate_session_state_transition(
                    session.state,
                    WorkflowState.IDEA_EVALUATION,
                )
                session.active_application_id = next_eval_app.idea_id
                await self.memory_manager.store_session(session)
                if should_transition:
                    await self._update_session_state(session, WorkflowState.IDEA_EVALUATION)
                else:
                    logger.info(
                        f"Advance hook: continue IDEA_EVALUATION with application {next_eval_app.idea_id} "
                        f"for session {session.id}"
                    )
                return

            if not self._all_applications_completed_evaluation(application_records):
                raise ValueError(
                    "Cannot complete session while some applications still lack idea_evaluation results"
                )

            ranking = self._build_application_ranking(application_records)
            self._validate_session_state_transition(
                session.state,
                WorkflowState.COMPLETED,
            )
            setattr(session, "application_ranking", ranking)
            session.active_application_id = None
            logger.info(
                f"Advance hook: all applications evaluated for session {session.id}; "
                f"ranking_size={len(ranking)}"
            )
            await self.memory_manager.store_session(session)
            await self._update_session_state(session, WorkflowState.COMPLETED)
            return

        if current_app.status not in {"ready_for_evaluation", "failed_before_evaluation"}:
            raise ValueError(
                f"Advance hook only supports planning/evaluation terminal states, got {current_app.status}"
            )

        next_pending_app = self._find_next_pending_application(application_records)
        if next_pending_app is not None:
            if not next_pending_app.task_plan:
                raise ValueError(
                    f"Pending application {next_pending_app.idea_id} has no task_plan for RESOURCE_CHECK"
                )
            should_transition = self._validate_session_state_transition(
                session.state,
                WorkflowState.RESOURCE_CHECK,
            )
            session.active_application_id = next_pending_app.idea_id
            logger.info(
                f"Advance hook: switch planning to next pending application {next_pending_app.idea_id} "
                f"for session {session.id}"
            )
            await self.memory_manager.store_session(session)
            if should_transition:
                await self._update_session_state(session, WorkflowState.RESOURCE_CHECK)
            else:
                logger.info(
                    f"Advance hook: continue RESOURCE_CHECK with application {next_pending_app.idea_id} "
                    f"for session {session.id}"
                )
            return

        if not self._all_applications_finished_planning(application_records):
            raise ValueError(
                "Cannot enter batch IDEA_EVALUATION while some applications remain pending or replanning"
            )

        # no_refinement 模式：planning 完成即终止，不进入 IDEA_EVALUATION
        if not self.enable_plan_refinement_phase:
            self._validate_session_state_transition(
                session.state,
                WorkflowState.COMPLETED,
            )
            session.active_application_id = None
            logger.info(
                f"Advance hook: planning finished under no_refinement mode for session {session.id}; "
                f"marking session COMPLETED without IDEA_EVALUATION"
            )
            await self.memory_manager.store_session(session)
            await self._update_session_state(session, WorkflowState.COMPLETED)
            return

        next_eval_app = self._find_next_application_pending_evaluation(application_records)
        if next_eval_app is None:
            raise ValueError(
                "Planning finished but no application is available for IDEA_EVALUATION"
            )

        self._validate_session_state_transition(
            session.state,
            WorkflowState.IDEA_EVALUATION,
        )
        session.active_application_id = next_eval_app.idea_id
        logger.info(
            f"Advance hook: planning finished for session {session.id}; "
            f"enter IDEA_EVALUATION with application {next_eval_app.idea_id}"
        )
        await self.memory_manager.store_session(session)
        await self._update_session_state(session, WorkflowState.IDEA_EVALUATION)

    async def _run_idea_evaluation_phase(self, session: WorkflowSession) -> None:
        """Run single-application idea evaluation and write back application_records[active_id].idea_evaluation."""
        logger.info(f"Idea Evaluation Agent: Starting idea evaluation for session {session.id}")

        try:
            from ..agents.idea_evaluation_agent import IdeaQualityCalculator

            application_records = list(getattr(session, "application_records", []) or [])
            if application_records and isinstance(application_records[0], dict):
                application_records = [
                    ApplicationRecord.from_dict(item)
                    for item in application_records
                    if isinstance(item, dict)
                ]
                session.application_records = application_records

            active_application_id = getattr(session, "active_application_id", None)
            if not active_application_id:
                raise ValueError("IDEA_EVALUATION requires active_application_id")

            current_app = next(
                (app for app in application_records if app.idea_id == active_application_id),
                None,
            )
            if current_app is None:
                raise ValueError(
                    f"Active application {active_application_id} not found in application_records"
                )

            idea_evaluation_agent = self._get_agent("idea_evaluation")
            if not idea_evaluation_agent:
                raise ValueError("Idea evaluation agent initialization failed")
            allowed_statuses = {"ready_for_evaluation", "failed_before_evaluation"}
            if current_app.status not in allowed_statuses:
                raise ValueError(
                    "IDEA_EVALUATION may only evaluate applications in "
                    + " or ".join(sorted(allowed_statuses))
                    + f" status, got {current_app.status}"
                )
            context = {
                "application": current_app.to_dict(),
                "task_plan": current_app.task_plan,
                "dataset_info": (
                    current_app.resource_check.get("dataset_info", {})
                    if isinstance(current_app.resource_check, dict)
                    and current_app.resource_check.get("dataset_info")
                    else session.task.spatial_extent
                ),
                "dataset_portrait": getattr(session, "dataset_portrait", {}) or {},
                "resource_check": current_app.resource_check,
                "literature_review": getattr(session, "literature_review", []) or [],
            }
            agent_result = await idea_evaluation_agent.execute(context, {})
            evaluated_result = IdeaQualityCalculator.build_result(
                idea_id=current_app.idea_id,
                evaluation_stage=agent_result.get("evaluation_stage", ""),
                dimension_judgments=agent_result.get("dimension_judgments", {}),
            )
            current_app.idea_evaluation = evaluated_result.to_dict()
            self._append_idea_evaluation_history(
                current_app,
                stage="evaluated",
                payload={
                    "evaluation_version": agent_result.get("evaluation_version", ""),
                    "evaluation_stage": agent_result.get("evaluation_stage", ""),
                    "dimension_judgments": agent_result.get("dimension_judgments", {}),
                    "final_result": evaluated_result.to_dict(),
                },
            )
            if current_app.status == "ready_for_evaluation":
                current_app.status = "evaluated"
            await self.memory_manager.store_session(session)
            await self._advance_after_application_terminal_state(session)

        except Exception as e:
            logger.error(f"Error in idea evaluation phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)
