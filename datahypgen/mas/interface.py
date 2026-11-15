"""
DataGenHyp Interface

This module implements the main interface for the DataGenHyp system.
The DataGenHypInterface manages system lifecycle, session coordination,
and provides a clean API for research workflow interactions.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dotenv import dotenv_values, load_dotenv

from .models.model_factory import ModelFactory
from .agents.agent_factory import AgentFactory
from .memory.memory_manager import MemoryManager
from .workflow.orchestration_agent import OrchestrationAgent
from .workflow.data_type import WorkflowState

logger = logging.getLogger(__name__)


class DataGenHypInterface:
    """
    Main interface for the DataGenHyp system.

    This class provides a high-level API for managing the research workflow system,
    including session management, orchestration agent coordination, and system lifecycle.
    It serves as the primary entry point for external applications.
    """

    def __init__(self, config_path: str = None, config: Dict[str, Any] = None, work_dir: str = 'example', task_name: str = None):
        """
        Initialize the DataGenHyp interface.

        Args:
            config_path: Path to the configuration file
            config: Configuration dictionary (takes precedence over config_path)
            work_dir: Working directory for the system
            task_name: Name of the task (from --task parameter)
        """
        self.work_dir = work_dir
        self.task_name = task_name or "DefaultTask"
        self._load_runtime_env(config_path=config_path, work_dir=work_dir)
        self.config = self._load_config(config_path, config)
        self.config['config_path'] = config_path
        self.config['work_dir'] = self.work_dir
        self.config['task_name'] = self.task_name

        # Core components
        self.model_factory = ModelFactory()
        self.memory_manager = self._init_memory_manager()
        self.agent_factory = AgentFactory()

        # Create specialized agents
        self.agents = self.agent_factory.create_all_agents(
            config=self.config,
            model_factory=self.model_factory
        )

        # Initialize orchestration agent
        self.orchestration_agent = self._init_orchestration_agent()

        # Session tracking
        self.active_sessions = {}

        # System state
        self.system_ready = False

        logger.info("DataGenHyp interface initialized")

    async def startup(self) -> None:
        """Initialize system components and mark as ready."""
        try:
            await self.memory_manager.startup()
            self.system_ready = True
            logger.info("DataGenHyp system started successfully")
        except Exception as e:
            self.system_ready = False
            logger.error(f"Error during system startup: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Clean shutdown of system components."""
        try:
            if hasattr(self.memory_manager, 'shutdown'):
                await self.memory_manager.shutdown()
            self.system_ready = False
            logger.info("DataGenHyp system shut down successfully")
        except Exception as e:
            logger.error(f"Error during system shutdown: {str(e)}")
            raise

    async def create_session(self,
                           task_description: str,
                           background: str = "",
                           spatial_extent: Dict[str, Any] = None,
                           data_requirements: List[str] = None,
                           analysis_methods: List[str] = None) -> str:
        """
        Create a new research session.

        Args:
            task_description: Description of the research goal
            background: Background information for context
            spatial_extent: Geographic extent information (for geography tasks)
            data_requirements: Initial data requirements (can be empty, will be determined by agents)
            analysis_methods: Initial analysis methods (can be empty, will be determined by agents)

        Returns:
            Session ID for the created session

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()

        session_id = await self.orchestration_agent.create_session(
            task_description=task_description,
            background=background,
            spatial_extent=spatial_extent or {},
            data_requirements=data_requirements or [],
            analysis_methods=analysis_methods or []
        )

        # Track session
        self._track_session(session_id, task_description)

        logger.info(f"Created session {session_id}: {task_description}")
        return session_id

    async def run_session(self,
                        session_id: str,
                        status_callback: Optional[Callable[[str, str, str], None]] = None) -> Dict[str, Any]:
        """
        Execute a research session.

        Args:
            session_id: ID of the session to run
            status_callback: Optional callback for state change notifications
                           Signature: (session_id, old_state, new_state) -> None

        Returns:
            Current session status

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()

        # Create state change handler
        async def on_state_change(session, old_state, new_state):
            self._update_session_tracking(session_id, new_state.value)

            if status_callback:
                try:
                    await status_callback(session_id, old_state.value, new_state.value)
                except Exception as e:
                    logger.error(f"Error in status callback: {str(e)}")

        # Execute session
        await self.orchestration_agent.run_session(
            session_id=session_id,
            on_state_change=on_state_change
        )

        return await self.get_session_status(session_id)

    async def add_feedback(self,
                         session_id: str,
                         feedback: dict,
                         target_finding_ids: List[str] = None,
                         auto_resume: bool = True) -> Dict[str, Any]:
        """
        Add feedback to a session and optionally resume execution.

        Args:
            session_id: Session ID
            feedback: Feedback content
            target_finding_ids: Specific findings the feedback applies to
            auto_resume: Whether to automatically resume session after feedback

        Returns:
            Updated session status

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()

        await self.orchestration_agent.add_feedback(
            session_id=session_id,
            feedback=feedback,
            target_finding_ids=target_finding_ids
        )

        self._update_session_tracking(session_id)

        # Auto-resume if requested and session is waiting for feedback
        if auto_resume:
            status = await self.get_session_status(session_id)
            if status.get("state") == WorkflowState.AWAITING_FEEDBACK.value:
                await self.resume_session(session_id)

        return await self.get_session_status(session_id)

    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """
        Resume a paused session.

        Args:
            session_id: Session ID to resume

        Returns:
            Updated session status

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()

        await self.orchestration_agent.run_session(session_id)
        self._update_session_tracking(session_id)

        return await self.get_session_status(session_id)

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current status of a session.

        Args:
            session_id: Session ID

        Returns:
            Status dictionary with session information

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()
        return await self.orchestration_agent.get_session_status(session_id)

    async def get_top_findings(self,
                               session_id: str,
                               include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Get ranked application results from a session.

        Args:
            session_id: Session ID
            include_all: Backward-compatible flag, ignored by the current pipeline.

        Returns:
            Ranked application entries

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()
        return await self.orchestration_agent.get_top_findings(
            session_id=session_id,
            include_all=include_all
        )

    async def get_all_findings(self,
                               session_id: str,
                               limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get ranked application results from a session.

        Args:
            session_id: Session ID
            limit: Maximum number of ranked entries to return

        Returns:
            Ranked application entries

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()
        return await self.orchestration_agent.get_findings(
            session_id=session_id,
            limit=limit,
            include_all=True
        )

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of all active sessions.

        Returns:
            List of session information dictionaries
        """
        return list(self.active_sessions.values())

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and status.

        Returns:
            System information dictionary
        """
        return {
            "ready": self.system_ready,
            "work_dir": self.work_dir,
            "active_sessions": len(self.active_sessions),
            "agents_loaded": len(self.agents),
            "config_version": self.config.get("version", "unknown")
        }

    # Private helper methods
    def _ensure_system_ready(self) -> None:
        """Ensure system is ready for operations."""
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")

    def _track_session(self, session_id: str, goal: str) -> None:
        """Add session to tracking."""
        self.active_sessions[session_id] = {
            "id": session_id,
            "goal": goal,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "state": WorkflowState.INITIAL.value
        }

    def _update_session_tracking(self, session_id: str, state: str = None) -> None:
        """Update session tracking information."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            if state:
                self.active_sessions[session_id]["state"] = state

    def _init_memory_manager(self) -> MemoryManager:
        """Initialize memory manager from configuration."""
        from .memory.memory_manager import create_memory_manager
        return create_memory_manager(self.config)

    def _init_orchestration_agent(self) -> OrchestrationAgent:
        """Initialize orchestration agent with all components."""
        return OrchestrationAgent(
            config=self.config,
            memory_manager=self.memory_manager,
            model_factory=self.model_factory,
            agent_registry=self.agents
        )

    def _load_config(self, config_path: Optional[str], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load configuration from file or dictionary.

        Args:
            config_path: Path to configuration file
            config: Configuration dictionary

        Returns:
            Loaded configuration dictionary
        """
        # Use provided config if available
        if config is not None:
            return config

        # Load from file if path provided and file exists
        if config_path and os.path.exists(config_path):
            logger.info(f"Attempting to load config from: {config_path}")
            try:
                if config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    logger.info(f"Successfully loaded YAML config with keys: {list(config_data.keys()) if config_data else 'None'}")
                    return config_data
                else:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    logger.info(f"Successfully loaded JSON config with keys: {list(config_data.keys()) if config_data else 'None'}")
                    return config_data
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
        elif config_path:
            logger.warning(f"Config path provided but file doesn't exist: {config_path}")
        else:
            logger.info("No config path provided")

        # Load default configuration
        default_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default_config.yaml")
        default_config_path = os.path.abspath(default_config_path)
        logger.info(f"Loading default configuration from: {default_config_path}")

        try:
            import yaml
            with open(default_config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"Successfully loaded default config with keys: {list(config_data.keys()) if config_data else 'None'}")
            return config_data
        except Exception as e:
            logger.error(f"Failed to load default config from {default_config_path}: {str(e)}")
            raise RuntimeError(f"Could not load default configuration: {str(e)}")

    def _load_runtime_env(self, config_path: Optional[str], work_dir: str) -> None:
        """
        Load .env files before model/agent initialization so all providers can read env vars.

        Project .env files take precedence over inherited shell environment variables.
        Among multiple candidate .env files, earlier paths in the search order win.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate_paths: List[str] = []

        if config_path:
            candidate_paths.append(
                os.path.join(os.path.dirname(os.path.abspath(config_path)), ".env")
            )

        if work_dir:
            candidate_paths.append(os.path.join(os.path.abspath(work_dir), ".env"))

        candidate_paths.extend(
            [
                os.path.join(os.getcwd(), ".env"),
                os.path.join(project_root, ".env"),
            ]
        )

        loaded_any = False
        seen_paths = set()
        loaded_keys = set()
        for env_path in candidate_paths:
            normalized_path = os.path.abspath(env_path)
            if normalized_path in seen_paths or not os.path.exists(normalized_path):
                continue
            seen_paths.add(normalized_path)
            env_values = dotenv_values(normalized_path)
            applied_any = False
            for key, value in env_values.items():
                if value is None or key in loaded_keys:
                    continue
                os.environ[key] = value
                loaded_keys.add(key)
                applied_any = True
            if applied_any:
                loaded_any = True
                logger.info(f"Loaded environment variables from {normalized_path}")

        if not loaded_any:
            logger.info("No .env file loaded during DataGenHyp initialization")
