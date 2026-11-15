"""
Memory Manager for DataGenHyp

This module implements the persistent context storage for session data,
hypotheses, and other information that needs to be shared across agents
and retained between system restarts.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryManager(ABC):
    """
    Abstract base class for memory management.
    """
    
    def open_research_memory(self, scope: str, *, directory=None):
        """Open one session/case namespace; storage is distinct from prompt selection."""
        from .research_memory import ResearchMemory, validate_scope
        validate_scope(scope)
        if not hasattr(self, "_research_memories"):
            self._research_memories = {}
        key = (scope, str(directory))
        if key not in self._research_memories:
            self._research_memories[key] = ResearchMemory(scope, directory)
        return self._research_memories[key]

    @abstractmethod
    async def startup(self) -> None:
        """
        Initialize and start the memory manager.
        This method should be called before using the memory manager.
        """
        pass
    
    @abstractmethod
    async def load_session(self, session_id: str) -> Any:
        """
        Load a workflow session from storage.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Loaded session object
        """
        pass
    
    @abstractmethod
    async def store_session(self, session: Any) -> None:
        """
        Store a workflow session.
        
        Args:
            session: Workflow session object
        """
        pass
    
    @abstractmethod
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data to persistent storage.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to store
        """
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data from storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or empty dict if not found
        """
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions in storage.
        
        Returns:
            List of session summaries
        """
        pass
    
    
    @abstractmethod
    async def save_hypothesis(self, hypothesis_id: str, hypothesis_data: Dict[str, Any]) -> None:
        """
        Save a hypothesis to persistent storage.
        
        Args:
            hypothesis_id: Unique hypothesis identifier
            hypothesis_data: Hypothesis data to store
        """
        pass

    @abstractmethod
    async def get_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Retrieve a hypothesis from storage.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            Hypothesis data or empty dict if not found
        """
        pass
    
    @abstractmethod
    async def get_hypotheses_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all hypotheses for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of hypothesis data
        """
        pass
    
    @abstractmethod
    async def save_finding(self, session_id: str, finding_data: Dict[str, Any]) -> None:
        """
        Save a finding to persistent storage.
        
        Note: Findings are part of a session, so session_id is required.
        The finding_data should contain the finding_id in its 'id' field.
            
        Args:
            session_id: Session identifier (findings belong to a session)
            finding_data: Finding data to store (Dict representation of Finding object, must contain 'id' field)
        """
        pass

    @abstractmethod
    async def get_finding(self, session_id: str, finding_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a finding by session and finding ID.
        
        Args:
            session_id: Session identifier
            finding_id: Finding identifier
            
        Returns:
            Finding data dict or None if not found
        """
        pass

    @abstractmethod
    async def get_findings_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all findings for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of finding data dicts
        """
        pass

    @abstractmethod
    async def update_finding_score(self, session_id: str, finding_id: str, scores: Dict[str, float]) -> bool:
        """
        Update scores for a specific finding.
        
        Args:
            session_id: Session identifier
            finding_id: Finding identifier
            scores: Dictionary of scores (e.g., novelty, scientific_value)
            
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def add_finding_evaluation_comments(self, session_id: str, finding_id: str, comments: str) -> bool:
        """
        Add evaluation comments to a finding.
        
        Args:
            session_id: Session identifier
            finding_id: Finding identifier
            comments: The comments string to add
            
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def update_hypothesis_score(self, hypothesis_id: str, scores: Dict[str, float]) -> bool:
        """
        Update scores for a specific hypothesis.
        Requirement: [cite: 48] update_hypothesis_score(hypothesis_id: str, scores: Dict) -> bool
        
        Args:
            hypothesis_id: hypothesis identifier
            scores: Dictionary of scores (e.g., novelty, scientific_value)
            
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def add_evaluation_comments(self, hypothesis_id: str, comments: str) -> bool:
        """
        Add an evaluation comments to a hypothesis.
        Requirement: [cite: 49] add_evaluation_comments(hypothesis_id: str, comments: str) -> bool
        
        Args:
            hypothesis_id: hypothesis identifier
            comments: The comments string to add
            
        Returns:
            True if successful
        """
        pass


class InMemoryMemoryManager(MemoryManager):
    """
    In-memory implementation of the memory manager.
    
    Warning: Data is lost when the application restarts.
    """
    
    def __init__(self):
        """Initialize the in-memory memory manager."""
        self.sessions = {}
        self.hypotheses = {}
    
    async def startup(self) -> None:
        """
        Initialize and start the in-memory memory manager.
        """
        pass
    
    async def load_session(self, session_id: str) -> Any:
        """
        Load a workflow session from memory.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Loaded session object
        """
        session_data = await self.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
        
        # Here you would typically reconstruct a session object from the data
        # For now, we'll just return the session data dictionary
        return session_data
    
    async def store_session(self, session: Any) -> None:
        """
        Store a workflow session in memory.
        
        Args:
            session: Workflow session object
        """
        # Convert session to dictionary and store it
        session_data = session.to_dict() if hasattr(session, 'to_dict') else session
        await self.save_session(session.id if hasattr(session, 'id') else session_data.get('id'), session_data)
    
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data to in-memory storage.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to store
        """
        self.sessions[session_id] = session_data.copy()
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data from in-memory storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or empty dict if not found
        """
        return self.sessions.get(session_id, {}).copy()
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from in-memory storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            # Also delete associated hypotheses
            self.hypotheses = {k: v for k, v in self.hypotheses.items() 
                              if v.get("session_id") != session_id}
            return True
        return False
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions in in-memory storage.
        
        Returns:
            List of session summaries
        """
        return [
            {
                "id": session_id,
                "research_goal": session.get("research_goal", ""),
                "status": session.get("status", "unknown"),
                "created_at": session.get("start_time", 0),
                "updated_at": session.get("update_time", 0),
                "hypothesis_count": len([h for h in self.hypotheses.values() 
                                       if h.get("session_id") == session_id])
            }
            for session_id, session in self.sessions.items()
        ]
    
    async def save_finding(self, session_id: str, finding_data: Dict[str, Any]) -> None:
        """
        Save a finding to in-memory storage.
        
        Args:
            session_id: Session identifier (findings belong to a session)
            finding_data: Finding data to store (must contain 'id' field)
        """
        finding_id = finding_data.get("id")
        if not finding_id:
            logger.error("Finding data must contain 'id' field")
            return

        # 只在调用方没标注类型时才默认成 finding。写死 "finding" 会覆盖
        # save_hypothesis 刚设的 "hypothesis"——而 get_hypothesis 恰好过滤掉
        # _type == "finding" 的条目，于是经 save_hypothesis 存的数据一律取不回来。
        finding_data.setdefault("_type", "finding")
        finding_data["session_id"] = session_id  # Ensure session_id is set
        # Reuse hypotheses storage (unified storage)
        self.hypotheses[finding_id] = finding_data.copy()

    async def get_finding(self, session_id: str, finding_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a finding from in-memory storage.
        
        Args:
            session_id: Session identifier
            finding_id: Finding identifier
            
        Returns:
            Finding data or None if not found
        """
        item = self.hypotheses.get(finding_id, {})
        # Check if it belongs to the session and is a finding type
        if item.get("session_id") == session_id and item.get("_type") == "finding":
            return item.copy()
        return None

    async def get_findings_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all findings for a session from in-memory storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of finding data
        """
        return [h.copy() for h in self.hypotheses.values() 
                if h.get("session_id") == session_id and h.get("_type") == "finding"]

    async def update_finding_score(self, session_id: str, finding_id: str, scores: Dict[str, float]) -> bool:
        """
        Update scores for a specific finding.
        
        Args:
            session_id: Session identifier
            finding_id: Finding identifier
            scores: Dictionary of scores
        """
        # Verify the finding belongs to the session
        finding = await self.get_finding(session_id, finding_id)
        if not finding:
            return False
        
        # Reuse update_hypothesis_score logic (it works with finding_id)
        return await self.update_hypothesis_score(finding_id, scores)

    async def add_finding_evaluation_comments(self, session_id: str, finding_id: str, comments: str) -> bool:
        """
        Add evaluation comments to a finding.
        
        Args:
            session_id: Session identifier
            finding_id: Finding identifier
            comments: The comments string to add
        """
        # Verify the finding belongs to the session
        finding = await self.get_finding(session_id, finding_id)
        if not finding:
            return False
        
        # Reuse add_evaluation_comments logic (it works with finding_id)
        return await self.add_evaluation_comments(finding_id, comments)
    
    async def save_hypothesis(self, hypothesis_id: str, hypothesis_data: Dict[str, Any]) -> None:
        """
        Save a hypothesis to in-memory storage.
        
        Note: This method is kept for backward compatibility.
        For new code, use save_finding() instead.
        Internally, it calls save_finding() to maintain consistency.
        """
        session_id = hypothesis_data.get("session_id")
        if not session_id:
            logger.error(f"No session_id found for hypothesis {hypothesis_id}")
            return
        
        hypothesis_data["_type"] = "hypothesis"
        await self.save_finding(session_id, hypothesis_data)
    
    async def get_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Retrieve a hypothesis from in-memory storage.
        
        Note: This method is kept for backward compatibility.
        For new code, use get_finding() instead.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            Hypothesis data or empty dict if not found
        """
        item = self.hypotheses.get(hypothesis_id, {})
        # Return only if it's not a finding
        if item.get("_type") != "finding":
            return item.copy()
        return {}
    
    async def get_hypotheses_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all hypotheses for a session from in-memory storage.
        
        Note: This method is kept for backward compatibility.
        For new code, use get_findings_by_session() instead.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of hypothesis data
        """
        return [h.copy() for h in self.hypotheses.values() 
                if h.get("session_id") == session_id and h.get("_type") != "finding"]
    

    async def update_hypothesis_score(self, hypothesis_id: str, scores: Dict[str, float]) -> bool:
        """
        Update scores for a specific hypothesis.
        """
        if not hypothesis_id or not isinstance(scores, dict):
            return False

        hypo = self.hypotheses.get(hypothesis_id)
        if not hypo:
            return False

        # merge scores
        existing_scores = hypo.get("scores")
        if not isinstance(existing_scores, dict):
            existing_scores = {}

        for k, v in scores.items():
            if isinstance(v, (int, float)):
                existing_scores[k] = float(v)

        hypo["scores"] = existing_scores
        hypo["update_time"] = time.time()

        self.hypotheses[hypothesis_id] = hypo.copy()
        return True

    async def add_evaluation_comments(self, hypothesis_id: str, comments: str) -> bool:
        """
        Add an evaluation comments to a hypothesis.
        """
        if not hypothesis_id or not isinstance(comments, str) or not comments.strip():
            return False

        hypo = self.hypotheses.get(hypothesis_id)
        if not hypo:
            return False

        existing_comments = hypo.get("comments")
        if not isinstance(existing_comments, list):
            existing_comments = []

        existing_comments.append(comments.strip())
        hypo["comments"] = existing_comments
        hypo["update_time"] = time.time()

        self.hypotheses[hypothesis_id] = hypo.copy()
        return True



class FileSystemMemoryManager(MemoryManager):
    """
    File system implementation of the memory manager.
    
    Data is stored as JSON files in the specified directory.
    """
    
    def open_research_memory(self, scope: str, *, directory=None):
        from pathlib import Path
        from .research_memory import validate_scope
        validate_scope(scope)
        target = Path(self._get_task_dir()) / "research_memory" / scope if directory is None else Path(directory)
        return super().open_research_memory(scope, directory=target)

    def __init__(self, data_dir: str = "results", task_name: str = None):
        """
        Initialize the file system memory manager.

        Args:
            data_dir: Directory to store data files
            task_name: Name of the task (from --task parameter)
        """
        self.data_dir = data_dir
        self.task_name = task_name or "DefaultTask"

        # Ensure base directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_task_dir(self, session_data: Dict[str, Any] = None) -> str:
        """
        Get the task directory using the provided task name.

        Args:
            session_data: Session data (unused, kept for compatibility)

        Returns:
            Task directory path
        """
        task_dir = os.path.join(self.data_dir, self.task_name)
        os.makedirs(task_dir, exist_ok=True)
        return task_dir

    def _reconstruct_item(self, item_data: Dict[str, Any]):
        """
        Reconstruct a Finding object from dictionary data.
        
        
        Args:
            item_data: Dictionary representation of the Finding
            
        Returns:
            Finding object
        """
        from ..workflow.data_type import Finding
        
        try:
            return Finding.from_dict(item_data)
        except Exception as e:
            logger.warning(f"Failed to load as Finding: {e}, creating minimal Finding")
            return Finding(
                id=item_data.get("id", "unknown"),
                text=item_data.get("text", item_data.get("description", ""))
            )



    async def startup(self) -> None:
        """
        Initialize and start the file system memory manager.
        """
        pass
    
    async def load_session(self, session_id: str) -> Any:
        """
        Load a workflow session from the file system.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Loaded session object
        """
        session_data = await self.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
        
        # Import here to avoid circular imports
        from ..workflow.data_type import (
            ApplicationRecord,
            Task,
            WorkflowSession,
            WorkflowState,
        )
        
        # Reconstruct the task object
        task_data = session_data.get("task", session_data.get("goal", {}))
        task = Task.from_dict(task_data)
        
        # Reconstruct findings (use the canonical 'findings' field only)
        findings = []
        for f_data in session_data.get("findings", []):
            item = self._reconstruct_item(f_data)
            findings.append(item)

        application_records = []
        for record_data in session_data.get("application_records", []):
            if isinstance(record_data, dict):
                application_records.append(ApplicationRecord.from_dict(record_data))
        
        # Convert state string to enum
        state_str = session_data.get("state", "initial")
        state = WorkflowState(state_str)
        
        # Reconstruct the session object
        session = WorkflowSession(
            id=session_data.get("id"),
            task=task,
            findings=findings,
            iterations_completed=session_data.get("iterations_completed", 0),
            max_iterations=session_data.get("max_iterations", 2),
            state=state,
            feedback_history=session_data.get("feedback_history", []),
            top_findings=session_data.get("top_findings", []),
            tool_usage=session_data.get("tool_usage", {}),
            # Support new fields for geography discovery workflow
            literature_review=session_data.get("literature_review", []),
            literature_review_raw=session_data.get("literature_review_raw", session_data.get("literature_review", [])),
            literature_filter_params=session_data.get("literature_filter_params", {}),
            survey_results=session_data.get("survey_results", {}),
            domain_knowledge=session_data.get("domain_knowledge", {}),
            dataset_portrait=session_data.get("dataset_portrait", {}),
            application_records=application_records,
            active_application_id=session_data.get("active_application_id"),
            planning_audit=session_data.get("planning_audit", {}),
        )
        
        # Set timestamps if available
        if "started_at" in session_data and session_data["started_at"]:
            session.started_at = datetime.fromisoformat(session_data["started_at"])
        
        if "completed_at" in session_data and session_data["completed_at"]:
            session.completed_at = datetime.fromisoformat(session_data["completed_at"])
        
        return session
    
    async def store_session(self, session: Any) -> None:
        """
        Store a workflow session in the file system.
        
        Args:
            session: Workflow session object
        """
        # Convert session to dictionary and store it
        session_data = session.to_dict() if hasattr(session, 'to_dict') else session
        await self.save_session(session.id if hasattr(session, 'id') else session_data.get('id'), session_data)
    
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data to file system.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to store
        """
        # Add update timestamp
        session_data["update_time"] = time.time()

        # Get dynamic task directory based on session data
        task_dir = self._get_task_dir(session_data)

        # Create session subdirectory: results/{task_name}/traj_{session_id}/
        traj_dir = os.path.join(task_dir, f"traj_{session_id}")
        os.makedirs(traj_dir, exist_ok=True)

        # Save to file using new path structure: results/{task_name}/traj_{session_id}/traj_{session_id}.json
        file_path = os.path.join(traj_dir, f"traj_{session_id}.json")
        await self._write_json_file(file_path, session_data)
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data from file system.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or empty dict if not found
        """
        task_dir = os.path.join(self.data_dir, self.task_name)
        traj_dir = os.path.join(task_dir, f"traj_{session_id}")
        file_path = os.path.join(traj_dir, f"traj_{session_id}.json")
        return await self._read_json_file(file_path)
    
    def get_session_file_path(self, session_id: str) -> str:
        """
        Get the file path for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Full path to the session file
        """
        task_dir = os.path.join(self.data_dir, self.task_name)
        traj_dir = os.path.join(task_dir, f"traj_{session_id}")
        return os.path.join(traj_dir, f"traj_{session_id}.json")
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from file system.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        
        task_dir = os.path.join(self.data_dir, self.task_name)
        traj_dir = os.path.join(task_dir, f"traj_{session_id}")
        # 会话的 finding 落在 idea_{session_id}.json，与 traj 目录同级。只删 traj 目录会
        # 留下孤儿 finding；InMemoryMemoryManager.delete_session 已级联删除关联 finding，
        # 两个实现共用同一抽象，此处保持一致。
        findings_path = os.path.join(task_dir, f"idea_{session_id}.json")

        try:
            if not os.path.exists(traj_dir):
                return False
            import shutil
            shutil.rmtree(traj_dir)
            if os.path.exists(findings_path):
                os.remove(findings_path)
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions in file system.
        
        Returns:
            List of session summaries
        """
        sessions = []

        try:
            task_path = os.path.join(self.data_dir, self.task_name)
            if not os.path.exists(task_path):
                return sessions

            # Look for traj_{session_id} directories
            session_dirs = [d for d in os.listdir(task_path) if d.startswith("traj_") and os.path.isdir(os.path.join(task_path, d))]

            for dir_name in session_dirs:
                try:
                    session_id = dir_name.replace("traj_", "")
                    file_path = os.path.join(task_path, dir_name, f"traj_{session_id}.json")
                    session = await self._read_json_file(file_path)
                    
                    # Get finding count for this session (based on findings stored in the session)
                    finding_count = len(session.get("findings", []))
                    
                    sessions.append({
                        "id": session_id,
                        "task_name": self.task_name,
                        "research_goal": session.get("research_goal", ""),
                        "status": session.get("status", "unknown"),
                        "created_at": session.get("start_time", 0),
                        "updated_at": session.get("update_time", 0),
                        "finding_count": finding_count
                    })
                except Exception as e:
                    logger.error(f"Error reading session file {dir_name}: {e}")
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")

        return sessions
    
    async def save_finding(self, session_id: str, finding_data: Dict[str, Any]) -> None:
        """
        Save a finding to file system.
        Note: Findings are stored per session in idea_{session_id}.json format.

        Args:
            session_id: Session identifier (findings belong to a session)
            finding_data: Finding data to store (must contain 'id' field)
        """
        finding_id = finding_data.get("id")
        if not finding_id:
            logger.error("Finding data must contain 'id' field")
            return

        task_dir = os.path.join(self.data_dir, self.task_name)
        # Keep using idea_ prefix for file name, but the internal key is now 'findings'
        findings_path = os.path.join(task_dir, f"idea_{session_id}.json")
        existing_data = await self._read_json_file(findings_path)
        if not existing_data:
            existing_data = {"findings": [], "session_id": session_id, "data_type": "finding"}
        
        finding_data["update_time"] = time.time()
        # 与 InMemory 实现同理：只在未标注时默认成 finding，否则会覆盖
        # save_hypothesis 设的 "hypothesis"，导致 get_hypothesis 取不回数据。
        finding_data.setdefault("_type", "finding")
        finding_data["session_id"] = session_id  # Ensure session_id is set
        
        findings_list = existing_data.get("findings", [])
        if not isinstance(findings_list, list):
            findings_list = []
        for i, item in enumerate(findings_list):
            if item.get("id") == finding_id:
                findings_list[i] = finding_data
                break
        else:
            findings_list.append(finding_data)
        
        existing_data["findings"] = findings_list
        existing_data["data_type"] = "finding"  # Mark entire file as finding type
        await self._write_json_file(findings_path, existing_data)
    
    async def save_hypothesis(self, hypothesis_id: str, hypothesis_data: Dict[str, Any]) -> None:
        """
        Save a hypothesis to file system.
        
        Note: This method is kept for backward compatibility.
        For new code, use save_finding() instead.
        Internally, it calls save_finding() to maintain consistency.
        """
        session_id = hypothesis_data.get("session_id")
        if not session_id:
            logger.error(f"No session_id found for hypothesis {hypothesis_id}")
            return

        # Mark as hypothesis type (backward compatibility)
        hypothesis_data["_type"] = "hypothesis"
        # Internally call save_finding for unified implementation
        await self.save_finding(session_id, hypothesis_data)
    
    async def get_finding(self, session_id: str, finding_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a finding from file system.

        Args:
            session_id: Session identifier
            finding_id: Finding identifier

        Returns:
            Finding data or None if not found
        """
        task_dir = os.path.join(self.data_dir, self.task_name)
        if not os.path.exists(task_dir):
            return None
        
        findings_path = os.path.join(task_dir, f"idea_{session_id}.json")
        data = await self._read_json_file(findings_path)
        
        for item in data.get("findings", []):
            if item.get("id") == finding_id and item.get("_type") == "finding":
                return item
        
        return None
    
    async def get_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Retrieve a hypothesis from file system.

        Note: This method is kept for backward compatibility.
        For new code, use get_finding() instead.

        Args:
            hypothesis_id: Hypothesis identifier

        Returns:
            Hypothesis data or empty dict if not found
        """
        # Search through all idea_ files (backward compatible)
        task_dir = os.path.join(self.data_dir, self.task_name)
        if not os.path.exists(task_dir):
            return {}
        idea_files = [f for f in os.listdir(task_dir) if f.startswith("idea_") and f.endswith(".json")]
        
        for file_name in idea_files:
            try:
                file_path = os.path.join(task_dir, file_name)
                data = await self._read_json_file(file_path)
                for item in data.get("findings", []):
                    if item.get("id") == hypothesis_id and item.get("_type") != "finding":
                        return item
            except Exception as e:
                logger.error(f"Error reading file {file_name}: {e}")

        return {}
    
    async def get_findings_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all findings for a session from file system.

        Args:
            session_id: Session identifier

        Returns:
            List of finding data
        """
        task_dir = os.path.join(self.data_dir, self.task_name)
        findings_path = os.path.join(task_dir, f"idea_{session_id}.json")
        data = await self._read_json_file(findings_path)
        
        # If file is marked as finding type, return all items
        if data.get("data_type") == "finding":
            return data.get("findings", [])
        
        # Otherwise filter out finding-type items
        findings = []
        for item in data.get("findings", []):
            if item.get("_type") == "finding":
                findings.append(item)
        
        return findings
    
    async def get_hypotheses_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all hypotheses for a session from file system.

        Note: This method is kept for backward compatibility.
        For new code, use get_findings_by_session() instead.

        Args:
            session_id: Session identifier

        Returns:
            List of hypothesis data
        """
        task_dir = os.path.join(self.data_dir, self.task_name)
        findings_path = os.path.join(task_dir, f"idea_{session_id}.json")
        data = await self._read_json_file(findings_path)
        
        # Return all items that are not findings (legacy hypothesis view)
        hypotheses = []
        for item in data.get("findings", []):
            if item.get("_type") != "finding":
                hypotheses.append(item)
        
        return hypotheses
    
    async def _write_json_file(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Write data to a JSON file asynchronously.
        
        Args:
            file_path: Path to the file
            data: Data to write
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_json_sync, file_path, data)
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            raise
    
    def _normalize_json_compatible(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._normalize_json_compatible(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize_json_compatible(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize_json_compatible(item) for item in value]
        if isinstance(value, set):
            return [self._normalize_json_compatible(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()

        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return self._normalize_json_compatible(item_method())
            except Exception:
                pass

        return value

    def _write_json_sync(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Synchronous implementation of JSON file writing.
        
        Args:
            file_path: Path to the file
            data: Data to write
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        normalized_data = self._normalize_json_compatible(data)
        with open(file_path, 'w') as f:
            json.dump(normalized_data, f, indent=2)
    
    async def _read_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read data from a JSON file asynchronously.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Data from the file or empty dict if file doesn't exist
        """
        if not os.path.exists(file_path):
            return {}
            
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._read_json_sync, file_path)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {}
    
    def _read_json_sync(self, file_path: str) -> Dict[str, Any]:
        """
        Synchronous implementation of JSON file reading.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Data from the file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    async def update_finding_score(self, session_id: str, finding_id: str, scores: Dict[str, float]) -> bool:
        """
        Update scores for a specific finding (stored inside idea_{session_id}.json).
        
        Args:
            session_id: Session identifier
            finding_id: Finding identifier
            scores: Dictionary of scores
        """
        # Verify the finding belongs to the session
        finding = await self.get_finding(session_id, finding_id)
        if not finding:
            return False
        
        # Reuse update_hypothesis_score logic
        return await self.update_hypothesis_score(finding_id, scores)

    async def add_finding_evaluation_comments(self, session_id: str, finding_id: str, comments: str) -> bool:
        """
        Add evaluation comments to a finding (stored inside idea_{session_id}.json).
        
        Args:
            session_id: Session identifier
            finding_id: Finding identifier
            comments: The comments string to add
        """
        # Verify the finding belongs to the session
        finding = await self.get_finding(session_id, finding_id)
        if not finding:
            return False
        
        # Reuse add_evaluation_comments logic
        return await self.add_evaluation_comments(finding_id, comments)
        
    async def update_hypothesis_score(self, hypothesis_id: str, scores: Dict[str, float]) -> bool:
        """
        Update scores for a specific hypothesis (stored inside idea_{session_id}.json).
        
        Note: This method is kept for backward compatibility.
        For new code, use update_finding_score() instead.
        """
        if not hypothesis_id or not isinstance(scores, dict):
            return False

        task_dir = os.path.join(self.data_dir, self.task_name)
        if not os.path.exists(task_dir):
            return False

        # Use idea_ prefix (files), but operate on 'findings' records
        idea_files = [f for f in os.listdir(task_dir) if f.startswith("idea_") and f.endswith(".json")]
        
        for file_name in idea_files:
            file_path = os.path.join(task_dir, file_name)
            try:
                data = await self._read_json_file(file_path)
                findings_list = data.get("findings", [])
                if not isinstance(findings_list, list):
                    continue
                
                for i, item in enumerate(findings_list):
                    if isinstance(item, dict) and item.get("id") == hypothesis_id:
                        existing_scores = item.get("scores")
                        if not isinstance(existing_scores, dict):
                            existing_scores = {}

                        for k, v in scores.items():
                            if isinstance(v, (int, float)):
                                existing_scores[k] = float(v)

                        item["scores"] = existing_scores
                        item["update_time"] = time.time()
                        findings_list[i] = item
                        
                        data["findings"] = findings_list
                        await self._write_json_file(file_path, data)
                        return True
            except Exception as e:
                logger.error(f"Error updating hypothesis score in {file_name}: {e}")

        return False

    async def add_evaluation_comments(self, hypothesis_id: str, comments: str) -> bool:
        """
        Add an evaluation comments to a hypothesis (stored inside idea_{session_id}.json).
        
        Note: This method is kept for backward compatibility.
        For new code, use add_finding_evaluation_comments() instead.
        """
        if not hypothesis_id or not isinstance(comments, str) or not comments.strip():
            return False

        task_dir = os.path.join(self.data_dir, self.task_name)
        if not os.path.exists(task_dir):
            return False

        # Use idea_ prefix (backward compatible file naming), operate on 'findings'
        idea_files = [f for f in os.listdir(task_dir) if f.startswith("idea_") and f.endswith(".json")]
        
        for file_name in idea_files:
            file_path = os.path.join(task_dir, file_name)
            try:
                data = await self._read_json_file(file_path)
                findings_list = data.get("findings", [])
                if not isinstance(findings_list, list):
                    continue
                
                for i, item in enumerate(findings_list):
                    if isinstance(item, dict) and item.get("id") == hypothesis_id:
                        item_comments = item.get("comments", [])
                        if not isinstance(item_comments, list):
                            item_comments = []

                        item_comments.append(comments.strip())
                        item["comments"] = item_comments
                        item["update_time"] = time.time()
                        findings_list[i] = item
                        
                        data["findings"] = findings_list
                        await self._write_json_file(file_path, data)
                        return True
            except Exception as e:
                logger.error(f"Error adding evaluation comments in {file_name}: {e}")

        return False

def create_memory_manager(config: Dict[str, Any]) -> MemoryManager:
    """
    Factory function to create the appropriate memory manager based on configuration.

    Args:
        config: Debug configuration dictionary

    Returns:
        Configured MemoryManager instance
    """
    backend = (config.get("memory") or {}).get("backend", "file_system")

    if backend == "file_system":
        file_dir = (config.get("memory") or {}).get("file_dir", "results")
        task_name = config.get("task_name", "DefaultTask")
        return FileSystemMemoryManager(
            data_dir=file_dir,
            task_name=task_name
        )
    else:
        # Default to in-memory
        return InMemoryMemoryManager() 
