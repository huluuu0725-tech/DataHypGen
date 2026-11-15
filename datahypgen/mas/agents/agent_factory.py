"""
Agent Factory for DataGenHyp

This module provides functionality for registering and creating specialized
agent instances based on configuration.
"""
import logging
from typing import Dict, Any, Type

logger = logging.getLogger(__name__)

from ..models.model_factory import ModelFactory
from .base_agent import BaseAgent

from .survey_agent import SurveyAgent
from .refinement_agent import RefinementAgent
from .idea_generation_agent import IdeaGenerationAgent
from .idea_maturation_agent import IdeaMaturationAgent
from .step_plan_agent import StepPlanAgent
from .resourcescheck_agent import ResourcesCheck_Agent
from .domainknowledge_agent import DomainKnowledgeAgent
from .idea_evaluation_agent import IdeaEvaluationAgent

try:
    from .dataset_profiler_agent import DatasetProfilerAgent
except ImportError as exc:
    DatasetProfilerAgent = None
    logger.debug(f"DatasetProfilerAgent unavailable during import: {exc}")


class AgentFactory:
    """
    Factory for creating agent instances based on configuration.
    
    This factory maintains a registry of available agent types
    and creates properly configured instances as needed.
    """
    
    # Registry of agent types
    _agent_registry: Dict[str, Type[BaseAgent]] = {
        "refinement": RefinementAgent,
        "survey": SurveyAgent,
        "domain_knowledge": DomainKnowledgeAgent,
        "idea_generation": IdeaGenerationAgent,
        "idea_maturation": IdeaMaturationAgent,
        "step_plan": StepPlanAgent,
        "resources_check": ResourcesCheck_Agent,
        "idea_evaluation": IdeaEvaluationAgent,
    }
    if DatasetProfilerAgent is not None:
        _agent_registry["dataset_profiler"] = DatasetProfilerAgent
    
    # Cache of created agent instances
    _agent_cache: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent type.
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class to register
        """
        if agent_type in cls._agent_registry:
            logger.warning(f"Overriding existing agent type: {agent_type}")
            
        cls._agent_registry[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
    
    @classmethod
    def create_agent(cls, 
                  agent_type: str, 
                  config: Dict[str, Any],
                  model_factory: 'ModelFactory') -> BaseAgent:
        """
        Create an agent instance of the specified type.
        
        Args:
            agent_type: Type of agent to create
            config: Debug configuration for the agent
            model_factory: ModelFactory instance for creating models
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If the agent type is not registered
        """
        # Check if agent type is registered
        if agent_type not in cls._agent_registry:
            raise ValueError(f"Agent type not registered: {agent_type}")
        
        # Create a cache key
        cache_key = cls._create_cache_key(agent_type, config)
        
        # Check if we have a cached instance
        if cache_key in cls._agent_cache:
            logger.debug(f"Using cached agent instance for {agent_type}")
            return cls._agent_cache[cache_key]
        
        # Get the agent class
        agent_class = cls._agent_registry[agent_type]

        # Create the model instance using model_provider from agent config
        model_provider = config.get("model_provider", "default")
        
        try:
            if agent_type == "dataset_profiler":
                agent = agent_class(config)
                cls._agent_cache[cache_key] = agent
                logger.info(f"Created agent instance of type: {agent_type} without model")
                return agent

            # Use the model factory to create a model for this agent
            model = model_factory.create_model_for_agent(agent_type, config)
            
            # Create the agent instance
            agent = agent_class(model, config)
            
            # Cache the instance
            cls._agent_cache[cache_key] = agent
            
            logger.info(f"Created agent instance of type: {agent_type} with provider: {model_provider}")
            return agent
        except Exception as e:
            logger.error(f"Error creating agent {agent_type}: {e}")
            raise
    
    @classmethod
    def create_all_agents(cls, 
                       config: Dict[str, Any],
                       model_factory: 'ModelFactory') -> Dict[str, BaseAgent]:
        """
        Create all configured agent instances.
        
        Args:
            config: Debug configuration dictionary with agent configurations
            model_factory: ModelFactory instance for creating models
            
        Returns:
            Dictionary mapping agent types to agent instances
        """
        agents = {}
        agent_configs = config.get("agents", {})
        logger.info(f"Creating agents from config. Found {len(agent_configs)} agent configs: {list(agent_configs.keys())}")
        
        for agent_type, agent_config in agent_configs.items():
            if agent_type in cls._agent_registry:
                try:
                    if isinstance(agent_config, dict) and agent_config.get("enabled", True) is False:
                        logger.info(f"Skipping disabled agent: {agent_type}")
                        continue

                    # Create a merged config with agent-specific settings
                    merged_config = agent_config.copy()

                    # Add reference to global config sections that might be needed
                    merged_config["_global_config"] = config

                    # 透传运行时上下文（work_dir / task_name）到 agent 顶层。
                    if "work_dir" not in merged_config:
                        merged_config["work_dir"] = config.get("work_dir", "")
                    if "task_name" not in merged_config:
                        merged_config["task_name"] = config.get("task_name", "")

                    # 透传 LLM 调用重试预算：agent 段未显式声明时用全局 workflow.agent_max_retries，
                    # 避免落到 BaseAgent 的兜底值 10（会与 replan 轮次相乘放大退避等待）。
                    if "max_retries" not in merged_config:
                        global_max_retries = (config.get("workflow") or {}).get("agent_max_retries")
                        if global_max_retries is not None:
                            merged_config["max_retries"] = global_max_retries

                    agents[agent_type] = cls.create_agent(
                        agent_type=agent_type,
                        config=merged_config,
                        model_factory=model_factory
                    )
                    logger.info(f"Successfully created agent: {agent_type}")
                except Exception as e:
                    logger.error(f"Error creating agent {agent_type}: {str(e)}", exc_info=True)
            else:
                logger.warning(f"Agent type {agent_type} not found in registry. Available types: {list(cls._agent_registry.keys())}")
        
        logger.info(f"Created {len(agents)} agents: {list(agents.keys())}")
        return agents
    
    @classmethod
    def get_available_agent_types(cls) -> Dict[str, str]:
        """
        Get available agent types.
        
        Returns:
            Dictionary mapping agent types to their class names
        """
        return {agent_type: agent_class.__name__ 
                for agent_type, agent_class in cls._agent_registry.items()}
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the agent cache."""
        cls._agent_cache.clear()
        logger.info("Agent cache cleared")
    
    @staticmethod
    def _create_cache_key(agent_type: str, config: Dict[str, Any]) -> str:
        """
        Create a cache key for an agent configuration.

        Args:
            agent_type: Type of agent
            config: Debug configuration for the agent

        Returns:
            Cache key string
        """
        model_provider = (
            config.get("model_provider")
            or config.get("model", {}).get("provider")
            or "default"
        )

        return f"{agent_type}_{model_provider}"
