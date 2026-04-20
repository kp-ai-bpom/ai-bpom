"""
Agent module untuk multi-agent architecture menggunakan strands-agents.

Module ini menyediakan:
- AgentAdapter: Dataclass untuk menyimpan semua instance Agent
- AgentManager: Singleton untuk mengelola lifecycle Agent
- init_agents: Dependency injection factory
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from strands import Agent
from strands.models.openai import OpenAIModel
from strands.multiagent import Swarm

from app.core.config import settings
from app.core.logger import log

from .config import settings as local_settings

ANALYSIS_POOL_SIZE = 5  # Number of analysis agents for concurrent evaluation


@dataclass
class AgentAdapter:
    """
    Adapter yang menyimpan semua instance Agent default.
    Dapat di-inject ke service/dependency manapun yang membutuhkan multi-agent.
    """

    orchestrator: Agent
    search: Agent
    analysis: Agent  # first analysis agent (backward compat)
    analysis_pool: List[Agent]  # pool for concurrent evaluations
    synthesis: Agent
    reviewer: Agent

    def get(self, name: str) -> Optional[Agent]:
        """Get agent by name from adapter."""
        return getattr(self, name, None)


# Default system prompts for each agent type — Pemetaan Suksesor flow
DEFAULT_PROMPTS = {
    "orchestrator": local_settings.AGENT_ORCHESTRATOR_PROMPT,
    "search": local_settings.AGENT_SEARCH_PROMPT,
    "analysis": local_settings.AGENT_ANALYSIS_PROMPT,
    "synthesis": local_settings.AGENT_SYNTHESIS_PROMPT,
    "reviewer": local_settings.AGENT_REVIEWER_PROMPT,
}

# Model tier mappings from config
MODEL_TIER_CONFIG = {
    "orchestrator": "AGENT_ORCHESTRATOR_MODEL",
    "search": "AGENT_SEARCH_MODEL",
    "analysis": "AGENT_ANALYSIS_MODEL",
    "synthesis": "AGENT_SYNTHESIS_MODEL",
    "reviewer": "AGENT_REVIEWER_MODEL",
}


class AgentManager:
    """
    Singleton class untuk mengelola instance Strands Agent.
    Memastikan hanya ada satu instance Agent di memori selama aplikasi berjalan.
    """

    _instance = None
    _agents: Dict[str, Agent] = {}

    def __new__(cls):
        """Override __new__ untuk memastikan hanya satu instance AgentManager yang dibuat."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._agents = {}
        return cls._instance

    def _create_model(self, model_tier: str) -> OpenAIModel:
        """
        Create Strands OpenAI model based on tier.
        Uses existing settings for API keys and base URLs.
        """
        model = OpenAIModel(
            client_args={
                "api_key": settings.OPENAI_API_KEY,
                "base_url": settings.AI_BASE_URL,
            }
            if settings.AI_BASE_URL
            else {"api_key": settings.OPENAI_API_KEY},
            model_id=settings.AI_INSTRUCT_MODEL_NAME
            if model_tier == "instruct"
            else settings.AI_THINK_MODEL_NAME
            if model_tier == "think"
            else settings.AI_DEEP_THINK_MODEL_NAME,
            # params={"temperature": 0.7},
        )
        return model

    def _get_model_tier(self, agent_name: str) -> str:
        """Get model tier for agent from local config."""
        config_key = MODEL_TIER_CONFIG.get(agent_name, "AGENT_ORCHESTRATOR_MODEL")
        return getattr(local_settings, config_key, "think")

    def _create_agent(
        self,
        name: str,
        model_tier: str,
        system_prompt: str,
        tools: Optional[List] = None,
    ) -> Agent:
        """
        Create a single agent with configuration.
        """
        model = self._create_model(model_tier)

        agent = Agent(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools or [],
        )

        model_name = (
            settings.AI_INSTRUCT_MODEL_NAME
            if model_tier == "instruct"
            else settings.AI_THINK_MODEL_NAME
            if model_tier == "think"
            else settings.AI_DEEP_THINK_MODEL_NAME
        )
        log.info(
            f"🤖 Agent '{name}' initialized — tier: {model_tier}, model: {model_name}"
        )
        return agent

    def _initialize_default_agents(self):
        """
        Initialize 5 default BPOM agents + analysis pool.
        Called lazily when first agent is requested.
        """
        if self._agents:
            return

        default_agents = ["orchestrator", "search", "analysis", "synthesis", "reviewer"]

        for agent_name in default_agents:
            model_tier = self._get_model_tier(agent_name)
            system_prompt = DEFAULT_PROMPTS[agent_name]

            self._agents[agent_name] = self._create_agent(
                name=agent_name,
                model_tier=model_tier,
                system_prompt=system_prompt,
                tools=[],
            )

        # Create analysis agent pool for concurrent evaluations
        for i in range(ANALYSIS_POOL_SIZE):
            pool_name = f"analysis-{i}"
            model_tier = self._get_model_tier("analysis")
            self._agents[pool_name] = self._create_agent(
                name=pool_name,
                model_tier=model_tier,
                system_prompt=DEFAULT_PROMPTS["analysis"],
                tools=[],
            )

        log.info(
            f"🚀 All default agents initialized (analysis pool: {ANALYSIS_POOL_SIZE})"
        )

    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Get agent by name.
        Initializes default agents if not yet created.
        """
        self._initialize_default_agents()
        return self._agents.get(name)

    def register_agent(
        self,
        name: str,
        model_tier: str,
        system_prompt: str,
        tools: Optional[List] = None,
    ) -> Agent:
        """
        Register custom agent dynamically.
        """
        agent = self._create_agent(
            name=name,
            model_tier=model_tier,
            system_prompt=system_prompt,
            tools=tools or [],
        )

        self._agents[name] = agent
        log.info(f"✅ Custom agent '{name}' registered")
        return agent

    def get_adapter(self) -> AgentAdapter:
        """
        Get AgentAdapter with all default agents.
        """
        self._initialize_default_agents()

        analysis_pool = [
            self._agents[f"analysis-{i}"] for i in range(ANALYSIS_POOL_SIZE)
        ]

        return AgentAdapter(
            orchestrator=self._agents["orchestrator"],
            search=self._agents["search"],
            analysis=self._agents["analysis"],
            analysis_pool=analysis_pool,
            synthesis=self._agents["synthesis"],
            reviewer=self._agents["reviewer"],
        )

    def create_swarm(
        self,
        agent_names: List[str],
        entry_point: str = "orchestrator",
    ) -> Swarm:
        """
        Create a Swarm for multi-agent orchestration.

        Args:
            agent_names: List of agent names to include in the swarm
            entry_point: Starting agent name

        Returns:
            Swarm instance for orchestrated execution
        """
        self._initialize_default_agents()

        agents = [self._agents[name] for name in agent_names if name in self._agents]

        entry_agent = self._agents.get(entry_point, agents[0] if agents else None)

        swarm = Swarm(
            agents,
            entry_point=entry_agent,
            max_handoffs=local_settings.SWARM_MAX_HANDOFFS,
            max_iterations=local_settings.SWARM_MAX_ITERATIONS,
            execution_timeout=local_settings.SWARM_EXECUTION_TIMEOUT,
        )

        log.info(f"🐝 Swarm created with {len(agents)} agents, entry: {entry_point}")
        return swarm


# Dependency Injection Factory
def init_agents() -> AgentAdapter:
    """
    Dependency Injection untuk mendapatkan semua instance Agent.
    Bisa disuntikkan ke service mana pun yang membutuhkan multi-agent.
    """
    manager = AgentManager()
    return manager.get_adapter()
