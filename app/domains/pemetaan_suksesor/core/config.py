import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    SWARM_MAX_HANDOFFS: int = 20
    SWARM_MAX_ITERATIONS: int = 20
    SWARM_EXECUTION_TIMEOUT: float = 900.0

    # Agent Model Tier Configuration
    AGENT_ORCHESTRATOR_MODEL: str = os.getenv("AGENT_ORCHESTRATOR_MODEL", "think")
    AGENT_SEARCH_MODEL: str = os.getenv("AGENT_SEARCH_MODEL", "instruct")
    AGENT_ANALYSIS_MODEL: str = os.getenv("AGENT_ANALYSIS_MODEL", "deep_think")
    AGENT_SYNTHESIS_MODEL: str = os.getenv("AGENT_SYNTHESIS_MODEL", "think")
    AGENT_REVIEWER_MODEL: str = os.getenv("AGENT_REVIEWER_MODEL", "think")


settings = Settings()
