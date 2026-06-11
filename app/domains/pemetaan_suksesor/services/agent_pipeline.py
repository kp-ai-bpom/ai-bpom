import asyncio
import json
import logging
from typing import Any, Dict, Optional

from app.domains.pemetaan_suksesor.agents.planner.main import create_planner_agent
from app.domains.pemetaan_suksesor.agents.analysis.main import create_analysis_agent
from app.domains.pemetaan_suksesor.agents.synthesis.main import create_synthesis_agent
from app.domains.pemetaan_suksesor.agents.reviewer.main import create_reviewer_agent
from app.domains.pemetaan_suksesor.dto.pipeline import AgentName

logger = logging.getLogger(__name__)


def _serialize_structured_output(obj: Any) -> Optional[dict]:
    """Safely serialize a structured output (Pydantic model or dict) to a dict.

    Handles Pydantic models via .model_dump(), dicts via json round-trip
    to ensure JSON-serializable types, and returns None for anything else.
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    # Fallback: try JSON round-trip to ensure serializability
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        logger.warning("Could not serialize agent output of type %s", type(obj).__name__)
        return None


def _extract_usage(metrics: Any) -> Dict[str, Optional[int]]:
    """Extract token usage from Strands AgentResult.metrics.accumulated_usage."""
    if metrics is None:
        return {}
    try:
        accumulated = getattr(metrics, "accumulated_usage", None)
        if accumulated and isinstance(accumulated, dict):
            return {
                "input_tokens": accumulated.get("inputTokens"),
                "output_tokens": accumulated.get("outputTokens"),
                "total_tokens": accumulated.get("totalTokens"),
            }
    except Exception:
        logger.debug("Could not extract usage metrics from agent result")
    return {}


class AgentPipelineService:
    """Orchestrates Strands agent invocations with async safety and result serialization."""

    # Map agent names to their factory functions for lazy caching
    _AGENT_FACTORIES = {
        AgentName.PLANNER: create_planner_agent,
        AgentName.ANALYSIS: create_analysis_agent,
        AgentName.SYNTHESIS: create_synthesis_agent,
        AgentName.REVIEWER: create_reviewer_agent,
    }

    def __init__(self):
        # Cached agent instances — created on first use, then reused
        self._agents: Dict[AgentName, Any] = {}

    def _get_agent(self, name: AgentName):
        """Get or create a cached agent instance."""
        if name not in self._agents:
            factory = self._AGENT_FACTORIES[name]
            self._agents[name] = factory()
        return self._agents[name]

    def _invoke_agent_sync(self, agent_name: AgentName, input_text: str) -> Dict[str, Any]:
        """Synchronous agent invocation — called via asyncio.to_thread()."""
        agent = self._get_agent(agent_name)
        result = agent(input_text)

        # Extract structured output if available
        structured = getattr(result, "structured_output", None)
        output = _serialize_structured_output(structured)

        # Extract raw text message as fallback
        message = None
        if output is None:
            msg_obj = getattr(result, "message", None)
            message = str(msg_obj) if msg_obj else None

        # Extract usage metrics
        metrics = getattr(result, "metrics", None)
        usage = _extract_usage(metrics)

        return {
            "agent_name": agent_name.value,
            "message": message,
            "output": output,
            "usage": usage or None,
        }

    async def run_agent(
        self,
        agent_name: AgentName,
        input_text: Optional[str] = None,
        input_json: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run a named agent with input_text or input_json.

        Handles:
        - Async safety via asyncio.to_thread()
        - Support for both input_text and input_json
        - Result serialization and usage extraction
        """
        # Build the prompt from available inputs
        if input_text and input_json:
            prompt = f"{input_text}\n\n{json.dumps(input_json, default=str)}"
        elif input_json:
            prompt = json.dumps(input_json, default=str)
        elif input_text:
            prompt = input_text
        else:
            raise ValueError("Either input_text or input_json must be provided")

        # Run agent in a thread to avoid blocking the event loop
        return await asyncio.to_thread(self._invoke_agent_sync, agent_name, prompt)

    async def run_planner(self, input_text: Optional[str] = None, input_json: Optional[Any] = None):
        """Run the Planner agent."""
        return await self.run_agent(AgentName.PLANNER, input_text=input_text, input_json=input_json)

    async def run_analysis(self, input_text: Optional[str] = None, input_json: Optional[Any] = None):
        """Run the Analysis agent."""
        return await self.run_agent(AgentName.ANALYSIS, input_text=input_text, input_json=input_json)

    async def run_synthesis(self, input_text: Optional[str] = None, input_json: Optional[Any] = None):
        """Run the Synthesis agent."""
        return await self.run_agent(AgentName.SYNTHESIS, input_text=input_text, input_json=input_json)

    async def run_reviewer(self, input_text: Optional[str] = None, input_json: Optional[Any] = None):
        """Run the Reviewer agent."""
        return await self.run_agent(AgentName.REVIEWER, input_text=input_text, input_json=input_json)


def get_agent_pipeline_service() -> AgentPipelineService:
    return AgentPipelineService()