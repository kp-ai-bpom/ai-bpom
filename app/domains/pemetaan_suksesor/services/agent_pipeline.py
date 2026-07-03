import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from app.domains.pemetaan_suksesor.agents.planner.main import create_planner_agent
from app.domains.pemetaan_suksesor.agents.analysis.main import create_analysis_agent
from app.domains.pemetaan_suksesor.agents.synthesis.main import create_synthesis_agent
from app.domains.pemetaan_suksesor.agents.reviewer.main import create_reviewer_agent
from app.domains.pemetaan_suksesor.dto.pipeline import AgentName
from app.domains.pemetaan_suksesor.rag.vector.embed import (
    clear_embedding_cache,
    precompute_embeddings,
)

logger = logging.getLogger(__name__)


def _serialize_structured_output(obj: Any) -> Optional[dict]:
    """Safely serialize a structured output (Pydantic model or dict) to a dict."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        logger.warning("Could not serialize agent output of type %s", type(obj).__name__)
        return None


def _extract_text_from_message(msg_obj: Any) -> Optional[str]:
    """Extract plain text string from a Strands AgentResult.message object."""
    if msg_obj is None:
        return None
    if isinstance(msg_obj, str):
        return msg_obj
    if isinstance(msg_obj, dict):
        content = msg_obj.get("content")
        if isinstance(content, list):
            parts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and "text" in c
            ]
            return "".join(parts).strip() or None
        if isinstance(content, str):
            return content.strip() or None
    return str(msg_obj)


def _try_parse_json_output(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM text output.

    Handles: bare JSON, ```json fences, preamble text before JSON,
    trailing text after JSON, and multiple content blocks.
    """
    if not text:
        return None

    stripped = text.strip()

    # Strategy 1: Strip markdown code fences
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        stripped = "\n".join(lines[1:end]).strip()

    # Strategy 2: Try direct parse
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: Find outermost { ... } using brace counting
    # This handles preamble text, trailing text, and mixed content
    first_brace = stripped.find("{")
    if first_brace == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(first_brace, len(stripped)):
        ch = stripped[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Found the matching closing brace
                candidate = stripped[first_brace : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    # Braces balanced but not valid JSON — keep searching
                    # for a later { that might be the real start
                    pass
                # Don't break — there might be another { ... } block later

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

        # Try Pydantic structured_output first (when output_schema is used)
        structured = getattr(result, "structured_output", None)
        output = _serialize_structured_output(structured)

        # Extract raw text from the message object
        msg_obj = getattr(result, "message", None)
        raw_text = _extract_text_from_message(msg_obj)

        # If no structured output, try parsing JSON from text (strips ```json fences)
        if output is None and raw_text:
            output = _try_parse_json_output(raw_text)

        # Only keep message if JSON parsing also failed (plain-text fallback)
        message = raw_text if output is None else None

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

    async def run_agent_job(
        self,
        agent_name: AgentName,
        input_json: Any,
        job_id: str,
        job_service,
    ) -> None:
        """Background task wrapper: runs agent, then updates job status + result."""
        try:
            result = await self.run_agent(agent_name, input_json=input_json)
            job_service.update_job(job_id, status="completed", result=result)
        except Exception as e:
            logger.error("Agent %s job %s failed: %s", agent_name.value, job_id, e)
            job_service.update_job(job_id, status="failed", error=str(e))

    # ── Full Pipeline ──────────────────────────────────────────────────

    async def run_full_pipeline(
        self,
        input_json: Any,
        job_id: str,
        job_service,
    ) -> None:
        """Run full XAI-MENTARI pipeline:

        Planner → Embeddings Pre-compute → Analysis → Synthesis → Reviewer
        """
        # Clear any stale embedding cache from previous runs
        clear_embedding_cache()
        pipeline_log: List[Dict[str, Any]] = []

        try:
            # ── Phase 1: Planner ────────────────────────────────────
            logger.info("Pipeline %s: Starting Planner phase", job_id)
            planner_result = await self.run_agent(
                AgentName.PLANNER, input_json=input_json
            )
            pipeline_log.append({"agent": "planner", "status": "completed"})
            planner_output = planner_result.get("output") or {}

            # ── Inter-phase: Pre-compute embeddings ─────────────────
            queries = _extract_vector_rag_queries(planner_output)
            if queries:
                count = await precompute_embeddings(queries)
                pipeline_log.append({
                    "agent": "embedding_precompute",
                    "queries_total": len(queries),
                    "queries_new": count,
                })
                logger.info(
                    "Pipeline %s: Pre-computed %d/%d embedding queries",
                    job_id, count, len(queries),
                )

            # ── Phase 2a: Analysis ───────────────────────────────────
            logger.info("Pipeline %s: Starting Analysis phase", job_id)
            analysis_input = {
                "blueprint": planner_output,
                "input": input_json,
            }
            analysis_result = await self.run_agent(
                AgentName.ANALYSIS, input_json=analysis_input
            )
            pipeline_log.append({"agent": "analysis", "status": "completed"})
            analysis_output = analysis_result.get("output") or {}

            # ── Phase 2b: Synthesis ─────────────────────────────────
            logger.info("Pipeline %s: Starting Synthesis phase", job_id)
            synthesis_input = {
                "xai_justification_report": analysis_output,
                "blueprint_context": planner_output,
            }
            synthesis_result = await self.run_agent(
                AgentName.SYNTHESIS, input_json=synthesis_input
            )
            pipeline_log.append({"agent": "synthesis", "status": "completed"})
            synthesis_output = synthesis_result.get("output") or {}

            # ── Phase 2c: Reviewer ──────────────────────────────────
            logger.info("Pipeline %s: Starting Reviewer phase", job_id)
            reviewer_input = {
                "synthesis_report": synthesis_output,
                "analysis_report": analysis_output,
                "planner_blueprint": planner_output,
            }
            reviewer_result = await self.run_agent(
                AgentName.REVIEWER, input_json=reviewer_input
            )
            pipeline_log.append({"agent": "reviewer", "status": "completed"})

            # ── Done ─────────────────────────────────────────────────
            result = {
                "status": "completed",
                "pipeline_log": pipeline_log,
                "planner": planner_result,
                "analysis": analysis_result,
                "synthesis": synthesis_result,
                "reviewer": reviewer_result,
            }
            job_service.update_job(job_id, status="completed", result=result)
            logger.info("Pipeline %s: Completed successfully", job_id)

        except Exception as e:
            logger.error("Pipeline %s failed: %s", job_id, e, exc_info=True)
            pipeline_log.append({"agent": "pipeline", "status": "failed", "error": str(e)})
            job_service.update_job(job_id, status="failed", error=str(e))


def _extract_vector_rag_queries(planner_output: dict) -> List[str]:
    """Extract fungsi_utama and kompetensi_spesifik from planner output
    for embedding pre-computation.

    These strings are passed verbatim to search_vector_rag by the Analysis Agent,
    so pre-computing their embeddings eliminates redundant API calls.
    """
    queries: List[str] = []
    blueprint = planner_output.get("xai_blueprint", planner_output)

    # Retrieved context from blueprint
    retrieved_context = blueprint.get("retrieved_context", {})
    fungsi_utama = retrieved_context.get("fungsi_utama", [])
    kompetensi_spesifik = retrieved_context.get("kompetensi_spesifik", [])

    if isinstance(fungsi_utama, list):
        queries.extend(fungsi_utama)
    elif isinstance(fungsi_utama, str):
        queries.append(fungsi_utama)

    if isinstance(kompetensi_spesifik, list):
        queries.extend(kompetensi_spesifik)
    elif isinstance(kompetensi_spesifik, str):
        queries.append(kompetensi_spesifik)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        if isinstance(q, str) and q.strip() and q not in seen:
            seen.add(q)
            unique.append(q)

    return unique


def get_agent_pipeline_service() -> AgentPipelineService:
    return AgentPipelineService()