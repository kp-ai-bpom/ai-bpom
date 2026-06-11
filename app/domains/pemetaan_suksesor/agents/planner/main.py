from strands import Agent

from .prompt import SYSTEM_PROMPT
from .schema import PlannerOutput
from .tools import query_jabatan_profile, query_neo4j_depth2, search_neo4j_entities
from ..llm_adapter import create_strands_model


def create_planner_agent() -> Agent:
    model = create_strands_model(tier="flash")
    return Agent(
        name="planner",
        model=model,
        tools=[query_jabatan_profile, query_neo4j_depth2, search_neo4j_entities],
        system_prompt=SYSTEM_PROMPT,
        output_schema=PlannerOutput,
    )
