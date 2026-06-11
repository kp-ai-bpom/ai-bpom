from strands import Agent
from .prompt import ANALYSIS_SYSTEM_PROMPT
from .tools import search_vector_rag
from ..llm_adapter import create_strands_model
from ..planner.tools import query_jabatan_profile, query_neo4j_depth2, search_neo4j_entities

def create_analysis_agent() -> Agent:
    model = create_strands_model(tier="pro")

    return Agent(
        name="analysis",
        model=model,
        tools=[
            search_vector_rag,
            query_jabatan_profile,
            query_neo4j_depth2,
            search_neo4j_entities,
        ],
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
    )
