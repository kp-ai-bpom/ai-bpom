from strands import Agent
from .prompt import SYNTHESIS_SYSTEM_PROMPT
from .tools import comparative_matrix_aggregator
from ..llm_adapter import create_strands_model

def create_synthesis_agent() -> Agent:
    model = create_strands_model(tier="flash")

    return Agent(
        name="synthesis",
        model=model,
        tools=[comparative_matrix_aggregator],
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
    )
