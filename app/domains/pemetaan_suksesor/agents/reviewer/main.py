from strands import Agent
from .prompt import REVIEWER_SYSTEM_PROMPT
from ..llm_adapter import create_strands_model

def create_reviewer_agent() -> Agent:
    model = create_strands_model(tier="pro")

    return Agent(
        name="reviewer",
        model=model,
        tools=[],
        system_prompt=REVIEWER_SYSTEM_PROMPT,
    )
