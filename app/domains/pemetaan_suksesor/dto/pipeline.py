from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field

class AgentName(str, Enum):
    PLANNER = "planner"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REVIEWER = "reviewer"

class AgentRunRequest(BaseModel):
    input_text: Optional[str] = None
    input_json: Optional[Any] = None
    retries: int = Field(default=3, ge=0)

class AgentEvaluateRequest(BaseModel):
    agent_output: Any
    input_json: Any
    tool_call_log: Optional[str] = None
