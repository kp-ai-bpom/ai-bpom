import os

from strands.models.openai import OpenAIModel

from app.core.config import settings


def create_strands_model(tier: str = "flash") -> OpenAIModel:
    """
    Creates a Strands OpenAIModel using ai-bpom settings.

    Tiers:
      flash → AI_INSTRUCT_MODEL_NAME (orchestrator, planner, synthesis)
      pro   → AI_THINK_MODEL_NAME (analysis, reviewer)
      deep  → AI_DEEP_THINK_MODEL_NAME (complex tasks)
    """
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")

    if tier == "pro":
        model_id = settings.AI_THINK_MODEL_NAME
    elif tier == "deep":
        model_id = settings.AI_DEEP_THINK_MODEL_NAME
    else:
        model_id = settings.AI_INSTRUCT_MODEL_NAME

    client_args: dict = {
        "api_key": api_key,
        "default_headers": {"User-Agent": "ai-bpom/1.0"},
    }
    if settings.AI_BASE_URL:
        client_args["base_url"] = settings.AI_BASE_URL

    return OpenAIModel(
        model_id=model_id,
        client_args=client_args,
        params={"temperature": 0},
    )
