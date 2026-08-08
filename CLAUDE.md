# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Services BPOM is a FastAPI-based API that provides LLM-powered services using LangChain with OpenAI and Anthropic providers. It uses PostgreSQL via SQLAlchemy async ORM.

## Architecture

### Domain-Driven Design (DDD) with N-Layer Pattern

```
app/
├── api/router.py              # Main router aggregator
├── core/                      # Core infrastructure
│   ├── config.py             # Settings via pydantic-settings
│   ├── llm.py                # LLMManager singleton + LLMAdapter
│   └── logger.py             # Custom logger with emoji prefixes
├── db/database.py            # SQLAlchemy async setup
├── domains/                  # Business domains
│   └── {domain}/
       ├── api.py              # FastAPI routes
       ├── services.py         # Business logic
       ├── repositories.py     # Data access layer
       ├── models.py           # SQLAlchemy models
       ├── schemas.py          # Pydantic request/response schemas
       └── dto/                # Data transfer objects (optional)
└── server.py                 # App factory with lifespan management
```

**Flow**: `api.py` → `services.py` → `repositories.py` → `models.py`

### Key Patterns

1. **Dependency Injection**: All services use FastAPI `Depends()` for loose coupling
2. **Singleton Pattern**: `LLMManager` ensures single instance of LLM clients
3. **LLM Adapter**: `LLMAdapter` dataclass bundles three model types (instruct, think, deep_think)
4. **Dual Provider Support**: Automatic fallback OpenAI → Anthropic based on available API keys

## Development Commands

This project uses `uv` for package management and `poethepoet` (poe) for task running.

```bash
# Install dependencies
uv sync

# Development mode (auto-reload)
poe dev
# Equivalent: fastapi dev main.py --port 8080

# Production mode
poe start
# Equivalent: fastapi run main.py --port 8080

# Production with workers
poe prod
# Equivalent: fastapi run main.py --port 8080 --workers 5

# Run tests
poe test
# Equivalent: pytest

# Lint code
poe check
# Equivalent: ruff check .

# Format code
poe format
# Equivalent: ruff format .
```

## Environment Setup

Copy `.env.example` to `.env` and configure:

```bash
ENV=development

# At least one provider is required
OPENAI_API_KEY=sk-xxxxxxx
ANTHROPIC_API_KEY=

AI_BASE_URL=https://xxx.com/api
AI_INSTRUCT_MODEL_NAME=gpt-4.1    # General purpose
AI_THINK_MODEL_NAME=gpt-4.1       # Reasoning tasks
AI_DEEP_THINK_MODEL_NAME=gpt-4.1  # Complex analysis

POSTGRES_URI=postgresql+asyncpg://username:password@localhost:5452/
```

## LLM Architecture

The `LLMManager` singleton (`app/core/llm.py`) manages three model tiers:

- **instruct**: General purpose Q&A
- **think**: Reasoning and analysis
- **deep_think**: Complex multi-step tasks

Provider priority: OpenAI → Anthropic (falls back if API key not set)

Usage in services:
```python
from app.core.llm import LLMAdapter

class SomeService:
    def __init__(self, llm_adapter: LLMAdapter):
        self._llm = llm_adapter

    def analyze(self, messages: list):
        response = self._llm.think.invoke(messages)
        return response.content
```

## Database

PostgreSQL with SQLAlchemy async (`asyncpg`). Models inherit from `Base` in `app/db/database.py`.

Repository pattern with dependency injection:
```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

# In service
def get_service(db: AsyncSession = Depends(get_db)) -> MyService:
    return MyService(MyRepository(db))
```

**Note**: Use Alembic for database migrations. Run `poe migrate` before starting the server. Do NOT use `init_db()` in production.

## Adding a New Domain

1. Create domain folder: `app/domains/{domain_name}/`
2. Create files: `{api,services,repositories,models,schemas}.py`
3. Follow existing pattern from `chatbot` domain
4. Import model in `alembic/env.py` for autogenerate support
5. Run `poe migration-new "add {domain} table"` to create migration
6. Register router in `app/api/router.py`

## Logging Convention

Use emoji prefixes from `app/core/logger.py`:

- 🚀 Start/Begin
- ✅ Success
- ❌ Error
- 💾 Save/Write
- 🧠 Model Load
- 🔄 Reload/Refresh

```python
from app.core.logger import log
log.info("🚀 Starting process...")
log.exception(f"❌ Error: {e}")
```

## Testing

Tests use `pytest` and `pytest-asyncio`. Run with:
```bash
pytest
```

## Docker Services

PostgreSQL container available via `docker-compose.yml`:
```bash
docker-compose up -d
```
- Port: 5452 (mapped to container's 5432)
- User: light_postgres / light_postgres_root
