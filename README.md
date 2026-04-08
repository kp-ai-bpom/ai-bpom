<div align="center">
  <h1>🚀 AI Services BPOM</h1>
  <p>FastAPI + LLM Integration untuk Sistem Kepegawaian BPOM</p>
  
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-green.svg)](https://fastapi.tiangolo.com)
  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
  [![MongoDB](https://img.shields.io/badge/MongoDB-7.0+-green.svg)](https://www.mongodb.com)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Model Management](#-model-management)
- [Architecture](#-architecture)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🤖 LLM-Powered Services

- **Triple Model Architecture**: Instruct, Think, Deep Think
- Multi-provider support (OpenAI & Anthropic) dengan automatic fallback
- Async/Sync invocation support
- Temperature control dan max_tokens konfigurasi

### 🏢 Domain Kepegawaian BPOM

- **Pemetaan Suksesor**: Sistem pemetaan calon penerus jabatan
- **Penilaian Suksesor**: Evaluasi dan penilaian kompetensi
- Modular domain-driven design untuk ekspansi fitur

### 🔄 System Management

- Health check endpoints
- Centralized configuration management
- Dual provider LLM dengan fallback otomatis

---

## 🛠️ Tech Stack

### Core Framework

- **FastAPI** - Modern async web framework
- **Python 3.11+** - Programming language
- **uvicorn** - ASGI server

### Database

- **MongoDB** - NoSQL database (shared with NestJS backend)
- **Beanie ODM** - Async MongoDB ODM untuk managed collections
- **PyMongo** - Native driver untuk external collections

### LLM Integration

- **LangChain** - LLM orchestration framework
- **OpenAI GPT** - Primary LLM provider
- **Anthropic Claude** - Fallback LLM provider

### Database

- **PostgreSQL** - Relational database dengan SQLAlchemy async
- **asyncpg** - Async PostgreSQL driver

### Package Management

- **uv** - Modern Python package manager (faster than pip)

---

## 📦 Prerequisites

- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **PostgreSQL** ([Download](https://www.postgresql.org/download/))
- **uv** package manager ([Installation](https://github.com/astral-sh/uv))
- **OpenAI API Key** atau **Anthropic API Key** (untuk LLM features)

Optional:

- **Docker** (untuk PostgreSQL container)

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/kp-ai-bpom/ai-bpom.git
cd ai-bpom
```

### 2. Install Dependencies dengan uv

```bash
uv sync
```

### 3. Setup PostgreSQL

Pastikan PostgreSQL sudah running di `localhost:5452` atau update connection string di `.env`.

```bash
# Dengan Docker
docker-compose up -d

# Atau cek PostgreSQL lokal
psql --version
```

---

## ⚙️ Configuration

### 1. Copy Environment Template

```bash
cp .env.example .env
```

### 2. Edit `.env` File

```bash
# Environment
ENV=development

# PostgreSQL Configuration
POSTGRES_URI=postgresql+asyncpg://username:password@localhost:5452/

# LLM Provider Configuration (Minimal satu provider diperlukan)
OPENAI_API_KEY=sk-your-api-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# AI Base URL (untuk custom endpoint atau proxy)
AI_BASE_URL=https://api.openai.com/v1/

# Model Names
AI_INSTRUCT_MODEL_NAME=gpt-4.1      # General purpose
AI_THINK_MODEL_NAME=gpt-4.1         # Reasoning tasks
AI_DEEP_THINK_MODEL_NAME=gpt-4.1    # Complex analysis
```

---

## 🎬 Running the Application

Gunakan `poethepoet` (poe) untuk menjalankan perintah:

### Development Mode (Auto-reload)

```bash
poe dev
```

Server akan berjalan di: `http://localhost:8080`

### Production Mode

```bash
poe start
```

Atau dengan multiple workers:

```bash
poe prod
```

### Verify Installation

Akses endpoints berikut:

- **API Docs (Swagger)**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

---

## 📚 API Documentation

### Interactive Documentation

Setelah server running, akses Swagger UI untuk interactive API documentation:

**Swagger UI**: `http://localhost:8000/docs`

### Endpoint Overview

#### Chatbot

- `POST /api/chatbot/instruct` - General purpose Q&A dengan model instruct
- `POST /api/chatbot/think` - Reasoning tasks dengan model think
- `POST /api/chatbot/deep-think` - Complex analysis dengan model deep-think

#### Pemetaan Suksesor

- Endpoints untuk pemetaan calon penerus jabatan

#### Penilaian Suksesor

- Endpoints untuk penilaian kompetensi suksesor

### Example Request

```bash
# Chatbot Instruct Model
curl -X POST "http://localhost:8080/api/chatbot/instruct" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Apa itu sistem kepegawaian BPOM?"
  }'

# Chatbot Think Model
curl -X POST "http://localhost:8080/api/chatbot/think" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Analisis kompetensi yang dibutuhkan untuk jabatan direktur"
  }'
```

---

## 📁 Project Structure

```
ai-services-bpom/
├── app/
│   ├── api/                        # API routing
│   │   └── router.py              # Main router aggregator
│   ├── core/                       # Core configuration
│   │   ├── config.py              # Settings via pydantic-settings
│   │   ├── logger.py              # Custom logger
│   │   └── llm.py                 # LLMManager singleton
│   ├── db/                         # Database setup
│   │   └── database.py            # PostgreSQL connection & SQLAlchemy
│   ├── domains/                    # Domain-driven modules
│   │   ├── chatbot/
│   │   │   ├── api.py             # Routes
│   │   │   ├── services.py        # Business logic
│   │   │   ├── repositories.py    # Data access
│   │   │   ├── models.py          # SQLAlchemy models
│   │   │   └── dto/               # Data transfer objects
│   │   ├── pemetaan_suksesor/     # Pemetaan calon penerus jabatan
│   │   │   ├── api.py
│   │   │   ├── services.py
│   │   │   ├── repositories.py
│   │   │   ├── models.py
│   │   │   └── schemas.py
│   │   └── penilaian_suksesor/    # Penilaian kompetensi suksesor
│   │       ├── api.py
│   │       ├── services.py
│   │       ├── repositories.py
│   │       ├── models.py
│   │       └── schemas.py
│   └── server.py                  # App factory with lifespan
├── docker-compose.yml              # PostgreSQL service
├── main.py                         # Application entry point
├── pyproject.toml                  # Dependencies & project config
├── README.md                       # This file
└── CLAUDE.md                       # AI assistant guidance
```

---

## 💻 Development

### Code Style & Conventions

- **Domain-Driven Design** - Modular architecture per business domain
- **N-Layer Pattern** - api.py → services.py → repositories.py → models.py
- **Dependency Injection** - FastAPI `Depends()` untuk loose coupling
- **Singleton Pattern** - LLM connections via LLMManager
- **Async/Await** - I/O operations menggunakan async

### Logging Convention

```python
from app.core.logger import log

log.info("🚀 Starting process...")
log.error("❌ Error occurred")
log.exception(f"💥 Exception: {e}")
```

Emoji prefixes:

- 🚀 Start/Begin
- ✅ Success
- ❌ Error
- 💾 Save/Write
- 🧠 Model Load
- 🔄 Reload/Refresh

### Adding New Domain

1. Create domain structure:

```bash
mkdir -p app/domains/new_domain
touch app/domains/new_domain/{api,services,repositories,models,schemas}.py
```

2. Follow canonical pattern dari `chatbot`

3. Register models di `app/db/database.py`

4. Add router ke `app/api/router.py`

### Running Tests

```bash
poe test
```

---

## 🔧 Development Commands

| Command | Description |
|---------|-------------|
| `poe dev` | Run development server with auto-reload |
| `poe start` | Run production server |
| `poe prod` | Run with 5 workers |
| `poe test` | Run pytest |
| `poe check` | Run ruff linter |
| `poe format` | Format code with ruff |

---

## 🏗️ Architecture

### Architectural Patterns

1. **Domain-Driven Design (DDD)** - Modular monolith per domain
2. **N-Layer Architecture** - Separation of concerns (API, Service, Repository)
3. **Dependency Injection** - FastAPI Depends()
4. **Singleton Pattern** - LLM connections via LLMManager
5. **Async Pattern** - I/O operations menggunakan async/await

### Database Strategy

- **PostgreSQL** dengan SQLAlchemy Async ORM
- Repository pattern dengan dependency injection

  ```python
  # Repository
  class MyRepository:
      def __init__(self, db: AsyncSession):
          self._db = db
  
  # Service
  class MyService:
      def __init__(self, repository: MyRepository):
          self._repo = repository
  ```

### Request Flow

```
User Request
    ↓
FastAPI Router (api.py)
    ↓
Service Layer (services.py)
    ├→ Repository (repositories.py) → PostgreSQL
    └→ LLM Service (llm.py) → OpenAI/Anthropic
    ↓
Response
```

---

## 📖 Documentation

- **[CLAUDE.md](CLAUDE.md)** - AI assistant guidance dan development reference

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Follow existing code patterns dan conventions
4. Commit changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to branch (`git push origin feature/AmazingFeature`)
6. Open Pull Request

### Code Review Checklist

- [ ] Follow N-Layer architecture pattern
- [ ] Use Dependency Injection
- [ ] Add logging dengan emoji prefixes
- [ ] Handle errors properly (HTTPException → generic Exception)
- [ ] Update CLAUDE.md jika ada perubahan architecture

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

**BPOM AI Services Team**

---

## 🙏 Acknowledgments

- FastAPI framework
- LangChain community
- SQLAlchemy team

---

## 📞 Support

Untuk pertanyaan atau dukungan:

- 🐛 Issues: [GitHub Issues](https://github.com/kp-ai-bpom/ai-bpom/issues)
- 📖 Docs: [CLAUDE.md](CLAUDE.md)

---

**Made with ❤️ for BPOM Kepegawaian System**
