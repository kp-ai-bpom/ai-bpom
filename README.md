<div align="center">
  <h1>🚀 AI BPOM</h1>
  <p>FastAPI + Machine Learning + LLM Integration</p>
  
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

### 🔍 Topic Modeling

- **Embedded Topic Model (ETM)** dengan Word2Vec embeddings
- 10-step Indonesian NLP preprocessing pipeline
- LLM-powered data augmentation untuk meningkatkan kualitas topik
- Automatic topic labeling dan contextualization
- Support untuk Twitter/social media text

### 😊 Sentiment Analysis

- **Dual model architecture**: CNN dan CNN-LSTM
- Binary classification: Positif/Negatif
- 12-step preprocessing khusus Indonesian text
- Confidence scoring untuk setiap prediksi
- Per-topic sentiment aggregation

### 💭 Emotion Classification

- **Multi-class emotion detection**: Anger, Fear, Joy, Love, Sad, Neutral
- Dual model: CNN dan BiLSTM
- Probability distribution untuk semua emotion classes
- Topic-based emotion analysis

### 🔄 System Management

- **Hot-reload ML models** tanpa restart server
- Model status monitoring
- Health check endpoints
- Centralized configuration management

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

### Machine Learning & NLP

- **TensorFlow/Keras** - Deep learning models (CNN, LSTM, BiLSTM)
- **OCTIS** - Topic modeling framework (ETM)
- **Sastrawi** - Indonesian stemming
- **scikit-learn** - NLP utilities (TF-IDF, preprocessing)
- **mpstemmer** - Modified Porter Stemmer untuk Indonesian
- **NLTK** - Natural language toolkit

### LLM Integration

- **LangChain** - LLM orchestration framework
- **OpenAI GPT** - Data augmentation dan context generation

### Package Management

- **uv** - Modern Python package manager (faster than pip)

### Storage

- **MinIO** - Object storage untuk model artifacts (optional)

---

## 📦 Prerequisites

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **MongoDB 7.0+** ([Download](https://www.mongodb.com/try/download/community))
- **uv** package manager ([Installation](https://github.com/astral-sh/uv))
- **OpenAI API Key** (untuk LLM features)

Optional:

- **Docker** (untuk MinIO)

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/codelabs-socialabs/socialabs-be-ai.git
cd socialabs-be-ai
```

### 2. Install Dependencies dengan uv

```bash
uv sync
```

### 3. Setup MongoDB

Pastikan MongoDB sudah running di `localhost:27017` atau update connection string di `.env`.

```bash
mongosh --eval "db.version()"
```

### 4. (Optional) Start MinIO untuk Model Storage

```bash
docker-compose up -d
```

MinIO Console: `http://localhost:9001` (admin/password123)

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

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGO_DB_NAME=socialabs_ai_db

# OpenAI Configuration (Required untuk LLM features)
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1/

# Path Configuration (relative to project root)
MODELS_BASE_PATH=models/raw
```

### 3. Prepare Model Files

Place model files di direktori yang sesuai:

```
models/raw/
├── sentiment/
│   ├── latest_model-cnn-sentiment.h5
│   ├── latest_tokenizer-cnn-sentiment.pickle
│   ├── latest_model-cnn-lstm-sentiment.h5
│   ├── latest_tokenizer-cnn-lstm-sentiment.pickle
│   └── utils/
│       ├── kamus.csv
│       └── stopwords.txt
├── emotion/
│   ├── latest_model-cnn-emotion.h5
│   ├── latest_tokenizer-cnn-emotion.pickle
│   ├── latest_model-bilstm-emotion.h5
│   ├── latest_tokenizer-bilstm-emotion.pickle
│   └── utils/
│       ├── kamus.csv
│       └── stopwords.txt
└── topic_modeling/
    ├── preprocessing/
    └── utils/
        └── kbba.txt
```

---

## 🎬 Running the Application

### Development Mode (Auto-reload)

```bash
fastapi dev
```

Server akan berjalan di: `http://localhost:8000`

### Production Mode

```bash
fastapi run --workers 5
```

Gunakan jumlah worker sebanyak 2×jumlah core CPU+1.

### Verify Installation

Akses endpoints berikut:

- **Health Check**: `http://localhost:8000/system/health`
- **API Docs (Swagger)**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📚 API Documentation

### Interactive Documentation

Setelah server running, akses Swagger UI untuk interactive API documentation:

**Swagger UI**: `http://localhost:8000/docs`

### Endpoint Overview

#### Topic Modeling

- `POST /topics/process` - Process topic modeling dengan ETM
- `GET /topics/{project_id}` - Get topics untuk project tertentu

#### Sentiment Analysis

- `POST /sentiments/classify` - Klasifikasi sentiment (Positif/Negatif)
- `GET /sentiments/{project_id}` - Get sentiment results

#### Emotion Classification

- `POST /emotions/classify` - Klasifikasi emotion (6 classes)
- `GET /emotions/{project_id}` - Get emotion results

#### System Management

- `POST /system/reload-models` - Hot-reload semua ML models
- `GET /system/models-status` - Cek status loading models
- `GET /system/health` - Health check

### Example Request

```bash
# Topic Modeling
curl -X POST "http://localhost:8000/topics/process" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "proj123",
    "keyword": "pemilu",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'

# Sentiment Analysis
curl -X POST "http://localhost:8000/sentiments/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "proj123"
  }'

# Reload Models
curl -X POST "http://localhost:8000/system/reload-models"
```

---

## 📁 Project Structure

```
new-socialabs-ai/
├── app/
│   ├── api/                        # API routing
│   │   ├── router.py              # Main router
│   │   └── system.py              # System management endpoints
│   ├── core/                       # Core configuration
│   │   ├── config.py              # Settings & path management
│   │   ├── logger.py              # Custom logger
│   │   ├── engine.py              # Centralized model manager
│   │   └── security.py            # Security utilities
│   ├── db/                         # Database setup
│   │   └── database.py            # MongoDB connection & Beanie init
│   ├── domains/                    # Domain-driven modules
│   │   ├── topic_modeling/
│   │   │   ├── api.py             # Routes
│   │   │   ├── services.py        # Business logic
│   │   │   ├── repositories.py    # Data access
│   │   │   ├── models.py          # Beanie models
│   │   │   └── schemas.py         # Pydantic schemas
│   │   ├── sentiment/
│   │   │   ├── api.py
│   │   │   ├── services.py
│   │   │   ├── repositories.py
│   │   │   ├── models.py
│   │   │   ├── schemas.py
│   │   │   └── engine.py          # ML model singleton
│   │   ├── emotion/
│   │   │   └── ... (sama seperti sentiment)
│   │   └── chatbot/
│   └── shared/                     # Shared utilities
│       ├── llm.py                 # LangChain LLM singleton
│       └── deps.py                # Shared dependencies
├── models/                         # Model artifacts
│   └── raw/
│       ├── sentiment/
│       ├── emotion/
│       └── topic_modeling/
├── docs/                           # Documentation
│   ├── DOCS.md                    # Complete architecture docs
│   └── MODEL_RELOAD_API.md        # Model reload API guide
├── test/                           # Tests (coming soon)
├── .env.example                    # Environment template
├── .github/
│   └── copilot-instructions.md    # AI coding guidelines
├── docker-compose.yml              # MinIO service
├── main.py                         # Application entry point
├── pyproject.toml                  # Dependencies & project config
└── README.md                       # This file
```

---

## 💻 Development

### Code Style & Conventions

- **Domain-Driven Design** - Modular architecture per business domain
- **N-Layer Pattern** - api.py → services.py → repositories.py → models.py
- **Dependency Injection** - FastAPI `Depends()` untuk loose coupling
- **Singleton Pattern** - ML models dan LLM connections
- **Async/Await** - I/O operations menggunakan async
- **Thread Pool** - CPU-bound operations dengan `asyncio.to_thread()`

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

2. Follow canonical pattern dari `topic_modeling` atau `sentiment`

3. Register Beanie models di `app/db/database.py`

4. Add router ke `app/api/router.py`

### Running Tests

```bash
# Tests belum diimplementasi
# Future: pytest dan pytest-asyncio
pytest
```

---

## 🔄 Model Management

### Update Model Baru

**Workflow**:

1. Upload model files ke folder:
   - Sentiment: `models/raw/sentiment/`
   - Emotion: `models/raw/emotion/`

2. Rename sesuai konfigurasi:
   - `latest_model-cnn-sentiment.h5`
   - `latest_tokenizer-cnn-sentiment.pickle`
   - dll.

3. Reload via API:

   ```bash
   curl -X POST http://localhost:8000/system/reload-models
   ```

4. Verify:
   ```bash
   curl http://localhost:8000/system/models-status
   ```

### Best Practices

- ⏰ Reload saat traffic rendah (misal: tengah malam)
- 💾 Backup model lama sebelum replace
- 🧪 Test model baru di development dulu
- 📊 Monitor logs saat reload

Lihat [MODEL_RELOAD_API.md](docs/MODEL_RELOAD_API.md) untuk detail lengkap.

---

## 🏗️ Architecture

### Architectural Patterns

1. **Domain-Driven Design (DDD)** - Modular monolith per domain
2. **N-Layer Architecture** - Separation of concerns (API, Service, Repository)
3. **Dependency Injection** - FastAPI Depends()
4. **Singleton Pattern** - ML models & LLM connections
5. **Async/Threading** - I/O dengan async, CPU-bound dengan threading

### Database Strategy (Dual DB Pattern)

- **External Collections** (dari NestJS): PyMongo native

  ```python
  self.tweets_collection = db["tweets"]
  cursor = await self.tweets_collection.aggregate(pipeline)
  ```

- **AI-generated Collections**: Beanie ODM
  ```python
  topics = await TopicsModel.find(TopicsModel.projectId == project_id).to_list()
  ```

### ML Pipeline Flow

```
User Request
    ↓
FastAPI Router (api.py)
    ↓
Service Layer (services.py)
    ├→ Repository (repositories.py) → MongoDB
    ├→ ML Engine (engine.py) → Keras Models
    └→ LLM Service (llm.py) → OpenAI
    ↓
Response
```

---

## 📖 Documentation

Dokumentasi lengkap tersedia di folder `docs/`:

- **[DOCS.md](docs/DOCS.md)** - Complete architecture & implementation guide
- **[MODEL_RELOAD_API.md](docs/MODEL_RELOAD_API.md)** - Model reload API documentation
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI coding agent guidelines

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
- [ ] Use `asyncio.to_thread()` untuk CPU-bound operations
- [ ] Update documentation jika diperlukan

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

**SociaLabs** - Social Media Analytics Platform

---

## 🙏 Acknowledgments

- FastAPI framework
- TensorFlow/Keras team
- LangChain community
- OCTIS library maintainers
- Sastrawi Indonesian NLP toolkit

---

## 📞 Support

Untuk pertanyaan atau dukungan:

- 📧 Email: support@socialabs.io (example)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/new-socialabs-ai/issues)
- 📖 Docs: [Documentation](docs/DOCS.md)

---

**Made with ❤️ for Indonesian Social Media Analytics**
