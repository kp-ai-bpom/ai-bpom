# 📝 Socialabs AI Backend - Architecture & Implementation Plan

## 1\. 🚀 Project Overview

Project ini adalah _AI Backend Service_ berbasis **FastAPI** yang bertugas menangani pemrosesan Machine Learning (Topic Modeling dengan ETM, Sentiment Analysis dengan CNN/LSTM) dan Generative AI (LangChain/LLM) untuk analitik media sosial berbahasa Indonesia.

Aplikasi ini beroperasi dalam arsitektur **Semi-Microservices**, di mana backend utama (CRUD) menggunakan NestJS, dan backend AI (FastAPI) berbagi _database_ yang sama (_Shared Database Pattern_) melalui MongoDB.

### Tech Stack Utama

- **Framework**: FastAPI dengan async/await patterns
- **Database**: MongoDB (Dual approach: Beanie ODM + PyMongo native)
- **Package Manager**: `uv` (modern Python package manager)
- **ML/NLP**: scikit-learn, TensorFlow/Keras, OCTIS (ETM), Sastrawi (Indonesian NLP)
- **LLM**: LangChain dengan OpenAI (singleton pattern)
- **Storage**: MinIO untuk model artifacts
- **Custom Dependencies**:
  - `mpstemmer` (fork from GitHub untuk Indonesian stemming)
  - `octis` (fork from GitHub untuk topic modeling)

---

## 2\. 🏛️ Architectural Patterns

Project ini sangat mengedepankan skalabilitas, performa, dan kemudahan pengujian (_testability_) dengan menerapkan pattern berikut:

1. **Domain-Driven Design (DDD) / Modular Monolith**: Kode dipisah berdasarkan _business domain_ (contoh: `topic_modeling`, `sentiment`, `chatbot`). Jika suatu saat domain tertentu butuh di- _scale_ terpisah, transisi ke _Microservices_ murni akan sangat mudah.
2. **N-Layer Architecture**: Pemisahan tanggung jawab yang ketat di dalam setiap domain:
   - **Controller (`api.py`)**: Hanya mengurus _Routing_, Validasi Request/Response (Pydantic), dan HTTP Status Codes.
   - **Service (`services.py`)**: Menyimpan _Business Logic_, NLP Preprocessing, dan orkestrasi ML/AI.
   - **Repository (`repositories.py`)**: Abstraksi akses ke Database (baik via Beanie ODM maupun PyMongo murni).
3. **Dependency Injection (DI)**: Menggunakan fitur bawaan FastAPI (`Depends()`) untuk menyuntikkan _Repository_, _Model Engine_, dan konfigurasi ke dalam _Service_ atau _Router_. Memudahkan proses _Mocking_ saat _Unit Testing_.
4. **Singleton Pattern**: Diterapkan pada objek yang berat/mahal seperti koneksi LLM LangChain dan model Keras/TensorFlow untuk mencegah _Out of Memory_ (OOM) dan _Race Conditions_.
5. **Asynchronous & Threading**: Menggunakan `async/await` untuk operasi I/O (Database/API) dan `asyncio.to_thread` untuk operasi CPU-bound (ML Training/Inference) agar tidak memblokir _Event Loop_ FastAPI.

---

## 3\. 📂 Directory Structure

```plaintext
├── app/
│   ├── api/                   # Entry point API global
│   │   └── router.py          # Menggabungkan router dari tiap domain
│   │
│   ├── core/                  # Konfigurasi dan Setup Inti
│   │   ├── config.py          # Environment variables (Pydantic BaseSettings)
│   │   ├── logger.py          # Custom logger (pengganti print)
│   │   └── security.py        # Security utilities
│   │
│   ├── db/                    # Setup Database
│   │   └── database.py        # AsyncMongoClient & init_beanie
│   │
│   ├── domains/               # Modul Domain Utama (DDD)
│   │   ├── topic_modeling/    # ✅ IMPLEMENTED
│   │   │   ├── api.py         # Controller / Routes
│   │   │   ├── schemas.py     # Pydantic DTOs (Request/Response)
│   │   │   ├── services.py    # Business Logic & 10-Step NLP Pipeline
│   │   │   ├── repositories.py# Data Access Layer (Beanie/PyMongo)
│   │   │   └── models.py      # Beanie Documents (TopicsModel, DocumentsModel)
│   │   │
│   │   ├── sentiment/         # ✅ IMPLEMENTED
│   │   │   ├── api.py
│   │   │   ├── schemas.py
│   │   │   ├── services.py    # Sentiment classification service
│   │   │   ├── repositories.py
│   │   │   ├── models.py
│   │   │   └── engine.py      # Singleton untuk Model Keras/TF (CNN, CNN-LSTM)
│   │   │
│   │   ├── chatbot/           # 🚧 STUB (Future implementation)
│   │   ├── emotion/           # 🚧 STUB (Future implementation)
│   │   └── sna/               # 🚧 STUB (Social Network Analysis - Future)
│   │
│   ├── shared/                # Utilities lintas domain
│   │   ├── deps.py            # Global DI utilities
│   │   └── llm.py             # Singleton LangChain LLM Manager
│   │
│   └── server.py              # Application Factory (create_app & lifespan)
│
├── models/                    # ⚠️ Di luar /app untuk model artifacts (exclude from app package)
│   ├── minio/                 # MinIO storage mount point
│   └── raw/                   # Trained model files
│       ├── sentiment/
│       │   ├── model-cnn-no-testing.h5
│       │   ├── model-cnn-lstm-no-testing.h5
│       │   ├── tokenizer-cnn-no-testing.pickle
│       │   ├── tokenizer-cnn-lstm-no-testing.pickle
│       │   └── utils/         # Dictionary & Stopwords khusus sentiment
│       │       ├── kamus.csv
│       │       └── stopwords.txt
│       │
│       └── topic_modeling/
│           ├── preprocessing/
│           │   └── vocabs/    # Generated OCTIS datasets (runtime)
│           └── utils/
│               └── kbba.txt   # Indonesian slang dictionary
│
├── docs/                      # Dokumentasi
│   └── DOCS.md                # File ini
│
├── test/                      # Unit tests (belum diimplementasi)
│
├── .github/                   # GitHub configurations
│   └── copilot-instructions.md # AI agent instructions
│
├── main.py                    # Entry point runner (`uv run fastapi dev main.py`)
├── pyproject.toml             # Dependencies (managed by uv)
├── docker-compose.yml         # MinIO service
└── .env                       # Environment variables (credentials)
```

---

## 4\. 💻 Core Implementation & Code Snippets

### A. Application Factory & Lifespan Pre-warming (`app/server.py`)

Model ML dan Database diinisialisasi sebelum server menerima _request_ (_Fail-Fast_ & menghilangkan _latency_ di _request_ pertama).

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.logger import log
from app.db.database import init_db
from app.shared.llm import init_llm
from app.domains.sentiment.engine import get_sentiment_models

def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan Manager untuk startup dan shutdown events."""
        log.info("🚀 Starting up server...")

        try:
            # 1. Inisialisasi Database
            await init_db()

            # 2. Pre-warm LLM Singleton
            init_llm()

            # 3. Pre-warm ML Models Singleton
            get_sentiment_models().load_models()
        except Exception as e:
            log.error(f"❌ Startup failed: {e}")
            raise e  # Hentikan server jika gagal

        yield  # Aplikasi berjalan

        log.info("🛑 Shutting down server...")

    app = FastAPI(
        title="AI Services API",
        description="API untuk Topic Modeling, Sentiment, SNA, dan Chatbot",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ... middleware dan routing
    return app
```

---

### B. Singleton Pattern untuk Model ML (`app/domains/sentiment/engine.py`)

Mencegah memori meledak karena model dimuat berulang kali dengan override `__new__()`.

```python
import os
import pickle
from keras.models import load_model
from app.core.logger import log

class SentimentModelManager:
    """Singleton Engine untuk Keras models (.h5) dan tokenizers (.pickle)."""
    _instance = None

    def __new__(cls):
        """Override __new__ untuk memastikan hanya satu instance."""
        if cls._instance is None:
            cls._instance = super(SentimentModelManager, cls).__new__(cls)
            cls._instance._is_loaded = False
        return cls._instance

    def load_models(self):
        """Lazy loading: hanya load jika belum pernah di-load."""
        if not self._is_loaded:
            log.info("🧠 Loading ML models...")

            # Setup Paths (relative dari domain)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            models_path = os.path.join(root_dir, "models", "raw", "sentiment")

            # Load CNN Model
            self.cnn_model, self.cnn_tokenizer = self._load_keras_model(
                models_path, "model-cnn-no-testing.h5",
                "tokenizer-cnn-no-testing.pickle"
            )

            # Load CNN-LSTM Model
            self.cnn_lstm_model, self.cnn_lstm_tokenizer = self._load_keras_model(
                models_path, "model-cnn-lstm-no-testing.h5",
                "tokenizer-cnn-lstm-no-testing.pickle"
            )

            self._is_loaded = True
            log.info("✅ All models loaded successfully.")

    def _load_keras_model(self, base_path, model_file, tokenizer_file):
        try:
            m_path = os.path.join(base_path, model_file)
            t_path = os.path.join(base_path, tokenizer_file)

            model = load_model(m_path)
            with open(t_path, "rb") as f:
                tokenizer = pickle.load(f)

            log.info(f"✅ Model loaded: {model_file}")
            return model, tokenizer
        except Exception as e:
            log.error(f"❌ Failed loading {model_file}: {e}")
            return None, None

# Factory Function untuk Dependency Injection
def get_sentiment_models() -> SentimentModelManager:
    manager = SentimentModelManager()
    manager.load_models()
    return manager
```

---

### C. Stateless Service & Asyncio Threading (`app/domains/sentiment/services.py`)

⚠️ **CRITICAL**: Service menerima _state_ dari argumen fungsi, **BUKAN** via `self.data`. Operasi CPU-bound dibungkus `asyncio.to_thread()`.

```python
import asyncio
from fastapi import Depends

class SentimentService:
    def __init__(self, repository: SentimentRepository, models: SentimentModelManager):
        self.repository = repository

        # Ambil referensi model dari Singleton (read-only)
        self.cnn_model = models.cnn_model
        self.cnn_tokenizer = models.cnn_tokenizer
        self.cnn_lstm_model = models.cnn_lstm_model
        self.cnn_lstm_tokenizer = models.cnn_lstm_tokenizer

        # Initialize paths (read-only)
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.utils_path = os.path.join(self.base_path, "utils")

        # Load utilities (read-only)
        self.normalization_dict = self._load_normalization_dict()
        self.stop_words = self._load_stopwords()

    async def process_sentiment(self, project_id: str) -> dict:
        # 1. Tarik data dari DB (I/O Bound -> await biasa)
        documents = await self.repository.get_documents(project_id)

        # 2. Preprocessing (CPU Bound -> to_thread agar Event Loop tidak macet)
        texts = await asyncio.to_thread(self._run_preprocessing, documents)

        # 3. Keras Inference (CPU/GPU Bound -> to_thread)
        predictions = await asyncio.to_thread(
            self._run_inference, texts, self.cnn_model, self.cnn_tokenizer
        )

        return predictions

    def _run_preprocessing(self, documents: list) -> list:
        """CPU-bound preprocessing (dijalankan di thread pool)."""
        # Pandas, regex, stemming operations here
        return processed_texts

    def _run_inference(self, texts, model, tokenizer):
        """CPU-bound model inference (dijalankan di thread pool)."""
        # Keras prediction operations here
        return predictions

# Factory dengan Dependency Injection
def get_sentiment_service(
    repo: SentimentRepository = Depends(get_sentiment_repository),
    models: SentimentModelManager = Depends(get_sentiment_models)
) -> SentimentService:
    """Factory function untuk inject dependencies ke Service."""
    return SentimentService(repo, models)
```

**Pattern Penting**:

- ✅ `self.base_path` → read-only, inisialisasi di `__init__`
- ✅ `self.cnn_model` → referensi ke singleton, read-only
- ❌ `self.current_data = []` → JANGAN! Mutable state causes race conditions
- ✅ State diterima via parameter fungsi: `process_sentiment(project_id: str)`

---

### D. Clean Controller (`app/domains/sentiment/api.py`)

Controller **HANYA** mengurus routing. Tidak tahu menahu soal ML atau DB. Semua logic ada di _Service_.

```python
from fastapi import APIRouter, HTTPException, Depends, status
from app.core.logger import log

router = APIRouter()

@router.post("/predict", response_model=ClassifySentimentResponse)
async def predict_sentiment(
    request: ClassifySentimentRequest,
    service: SentimentService = Depends(get_sentiment_service)  # DI Magic ✨
):
    """
    Endpoint untuk sentiment classification.
    Service di-inject otomatis oleh FastAPI via Depends().
    """
    try:
        result = await service.process_sentiment(request.project_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Data not found for the given project_id"
            )

        return ClassifySentimentResponse(status="success", data=result)

    except HTTPException:
        raise  # Re-raise HTTPException agar status code tidak berubah jadi 500

    except Exception as e:
        log.exception(f"Error in predict_sentiment endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan: {str(e)}"
        )
```

**Error Handling Pattern**:

1. Tangkap `HTTPException` pertama dan re-raise (preserve status code)
2. Tangkap `Exception` generik, log dengan `log.exception()`, raise sebagai 500

---

### E. Repository Pattern - Dual Database Strategy (`app/domains/topic_modeling/repositories.py`)

Mengatasi _Shared Database_ dengan NestJS menggunakan **Dual DB Pattern**:

- **Unmanaged collections** (milik NestJS): PyMongo native
- **Managed collections** (milik AI): Beanie ODM

```python
from typing import List, Dict, Optional
from pymongo.asynchronous.database import AsyncDatabase
from fastapi import Depends
from datetime import datetime

from app.core.logger import log
from app.db.database import get_db
from .models import TopicsModel, DocumentsModel

class TopicRepository:
    def __init__(self, db: AsyncDatabase):
        # Unmanaged collection (PyMongo native) - Data dari NestJS
        self.tweets_collection = db["tweets"]

    async def get_tweet_by_keyword(self, topic_data: dict) -> Optional[Dict]:
        """
        Ambil tweets dari koleksi NestJS menggunakan native PyMongo aggregation.
        """
        try:
            keyword = topic_data.get("keyword")
            start_date = topic_data.get("start_date")
            end_date = topic_data.get("end_date")

            # Build aggregation pipeline
            match_stage = {
                "$match": {
                    "full_text": {"$regex": keyword.replace(" ", "|"), "$options": "i"}
                }
            }

            pipeline = [match_stage]

            # Date filtering jika ada
            if start_date and end_date:
                start_datetime = datetime.strptime(
                    f"{start_date} 00:00:00 +0000", "%Y-%m-%d %H:%M:%S %z"
                )
                end_datetime = datetime.strptime(
                    f"{end_date} 23:59:59 +0000", "%Y-%m-%d %H:%M:%S %z"
                )

                pipeline.extend([
                    {"$addFields": {"parsed_date": {"$toDate": "$created_at"}}},
                    {"$match": {"parsed_date": {"$gte": start_datetime, "$lte": end_datetime}}}
                ])

            # Projection
            pipeline.append({
                "$project": {
                    "_id": 0, "full_text": 1, "username": 1, "tweet_url": 1
                }
            })

            # Execute aggregation (native PyMongo)
            cursor = await self.tweets_collection.aggregate(pipeline)
            tweets = await cursor.to_list(length=None)

            return {
                "keyword": keyword,
                "total_tweets": len(tweets),
                "tweets": tweets
            }

        except Exception as e:
            log.exception(f"Error getting tweets: {e}")
            return None

    async def create_topics(self, topics_data: List[dict]) -> List[TopicsModel]:
        """
        Simpan topics menggunakan Beanie ODM (managed collection).
        """
        try:
            # Delete existing untuk project ini
            project_id = topics_data[0].get("projectId")
            await TopicsModel.find(TopicsModel.projectId == project_id).delete()

            # Insert new topics
            docs = [TopicsModel(**t) for t in topics_data]
            await TopicsModel.insert_many(docs)

            log.info(f"💾 Saved {len(docs)} topics for project {project_id}")
            return docs

        except Exception as e:
            log.exception(f"Error creating topics: {e}")
            return []

    async def get_topics_by_project_id(self, project_id: str) -> List[TopicsModel]:
        """Retrieve topics using Beanie ODM queries."""
        return await TopicsModel.find(
            TopicsModel.projectId == project_id
        ).to_list()

# Factory Function untuk DI
def get_topic_repository(db: AsyncDatabase = Depends(get_db)) -> TopicRepository:
    return TopicRepository(db)
```

**Key Points**:

- Koleksi `tweets` → PyMongo native (external, read-only)
- Koleksi `topics`, `documents` → Beanie ODM (AI-generated, managed)
- Beanie models **MUST** didaftarkan di `app/db/database.py::init_db()`

---

### F. Lazy Loading Pattern dengan `@property` (`app/domains/topic_modeling/services.py`)

Untuk resource yang mahal/berat tapi tidak selalu dipakai, gunakan lazy loading via `@property`.

```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class TopicModelingService:
    def __init__(self, repository: TopicRepository, llm: BaseChatModel):
        self.repository = repository
        self.llm = llm

        # Path initialization (read-only)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        self.preprocessing_path = os.path.join(root_dir, "models", "raw", "topic_modeling", "preprocessing")

        # Lazy-loaded resources (via @property)
        self._stemmer = None  # Private variable

    @property
    def stemmer(self):
        """Lazy load Sastrawi stemmer (hanya load saat pertama kali diakses)."""
        if self._stemmer is None:
            self._stemmer = StemmerFactory().create_stemmer()
        return self._stemmer

    # Sekarang bisa pakai self.stemmer di mana saja
    def _stem_tokens(self, tokens: list) -> list:
        return [self.stemmer.stem(tok) for tok in tokens]
```

**Kenapa Lazy Loading?**

- Sastrawi stemmer memakan waktu untuk inisialisasi
- Tidak semua request butuh stemming
- Hindari overhead di `__init__` yang dipanggil setiap request

---

## 5\. 🛡️ Guardrails & Best Practices

### ❌ CRITICAL - Hal yang HARUS Dihindari

1. **NO Stateful Services**:

   ```python
   # ❌ SALAH - Mutable state dalam service
   class BadService:
       def __init__(self):
           self.current_data = []  # Race condition!

   # ✅ BENAR - Stateless, state via parameter
   class GoodService:
       async def process(self, data: list):
           processed = self._transform(data)
           return processed
   ```

2. **Thread Pool for CPU Intensive Tasks**:
   - ⚠️ `Pandas`, `Scikit-Learn`, `Sastrawi`, `Joblib`, `Keras/TensorFlow` adalah **blocking**
   - Wajib gunakan `await asyncio.to_thread()` untuk mencegah API _hang/timeout_

   ```python
   # ❌ SALAH - Blocks event loop
   df = pd.DataFrame(data).apply(heavy_function)

   # ✅ BENAR - Non-blocking
   df = await asyncio.to_thread(lambda: pd.DataFrame(data).apply(heavy_function))
   ```

3. **Joblib Threading Backend** (CRITICAL ⚠️):

   ```python
   # ❌ SALAH - Default multiprocessing/loky tidak bisa serialize async DB connections
   results = Parallel(n_jobs=-1)(delayed(train_model)(n) for n in range)

   # ✅ BENAR - Gunakan threading backend
   results = Parallel(n_jobs=-1, backend="threading")(delayed(train_model)(n) for n in range)
   ```

4. **Hanya Gunakan DI (`Depends`)**:

   ```python
   # ❌ SALAH - Global instantiation
   service = MyService()

   # ✅ BENAR - Factory function + DI
   def get_my_service(repo = Depends(get_repo)) -> MyService:
       return MyService(repo)
   ```

5. **Path Initialization Pattern**:

   ```python
   # ✅ BENAR - Gunakan centralized config
   from app.core.config import settings

   def __init__(self):
       self.models_path = str(settings.sentiment_models_path)
       self.utils_path = str(settings.sentiment_utils_path)

   # ❌ SALAH - Manual path construction (deprecated)
   def __init__(self):
       current_dir = os.path.dirname(os.path.abspath(__file__))
       root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
       self.utils_path = os.path.join(root_dir, "models", "raw", "sentiment", "utils")
   ```

   **Path yang tersedia di config**:
   - `settings.topic_modeling_path` - Topic modeling models & preprocessing
   - `settings.topic_modeling_utils_path` - KBBA dictionary, stopwords
   - `settings.topic_modeling_preprocessing_path` - OCTIS dataset artifacts
   - `settings.sentiment_models_path` - Sentiment CNN/LSTM models
   - `settings.sentiment_utils_path` - Kamus normalisasi, stopwords
   - `settings.emotion_models_path` - Emotion CNN/BiLSTM models
   - `settings.emotion_utils_path` - Emotion preprocessing utils

6. **Logging Convention**:

   ```python
   # ❌ SALAH
   print("Processing data...")

   # ✅ BENAR - Gunakan logger dengan emoji
   from app.core.logger import log
   log.info("🚀 Starting data processing...")
   log.exception(f"❌ Error: {e}")  # Captures stack trace
   ```

7. **Register Beanie Models**:

   ```python
   # File: app/db/database.py
   from app.domains.topic_modeling.models import DocumentsModel, TopicsModel
   from app.domains.sentiment.models import SentimentModel

   async def init_db():
       await init_beanie(
           database=get_db(),
           document_models=[
               DocumentsModel,
               TopicsModel,
               SentimentModel,
               # ⚠️ WAJIB daftarkan semua Beanie models di sini!
           ]
       )
   ```

8. **LLM Error Recovery Pattern**:

   ```python
   async def _augment_batch(self, batch: list, batch_num: int) -> list:
       try:
           res = await self.llm.ainvoke([{"role": "user", "content": prompt}])
           return parse_llm_response(res.content)
       except Exception as e:
           log.error(f"Batch {batch_num} failed: {e}. Using original data.")
           return batch  # Fallback ke data original
   ```

---

## 6\. 🚀 Development Workflows

### Running the Application

```bash
# 1. Install dependencies via uv
uv sync

# 2. Start MinIO (untuk model storage)
docker-compose up -d

# 3. Setup environment variables
cp .env.example .env
# Edit .env dan isi MONGODB_URI, OPENAI_API_KEY

# 4. Run development server (auto-reload enabled)
uv run fastapi dev main.py
```

Server akan berjalan di `http://localhost:8000` dengan OpenAPI docs di `/docs`.

### Environment Variables Required

```bash
# .env file
MONGODB_URI=mongodb://localhost:27017  # MongoDB dari NestJS
MONGO_DB_NAME=socialabs_ai_db
OPENAI_API_KEY=sk-...                  # Wajib untuk LLM operations
OPENAI_MODEL_NAME=gpt-4o-mini          # Default model
ENV=development

# Path Configuration (relative to project root)
MODELS_BASE_PATH=models/raw            # Base path untuk semua model artifacts
```

**Path Configuration**:

- Semua path dikonfigurasi melalui `.env` dan di-load di `app/core/config.py`
- Services dan Engine menggunakan `settings` dari config untuk mendapatkan path
- Computed properties di config: `topic_modeling_path`, `sentiment_models_path`, `emotion_models_path`, dll.
- Pattern: `from app.core.config import settings` → `settings.sentiment_models_path`

### Testing (Belum Diimplementasi)

- Folder `test/` masih kosong
- Ketika menambahkan tests, gunakan `pytest` dan `pytest-asyncio` (sudah ada di dev dependencies)
- Pattern untuk mocking: Override `Depends()` di test fixtures

---

## 7\. 📚 Domain-Specific Knowledge

### Topic Modeling - 10-Step Indonesian NLP Pipeline

Pipeline preprocessing untuk teks media sosial berbahasa Indonesia:

1. **URL Removal** → `re.sub(r"(?:https?://|www\.)...", "", text)`
2. **Emoticon Replacement** → `{r":\)": "emot-senyum", ...}` (semantic tokens)
3. **Twitter Symbol Removal** → `re.sub(r"#\w+|@\w+|\bRT\b", " ", text)`
4. **Symbol/Punctuation Cleaning** → `re.sub(r"[^a-zA-Z\s]", " ", text)`
5. **Tokenization + Case Folding** → `[token.lower() for token in text.split()]`
6. **Delete Extra Letters** → `re.sub(r"([A-Za-z])\1{2,}", r"\1", token)` (collapse repeats)
7. **Normalization** → KBBA dictionary (`models/raw/topic_modeling/utils/kbba.txt`) untuk Indonesian slang
8. **Stemming** → Sastrawi via `StemmerFactory()` (lazy-loaded dengan `@property`)
9. **Stopword Curation** → TF-IDF (>0.7) + rare words + pronouns + manual list
10. **Final Filtering** → remove tokens <3 chars, deduplicate empty documents

**OCTIS Dataset Format**: Save sebagai `corpus.tsv` (tab-separated) + `vocabulary.txt` di `preprocessing/vocabs/{keyword}/`

**ETM Hyperparameters**:

```python
ETM(num_topics=num_topics, num_epochs=100, batch_size=256, dropout=0.3,
    activation="tanh", embeddings_path="wiki/idwiki_word2vec_100_new_lower.txt",
    embeddings_type="word2vec", t_hidden_size=512, wdecay=1e-5,
    lr=0.001, optimizer="SGD")
```

### LangChain Integration

- **Temperature**: `0.7` (lihat `app/shared/llm.py`)
- **Use cases**: Data augmentation (rephrase/translate), topic context generation
- **Batch processing pattern**:

  ```python
  batches = [data[i:i+10] for i in range(0, len(data), 10)]
  results = await asyncio.gather(*[self._augment_batch(b, i) for i, b in enumerate(batches)])
  ```

- **Fallback strategy**: Always return original data if LLM fails

---

## 8\. 🔗 Integration Points

### External MongoDB (Shared Database dengan NestJS)

- Koleksi `tweets` di-manage oleh NestJS backend
- Schema: `{full_text, username, tweet_url, created_at, ...}`
- Access via PyMongo native di Repository: `self.tweets_collection = db["tweets"]`
- Query menggunakan aggregation pipelines untuk filtering

### MinIO Storage

- Docker service di `docker-compose.yml`
- Ports: 9000 (API) / 9001 (Console)
- Credentials: `admin / password123`
- Mount: `./models/minio:/data`
- Digunakan untuk menyimpan model artifacts

---

## 9\. 📖 Key Files Reference

**Core Architecture**:

- `app/server.py` - Application factory dengan lifespan pre-warming
- `app/db/database.py` - Beanie initialization dan DB connection
- `app/shared/llm.py` - LangChain LLM singleton manager
- `app/api/router.py` - Main API router menggabungkan semua domains
- `app/core/config.py` - Environment configuration
- `app/core/logger.py` - Custom logging setup

**Domain Examples (Canonical Patterns)**:

- `app/domains/topic_modeling/services.py` - Complete ML pipeline dengan async/threading
- `app/domains/topic_modeling/repositories.py` - Dual DB pattern (pymongo + Beanie)
- `app/domains/topic_modeling/api.py` - Route handlers dengan DI
- `app/domains/sentiment/engine.py` - ML model singleton pattern

**Dependencies**:

- `pyproject.toml` - Custom fork dependencies via `uv` (mpstemmer, octis)
- `docker-compose.yml` - MinIO service configuration
