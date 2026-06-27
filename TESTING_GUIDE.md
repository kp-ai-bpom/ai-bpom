# Setup & Testing Guide untuk Penilaian Makalah REST API

## 1. Start Infrastructure (Docker Compose)

```bash
docker-compose up -d
```

Pastikan 3 service sudah running:
- PostgreSQL: localhost:5452 (user: light_postgres, pass: light_postgres_root)
- Neo4j: localhost:7474 (user: neo4j, pass: lightrag_neo4j_root)
- MinIO: localhost:9000 (user: admin, pass: password123)

## 2. Setup Environment

Copy `.env.example` ke `.env` dan pastikan sudah diisi:

```bash
ENV=development
POSTGRES_URI=postgresql+asyncpg://light_postgres:light_postgres_root@localhost:5452/lightrag
OPENAI_API_KEY=sk-your-key
AI_INSTRUCT_MODEL_NAME=gpt-4o-mini
```

## 3. Install Dependencies

```bash
uv sync
```

## 4. Run Database Migrations

```bash
# Create tables
poe migrate

# Or manually with Alembic
alembic upgrade head
```

## 5. Seed Sample Data (Optional)

```bash
python -m scripts.seed_penilaian_makalah
```

## 6. Start Server

```bash
# Development mode (auto-reload)
poe dev

# Or manually
fastapi dev main.py --port 8080
```

Server akan berjalan di: `http://localhost:8080`

## 7. Test Endpoints

### Quick Manual Test

```bash
# Health check
curl http://localhost:8080/api/penilaian-makalah/health

# Get supported query modes
curl http://localhost:8080/api/penilaian-makalah/modes

# Get history (limit 10)
curl "http://localhost:8080/api/penilaian-makalah/history?limit=10"

# List papers
curl http://localhost:8080/api/penilaian-makalah/papers

# List tema files
curl http://localhost:8080/api/penilaian-makalah/tema
```

### Run Full Test Suite

```bash
# Unit tests + schema validation
python -m pytest tests/test_penilaian_makalah.py -v

# Manual HTTP tests (server harus running)
python tests/test_penilaian_makalah.py manual
```

## 8. API Documentation

Akses Swagger UI:
- **Swagger**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## Endpoints Overview

### Health & Configuration
- `GET /api/penilaian-makalah/health` - Health check
- `GET /api/penilaian-makalah/modes` - Supported query modes

### File Management
- `GET /api/penilaian-makalah/papers` - List papers in MinIO
- `GET /api/penilaian-makalah/tema` - List tema files in MinIO
- `POST /api/penilaian-makalah/upload/paper` - Upload paper
- `POST /api/penilaian-makalah/upload/knowledge` - Upload knowledge file

### History
- `GET /api/penilaian-makalah/history` - List evaluations (paginated)
- `GET /api/penilaian-makalah/history/{id}` - Get specific evaluation

### Evaluation
- `POST /api/penilaian-makalah/evaluate` - Evaluate from MinIO files
- `POST /api/penilaian-makalah/evaluate/upload` - Evaluate with file upload
- `POST /api/penilaian-makalah/evaluate/raw` - Evaluate with raw text

## Request/Response Examples

### Evaluate Request

```json
{
  "jabatan": "Analis Data Senior",
  "paper_filename": "paper_001.pdf",
  "tema_filename": "tema_2026.docx",
  "query_mode": "hybrid",
  "m_samples": 7,
  "temperature": 1.0
}
```

### Evaluate Response

```json
{
  "ringkasan": "Makalah berkualitas baik dengan analisis mendalam...",
  "final_score": 78.4,
  "scores": {
    "n1_kesesuaian_judul": 82.0,
    "n2_kesesuaian_isi": 78.0,
    "n3_sistematika": 75.0,
    "n4_ketajaman_analisis": 80.0,
    "n5_penggunaan_bahasa": 77.0
  },
  "justification": {...},
  "evidence": {...},
  "uncertainty": {
    "per_criteria": {...},
    "weighted_aggregate": 0.029,
    "overall_status": "✅ YAKIN (Konsisten)",
    "most_uncertain_criteria": "n4"
  },
  "valid_samples": 7,
  "total_samples": 7
}
```

## Troubleshooting

### Service not starting
- Pastikan Docker containers sudah running: `docker-compose ps`
- Cek logs: `docker-compose logs postgres`

### Migration failed
- Reset database: `alembic downgrade base`
- Rerun migration: `alembic upgrade head`

### Port already in use
- Check port: `netstat -ano | findstr :8080` (Windows)
- Kill process: `taskkill /PID <pid> /F`

### Schema validation errors
- Lihat pydantic errors di stdout
- Update request body sesuai schema di `/docs`
