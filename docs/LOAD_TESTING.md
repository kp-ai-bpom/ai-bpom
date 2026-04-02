# Load Testing Guide - SociaLabs AI Backend

Dokumentasi lengkap untuk melakukan load testing pada semua endpoint API menggunakan Locust.

## 📋 Daftar Isi

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [User Classes](#user-classes)
- [Skenario Testing](#skenario-testing)
- [Monitoring & Hasil](#monitoring--hasil)
- [Best Practices](#best-practices)

---

## Prerequisites

Pastikan dependencies sudah terinstall:

```bash
# Locust sudah ada di dev dependencies
uv sync --group dev

# Atau install manual
uv add --dev locust
```

**Requirements:**

- Python >= 3.10
- FastAPI server running di `http://localhost:8000`
- MongoDB connection untuk data persistence
- ML models sudah di-load (sentiment, emotion)

---

## Quick Start

### 1. Jalankan Server FastAPI

Pastikan server berjalan sebelum load testing:

```bash
# Terminal 1 - Start server
uv run fastapi dev main.py
```

### 2. Jalankan Load Test

```bash
# Terminal 2 - Load test dengan Web UI
uv run locust

# Akses Web UI di: http://localhost:8089
```

**Parameter Load Test:**

- **Number of users**: Total concurrent users (mulai dari 10-50)
- **Spawn rate**: Users spawned per detik (mulai dari 2-5)
- **Host**: `http://localhost:8000` (atau sesuai server Anda)

### 3. Headless Mode (Tanpa Web UI)

```bash
# Test singkat - 10 users, 2 spawn rate, 60 detik
uv run locust --headless -u 10 -r 2 -t 60s --host=http://localhost:8000

# Test medium - 50 users, 5 spawn rate, 5 menit
uv run locust --headless -u 50 -r 5 -t 5m --host=http://localhost:8000

# Test dengan HTML report
uv run locust --headless -u 100 -r 10 -t 10m --host=http://localhost:8000 --html=reports/load_test_report.html
```

---

## User Classes

File `locustfile.py` menyediakan 3 user classes dengan behavior berbeda:

### 1. **SociaLabsAIUser** (Default - Balanced)

User dengan mixed workload (read + write operations).

**Karakteristik:**

- Wait time: 1-5 detik antar request
- Task distribution:
  - 30% System endpoints (health check, status)
  - 15% Topic Modeling (3% POST, 12% GET)
  - 20% Sentiment Analysis (8% POST, 12% GET)
  - 20% Emotion Classification (8% POST, 12% GET)
  - 15% Social Network Analysis (6% POST, 9% GET)

**Cara menggunakan:**

```bash
uv run locust -f locustfile.py SociaLabsAIUser --host=http://localhost:8000
```

### 2. **ReadOnlyUser** (Read-Heavy Workload)

User yang hanya melakukan GET operations (testing caching & retrieval).

**Karakteristik:**

- Wait time: 0.5-2 detik (lebih cepat)
- 100% GET endpoints
- Cocok untuk mengetes performa database queries dan caching

**Cara menggunakan:**

```bash
uv run locust -f locustfile.py ReadOnlyUser --headless -u 50 -r 10 -t 5m
```

### 3. **HeavyProcessingUser** (ML Pipeline Stress Test)

User yang fokus pada heavy ML processing endpoints.

**Karakteristik:**

- Wait time: 5-15 detik (slow, heavy operations)
- Fokus pada POST endpoints (topic modeling, sentiment, emotion, SNA)
- Timeout lebih lama (120-180 detik)
- Cocok untuk stress testing ML pipelines

**Cara menggunakan:**

```bash
uv run locust -f locustfile.py HeavyProcessingUser --headless -u 5 -r 1 -t 10m
```

---

## Skenario Testing

### Skenario 1: Smoke Test (Verifikasi Dasar)

**Tujuan**: Memastikan semua endpoint bisa diakses tanpa error.

```bash
uv run locust --headless -u 5 -r 1 -t 2m --host=http://localhost:8000
```

**Expected Result**: Success rate > 95%

---

### Skenario 2: Normal Load (Simulasi Traffic Normal)

**Tujuan**: Menguji performa dengan beban normal (10-20 concurrent users).

```bash
# Mixed workload
uv run locust -f locustfile.py SociaLabsAIUser --headless -u 20 -r 5 -t 10m --html=reports/normal_load.html

# Read-only workload
uv run locust -f locustfile.py ReadOnlyUser --headless -u 30 -r 10 -t 10m --html=reports/read_heavy.html
```

**Expected Result**:

- Response time < 2s (GET endpoints)
- Response time < 30s (ML inference endpoints)
- Success rate > 98%

---

### Skenario 3: Stress Test (Menemukan Breaking Point)

**Tujuan**: Menemukan limit sistem sebelum performance degradasi.

```bash
# Gradual ramp-up
uv run locust --headless -u 100 -r 10 -t 15m --host=http://localhost:8000 --html=reports/stress_test.html
```

**Monitoring**:

- Perhatikan response time saat user meningkat
- Cek error rate mulai naik di berapa concurrent users
- Monitor CPU & Memory usage server

---

### Skenario 4: Spike Test (Traffic Surge Mendadak)

**Tujuan**: Menguji resilience saat traffic spike tiba-tiba.

```bash
# Spawn 50 users sekaligus dalam 5 detik
uv run locust --headless -u 50 -r 10 -t 5m --host=http://localhost:8000
```

**Expected Behavior**: Server tetap stabil meskipun ada traffic spike.

---

### Skenario 5: ML Pipeline Stress Test

**Tujuan**: Khusus testing heavy ML operations (CNN, LSTM, ETM).

```bash
uv run locust -f locustfile.py HeavyProcessingUser --headless -u 10 -r 2 -t 20m --html=reports/ml_stress.html
```

**Monitoring**:

- Model loading time
- Inference time per model
- Memory usage (models di-load ke RAM)
- Async task queue behavior

---

## Tag-Based Testing

Locust mendukung filtering task berdasarkan tag. Gunakan untuk testing endpoint spesifik:

### Test hanya System endpoints:

```bash
uv run locust --tags system --headless -u 20 -r 5 -t 5m
```

### Test hanya ML endpoints (sentiment + emotion):

```bash
uv run locust --tags sentiment emotion --headless -u 10 -r 2 -t 10m
```

### Test hanya GET endpoints (exclude heavy operations):

```bash
uv run locust --tags get --headless -u 50 -r 10 -t 5m
```

### Exclude heavy processing:

```bash
uv run locust --exclude-tags heavy modeling --headless -u 30 -r 10 -t 10m
```

**Available Tags**:

- `system`, `health`, `status`, `reload`
- `topics`, `modeling`, `get`
- `sentiment`, `predict`, `get`
- `emotion`, `predict`, `get`
- `sna`, `community`, `buzzer`, `get`
- `readonly`, `heavy`

---

## Monitoring & Hasil

### 1. Web UI Dashboard (Real-time)

Akses `http://localhost:8089` untuk monitoring real-time:

**Metrics yang ditampilkan:**

- Request count & RPS (Requests Per Second)
- Response time (min, max, median, 95th percentile)
- Failure rate
- User count & spawn rate
- Charts & graphs

### 2. HTML Report (Post-test Analysis)

Generate HTML report dengan flag `--html`:

```bash
uv run locust --headless -u 50 -r 10 -t 10m --html=reports/test_$(date +%Y%m%d_%H%M%S).html
```

**Report berisi:**

- Summary statistics
- Response time distribution
- Chart trends
- Failures breakdown

### 3. CSV Export (Raw Data)

Export data mentah untuk analisis lanjutan:

```bash
uv run locust --headless -u 50 -r 10 -t 10m \
  --csv=reports/test \
  --csv-full-history
```

**Output files:**

- `test_stats.csv` - Request statistics
- `test_stats_history.csv` - Time-series data
- `test_failures.csv` - Failure details

### 4. Command Line Stats

Mode headless akan print stats ke console setiap 2 detik.

---

## Metrics yang Perlu Diperhatikan

### 1. **Response Time**

| Endpoint Type            | Target Response Time |
| ------------------------ | -------------------- |
| Health Check             | < 100ms              |
| GET (Retrieval)          | < 2s                 |
| POST (Sentiment/Emotion) | < 30s                |
| POST (Topic Modeling)    | < 120s               |
| POST (SNA)               | < 60s                |

### 2. **Throughput (RPS)**

- Target: >= 10 RPS untuk mixed workload
- Target: >= 50 RPS untuk read-only workload

### 3. **Error Rate**

- Normal: < 1%
- Acceptable: < 5%
- **Action needed if > 10%**

### 4. **Percentile Response Times**

- 50th percentile (median) - typical user experience
- 95th percentile - outliers (slow requests)
- 99th percentile - worst case scenarios

---

## Best Practices

### 1. **Persiapan Data Test**

Sebelum load testing, pastikan:

```python
# Update test_project_ids di locustfile.py dengan project yang sudah ada data
test_project_ids = [
    "proj_real_001",  # Project dengan data real
    "proj_real_002",
]
```

Atau pre-populate data:

```bash
# Jalankan topic modeling dulu untuk beberapa project
curl -X POST http://localhost:8000/api/topics/topic-modelling \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "proj_test_001",
    "keyword": "test",
    "start_date": "2025-02-01",
    "end_date": "2025-02-08"
  }'
```

### 2. **Bertahap (Ramp-up)**

Jangan langsung spawn banyak user sekaligus:

```bash
# ❌ Bad - 100 users sekaligus (spawn rate = 100)
uv run locust --headless -u 100 -r 100 -t 10m

# ✅ Good - 100 users bertahap (10 users per detik)
uv run locust --headless -u 100 -r 10 -t 10m
```

### 3. **Baseline Measurement**

Ukur performa dengan 1 user dulu untuk baseline:

```bash
uv run locust --headless -u 1 -r 1 -t 5m --html=reports/baseline.html
```

### 4. **Monitor Resource Usage**

Pantau server resources saat load testing:

```bash
# CPU & Memory (Windows)
# Task Manager atau:
Get-Process -Name python | Select-Object CPU, WorkingSet

# Linux/Mac
htop
# atau
top -p $(pgrep -f fastapi)
```

### 5. **Isolasi Environment**

- Jalankan load test di environment terpisah (bukan production!)
- Gunakan database test yang terpisah
- Matikan logging verbose saat testing (impact performa)

### 6. **Timeout Configuration**

Sesuaikan timeout berdasarkan endpoint:

```python
# Quick endpoints
self.client.get("/api/system/health", timeout=5)

# ML inference
self.client.post("/api/sentiments/predict", json=payload, timeout=60)

# Heavy processing
self.client.post("/api/topics/topic-modelling", json=payload, timeout=180)
```

---

## Troubleshooting

### Problem: Banyak Timeout

**Solusi**:

1. Kurangi concurrent users
2. Tingkatkan timeout di locustfile.py
3. Cek server resources (CPU/Memory)
4. Periksa database connection pool

### Problem: Connection Refused

**Solusi**:

1. Pastikan FastAPI server running: `http://localhost:8000/docs`
2. Cek firewall/antivirus tidak blocking
3. Gunakan `--host` flag yang benar

### Problem: Error Rate Tinggi (>10%)

**Solusi**:

1. Cek logs FastAPI: `tail -f logs/app.log`
2. Periksa apakah data test tersedia (404 errors normal jika no data)
3. Validate request payloads di locustfile.py
4. Cek model loading status: `/api/system/models-status`

### Problem: Response Time Lambat

**Solusi**:

1. Profiling: gunakan monitoring tools
2. Cek database query performance
3. Review ML model inference time
4. Consider async optimization di services.py

---

## Advanced: Custom Scenarios

Buat custom scenario dengan Python:

```python
# custom_scenario.py
from locust import HttpUser, task, between

class CustomUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def my_custom_workflow(self):
        # 1. Topic modeling
        response = self.client.post("/api/topics/topic-modelling", json={...})
        project_id = response.json()["data"]["project_id"]

        # 2. Get topics
        self.client.get(f"/api/topics/topics-by-project/{project_id}")

        # 3. Run sentiment
        self.client.post("/api/sentiments/predict", json={"project_id": project_id, ...})

        # 4. Get results
        self.client.get(f"/api/sentiments/by-project/{project_id}")
```

Jalankan:

```bash
uv run locust -f custom_scenario.py
```

---

## Reporting & Analysis

### Generate Comprehensive Report

```bash
# Create reports directory
mkdir -p reports

# Run test dengan full reporting
uv run locust --headless \
  -u 50 -r 10 -t 10m \
  --host=http://localhost:8000 \
  --html=reports/full_test_$(date +%Y%m%d_%H%M%S).html \
  --csv=reports/test \
  --csv-full-history \
  --loglevel INFO
```

### Analisis Results

Import CSV ke tools analisis:

- **Excel/Google Sheets**: Untuk charting manual
- **Pandas**: Untuk analisis programmatic
- **Grafana**: Untuk monitoring dashboard

---

## Tips & Tricks

1. **Use Tags** untuk testing specific features
2. **Start Small** (5-10 users) lalu scale up
3. **Monitor DB** connection pool & query time
4. **Cache Strategy**: Test dengan/tanpa caching untuk comparison
5. **CI/CD Integration**: Automate load testing di pipeline

---

## Referensi

- [Locust Documentation](https://docs.locust.io/)
- [FastAPI Performance Guide](https://fastapi.tiangolo.com/deployment/concepts/)
- [Load Testing Best Practices](https://docs.locust.io/en/stable/running-locust-distributed.html)

---

**Happy Load Te! 🚀**

Jangan lupa monitor server health dan resources saat testing!
