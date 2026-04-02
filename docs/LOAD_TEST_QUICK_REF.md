# Load Testing Quick Reference

## 🚀 Quick Start

```bash
# Web UI mode (interactive)
uv run locust

# Headless mode (automated)
uv run locust --headless -u 10 -r 2 -t 60s --host=http://localhost:8000
```

---

## 📋 Pre-configured Scenarios

### Windows (PowerShell)

```powershell
# Smoke test
.\run_load_test.ps1 smoke

# Normal load
.\run_load_test.ps1 normal

# Stress test
.\run_load_test.ps1 stress

# Spike test
.\run_load_test.ps1 spike

# ML stress
.\run_load_test.ps1 ml

# Read-only
.\run_load_test.ps1 readonly

# Web UI
.\run_load_test.ps1 web
```

### Linux/Mac (Bash)

```bash
# Smoke test
./run_load_test.sh smoke

# Normal load
./run_load_test.sh normal

# Stress test
./run_load_test.sh stress

# Spike test
./run_load_test.sh spike

# ML stress
./run_load_test.sh ml

# Read-only
./run_load_test.sh readonly

# Web UI
./run_load_test.sh web
```

---

## 🎯 Common Commands

### Basic Test

```bash
# 10 users, 2 per second, 5 minutes
uv run locust --headless -u 10 -r 2 -t 5m --host=http://localhost:8000
```

### With HTML Report

```bash
uv run locust --headless -u 50 -r 10 -t 10m \
  --host=http://localhost:8000 \
  --html=reports/test_report.html
```

### With CSV Export

```bash
uv run locust --headless -u 50 -r 10 -t 10m \
  --host=http://localhost:8000 \
  --csv=reports/test \
  --csv-full-history
```

### Full Reporting

```bash
uv run locust --headless -u 50 -r 10 -t 10m \
  --host=http://localhost:8000 \
  --html=reports/full_report.html \
  --csv=reports/test \
  --csv-full-history \
  --loglevel INFO
```

---

## 🏷️ Tag-Based Testing

### Test Specific Endpoints

```bash
# Only system endpoints
uv run locust --tags system health --headless -u 20 -r 5 -t 5m

# Only ML endpoints
uv run locust --tags sentiment emotion --headless -u 10 -r 2 -t 10m

# Only GET endpoints
uv run locust --tags get --headless -u 50 -r 10 -t 5m

# Exclude heavy operations
uv run locust --exclude-tags heavy modeling --headless -u 30 -r 10 -t 10m
```

### Available Tags

- **System**: `system`, `health`, `status`, `reload`
- **Topics**: `topics`, `modeling`, `get`
- **Sentiment**: `sentiment`, `predict`, `get`
- **Emotion**: `emotion`, `predict`, `get`
- **SNA**: `sna`, `community`, `buzzer`, `get`
- **User Classes**: `readonly`, `heavy`

---

## 👥 User Classes

### Default (Balanced)

```bash
# Mixed read/write workload
uv run locust -f locustfile.py SociaLabsAIUser --headless -u 20 -r 5 -t 10m
```

### Read-Only

```bash
# Only GET endpoints
uv run locust -f locustfile.py ReadOnlyUser --headless -u 50 -r 10 -t 10m
```

### Heavy Processing

```bash
# ML pipeline stress test
uv run locust -f locustfile.py HeavyProcessingUser --headless -u 10 -r 2 -t 20m
```

---

## 📊 Typical Scenarios

### Development Testing

```bash
# Quick test during development
uv run locust --headless -u 5 -r 1 -t 2m
```

### Pre-Production Testing

```bash
# Normal load simulation
uv run locust --headless -u 50 -r 10 -t 15m \
  --html=reports/pre_prod_test.html
```

### Stress Testing

```bash
# Find breaking point
uv run locust --headless -u 100 -r 10 -t 20m \
  --html=reports/stress_test.html \
  --csv=reports/stress_test
```

### Spike Testing

```bash
# Sudden traffic surge
uv run locust --headless -u 50 -r 25 -t 5m \
  --html=reports/spike_test.html
```

---

## 🎛️ Environment Variables (Windows)

```powershell
# Custom parameters
.\run_load_test.ps1 normal -Users 30 -SpawnRate 5 -Duration '15m'

# Custom host
.\run_load_test.ps1 stress -Host 'https://api.socialabs.ai'

# Custom report dir
.\run_load_test.ps1 normal -ReportDir 'custom_reports'
```

## 🎛️ Environment Variables (Linux/Mac)

```bash
# Custom parameters
USERS=30 SPAWN_RATE=5 DURATION=15m ./run_load_test.sh normal

# Custom host
HOST=https://api.socialabs.ai ./run_load_test.sh stress

# Custom report dir
REPORT_DIR=custom_reports ./run_load_test.sh normal
```

---

## 📈 Key Metrics to Monitor

| Metric               | Description            | Target                           |
| -------------------- | ---------------------- | -------------------------------- |
| **RPS**              | Requests Per Second    | >= 10 (mixed), >= 50 (read-only) |
| **Response Time**    | Average latency        | < 2s (GET), < 30s (ML)           |
| **Error Rate**       | Failed requests %      | < 1% (normal), < 5% (acceptable) |
| **95th Percentile**  | Slow request threshold | < 2x average                     |
| **Concurrent Users** | Simultaneous users     | Varies by scenario               |

---

## 🔍 Troubleshooting

### Connection Refused

```bash
# Check if server is running
curl http://localhost:8000/api/system/health
```

### High Error Rate

```bash
# Check models are loaded
curl http://localhost:8000/api/system/models-status
```

### Timeout Errors

```bash
# Reduce users or increase timeout in locustfile.py
# Edit timeout parameter in task methods
```

---

## 📁 Report Files

After running tests, check these locations:

```
reports/
├── smoke_test_YYYYMMDD_HHMMSS.html
├── normal_load_YYYYMMDD_HHMMSS.html
├── normal_load_YYYYMMDD_HHMMSS_stats.csv
├── normal_load_YYYYMMDD_HHMMSS_stats_history.csv
├── normal_load_YYYYMMDD_HHMMSS_failures.csv
└── ...
```

---

## 🔗 Resources

- **Full Documentation**: [LOAD_TESTING.md](LOAD_TESTING.md)
- **Locust Docs**: https://docs.locust.io/
- **API Docs**: http://localhost:8000/docs

---

## 💡 Tips

1. **Start Small**: Begin with 5-10 users, then scale up
2. **Monitor Server**: Watch CPU/Memory during tests
3. **Use Tags**: Filter specific endpoints with `--tags`
4. **HTML Reports**: Always generate reports for analysis
5. **Baseline First**: Run 1-user test to establish baseline
6. **Gradual Ramp**: Don't spawn all users at once
7. **Isolate Environment**: Test on non-production

---

**Quick Help:**

```bash
# Windows
.\run_load_test.ps1 help

# Linux/Mac
./run_load_test.sh help

# Locust help
uv run locust --help
```
