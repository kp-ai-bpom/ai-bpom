#!/bin/bash
# Complete testing setup script

echo "=================================================="
echo "PENILAIAN MAKALAH API - COMPLETE TESTING SETUP"
echo "=================================================="

# Step 1: Docker Services
echo ""
echo "[1/5] Starting Docker services..."
docker-compose up -d
sleep 5

# Check services
if docker-compose ps | grep -q "Up"; then
    echo "✅ Docker services started"
else
    echo "❌ Docker services failed to start"
    exit 1
fi

# Step 2: Database Migration
echo ""
echo "[2/5] Running database migrations..."
alembic upgrade head
if [ $? -eq 0 ]; then
    echo "✅ Migration completed"
else
    echo "❌ Migration failed"
    exit 1
fi

# Step 3: Seed Data
echo ""
echo "[3/5] Seeding test data..."
python -m scripts.seed_penilaian_makalah
if [ $? -eq 0 ]; then
    echo "✅ Seed data inserted"
else
    echo "❌ Seed failed"
    exit 1
fi

# Step 4: Schema Validation Tests
echo ""
echo "[4/5] Running schema validation tests..."
python -m pytest tests/test_penilaian_makalah.py::TestPenilaianMakalahAPI::test_validate_evaluate_request -v
python -m pytest tests/test_penilaian_makalah.py::TestPenilaianMakalahAPI::test_validate_evaluate_response -v
if [ $? -eq 0 ]; then
    echo "✅ Schema validation tests passed"
else
    echo "❌ Schema tests failed"
    exit 1
fi

# Step 5: Start Server
echo ""
echo "[5/5] Starting FastAPI server..."
echo "🚀 Server will start on http://localhost:8080"
echo "📚 API docs available at http://localhost:8080/docs"
echo ""
echo "Press Ctrl+C to stop server"
echo ""

poe dev

