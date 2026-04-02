# Model Reload API Documentation

## Endpoints

### 1. POST `/system/reload-models`

Reload semua ML models (Sentiment & Emotion) tanpa restart server.

**Use Cases:**

- Setelah update/training model baru
- Troubleshooting jika model corrupt
- Hot-reload untuk production deployment

**Request:**

```bash
curl -X POST http://localhost:8000/system/reload-models
```

**Response (Success):**

```json
{
  "status": "success",
  "message": "All models reloaded successfully",
  "results": {
    "sentiment": "success",
    "emotion": "success"
  }
}
```

**Response (Partial Success):**

```json
{
  "status": "partial",
  "message": "Some models failed to reload. Check results for details.",
  "results": {
    "sentiment": "success",
    "emotion": "failed: Model file not found"
  }
}
```

---

### 2. GET `/system/models-status`

Mendapatkan status loading untuk semua ML models.

**Request:**

```bash
curl http://localhost:8000/system/models-status
```

**Response:**

```json
{
  "status": "success",
  "models": {
    "sentiment": {
      "loaded": true,
      "models": {
        "cnn": true,
        "cnn_lstm": true
      }
    },
    "emotion": {
      "loaded": true,
      "models": {
        "cnn": true,
        "bilstm": true
      }
    }
  }
}
```

---

### 3. GET `/system/health`

Simple health check endpoint.

**Request:**

```bash
curl http://localhost:8000/system/health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "SociaLabs AI Backend",
  "message": "Service is running"
}
```

---

## Workflow untuk Update Model Baru

1. **Upload model baru** ke folder yang sesuai:
   - Sentiment: `models/raw/sentiment/`
   - Emotion: `models/raw/emotion/`

2. **Rename file model** sesuai dengan konfigurasi di `.env` atau `config.py`:

   ```
   latest_model-cnn-sentiment.h5
   latest_tokenizer-cnn-sentiment.pickle
   latest_model-cnn-lstm-sentiment.h5
   latest_tokenizer-cnn-lstm-sentiment.pickle
   latest_model-cnn-emotion.h5
   latest_tokenizer-cnn-emotion.pickle
   latest_model-bilstm-emotion.h5
   latest_tokenizer-bilstm-emotion.pickle
   ```

3. **Panggil reload endpoint**:

   ```bash
   curl -X POST http://localhost:8000/system/reload-models
   ```

4. **Verify** model sudah ter-load:
   ```bash
   curl http://localhost:8000/system/models-status
   ```

---

## Important Notes

⚠️ **Warning:**

- Proses reload bisa memakan waktu 5-15 detik tergantung ukuran model
- Model lama akan di-unload dari memory terlebih dahulu
- Request inference yang sedang berjalan saat reload bisa gagal
- Pastikan file model baru sudah ter-upload sebelum reload

✅ **Best Practices:**

- Lakukan reload pada saat traffic rendah (misal: tengah malam)
- Gunakan versioning untuk backup model lama
- Test model baru di development environment dulu
- Monitor logs saat reload untuk memastikan tidak ada error

---

## Testing di Development

```bash
# 1. Start server
uv run fastapi dev main.py

# 2. Akses Swagger UI
# http://localhost:8000/docs

# 3. Cari section "System Management"
# 4. Try out endpoint "/system/reload-models"
```
