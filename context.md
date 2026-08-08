# Context: XAI-MENTARI Multi-Agent Pipeline Implementation

## Status

Implementasi agent pipeline sedang berjalan. Planner agent sudah selesai dan berjalan sebagai background job. Masih perlu mengimplementasikan:

1. **Embedding cache** (`rag/vector/embed.py`) — tambahkan `get_cached_embedding`, `cache_embedding`, `precompute_embeddings`, `clear_embedding_cache`
2. **Analysis Agent tools** (`agents/analysis/tools.py`) — integrasikan embedding cache ke `search_vector_rag`
3. **Synthesis Agent tools** (`agents/synthesis/tools.py`) — buat `comparative_matrix_aggregator` (adaptasi untuk in-memory data)
4. **Synthesis Agent main** (`agents/synthesis/main.py`) — tambahkan tool
5. **Reviewer Agent tools** (`agents/reviewer/tools.py`) — buat 3 read tools (in-memory)
6. **Reviewer Agent main** (`agents/reviewer/main.py`) — tambahkan tools
7. **Pipeline Orchestrator** — method baru di `AgentPipelineService` untuk run full pipeline sequential
8. **Pipeline API endpoint** — `POST /agents/pipeline` sebagai background job

---

## File-File Yang Sudah Selesai

### `app/domains/pemetaan_suksesor/services/agent_pipeline.py`
- `_extract_text_from_message()` — extract text dari Strands AgentResult.message (`{'role': 'assistant', 'content': [{'text': '...'}]}`)
- `_try_parse_json_output()` — strip ```json fences + json.loads
- `_invoke_agent_sync()` — try structured_output → try parse JSON from text → fallback to plain message
- `run_agent_job()` — background task wrapper: run agent → update job status/result
- All 4 agents: PLANNER, ANALYSIS, SYNTHESIS, REVIEWER sudah di-register

### `app/domains/pemetaan_suksesor/api.py`
- `POST /agents/planner` — background job (202 Accepted), returns `job_id`
- `GET /jobs/{job_id}` — poll status

### `app/domains/pemetaan_suksesor/agents/analysis/prompt.py`
- ANALYSIS_SYSTEM_PROMPT sudah lengkap (STEP 1-5, XAI schema)
- Tools yang direferensikan: `search_vector_rag`, `query_jabatan_profile`, `query_neo4j_depth2`, `search_neo4j_entities`

### `app/domains/pemetaan_suksesor/agents/synthesis/prompt.py`
- SYNTHESIS_SYSTEM_PROMPT sudah lengkap (STEP 1-5, comparison matrix schema)

### `app/domains/pemetaan_suksesor/agents/reviewer/prompt.py`
- REVIEWER_SYSTEM_PROMPT sudah lengkap (STEP 1-6, final XAI schema)

### `app/domains/pemetaan_suksesor/rag/vector/embed.py`
- `embed_texts(texts)` — batch embed via OpenAI
- `embed_single(text)` — single embed
- **BELUM ADA**: `get_cached_embedding`, `cache_embedding`, `precompute_embeddings`, `clear_embedding_cache`

---

## File-File Yang Perlu Dibuat/Dimodifikasi

### 1. `rag/vector/embed.py` — Tambah Embedding Cache

```python
# Module-level cache
_embedding_cache: dict[str, list[float]] = {}

def get_cached_embedding(query: str) -> list[float] | None:
    return _embedding_cache.get(query)

def cache_embedding(query: str, embedding: list[float]) -> None:
    _embedding_cache[query] = embedding

async def precompute_embeddings(queries: list[str]) -> None:
    """Batch embed queries not yet in cache, store results."""
    to_embed = [q for q in queries if q not in _embedding_cache]
    if not to_embed:
        return
    embeddings = await embed_texts(to_embed)
    for query, embedding in zip(to_embed, embeddings):
        _embedding_cache[query] = embedding

def clear_embedding_cache() -> None:
    _embedding_cache.clear()
```

### 2. `agents/analysis/tools.py` — Integrasikan Cache ke search_vector_rag

```python
from app.domains.pemetaan_suksesor.rag.vector.embed import (
    embed_single, get_cached_embedding, cache_embedding
)

@tool
def search_vector_rag(query: str, top_k: int = 5) -> str:
    async def _search():
        # Check cache first
        query_embedding = get_cached_embedding(query)
        if query_embedding is None:
            query_embedding = await embed_single(query)
            cache_embedding(query, query_embedding)
        # ... rest sama
```

### 3. `agents/synthesis/tools.py` — comparative_matrix_aggregator

Terima `analysis_output` sebagai dict (bukan file path) karena ai-bpom adalah in-memory API.

```python
@tool
def comparative_matrix_aggregator(analysis_output: str) -> str:
    """
    Hitung skor deterministik dari XAI Justification Report.
    analysis_output: JSON string dari output Analysis Agent.
    """
```

Sub-functions yang perlu diimplementasikan:
- `_compute_skor_rekam_jejak(candidate_data)` — 6 komponen: jabatan 5%, pendidikan 19%, pelatihan 19%, disiplin 19%, skp 19%, integritas 5%
  - Formula: `(skala/4) × (bobot_pct/100) × 100`
- `_compute_skor_komposit_parsial(skor_rekam_jejak, nilai_mansoskul)` — `(rj/100)×20 + (mansoskul/100)×25`
- `_sort_and_rank(candidates)` — sort by skor_domain_fit DESC, total_rj DESC, nilai_potensi DESC
- `_detect_systemic_gaps(candidates)` — 5 gap types: SG-DOMAIN-MISMATCH, SG-FUNCTIONAL-GAP, SG-STRUCTURAL-DISTANCE, SG-DIKLAT-GAP, SG-VECTOR-RAG-ERROR
- `_validate_consistency(candidates)` — 3 flags: CF-FUNGSI-UTAMA, CF-SYARAT-PENGALAMAN, CF-EXPERIENCE-GAP-METHOD
- `_check_regulatory_compliance(candidates, planner_blueprint)` — check: status PNS, S1, pengalaman 2 tahun, diklat PIM III, pangkat IV/a

Output format (JSON string):
```json
{
  "comparison_matrix": [...],
  "systemic_gaps": [...],
  "consistency_flags": [...],
  "regulatory_compliance": [...]
}
```

**Architectural decision**: Tool menerima `analysis_output` sebagai JSON string (bukan file path). Planner blueprint diambil dari field `blueprint_context` yang ikut di analysis_output, atau dari argument kedua opsional.

### 4. `agents/synthesis/main.py` — Tambah Tool

```python
from .tools import comparative_matrix_aggregator

def create_synthesis_agent() -> Agent:
    model = create_strands_model(tier="flash")
    return Agent(
        name="synthesis",
        model=model,
        tools=[comparative_matrix_aggregator],
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
    )
```

### 5. `agents/reviewer/tools.py` — 3 Read-Only Tools (In-Memory)

Reviewer tools di hybrid-rag membaca dari file. Di ai-bpom, data mengalir in-memory.
Reviewer hanya perlu tools untuk membaca output dari stage sebelumnya — tapi karena pipeline pass data as JSON string ke agent, reviewer bisa baca langsung dari input tanpa tools.

**Opsi A**: Tidak perlu tools untuk reviewer (baca data langsung dari input JSON yang dikirim ke agent).
**Opsi B**: Tools dummy yang parse JSON dari string yang di-pass.

Rekomendasi: **Opsi A** — reviewer menerima full synthesis report + analysis summary + planner blueprint sebagai JSON input, tidak perlu tools.

### 6. Pipeline Orchestrator — `run_full_pipeline`

Tambahkan method baru ke `AgentPipelineService`:

```python
async def run_full_pipeline(
    self,
    input_json: Any,
    job_id: str,
    job_service,
) -> None:
    """Run full XAI-MENTARI pipeline: Planner → Embeddings → Analysis → Synthesis → Reviewer"""
    try:
        pipeline_log = []
        
        # Phase 1: Planner
        planner_result = await self.run_agent(AgentName.PLANNER, input_json=input_json)
        pipeline_log.append({"agent": "planner", "status": "completed"})
        planner_output = planner_result.get("output") or {}
        
        # Inter-phase: Pre-compute embeddings
        queries = _extract_vector_rag_queries(planner_output)
        if queries:
            await precompute_embeddings(queries)
            pipeline_log.append({"agent": "embedding_precompute", "queries": len(queries)})
        
        # Phase 2: Analysis
        analysis_input = {"blueprint": planner_output, "input": input_json}
        analysis_result = await self.run_agent(AgentName.ANALYSIS, input_json=analysis_input)
        pipeline_log.append({"agent": "analysis", "status": "completed"})
        analysis_output = analysis_result.get("output") or {}
        
        # Synthesis
        synthesis_input = {"xai_justification_report": analysis_output, "blueprint": planner_output}
        synthesis_result = await self.run_agent(AgentName.SYNTHESIS, input_json=synthesis_input)
        pipeline_log.append({"agent": "synthesis", "status": "completed"})
        synthesis_output = synthesis_result.get("output") or {}
        
        # Reviewer
        reviewer_input = {
            "synthesis_report": synthesis_output,
            "analysis_report": analysis_output,
            "planner_blueprint": planner_output,
        }
        reviewer_result = await self.run_agent(AgentName.REVIEWER, input_json=reviewer_input)
        pipeline_log.append({"agent": "reviewer", "status": "completed"})
        
        result = {
            "status": "completed",
            "pipeline_log": pipeline_log,
            "planner": planner_result,
            "analysis": analysis_result,
            "synthesis": synthesis_result,
            "reviewer": reviewer_result,
        }
        job_service.update_job(job_id, status="completed", result=result)
        
    except Exception as e:
        logger.error("Pipeline job %s failed: %s", job_id, e)
        job_service.update_job(job_id, status="failed", error=str(e))
```

Helper untuk extract queries dari planner output:
```python
def _extract_vector_rag_queries(planner_output: dict) -> list[str]:
    queries = []
    retrieved_context = planner_output.get("xai_blueprint", {}).get("retrieved_context", {})
    queries.extend(retrieved_context.get("fungsi_utama", []))
    queries.extend(retrieved_context.get("kompetensi_spesifik", []))
    return [q for q in queries if isinstance(q, str) and q.strip()]
```

### 7. API Endpoint

```python
@router.post(
    "/agents/pipeline",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run Full XAI-MENTARI Pipeline (Background Job)",
)
async def run_full_pipeline(
    background_tasks: BackgroundTasks,
    payload: PlannerRequest,
    job_service: JobService = Depends(get_job_service),
    agent_service: AgentPipelineService = Depends(get_agent_pipeline_service),
) -> JobAcceptedResponse:
    job_id = job_service.create_job("full_pipeline")
    background_tasks.add_task(
        agent_service.run_full_pipeline,
        payload.model_dump(),
        job_id,
        job_service,
    )
    return JobAcceptedResponse(job_id=job_id, message="Full XAI-MENTARI pipeline started")
```

---

## Arsitektur Data Flow

```
POST /agents/pipeline (PlannerRequest)
  ↓
job_id returned (202)
  ↓ background
run_full_pipeline()
  ↓
Phase 1: Planner Agent
  input: {target_jabatan, kandidat[]}
  output: xai_blueprint (JSON dict)
  ↓
Inter-phase: precompute_embeddings(fungsi_utama + kompetensi_spesifik)
  ↓
Phase 2a: Analysis Agent
  input: {blueprint: planner_output, input: original_request}
  output: xai_justification_report (JSON dict)
  ↓
Phase 2b: Synthesis Agent
  input: {xai_justification_report: analysis_output, blueprint: planner_output}
  tools: [comparative_matrix_aggregator(analysis_output_json_str)]
  output: synthesis_report (JSON dict)
  ↓
Phase 2c: Reviewer Agent
  input: {synthesis_report, analysis_report, planner_blueprint}
  tools: [] (no tools needed, reads from input)
  output: final_xai_report (JSON dict)
  ↓
GET /jobs/{job_id} → result: {planner, analysis, synthesis, reviewer}
```

---

## Catatan Penting

### Strands SDK Constraints
- `Agent(name, model, tools, system_prompt)` — TIDAK ada `output_schema` parameter
- `@tool` decorator hanya untuk fungsi **synchronous** — gunakan `asyncio.run()` di dalam untuk async code
- Hasil agent: `result.message` adalah `{'role': 'assistant', 'content': [{'text': '...'}]}`
- LLM wrap JSON dalam ` ```json ``` ` fences — perlu strip sebelum `json.loads()`

### Proxy Fix
- Proxy `9router.zulfifazhar.dev` block default OpenAI SDK User-Agent
- Fix: `default_headers={"User-Agent": "ai-bpom/1.0"}` di semua `AsyncOpenAI(...)` call
- Sudah diterapkan di `rag/vector/embed.py`

### File-based vs In-Memory
- hybrid-rag tools: file-based (terima file path, auto-discover related files)
- ai-bpom: in-memory API (data mengalir sebagai JSON dict/string antar agents)
- Adaptation: `comparative_matrix_aggregator` terima JSON string bukan file path

### Job System
- `JobService.create_job(name)` → returns `job_id`
- `job_service.update_job(job_id, status, result)` — update setelah selesai
- `GET /jobs/{job_id}` — poll status

---

## Dependency Reference

### Imports Kunci di ai-bpom
```python
# Embedding
from app.domains.pemetaan_suksesor.rag.vector.embed import (
    embed_single, embed_texts, get_cached_embedding, cache_embedding, 
    precompute_embeddings, clear_embedding_cache
)

# Agent creation
from app.domains.pemetaan_suksesor.agents.analysis.main import create_analysis_agent
from app.domains.pemetaan_suksesor.agents.synthesis.main import create_synthesis_agent
from app.domains.pemetaan_suksesor.agents.reviewer.main import create_reviewer_agent
from app.domains.pemetaan_suksesor.agents.planner.main import create_planner_agent

# LLM
from app.domains.pemetaan_suksesor.agents.llm_adapter import create_strands_model

# DTO
from app.domains.pemetaan_suksesor.dto.pipeline import AgentName, PlannerRequest
```

### Path Structure
```
app/domains/pemetaan_suksesor/
├── api.py
├── services/
│   └── agent_pipeline.py
├── dto/
│   └── pipeline.py
├── rag/
│   └── vector/
│       └── embed.py          ← tambah cache functions
├── agents/
│   ├── llm_adapter.py
│   ├── planner/
│   │   ├── main.py
│   │   ├── prompt.py
│   │   └── tools.py
│   ├── analysis/
│   │   ├── main.py
│   │   ├── prompt.py         ← DONE
│   │   └── tools.py          ← perlu integrasikan cache
│   ├── synthesis/
│   │   ├── main.py           ← perlu tambah tool
│   │   ├── prompt.py         ← DONE
│   │   └── tools.py          ← BELUM ADA, perlu dibuat
│   └── reviewer/
│       ├── main.py           ← perlu tambah tools (jika ada)
│       ├── prompt.py         ← DONE
│       └── tools.py          ← BELUM ADA (opsional)
```

---

## Reference: hybrid-rag synthesis tools (untuk adaptasi)

Dari `D:\Zulfi\My-Skripsi-Gweh\KP\1-final-services\hybrid-rag\app\agents\synthesis\tools.py`:

Bobot rekam jejak:
```python
REKAM_JEJAK_WEIGHTS = {
    "jabatan": 5,       # skor_jabatan_sekarang
    "pendidikan": 19,   # skor_pendidikan
    "pelatihan": 19,    # skor_pelatihan_struktural
    "disiplin": 19,     # skor_hukuman_disiplin
    "skp": 19,          # skor_skp
    "integritas": 5,    # skor_integritas
}
# Formula: sum((skala/4) * (bobot/100) * 100) untuk setiap komponen
# Total bobot = 86% (bukan 100%, karena komponen lain di luar rekam jejak)
```

Skor komposit parsial (45%):
```python
# skor_komposit_parsial = (skor_rekam_jejak/100)*20 + (nilai_mansoskul/100)*25
```

Sumber data untuk synthesis tool datang dari `candidate_data` di planner blueprint (bukan dari file terpisah). Perlu extract dari analysis_output yang berisi reference ke blueprint context.
