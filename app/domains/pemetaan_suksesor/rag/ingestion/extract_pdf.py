import asyncio
import io
import re

import pdfplumber
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import log

ENTITY_TYPES_PDF = "REGULATION, CRITERIA, WEIGHT, PROCESS, POSITION, UNIT, FUNCTION, REQUIREMENT"

SYSTEM_PROMPT_PDF = f"""---Goal---
Given a regulatory document and a list of entity types, identify all entities and relationships.

---Entity Definitions---
- REGULATION: Peraturan, keputusan, atau kebijakan BPOM (contoh: PerBPOM No. 3/2020).
- CRITERIA: Kriteria atau indikator penilaian dalam regulasi (contoh: penilaian kinerja, rekam jejak).
- WEIGHT: Bobot atau skor penilaian dalam regulasi (contoh: bobot 30%, nilai minimum 70).
- PROCESS: Tahapan atau mekanisme proses seleksi/penilaian (contoh: uji kompetensi, wawancara).
- POSITION: Jabatan struktural atau fungsional di BPOM.
- UNIT: Unit kerja atau organisasi di BPOM.
- FUNCTION: Tugas, fungsi, atau tanggung jawab spesifik.
- REQUIREMENT: Syarat kualifikasi mutlak (pangkat/golongan, pendidikan, status PNS, dll).

---Steps---
1. Identify all entities of types: [{ENTITY_TYPES_PDF}]. Format: ("entity"|NAME|TYPE|DESCRIPTION)
2. Identify all relationships between entities. Format: ("relationship"|SOURCE|TARGET|DESCRIPTION|WEIGHT 1-10)
3. Return output in Indonesian.
4. When finished, output [DONE]

---Real Data---
Input:
{{{{input_text}}}}
Output:
"""

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        client_kwargs = {"api_key": settings.OPENAI_API_KEY}
        if settings.AI_BASE_URL:
            client_kwargs["base_url"] = settings.AI_BASE_URL
        _client = AsyncOpenAI(**client_kwargs)
    return _client


def parse_pdf_bytes(data: bytes, filename: str) -> list[dict]:
    """Extract text from PDF bytes and split into page-based chunks with 1-page overlap."""
    prefix = f"Dokumen: {filename}\n"
    pages: list[str] = []

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())

    if not pages:
        return []

    # Divide pages into 3 roughly equal chunks with 1-page overlap
    total = len(pages)
    chunk_size = max(1, total // 3)
    chunks = []
    starts = [0, chunk_size - 1, 2 * chunk_size - 1]  # -1 for overlap

    for i, start in enumerate(starts):
        if start >= total:
            break
        end = start + chunk_size + 1  # +1 for overlap at end
        chunk_pages = pages[start:end]
        chunk_text = prefix + "\n\n".join(chunk_pages)
        chunks.append({"chunk_index": i, "chunk_text": chunk_text})

    return chunks


async def extract_graph_elements_pdf(chunk_text: str) -> str:
    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=settings.AI_THINK_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_PDF},
                {"role": "user", "content": f"Input:\n{chunk_text}\nOutput:"},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        log.exception(f"❌ PDF graph extraction failed: {e}")
        return ""


def filter_kg_output(text: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, int]]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    raw_entities = re.findall(r'\("entity"\|(.*?)\|(.*?)\|(.*?)\)', text, re.DOTALL)
    raw_relationships = re.findall(r'\("relationship"\|(.*?)\|(.*?)\|(.*?)\|(.*?)\)', text, re.DOTALL)

    entities = [(n.strip(), t.strip(), d.strip()) for n, t, d in raw_entities]
    relationships = []
    for s, t, d, w in raw_relationships:
        try:
            weight = int(w.strip())
        except ValueError:
            weight = 1
        relationships.append((s.strip(), t.strip(), d.strip(), weight))

    return entities, relationships


async def extract_from_pdf_chunks(
    chunks: list[dict], max_concurrent: int = 3
) -> tuple[list[tuple], list[tuple]]:
    """Run LLM extraction over PDF chunks with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _extract_one(chunk_text: str):
        async with semaphore:
            return await extract_graph_elements_pdf(chunk_text)

    raw_results = await asyncio.gather(
        *[_extract_one(c["chunk_text"]) for c in chunks]
    )

    all_entities = []
    all_relationships = []
    for raw in raw_results:
        if raw:
            entities, relationships = filter_kg_output(raw)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

    seen_names = set()
    unique_entities = []
    for name, etype, desc in all_entities:
        if name not in seen_names:
            seen_names.add(name)
            unique_entities.append((name, etype, desc))

    return unique_entities, all_relationships
