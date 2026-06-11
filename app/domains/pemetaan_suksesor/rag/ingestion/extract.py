import asyncio
import re

import httpx

from app.core.config import settings
from app.core.logger import log

ENTITY_TYPES = "POSITION, UNIT, FUNCTION, REQUIREMENT"

SYSTEM_PROMPT = f"""---Goal---
Given a text document and a list of entity types, identify all entities and relationships.

---Entity Definitions---
- POSITION: Jabatan struktural atau fungsional di BPOM (contoh: Inspektur I, Inspektur Utama).
- UNIT: Unit kerja atau organisasi di BPOM (contoh: Inspektorat I, Biro SDM).
- FUNCTION: Tugas, fungsi, atau tanggung jawab spesifik dari suatu jabatan.
- REQUIREMENT: Syarat kualifikasi mutlak untuk menduduki jabatan (pangkat/golongan, pendidikan, status PNS, dll).

---Steps---
1. Identify all entities of types: [{ENTITY_TYPES}]. Format: ("entity"|NAME|TYPE|DESCRIPTION)
2. Identify all relationships between entities. Format: ("relationship"|SOURCE|TARGET|DESCRIPTION|WEIGHT 1-10)
3. Return output in Indonesian.
4. When finished, output [DONE]

---Real Data---
Input:
{{input_text}}
Output:
"""


def _base_url() -> str:
    url = settings.AI_BASE_URL or "https://api.openai.com/v1/"
    return url.rstrip("/")


async def extract_graph_elements(chunk_text: str) -> str:
    payload = {
        "model": settings.AI_THINK_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Input:\n{chunk_text}\nOutput:"},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{_base_url()}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"] or ""
    except Exception as e:
        log.exception(f"❌ Graph extraction failed: {e}")
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


async def extract_from_chunks(chunks: list[dict], max_concurrent: int = 1) -> tuple[list[tuple], list[tuple]]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _extract_one(chunk_text: str):
        async with semaphore:
            return await extract_graph_elements(chunk_text)

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
