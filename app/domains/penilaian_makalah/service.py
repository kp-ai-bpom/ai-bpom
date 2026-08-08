"""Application services for penilaian_makalah."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, HTTPException, status
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings as app_settings
from app.core.logger import log
from app.db.database import get_db
from app.domains.penilaian_makalah.constant import (
    CRITERIA_KEYS,
    CRITERIA_SHORT,
    DEFAULT_M_SAMPLES,
    DEFAULT_QUERY_MODE,
    DEFAULT_TEMPERATURE,
    MAX_SCORE,
    MIN_SCORE,
    NSV_THRESHOLD,
    SAMPLE_RETRY_LIMIT,
    SCORE_RANGE,
    SCORE_WEIGHTS,
    SUPPORTED_QUERY_MODE,
)
from app.domains.penilaian_makalah.core.config import settings as domain_settings
from app.domains.penilaian_makalah.parser import (
    compute_final_score,
    extract_text_from_bytes,
    extract_text_from_uploaded,
    load_all_skj,
    parse_response,
    score_color,
)
from app.domains.penilaian_makalah.prompt import PROMPT_KONTEKS, PROMPT_PENILAIAN, QUERY_KEYWORDS
from app.domains.penilaian_makalah.repositories import EvaluationRepository
from app.domains.penilaian_makalah.storage import minio_storage
from app.domains.penilaian_makalah.uncertainty import calculate_uncertainty_metrics


class PenilaianMakalahService:
    def __init__(self, db: AsyncSession):
        self._repo = EvaluationRepository(db)
        self._skj_cache: Optional[dict[str, dict[str, Any]]] = None

    def get_supported_query_modes(self) -> list[str]:
        return list(domain_settings.supported_query_modes)

    def load_skj(self) -> dict[str, dict[str, Any]]:
        if self._skj_cache is None:
            self._skj_cache = load_all_skj()
        return self._skj_cache

    async def list_history(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self._repo.get_history(limit=limit)
        return [self._serialize_history_row(row) for row in rows]

    async def get_history_item(self, history_id: int) -> dict[str, Any]:
        rows = await self._repo.get_history(limit=1000)
        for row in rows:
            if row.id == history_id:
                return self._serialize_history_row(row)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History item not found")

    async def get_minio_files(self, bucket: str, detailed: bool = False):
        if detailed:
            return minio_storage.list_minio_files_detailed(bucket)
        return minio_storage.list_minio_files(bucket)

    async def upload_document(self, bucket: str, filename: str, data: bytes) -> bool:
        return minio_storage.upload_to_minio(bucket, filename, data)

    async def download_document(self, bucket: str, filename: str):
        return minio_storage.download_from_minio(bucket, filename)

    async def retrieve_assessment_context(self, selected_jabatan: str, query_mode: str) -> str:
        return await self._retrieve_assessment_context(selected_jabatan, query_mode)

    async def evaluate_from_text(
        self,
        *,
        paper_filename: str,
        jabatan: str,
        makalah_text: str,
        tema_text: str,
        query_mode: str = DEFAULT_QUERY_MODE,
        m_samples: int = DEFAULT_M_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        save_result: bool = True,
    ) -> dict[str, Any]:
        if query_mode not in self.get_supported_query_modes():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported query_mode '{query_mode}'",
            )

        assessment_context = await self._retrieve_assessment_context(jabatan, query_mode)
        result = await self._evaluate_paper_with_uncertainty(
            makalah_text=makalah_text,
            assessment_context=assessment_context,
            tema_text=tema_text,
            m_samples=m_samples,
            temperature=temperature,
        )
        result["assessment_context"] = assessment_context
        result["query_mode"] = query_mode
        result["jabatan"] = jabatan
        result["paper_filename"] = paper_filename

        if save_result:
            await self._repo.save_evaluation(paper_filename, jabatan, result, query_mode)

        return result

    async def evaluate_from_minio(
        self,
        *,
        jabatan: str,
        paper_filename: str,
        tema_filename: str,
        query_mode: str = DEFAULT_QUERY_MODE,
        m_samples: int = DEFAULT_M_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        save_result: bool = True,
    ) -> dict[str, Any]:
        paper_data = await self.download_document(domain_settings.bucket_makalah, paper_filename)
        if not paper_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper file not found in MinIO")

        tema_data = await self.download_document(domain_settings.bucket_tema, tema_filename)
        if not tema_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tema file not found in MinIO")

        makalah_text = extract_text_from_bytes(paper_data, paper_filename)
        tema_text = extract_text_from_bytes(tema_data, tema_filename)
        return await self.evaluate_from_text(
            paper_filename=paper_filename,
            jabatan=jabatan,
            makalah_text=makalah_text,
            tema_text=tema_text,
            query_mode=query_mode,
            m_samples=m_samples,
            temperature=temperature,
            save_result=save_result,
        )

    async def evaluate_uploaded_file(
        self,
        *,
        jabatan: str,
        tema_filename: str,
        uploaded_filename: str,
        uploaded_data: bytes,
        query_mode: str = DEFAULT_QUERY_MODE,
        m_samples: int = DEFAULT_M_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        save_result: bool = True,
    ) -> dict[str, Any]:
        tema_data = await self.download_document(domain_settings.bucket_tema, tema_filename)
        if not tema_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tema file not found in MinIO")

        makalah_text = extract_text_from_bytes(uploaded_data, uploaded_filename)
        tema_text = extract_text_from_bytes(tema_data, tema_filename)
        return await self.evaluate_from_text(
            paper_filename=uploaded_filename,
            jabatan=jabatan,
            makalah_text=makalah_text,
            tema_text=tema_text,
            query_mode=query_mode,
            m_samples=m_samples,
            temperature=temperature,
            save_result=save_result,
        )

    async def evaluate_uploaded_stream(
        self,
        *,
        jabatan: str,
        tema_filename: str,
        uploaded_file,
        query_mode: str = DEFAULT_QUERY_MODE,
        m_samples: int = DEFAULT_M_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        save_result: bool = True,
    ) -> dict[str, Any]:
        tema_data = await self.download_document(domain_settings.bucket_tema, tema_filename)
        if not tema_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tema file not found in MinIO")

        makalah_text = extract_text_from_uploaded(uploaded_file)
        tema_text = extract_text_from_bytes(tema_data, tema_filename)
        return await self.evaluate_from_text(
            paper_filename=uploaded_file.name,
            jabatan=jabatan,
            makalah_text=makalah_text,
            tema_text=tema_text,
            query_mode=query_mode,
            m_samples=m_samples,
            temperature=temperature,
            save_result=save_result,
        )

    async def log_ingestion(self, filename: str, status_text: str, error_message: str | None = None) -> None:
        await self._repo.create_ingestion_log(filename=filename, status=status_text, error_message=error_message)

    async def _retrieve_assessment_context(self, selected_jabatan: str, query_mode: str) -> str:
        return await self._query_rag(selected_jabatan=selected_jabatan, query_mode=query_mode)

    async def _query_rag(self, selected_jabatan: str, query_mode: str) -> str:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs

        async def llm_model_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs) -> str:
            history_messages = history_messages or []
            return await openai_complete_if_cache(
                os.getenv("LLM_MODEL"),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("LLM_BINDING_API_KEY"),
                base_url=os.getenv("LLM_BINDING_HOST"),
                **kwargs,
            )

        embedding_dim = int(os.getenv("EMBEDDING_DIM", 1536))
        token_limit = int(os.getenv("EMBEDDING_TOKEN_LIMIT", 8192))
        model_name = os.getenv("EMBEDDING_MODEL")

        async def raw_embedding_func(texts):
            return await openai_embed.func(
                texts,
                api_key=os.getenv("LLM_BINDING_API_KEY"),
                base_url=os.getenv("LLM_BINDING_HOST"),
                model=model_name,
            )

        embedding_func = wrap_embedding_func_with_attrs(
            embedding_dim=embedding_dim,
            max_token_size=token_limit,
            model_name=model_name,
        )(raw_embedding_func)

        rag = LightRAG(
            working_dir=domain_settings.working_dir,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="Neo4JStorage",
            enable_llm_cache=False,
            enable_llm_cache_for_entity_extract=False,
        )
        await rag.initialize_storages()

        try:
            ctx = await rag.aquery(
                query=QUERY_KEYWORDS.format(selected_jabatan=selected_jabatan),
                param=QueryParam(
                    mode=query_mode,
                    user_prompt=PROMPT_KONTEKS.format(selected_jabatan=selected_jabatan),
                ),
            )
            return ctx if ctx else "Konteks jabatan tidak ditemukan dalam knowledge base."
        except Exception as exc:
            return f"Error retrieving context: {exc}"

    async def _evaluate_paper_with_uncertainty(
        self,
        *,
        makalah_text: str,
        assessment_context: str,
        tema_text: str,
        m_samples: int,
        temperature: float,
    ) -> dict[str, Any]:
        evaluation_prompt = PROMPT_PENILAIAN.format(
            assessment_context=assessment_context,
            makalah_text=makalah_text,
            tema_text=tema_text,
        )

        async def llm_sample(prompt: str, sample_id: int) -> dict | None:
            llm = ChatOpenAI(
                model=os.getenv("LLM_PENILAI", "gpt-5.4"),
                api_key=os.getenv("LLM_BINDING_API_KEY"),
                base_url=os.getenv("LLM_BINDING_HOST"),
                temperature=temperature,
            )
            try:
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                content = response.content.strip()
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if not json_match:
                    log.warning(f"Sample {sample_id}: No JSON found")
                    return None

                clean_json = re.sub(r",\s*([\]}])", r"\1", json_match.group(0))
                data = json.loads(clean_json)
                scores = data.get("scores", {})
                valid_scores = {}
                for index, key_full in enumerate(CRITERIA_KEYS):
                    short_key = CRITERIA_SHORT[index]
                    value = scores.get(key_full, scores.get(short_key))
                    if isinstance(value, (int, float)) and MIN_SCORE <= value <= MAX_SCORE:
                        valid_scores[key_full] = float(value)

                if len(valid_scores) == len(CRITERIA_KEYS):
                    return {
                        "sample_id": sample_id,
                        "scores": valid_scores,
                        "Ringkasan": data.get("Ringkasan", ""),
                        "justification": data.get("justification", {}),
                        "evidence": data.get("evidence", {}),
                        "raw_response": content,
                    }
                log.warning(f"Sample {sample_id}: Invalid scores")
                return None
            except Exception as exc:
                log.warning(f"Sample {sample_id}: Error - {exc}")
                return None

        async def sample_with_retry(prompt: str, sample_id: int, attempt: int = 0):
            result = await llm_sample(prompt, sample_id)
            if result is None and attempt < SAMPLE_RETRY_LIMIT:
                return await sample_with_retry(prompt, sample_id, attempt + 1)
            return result

        results = await asyncio.gather(*[sample_with_retry(evaluation_prompt, sample_id) for sample_id in range(m_samples)])
        valid_results = [item for item in results if item is not None]

        if not valid_results:
            return {
                "scores": {key: 0 for key in CRITERIA_KEYS},
                "final_score": 0,
                "uncertainty_metrics": {},
                "valid_samples": 0,
                "total_samples": m_samples,
                "raw_samples": [],
                "full_samples": [],
                "Ringkasan": "Gagal mendapatkan sampel yang valid",
                "justification": {},
                "evidence": {},
                "temperature_used": temperature,
                "m_samples_used": m_samples,
                "error": "No valid samples",
            }

        scores_list = [item["scores"] for item in valid_results]
        metrics = calculate_uncertainty_metrics(scores_list)
        consensus = metrics["consensus_scores"]
        final_score = compute_final_score(consensus)
        last_sample = sorted(valid_results, key=lambda item: item.get("sample_id", 0))[-1]

        best_sample = None
        best_score_diff = float("inf")
        for sample in valid_results:
            sample_scores = sample.get("scores", {})
            diff = sum((sample_scores.get(key, 0) - consensus.get(key, 0)) ** 2 for key in CRITERIA_KEYS)
            if diff < best_score_diff:
                best_score_diff = diff
                best_sample = sample

        return {
            "scores": consensus,
            "final_score": final_score,
            "uncertainty_metrics": metrics["uncertainty"],
            "valid_samples": len(valid_results),
            "total_samples": m_samples,
            "raw_samples": scores_list,
            "full_samples": valid_results,
            "Ringkasan": last_sample.get("Ringkasan", ""),
            "justification": last_sample.get("justification", {}),
            "evidence": last_sample.get("evidence", {}),
            "best_sample_by_consensus": {
                "sample_id": best_sample.get("sample_id") if best_sample else None,
                "scores": best_sample.get("scores") if best_sample else {},
                "Ringkasan": best_sample.get("Ringkasan", "") if best_sample else "",
            }
            if best_sample
            else None,
            "temperature_used": temperature,
            "m_samples_used": m_samples,
            "last_sample_id": last_sample.get("sample_id"),
        }

    def _serialize_history_row(self, row) -> dict[str, Any]:
        return {
            "id": row.id,
            "paper_filename": row.paper_filename,
            "jabatan": row.jabatan,
            "scores": row.scores if isinstance(row.scores, dict) else {},
            "final_score": row.final_score,
            "justification": row.justification if isinstance(row.justification, dict) else {},
            "evidence": row.evidence if isinstance(row.evidence, dict) else {},
            "ringkasan": row.ringkasan,
            "query_mode": row.query_mode,
            "uncertainty_metrics": row.uncertainty_metrics if isinstance(row.uncertainty_metrics, dict) else {},
            "created_at": row.created_at,
        }


async def get_penilaian_makalah_service(db: AsyncSession = Depends(get_db)) -> PenilaianMakalahService:
    return PenilaianMakalahService(db)
