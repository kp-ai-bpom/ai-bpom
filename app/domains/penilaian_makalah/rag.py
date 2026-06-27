# ── RAG Async Functions ───────────────────────────────────────────────────────
import asyncio
import threading
import os
from turtle import st

from app.domains.penilaian_makalah.prompt import PROMPT_KONTEKS, QUERY_KEYWORDS
from constant import WORKING_DIR

@st.cache_resource(show_spinner=False)
def get_async_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop

def run_async(coro):
    loop = get_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


async def _initialize_rag():
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
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
    token_limit   = int(os.getenv("EMBEDDING_TOKEN_LIMIT", 8192))
    model_name    = os.getenv("EMBEDDING_MODEL")

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
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="Neo4JStorage",
        enable_llm_cache=False, 
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    return rag


@st.cache_resource(show_spinner=False)
def get_rag():
    return run_async(_initialize_rag())

async def retrieve_assessment_context(rag, selected_jabatan: str, query_mode: str) -> str:
    from lightrag import QueryParam
    try:
        ctx = await rag.aquery(
            query=QUERY_KEYWORDS.format(selected_jabatan=selected_jabatan),
            param=QueryParam(
                mode=query_mode,
                user_prompt=PROMPT_KONTEKS.format(selected_jabatan=selected_jabatan),
            ),
        )
        return ctx if ctx else "Konteks jabatan tidak ditemukan dalam knowledge base."
    except Exception as e:
        return f"Error retrieving context: {str(e)}"
