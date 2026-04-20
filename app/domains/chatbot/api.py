from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.core.logger import log

from .dto.request import (
    ChatRequest,
    DeleteSessionRequest,
    EmbeddingRequest,
    GetSessionMessagesRequest,
    ImportBaseKnowledgeCsvRequest,
    ListSessionsRequest,
    SendMessageRequest,
)
from .dto.response import (
    ChatErrorResponse,
    ChatDataResponse,
    ChatResponse,
    DeleteSessionDataResponse,
    DeleteSessionResponse,
    EmbeddingDataResponse,
    EmbeddingResponse,
    ImportBaseKnowledgeCsvDataResponse,
    ImportBaseKnowledgeCsvResponse,
    SendMessageDataResponse,
    SendMessageResponse,
    SessionListDataResponse,
    SessionListItemResponse,
    SessionListResponse,
    SessionMessagesDataResponse,
    SessionMessagesResponse,
)
from .services import (
    ChatbotService,
    get_chatbot_pipeline_service,
    get_chatbot_service,
)

router = APIRouter()


def _error_response(status_code: int, error: str) -> JSONResponse:
    payload = ChatErrorResponse(status=status_code, error=error)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@router.post("/instruct", response_model=ChatResponse)
def instruct(
    request: ChatRequest, service: ChatbotService = Depends(get_chatbot_service)
):
    """
    Endpoint untuk model instruct.
    Cocok untuk tugas-tugas umum seperti menjawab pertanyaan.
    """
    messages = [{"role": "user", "content": request.input}]
    response = service.instruct(messages)
    return ChatResponse(
        message="instruct model responded", data=ChatDataResponse(response=response)
    )


@router.post("/think", response_model=ChatResponse)
def think(request: ChatRequest, service: ChatbotService = Depends(get_chatbot_service)):
    """
    Endpoint untuk model think.
    Cocok untuk tugas-tugas yang membutuhkan penalaran/berpikir.
    """
    messages = [{"role": "user", "content": request.input}]
    response = service.think(messages)
    return ChatResponse(
        message="think model responded", data=ChatDataResponse(response=response)
    )


@router.post("/deep-think", response_model=ChatResponse)
def deep_think(
    request: ChatRequest, service: ChatbotService = Depends(get_chatbot_service)
):
    """
    Endpoint untuk model deep_think.
    Cocok untuk tugas-tugas yang membutuhkan analisis mendalam.
    """
    messages = [{"role": "user", "content": request.input}]
    response = service.deep_think(messages)
    return ChatResponse(
        message="deep_think model responded", data=ChatDataResponse(response=response)
    )


@router.post("/embed", response_model=EmbeddingResponse)
def embedings(
    request: EmbeddingRequest, service: ChatbotService = Depends(get_chatbot_service)
):
    """
    Endpoint untuk embedding text.
    """
    embedding = service.embed(request.input)
    return EmbeddingResponse(
        message="embedding generated", data=EmbeddingDataResponse(embedding=embedding)
    )


@router.post(
    "/reembed/base-knowledge/import-csv",
    response_model=ImportBaseKnowledgeCsvResponse,
    responses={400: {"model": ChatErrorResponse}, 500: {"model": ChatErrorResponse}},
)
async def import_base_knowledge_csv(
    request: ImportBaseKnowledgeCsvRequest,
    service: ChatbotService = Depends(get_chatbot_pipeline_service),
):
    """Endpoint sederhana untuk import CSV base knowledge langsung ke pgvector."""
    try:
        result = await service.import_base_knowledge_csv(
            csv_path=request.csv_path,
            table_name=request.table_name,
            batch_size=request.batch_size,
            truncate_before_insert=request.truncate_before_insert,
        )
    except ValueError as exc:
        return _error_response(status_code=400, error=str(exc))
    except RuntimeError as exc:
        return _error_response(status_code=500, error=str(exc))
    except Exception:
        return _error_response(
            status_code=500,
            error="Failed to import base knowledge CSV",
        )

    return ImportBaseKnowledgeCsvResponse(
        status=200,
        data=ImportBaseKnowledgeCsvDataResponse(**result),
    )


@router.post(
    "/chat",
    response_model=SendMessageResponse,
    responses={400: {"model": ChatErrorResponse}, 500: {"model": ChatErrorResponse}},
)
async def send_message(
    request: SendMessageRequest,
    service: ChatbotService = Depends(get_chatbot_pipeline_service),
):
    """Endpoint untuk mengirim pesan dan menghasilkan query SQL."""
    try:
        result = await service.send_message(
            user_id=request.user_id,
            message=request.message,
            session_id=request.session_id,
        )
    except ValueError as exc:
        log.warning(
            "Invalid /chat request user_id=%s session_id=%s error=%s",
            request.user_id,
            request.session_id,
            str(exc),
        )
        return _error_response(status_code=400, error=str(exc))
    except Exception:
        log.exception(
            "Failed to process /chat request user_id=%s session_id=%s message=%r",
            request.user_id,
            request.session_id,
            request.message,
        )
        return _error_response(
            status_code=500,
            error="Failed to process message",
        )

    return SendMessageResponse(
        status=200,
        data=SendMessageDataResponse(**result),
    )


@router.get(
    "/chat",
    response_model=SessionMessagesResponse,
    responses={400: {"model": ChatErrorResponse}, 404: {"model": ChatErrorResponse}},
)
async def get_session_messages(
    user_id: str = Query(...),
    session_id: str = Query(...),
    service: ChatbotService = Depends(get_chatbot_pipeline_service),
):
    """Endpoint untuk mengambil percakapan berdasarkan user dan session."""
    request = GetSessionMessagesRequest(user_id=user_id, session_id=session_id)
    try:
        session_data = await service.get_session_messages(
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except ValueError as exc:
        return _error_response(status_code=400, error=str(exc))

    if session_data is None:
        return _error_response(status_code=404, error="Session not found")

    return SessionMessagesResponse(
        status=200,
        data=SessionMessagesDataResponse(**session_data),
    )


@router.get(
    "/chat/sessions",
    response_model=SessionListResponse,
    responses={400: {"model": ChatErrorResponse}},
)
async def list_sessions(
    user_id: str = Query(...),
    service: ChatbotService = Depends(get_chatbot_pipeline_service),
):
    """Endpoint untuk mengambil daftar session milik user."""
    request = ListSessionsRequest(user_id=user_id)
    try:
        sessions = await service.list_sessions(user_id=request.user_id)
    except ValueError as exc:
        return _error_response(status_code=400, error=str(exc))

    mapped_sessions = [SessionListItemResponse(**session) for session in sessions]
    return SessionListResponse(
        status=200,
        data=SessionListDataResponse(user_id=request.user_id, sessions=mapped_sessions),
    )


@router.delete(
    "/chat/session",
    response_model=DeleteSessionResponse,
    responses={400: {"model": ChatErrorResponse}, 404: {"model": ChatErrorResponse}},
)
async def delete_session(
    request: DeleteSessionRequest,
    service: ChatbotService = Depends(get_chatbot_pipeline_service),
):
    """Endpoint untuk menghapus session berdasarkan user dan session id."""
    try:
        deleted = await service.delete_session(
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except ValueError as exc:
        return _error_response(status_code=400, error=str(exc))

    if not deleted:
        return _error_response(status_code=404, error="Session not found")

    return DeleteSessionResponse(
        status=200,
        data=DeleteSessionDataResponse(message="Session deleted successfully"),
    )
