from fastapi import APIRouter, Depends

from .dto.request import ChatRequest, EmbeddingRequest
from .dto.response import (
    ChatDataResponse,
    ChatResponse,
    EmbeddingDataResponse,
    EmbeddingResponse,
)
from .services import ChatbotService, get_chatbot_service

router = APIRouter()


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
