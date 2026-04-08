from fastapi import APIRouter

from app.domains.chatbot.api import router as chatbot_router
from app.domains.pemetaan_suksesor.api import router as pemetaan_suksesor_router
from app.domains.penilaian_suksesor.api import router as penilaian_suksesor_router

api_router = APIRouter()

api_router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])
api_router.include_router(
    pemetaan_suksesor_router, prefix="/pemetaan-suksesor", tags=["pemetaan-suksesor"]
)
api_router.include_router(
    penilaian_suksesor_router, prefix="/penilaian-suksesor", tags=["penilaian-suksesor"]
)
