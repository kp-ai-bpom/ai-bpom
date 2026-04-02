from fastapi import APIRouter

from app.api.system import router as system_router
from app.domains.emotion.api import router as emotion_router
from app.domains.sentiment.api import router as sentiment_router
from app.domains.sna.api import router as sna_router
from app.domains.topic_modeling.api import router as topic_router

api_router = APIRouter()

api_router.include_router(topic_router, prefix="/topics", tags=["Topic Modeling"])
api_router.include_router(
    sentiment_router, prefix="/sentiments", tags=["Sentiment Analysis"]
)
api_router.include_router(
    emotion_router, prefix="/emotions", tags=["Emotion Classification"]
)
api_router.include_router(sna_router, prefix="/sna", tags=["Social Network Analysis"])
api_router.include_router(system_router, prefix="/system", tags=["System Management"])
