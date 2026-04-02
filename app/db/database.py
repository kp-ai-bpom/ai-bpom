from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from beanie import init_beanie
from app.core.config import settings

from app.core.logger import log
from app.domains.topic_modeling.models import DocumentsModel, TopicsModel
from app.domains.sentiment.models import SentimentModel
from app.domains.emotion.models import EmotionModel
from app.domains.sna.models import CommunityDetectionModel, BuzzerModel

client = AsyncMongoClient(settings.MONGODB_URI)

def get_db() -> AsyncDatabase:
    return client[settings.MONGO_DB_NAME]

async def init_db():
    await init_beanie(
        database=get_db(),
        document_models=[
            DocumentsModel,
            TopicsModel,
            SentimentModel,
            EmotionModel,
            CommunityDetectionModel,
            BuzzerModel,
        ]
    )
    log.info("✅ Database initialized.")
