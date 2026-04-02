from typing import List

from app.core.logger import log

from .models import SentimentModel


class SentimentRepository:
    """Repository untuk database operations Sentiment"""

    async def get_sentiments_by_project_id(
        self, project_id: str
    ) -> List[SentimentModel]:
        log.info(f"Mengambil data sentimen untuk project: {project_id}")
        return await SentimentModel.find(
            SentimentModel.projectId == project_id
        ).to_list()

    async def save_sentiments(
        self, sentiments_data: List[dict]
    ) -> List[SentimentModel]:
        """Menyimpan list of sentiments secara bulk."""
        if not sentiments_data:
            log.warning("Tidak ada data sentimen untuk disimpan.")
            return []

        try:
            # Karena ini data AI yang digenerate ulang per project,
            # pendekatan teraman adalah menghapus yang lama dan insert yang baru
            project_id = sentiments_data[0].get("projectId")
            if project_id:
                await SentimentModel.find(
                    SentimentModel.projectId == project_id
                ).delete()

            docs = [SentimentModel(**data) for data in sentiments_data]
            await SentimentModel.insert_many(docs)

            log.info(f"Berhasil menyimpan {len(docs)} data sentimen.")
            return docs
        except Exception as e:
            log.exception(f"Error menyimpan sentimen: {e}")
            return []


# Dependency Factory
def get_sentiment_repository() -> SentimentRepository:
    return SentimentRepository()
