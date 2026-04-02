from typing import List

from app.core.logger import log

from .models import EmotionModel


class EmotionRepository:
    """Repository untuk operasi database Emotion"""

    async def get_emotions_by_project_id(self, project_id: str) -> List[EmotionModel]:
        log.info(f"Mengambil data emosi untuk project: {project_id}")
        return await EmotionModel.find(EmotionModel.projectId == project_id).to_list()

    async def save_emotions(self, emotions_data: List[dict]) -> List[EmotionModel]:
        """Menyimpan list of emotions secara bulk, replace data lama."""
        if not emotions_data:
            log.warning("Tidak ada data emosi untuk disimpan.")
            return []

        try:
            project_id = emotions_data[0].get("projectId")
            if project_id:
                # Hapus data lama untuk project ini agar tidak double
                await EmotionModel.find(EmotionModel.projectId == project_id).delete()

            docs = [EmotionModel(**data) for data in emotions_data]
            await EmotionModel.insert_many(docs)

            log.info(f"Berhasil menyimpan {len(docs)} data emosi.")
            return docs
        except Exception as e:
            log.exception(f"Error menyimpan emosi: {e}")
            return []


def get_emotion_repository() -> EmotionRepository:
    return EmotionRepository()
