from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends
from pymongo.asynchronous.database import AsyncDatabase

from app.core.logger import log
from app.db.database import get_db

from .models import DocumentsModel, TopicsModel


class TopicRepository:
    """Repository for topic modeling related database operations"""

    def __init__(self, db: AsyncDatabase):
        """
        Initialize repository with MongoDB database instance.
        """
        self.tweets_collection = db["tweets"]

    # Unmanaged collection (pymongo native)
    async def get_tweet_by_keyword(self, topic_data: dict) -> Optional[Dict[str, Any]]:
        """Create and aggregate topic data based on keyword and date range"""
        try:
            keyword = topic_data.get("keyword")
            start_date = topic_data.get("start_date")
            end_date = topic_data.get("end_date")

            if not keyword:
                log.warning("Keyword is required but was None")
                return None

            match_stage = {
                "$match": {
                    "full_text": {"$regex": keyword.replace(" ", "|"), "$options": "i"}
                }
            }

            pipeline: List[Any] = [match_stage]

            if start_date and end_date:
                start_datetime = datetime.strptime(
                    f"{start_date} 00:00:00 +0000", "%Y-%m-%d %H:%M:%S %z"
                )
                end_datetime = datetime.strptime(
                    f"{end_date} 23:59:59 +0000", "%Y-%m-%d %H:%M:%S %z"
                )
                log.info(f"Filtering tweets from {start_datetime} to {end_datetime}")

                pipeline.extend(
                    [
                        {"$addFields": {"parsed_date": {"$toDate": "$created_at"}}},
                        {
                            "$match": {
                                "parsed_date": {
                                    "$gte": start_datetime,
                                    "$lte": end_datetime,
                                }
                            }
                        },
                    ]
                )

            project_stage = {
                "$project": {
                    "_id": 0,
                    "full_text": 1,
                    "username": 1,
                    "in_reply_to_screen_name": 1,
                    "tweet_url": 1,
                }
            }
            pipeline.append(project_stage)

            # Menggunakan collection native pymongo
            cursor = await self.tweets_collection.aggregate(pipeline)
            aggregate_data = await cursor.to_list(length=None)

            return {
                "keyword": keyword,
                "start_date": start_date,
                "end_date": end_date,
                "total_tweets": len(aggregate_data),
                "tweets": aggregate_data,
            }

        except Exception as e:
            log.exception(f"Error getting tweet by keyword data: {e}")
            return None

    # Managed collections (Beanie ODM)
    async def get_topics_by_project_id(self, project_id: str) -> List[TopicsModel]:
        """Retrieve all topics by project ID"""
        try:
            # Beanie secara otomatis mengembalikan list dari TopicsModel
            # Tidak perlu manual convert _id ke string! Beanie handle itu di property `.id`
            return await TopicsModel.find(TopicsModel.projectId == project_id).to_list()
        except Exception as e:
            log.exception(f"Error retrieving topics by projectId {project_id}: {e}")
            return []

    async def get_documents_by_topic(
        self, project_id: str, topic_id: int
    ) -> List[DocumentsModel]:
        try:
            return await DocumentsModel.find(
                DocumentsModel.projectId == project_id, DocumentsModel.topic == topic_id
            ).to_list()
        except Exception as e:
            log.exception(f"Error retrieving documents: {e}")
            return []

    async def create_topics(self, topics_data: List[dict]) -> List[TopicsModel]:
        """Create multiple topic data with duplicate checking"""
        try:
            if not topics_data:
                log.warning("No topics to create")
                return []

            project_id = topics_data[0].get("projectId")

            if project_id:
                # Menghitung jumlah dokumen menggunakan syntax Beanie
                existing_count = await TopicsModel.find(
                    TopicsModel.projectId == project_id
                ).count()

                if existing_count > 0:
                    log.info(
                        f"Topics already exist for projectId: {project_id}, found {existing_count}. Skipping insert."
                    )
                    return await TopicsModel.find(
                        TopicsModel.projectId == project_id
                    ).to_list()

            # Konversi list of dict menjadi list of Beanie Documents
            topic_docs = [TopicsModel(**topic) for topic in topics_data]

            # Insert massal menggunakan Beanie
            await TopicsModel.insert_many(topic_docs)
            log.info(f"Inserted {len(topic_docs)} topics for projectId: {project_id}")

            # Return object Beanie yang sudah otomatis memiliki ID dari MongoDB
            return topic_docs

        except Exception as e:
            log.exception(f"Error creating topics: {e}")
            return []

    async def create_documents(
        self, documents_data: List[dict]
    ) -> List[DocumentsModel]:
        """Create multiple document data with duplicate checking"""
        try:
            if not documents_data:
                log.warning("No documents to create")
                return []

            project_id = documents_data[0].get("projectId")

            if project_id:
                existing_count = await DocumentsModel.find(
                    DocumentsModel.projectId == project_id
                ).count()

                if existing_count > 0:
                    log.info(
                        f"Documents already exist for projectId: {project_id}, found {existing_count}. Skipping insert."
                    )
                    return await DocumentsModel.find(
                        DocumentsModel.projectId == project_id
                    ).to_list()

            doc_models = [DocumentsModel(**doc) for doc in documents_data]
            await DocumentsModel.insert_many(doc_models)

            log.info(
                f"Inserted {len(doc_models)} documents for projectId: {project_id}"
            )
            return doc_models

        except Exception as e:
            log.exception(f"Error creating documents: {e}")
            return []


def get_topic_repository(db: AsyncDatabase = Depends(get_db)) -> TopicRepository:
    """
    Dependency factory untuk menginisialisasi TopicRepository dengan database yang sudah diinisialisasi.
    """
    return TopicRepository(db)
