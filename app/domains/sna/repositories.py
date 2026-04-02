from datetime import datetime
from typing import Dict, List, Optional

from fastapi import Depends
from pymongo.asynchronous.database import AsyncDatabase

from app.core.logger import log
from app.db.database import get_db
from app.domains.sna.models import BuzzerModel, CommunityDetectionModel
from app.domains.topic_modeling.models import DocumentsModel


class SNARepository:
    """Repository for Social Network Analysis data access."""

    def __init__(self, db: AsyncDatabase):
        """
        Initialize SNA repository.

        Args:
            db: MongoDB database instance
        """
        self.db = db

    # ======================== Tweet Data Access (Beanie) ========================

    async def get_tweets_by_project(self, project_id: str) -> List[Dict]:
        """
        Retrieve all documents (tweets) for a project from topic_modeling collection.

        Args:
            project_id: Project identifier

        Returns:
            List of document dictionaries with username, full_text, tweet_url, topic, etc.
        """
        try:
            # Query using Beanie ODM from topic_modeling domain
            documents = await DocumentsModel.find(
                DocumentsModel.projectId == project_id
            ).to_list()

            if not documents:
                log.warning(f"⚠️ No documents found for project {project_id}")
                return []

            # Convert Beanie documents to dictionaries
            tweets = []
            for doc in documents:
                tweets.append(
                    {
                        "username": doc.username,
                        "full_text": doc.raw_text,  # Use raw_text for mention extraction (@username)
                        "tweet_url": doc.tweet_url,
                        "topic": str(
                            doc.topic
                        ),  # Convert int to string for consistency
                        "in_reply_to_screen_name": None,  # Not available in DocumentsModel
                        "created_at": None,  # Not available in DocumentsModel
                    }
                )

            log.info(f"📊 Retrieved {len(tweets)} documents for project {project_id}")
            return tweets

        except Exception as e:
            log.exception(f"❌ Error fetching documents for project {project_id}: {e}")
            raise

    # ======================== Community Detection (Beanie) ========================

    async def save_community_detection(
        self,
        project_id: str,
        nodes: List[Dict],
        links: List[Dict],
        total_communities: int,
    ) -> CommunityDetectionModel:
        """
        Save community detection results to database.

        Args:
            project_id: Project identifier
            nodes: List of node dictionaries with community assignments
            links: List of link dictionaries with edge data
            total_communities: Number of communities detected

        Returns:
            Saved CommunityDetectionModel document
        """
        try:
            # Delete existing results for this project
            await CommunityDetectionModel.find(
                CommunityDetectionModel.projectId == project_id
            ).delete()

            # Create new document
            community_doc = CommunityDetectionModel(
                projectId=project_id,
                nodes=nodes,
                links=links,
                total_communities=total_communities,
                total_nodes=len(nodes),
                total_links=len(links),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            await community_doc.insert()
            log.info(
                f"💾 Saved community detection for project {project_id} | Communities: {total_communities}"
            )
            return community_doc

        except Exception as e:
            log.exception(f"❌ Error saving community detection: {e}")
            raise

    async def get_community_detection(
        self, project_id: str
    ) -> Optional[CommunityDetectionModel]:
        """
        Retrieve community detection results for a project.

        Args:
            project_id: Project identifier

        Returns:
            CommunityDetectionModel document or None if not found
        """
        try:
            result = await CommunityDetectionModel.find_one(
                CommunityDetectionModel.projectId == project_id
            )
            return result

        except Exception as e:
            log.exception(f"❌ Error fetching community detection: {e}")
            raise

    # ======================== Buzzer Detection (Beanie) ========================

    async def save_buzzer_detection(
        self, project_id: str, buzzers: List[Dict]
    ) -> List[BuzzerModel]:
        """
        Save buzzer detection results to database.

        Args:
            project_id: Project identifier
            buzzers: List of buzzer dictionaries with centrality scores

        Returns:
            List of saved BuzzerModel documents
        """
        try:
            # Delete existing results for this project
            await BuzzerModel.find(BuzzerModel.projectId == project_id).delete()

            # Create new documents
            buzzer_docs = []
            for buzzer in buzzers:
                buzzer_doc = BuzzerModel(
                    node=buzzer["node"],
                    projectId=project_id,
                    BEC=buzzer["BEC"],
                    EVC=buzzer["EVC"],
                    BEC_Norm=buzzer["BEC_Norm"],
                    EVC_Norm=buzzer["EVC_Norm"],
                    final_measure=buzzer["final_measure"],
                    tweet_url=buzzer["tweet_url"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                buzzer_docs.append(buzzer_doc)

            if buzzer_docs:
                await BuzzerModel.insert_many(buzzer_docs)
                log.info(
                    f"💾 Saved {len(buzzer_docs)} buzzers for project {project_id}"
                )

            return buzzer_docs

        except Exception as e:
            log.exception(f"❌ Error saving buzzer detection: {e}")
            raise

    async def get_buzzer_detection(
        self, project_id: str, limit: int = 10
    ) -> List[BuzzerModel]:
        """
        Retrieve buzzer detection results for a project.

        Args:
            project_id: Project identifier
            limit: Maximum number of buzzers to return (default: 10)

        Returns:
            List of BuzzerModel documents sorted by final_measure descending
        """
        try:
            buzzers = (
                await BuzzerModel.find(BuzzerModel.projectId == project_id)
                .sort("-final_measure")
                .limit(limit)
                .to_list()
            )

            return buzzers

        except Exception as e:
            log.exception(f"❌ Error fetching buzzer detection: {e}")
            raise


def get_sna_repository(db: AsyncDatabase = Depends(get_db)) -> SNARepository:
    """
    Dependency injection factory for SNARepository.

    Args:
        db: MongoDB database instance from dependency

    Returns:
        SNARepository instance
    """
    return SNARepository(db)
