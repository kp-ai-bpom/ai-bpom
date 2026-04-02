from datetime import datetime
from typing import List

from beanie import Document
from pydantic import BaseModel, Field


class CommunityNodeDocument(BaseModel):
    """Embedded document for community graph nodes."""

    id: str = Field(..., description="User identifier (username)")
    name: str = Field(..., description="User display name")
    val: int = Field(..., description="Node degree (connections)")
    community: int = Field(..., description="Community ID from Louvain")
    profile_url: str = Field(..., description="Twitter/X profile URL")


class CommunityLinkDocument(BaseModel):
    """Embedded document for community graph edges."""

    source: str = Field(..., description="Source user")
    target: str = Field(..., description="Target user")
    full_text: str = Field(..., description="Tweet content")
    topic: str = Field(..., description="Topic/keyword")
    url_tweet: str = Field(..., description="Tweet URL")
    source_community: int = Field(..., description="Source community ID")
    target_community: int = Field(..., description="Target community ID")


class CommunityDetectionModel(Document):
    """
    Community detection results for a project.
    Stores graph data with nodes (users) and links (interactions),
    each annotated with community assignments from Louvain algorithm.
    """

    projectId: str = Field(..., description="Project identifier")
    nodes: List[dict] = Field(
        default_factory=list, description="List of graph nodes with community info"
    )
    links: List[dict] = Field(default_factory=list, description="List of graph edges")
    total_communities: int = Field(..., description="Number of communities detected")
    total_nodes: int = Field(..., description="Total number of nodes")
    total_links: int = Field(..., description="Total number of edges")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    class Settings:
        name = "sna_communities"
        indexes = [
            "projectId",
            "created_at",
        ]


class BuzzerModel(Document):
    """
    Buzzer/influencer detection results for a project.
    Stores ranked list of influential users based on network centrality measures.
    """

    node: str = Field(..., description="User username")
    projectId: str = Field(..., description="Project identifier")
    BEC: float = Field(..., description="Betweenness Centrality")
    EVC: float = Field(..., description="Eigenvector Centrality")
    BEC_Norm: float = Field(..., description="Normalized BEC (0-1)")
    EVC_Norm: float = Field(..., description="Normalized EVC (0-1)")
    final_measure: float = Field(..., description="Combined influence score")
    tweet_url: str = Field(..., description="User profile URL")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    class Settings:
        name = "sna_buzzers"
        indexes = [
            "projectId",
            "final_measure",  # For sorting by influence score
            "created_at",
        ]
