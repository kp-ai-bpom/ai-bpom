from typing import List

from pydantic import BaseModel, Field


# Request Model
class CommunityDetectionRequest(BaseModel):
    """Request model for community detection."""

    projectId: str = Field(
        ...,
        description="Project ID to analyze",
        json_schema_extra={"example": "proj_123456"},
    )


class BuzzerDetectionRequest(BaseModel):
    """Request model for buzzer identification."""

    projectId: str = Field(
        ...,
        description="Project ID to analyze",
        json_schema_extra={"example": "proj_123456"},
    )


# Response Model
class CommunityNode(BaseModel):
    """Node in community detection graph."""

    id: str = Field(
        ...,
        description="User identifier (username)",
        json_schema_extra={"example": "@johndoe"},
    )
    name: str = Field(
        ..., description="User display name", json_schema_extra={"example": "@johndoe"}
    )
    val: int = Field(
        ...,
        description="Node degree (number of connections)",
        json_schema_extra={"example": 15},
    )
    community: int = Field(
        ...,
        description="Community ID assigned by Louvain",
        json_schema_extra={"example": 0},
    )
    profile_url: str = Field(
        ...,
        description="Twitter/X profile URL",
        json_schema_extra={"example": "https://x.com/johndoe"},
    )


class CommunityLink(BaseModel):
    """Edge in community detection graph."""

    source: str = Field(
        ...,
        description="Source user username",
        json_schema_extra={"example": "@johndoe"},
    )
    target: str = Field(
        ...,
        description="Target user username",
        json_schema_extra={"example": "@janedoe"},
    )
    full_text: str = Field(
        ...,
        description="Tweet text content",
        json_schema_extra={"example": "@janedoe great insight!"},
    )
    topic: str = Field(
        ..., description="Topic/keyword", json_schema_extra={"example": "politik"}
    )
    url_tweet: str = Field(
        ...,
        description="Tweet URL",
        json_schema_extra={"example": "https://x.com/johndoe/status/123"},
    )
    source_community: int = Field(
        ...,
        description="Source node community ID",
        json_schema_extra={"example": 0},
    )
    target_community: int = Field(
        ...,
        description="Target node community ID",
        json_schema_extra={"example": 1},
    )


class CommunityData(BaseModel):
    """Community detection result data."""

    projectId: str = Field(
        ...,
        description="Project identifier",
        json_schema_extra={"example": "proj_123456"},
    )
    nodes: List[CommunityNode] = Field(
        ..., description="List of nodes with community assignments"
    )
    links: List[CommunityLink] = Field(..., description="List of edges between nodes")
    total_communities: int = Field(
        ...,
        description="Total number of communities detected",
        json_schema_extra={"example": 5},
    )
    total_nodes: int = Field(
        ...,
        description="Total number of nodes in graph",
        json_schema_extra={"example": 120},
    )
    total_links: int = Field(
        ...,
        description="Total number of edges in graph",
        json_schema_extra={"example": 350},
    )


class CommunityDetectionResponse(BaseModel):
    """Response model for community detection."""

    status: str = Field(
        default="success",
        description="Status of the operation",
        json_schema_extra={"example": "success"},
    )
    message: str = Field(
        default="Community detection completed successfully",
        description="Response message",
        json_schema_extra={"example": "Community detection completed successfully"},
    )
    data: CommunityData = Field(..., description="Community detection graph data")


class Buzzer(BaseModel):
    """Buzzer/influencer detection result."""

    node: str = Field(
        ...,
        description="User username",
        json_schema_extra={"example": "@influencer123"},
    )
    projectId: str = Field(
        ...,
        description="Project identifier",
        json_schema_extra={"example": "proj_123456"},
    )
    BEC: float = Field(
        ...,
        description="Betweenness Centrality raw score",
        json_schema_extra={"example": 0.125},
    )
    EVC: float = Field(
        ...,
        description="Eigenvector Centrality raw score",
        json_schema_extra={"example": 0.095},
    )
    BEC_Norm: float = Field(
        ...,
        description="Normalized Betweenness Centrality (0-1)",
        json_schema_extra={"example": 0.85},
    )
    EVC_Norm: float = Field(
        ...,
        description="Normalized Eigenvector Centrality (0-1)",
        json_schema_extra={"example": 0.78},
    )
    final_measure: float = Field(
        ...,
        description="Combined measure (avg of BEC_Norm and EVC_Norm)",
        json_schema_extra={"example": 0.815},
    )
    tweet_url: str = Field(
        ...,
        description="Profile URL",
        json_schema_extra={"example": "https://x.com/influencer123"},
    )


class BuzzerDetectionResponse(BaseModel):
    """Response model for buzzer identification."""

    status: str = Field(
        default="success",
        description="Status of the operation",
        json_schema_extra={"example": "success"},
    )
    message: str = Field(
        default="Buzzer detection completed successfully",
        description="Response message",
        json_schema_extra={"example": "Buzzer detection completed successfully"},
    )
    data: List[Buzzer] = Field(..., description="List of detected buzzers/influencers")
    totalData: int = Field(
        ...,
        description="Total number of buzzers detected",
        json_schema_extra={"example": 10},
    )
