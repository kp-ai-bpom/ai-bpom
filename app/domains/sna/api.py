from fastapi import APIRouter, Depends, HTTPException

from app.core.logger import log
from app.domains.sna.schemas import (
    BuzzerDetectionRequest,
    BuzzerDetectionResponse,
    CommunityDetectionRequest,
    CommunityDetectionResponse,
)
from app.domains.sna.services import SNAService, get_sna_service

router = APIRouter()


@router.post(
    "/community-detection",
    response_model=CommunityDetectionResponse,
    summary="Detect Communities in Social Network",
    description="""
    Perform community detection on social network graph using Louvain algorithm.

    This endpoint analyzes mentions and replies in tweets to construct a social network graph,
    then applies the Louvain community detection algorithm to identify clusters of users.

    **Algorithm Steps:**
    1. Extract mentions (@username) and replies from tweets
    2. Build NetworkX graph with users as nodes and interactions as edges
    3. Find largest connected component
    4. Apply Louvain algorithm for community detection
    5. Assign community IDs sorted by size
    6. Return graph with nodes (users) and links (interactions) labeled by community

    **Returns:**
    - `nodes`: List of users with community assignments
    - `links`: List of interactions between users
    - `total_communities`: Number of communities detected
    """,
)
async def detect_communities(
    request: CommunityDetectionRequest,
    service: SNAService = Depends(get_sna_service),
):
    """
    Endpoint for community detection on social network.

    Args:
        request: CommunityDetectionRequest with projectId
        service: SNAService instance from dependency injection

    Returns:
        Response wrapping CommunityDetectionResponse

    Raises:
        HTTPException: 404 if no tweets found, 400 if insufficient data, 500 on processing error
    """
    try:
        result = await service.detect_communities(request.projectId)
        return CommunityDetectionResponse(
            status="success",
            message="Community detection completed successfully",
            data=result,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"❌ Error in /community-detection endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/community-detection/{project_id}",
    response_model=CommunityDetectionResponse,
    summary="Get Saved Community Detection Results",
    description="""
    Retrieve previously calculated community detection results for a project.

    Use this endpoint to fetch cached results without re-running the algorithm.
    """,
)
async def get_community_detection(
    project_id: str,
    service: SNAService = Depends(get_sna_service),
):
    """
    Endpoint to retrieve saved community detection results.

    Args:
        project_id: Project identifier
        service: SNAService instance from dependency injection

    Returns:
        Response wrapping CommunityDetectionResponse

    Raises:
        HTTPException: 404 if results not found, 500 on processing error
    """
    try:
        result = await service.get_community_detection_result(project_id)
        return CommunityDetectionResponse(
            status="success",
            message="Community detection results retrieved successfully",
            data=result,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"❌ Error in GET /community-detection endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/buzzer-detection",
    response_model=BuzzerDetectionResponse,
    summary="Detect Influential Users (Buzzers)",
    description="""
    Identify influential users in social network using centrality measures.

    This endpoint analyzes the social network structure to find users with high influence
    based on two centrality metrics:
    - **Betweenness Centrality (BEC)**: Measures how often a user appears on shortest paths
    - **Eigenvector Centrality (EVC)**: Measures influence based on connections to influential users

    **Algorithm Steps:**
    1. Build social network graph from mentions and replies
    2. Find largest connected component
    3. Calculate BEC and EVC for each user
    4. Normalize scores to 0-1 range
    5. Compute final influence measure = (BEC_Norm + EVC_Norm) / 2
    6. Rank users by final measure
    7. Return top 10 buzzers

    **Returns:**
    - List of top influencers with centrality scores
    - Profile URLs for each buzzer
    """,
)
async def detect_buzzers(
    request: BuzzerDetectionRequest,
    service: SNAService = Depends(get_sna_service),
):
    """
    Endpoint for buzzer/influencer detection.

    Args:
        request: BuzzerDetectionRequest with projectId
        service: SNAService instance from dependency injection

    Returns:
        Response wrapping BuzzerDetectionResponse

    Raises:
        HTTPException: 404 if no tweets found, 400 if insufficient data, 500 on processing error
    """
    try:
        result = await service.detect_buzzers(request.projectId)
        return BuzzerDetectionResponse(
            status="success",
            message="Buzzer detection completed successfully",
            data=result,
            totalData=len(result),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"❌ Error in /buzzer-detection endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/buzzer-detection/{project_id}",
    response_model=BuzzerDetectionResponse,
    summary="Get Saved Buzzer Detection Results",
    description="""
    Retrieve previously calculated buzzer detection results for a project.

    Use this endpoint to fetch cached results without re-running the algorithm.

    **Query Parameters:**
    - `limit`: Maximum number of buzzers to return (default: 10)
    """,
)
async def get_buzzer_detection(
    project_id: str,
    limit: int = 10,
    service: SNAService = Depends(get_sna_service),
):
    """
    Endpoint to retrieve saved buzzer detection results.

    Args:
        project_id: Project identifier
        limit: Maximum number of buzzers to return (default: 10)
        service: SNAService instance from dependency injection

    Returns:
        Response wrapping BuzzerDetectionResponse

    Raises:
        HTTPException: 404 if results not found, 500 on processing error
    """
    try:
        result = await service.get_buzzer_detection_result(project_id, limit)
        return BuzzerDetectionResponse(
            status="success",
            message="Buzzer detection results retrieved successfully",
            data=result,
            totalData=len(result),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"❌ Error in GET /buzzer-detection endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
