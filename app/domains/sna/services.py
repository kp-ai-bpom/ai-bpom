import asyncio
import re
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from community import community_louvain
from fastapi import Depends, HTTPException

from app.core.logger import log
from app.domains.sna.repositories import SNARepository, get_sna_repository
from app.domains.sna.schemas import (
    Buzzer,
    CommunityData,
    CommunityLink,
    CommunityNode,
)


class SNAService:
    """Service for Social Network Analysis operations."""

    def __init__(self, repository: SNARepository):
        """
        Initialize SNA service.

        Args:
            repository: SNARepository instance for data access
        """
        self.repository = repository

    async def detect_communities(self, project_id: str) -> CommunityData:
        """
        Perform community detection on social network graph using Louvain algorithm.

        Algorithm steps:
        1. Fetch tweets for the project
        2. Extract mentions and replies to build edges
        3. Create NetworkX graph from edges
        4. Find largest connected component
        5. Apply Louvain community detection
        6. Assign community IDs and generate graph data
        7. Save results to database

        Args:
            project_id: Project identifier

        Returns:
            CommunityDetectionResponse with nodes, links, and statistics

        Raises:
            HTTPException: If no data found or processing fails
        """
        try:
            log.info(f"🚀 Starting Community Detection | Project: {project_id}")

            # Fetch tweets
            tweets = await self.repository.get_tweets_by_project(project_id)
            if not tweets:
                raise HTTPException(
                    status_code=404, detail=f"No tweets found for project {project_id}"
                )

            # Run CPU-intensive graph analysis in thread pool
            nodes, links, total_communities = await asyncio.to_thread(
                self._run_community_detection, tweets
            )

            if not nodes or not links:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to detect communities - insufficient network data",
                )

            # Convert to dictionaries for storage
            nodes_dict = [node.model_dump() for node in nodes]
            links_dict = [link.model_dump() for link in links]

            # Save to database
            await self.repository.save_community_detection(
                project_id=project_id,
                nodes=nodes_dict,
                links=links_dict,
                total_communities=total_communities,
            )

            # Construct and return inner data
            community_data = CommunityData(
                projectId=project_id,
                nodes=nodes,
                links=links,
                total_communities=total_communities,
                total_nodes=len(nodes),
                total_links=len(links),
            )

            log.info(
                f"✅ Community Detection Complete | Project: {project_id} | "
                f"Communities: {total_communities} | Nodes: {len(nodes)} | Links: {len(links)}"
            )

            return community_data

        except HTTPException:
            raise
        except Exception as e:
            log.exception(f"❌ Error in community detection: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _run_community_detection(
        self, tweets: List[Dict]
    ) -> Tuple[List[CommunityNode], List[CommunityLink], int]:
        """
        CPU-intensive community detection algorithm (runs in thread pool).
        Ported from legacy CommunityWorker.communityDetLouvain().

        Args:
            tweets: List of tweet documents

        Returns:
            Tuple of (nodes, links, total_communities)
        """
        # Step 1: Extract mentions from tweets
        edges = []
        for tweet in tweets:
            username = tweet.get("username")
            full_text = tweet.get("full_text", "")
            topic = tweet.get("topic", "")

            # Find @mentions using regex
            mentions = re.findall(r"@(\w+)", full_text)
            if mentions:
                for mention in mentions:
                    edges.append(
                        {"source": username, "target": mention, "topic": topic}
                    )

        edges_df = pd.DataFrame(edges)

        # Step 2: Extract replies
        reply_edges = []
        for tweet in tweets:
            source = tweet.get("username")
            target = tweet.get("in_reply_to_screen_name")
            topic = tweet.get("topic", "")

            if target:
                reply_edges.append({"source": source, "target": target, "topic": topic})

        reply_edges_df = pd.DataFrame(reply_edges)

        # Step 3: Combine edges and filter self-loops
        combined_edges_df = pd.concat([edges_df, reply_edges_df], ignore_index=True)

        # Check if DataFrame is empty or has no columns
        if combined_edges_df.empty or "source" not in combined_edges_df.columns:
            log.warning("⚠️ No valid edges found - no mentions or replies extracted")
            return [], [], 0

        combined_edges_df = combined_edges_df[
            combined_edges_df["source"] != combined_edges_df["target"]
        ]

        if combined_edges_df.empty:
            log.warning("⚠️ No valid edges found after filtering self-loops")
            return [], [], 0

        # Step 4: Build graph from edges
        G = nx.from_pandas_edgelist(
            combined_edges_df, source="source", target="target", edge_attr=True
        )

        # Step 5: Find largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest_cc = G.subgraph(largest_cc).copy()
        else:
            G_largest_cc = G

        # Step 6: Apply Louvain community detection
        partition = community_louvain.best_partition(G_largest_cc)

        # Step 7: Sort communities by size and assign IDs
        community_sizes = {}
        for node, comm_id in partition.items():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1

        sorted_communities = sorted(
            community_sizes.items(), key=lambda x: x[1], reverse=True
        )
        community_mapping = {
            old_id: new_id for new_id, (old_id, _) in enumerate(sorted_communities)
        }

        # Update partition with new IDs
        partition = {
            node: community_mapping[comm_id] for node, comm_id in partition.items()
        }

        # Step 8: Create nodes with community assignments
        nodes = []
        for node in G_largest_cc.nodes():
            degree = G_largest_cc.degree(node)
            community_id = partition[node]
            nodes.append(
                CommunityNode(
                    id=node,
                    name=node,
                    val=degree,
                    community=community_id,
                    profile_url=f"https://x.com/{node}",
                )
            )

        # Step 9: Create links with community labels
        # First, create a mapping of tweets for URLs
        tweet_url_map = {}
        for tweet in tweets:
            source = tweet.get("username")
            target = tweet.get("in_reply_to_screen_name")
            url = tweet.get("tweet_url", "")

            if target:
                tweet_url_map[(source, target)] = url

            # Also check mentions
            full_text = tweet.get("full_text", "")
            mentions = re.findall(r"@(\w+)", full_text)
            for mention in mentions:
                if (source, mention) not in tweet_url_map:
                    tweet_url_map[(source, mention)] = url

        links = []
        for u, v, data in G_largest_cc.edges(data=True):
            topic = data.get("topic", "")
            # Find matching tweet for full_text and URL
            matching_tweet = None
            for tweet in tweets:
                if (tweet.get("username") == u) and (
                    tweet.get("in_reply_to_screen_name") == v
                    or f"@{v}" in tweet.get("full_text", "")
                ):
                    matching_tweet = tweet
                    break

            full_text = matching_tweet.get("full_text", "") if matching_tweet else ""
            url_tweet = tweet_url_map.get((u, v), "")

            links.append(
                CommunityLink(
                    source=u,
                    target=v,
                    full_text=full_text,
                    topic=topic,
                    url_tweet=url_tweet,
                    source_community=partition[u],
                    target_community=partition[v],
                )
            )

        total_communities = len(set(partition.values()))

        return nodes, links, total_communities

    # ======================== Buzzer Detection ========================

    async def detect_buzzers(self, project_id: str, top_n: int = 10) -> List[Buzzer]:
        """
        Identify influential users (buzzers) in social network using centrality measures.

        Algorithm steps:
        1. Fetch tweets for the project
        2. Extract mentions and replies to build edges
        3. Create NetworkX graph
        4. Calculate Betweenness Centrality (BEC) and Eigenvector Centrality (EVC)
        5. Normalize scores and compute final measure
        6. Rank users by influence score
        7. Save top N buzzers to database

        Args:
            project_id: Project identifier
            top_n: Number of top buzzers to return (default: 10)

        Returns:
            BuzzerDetectionResponse with ranked buzzers

        Raises:
            HTTPException: If no data found or processing fails
        """
        try:
            log.info(f"🚀 Starting Buzzer Detection | Project: {project_id}")

            # Fetch tweets
            tweets = await self.repository.get_tweets_by_project(project_id)
            if not tweets:
                raise HTTPException(
                    status_code=404, detail=f"No tweets found for project {project_id}"
                )

            # Run CPU-intensive centrality analysis in thread pool
            buzzers_data = await asyncio.to_thread(
                self._run_buzzer_detection, tweets, top_n
            )

            if not buzzers_data:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to detect buzzers - insufficient network data",
                )

            # Save to database
            await self.repository.save_buzzer_detection(
                project_id=project_id, buzzers=buzzers_data
            )

            # Convert to Pydantic models (inject projectId not present in centrality dict)
            buzzers = [
                Buzzer(projectId=project_id, **buzzer) for buzzer in buzzers_data
            ]

            log.info(
                f"✅ Buzzer Detection Complete | Project: {project_id} | "
                f"Buzzers Detected: {len(buzzers)}"
            )

            return buzzers

        except HTTPException:
            raise
        except Exception as e:
            log.exception(f"❌ Error in buzzer detection: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _run_buzzer_detection(self, tweets: List[Dict], top_n: int) -> List[Dict]:
        """
        CPU-intensive buzzer detection algorithm (runs in thread pool).
        Ported from legacy BuzzerWorker.buzzerRec().

        Args:
            tweets: List of tweet documents
            top_n: Number of top buzzers to return

        Returns:
            List of buzzer dictionaries with centrality scores
        """
        # Step 1: Extract mentions from tweets
        edges = []
        for tweet in tweets:
            username = tweet.get("username")
            full_text = tweet.get("full_text", "")
            topic = tweet.get("topic", "")

            mentions = re.findall(r"@(\w+)", full_text)
            if mentions:
                for mention in mentions:
                    edges.append(
                        {"source": username, "target": mention, "topic": topic}
                    )

        edges_df = pd.DataFrame(edges)

        # Step 2: Extract replies
        reply_edges = []
        for tweet in tweets:
            source = tweet.get("username")
            target = tweet.get("in_reply_to_screen_name")
            topic = tweet.get("topic", "")

            if target:
                reply_edges.append({"source": source, "target": target, "topic": topic})

        reply_edges_df = pd.DataFrame(reply_edges)

        # Step 3: Combine and filter edges
        combined_edges_df = pd.concat([edges_df, reply_edges_df], ignore_index=True)

        # Check if DataFrame is empty or has no columns
        if combined_edges_df.empty or "source" not in combined_edges_df.columns:
            log.warning(
                "⚠️ No valid edges found for buzzer detection - no mentions or replies extracted"
            )
            return []

        combined_edges_df = combined_edges_df[
            combined_edges_df["source"] != combined_edges_df["target"]
        ]

        if combined_edges_df.empty:
            log.warning(
                "⚠️ No valid edges found for buzzer detection after filtering self-loops"
            )
            return []

        # Step 4: Group by source, target, topic
        grouped_edges_df = (
            combined_edges_df.groupby(["source", "target", "topic"])
            .first()
            .reset_index()
        )

        # Step 5: Build graph
        G = nx.from_pandas_edgelist(
            grouped_edges_df, source="source", target="target", edge_attr=True
        )

        # Step 6: Find largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            largest_subgraph = G.subgraph(largest_cc).copy()
        else:
            largest_subgraph = G

        # Step 7: Calculate centrality measures
        bec = nx.betweenness_centrality(largest_subgraph)
        evc = nx.eigenvector_centrality(largest_subgraph)

        # Step 8: Create centrality DataFrame
        bec_df = pd.DataFrame(bec.items(), columns=["node", "BEC"])  # type: ignore[misc]
        evc_df = pd.DataFrame(evc.items(), columns=["node", "EVC"])  # type: ignore[misc]
        centrality_df = pd.merge(bec_df, evc_df, on="node")

        # Step 9: Normalize scores
        max_bec = centrality_df["BEC"].max()
        max_evc = centrality_df["EVC"].max()

        # Avoid division by zero
        centrality_df["BEC_Norm"] = centrality_df["BEC"] / max_bec if max_bec > 0 else 0
        centrality_df["EVC_Norm"] = centrality_df["EVC"] / max_evc if max_evc > 0 else 0

        # Step 10: Calculate final measure (average of normalized scores)
        centrality_df["final_measure"] = (
            centrality_df["EVC_Norm"] + centrality_df["BEC_Norm"]
        ) / 2

        # Step 11: Sort by final measure and get top N
        centrality_df_sorted = centrality_df.sort_values(
            by="final_measure", ascending=False
        )
        top_buzzers = centrality_df_sorted.head(top_n).copy()

        # Step 12: Generate tweet URLs
        top_buzzers["tweet_url"] = top_buzzers["node"].apply(
            lambda x: f"https://x.com/{x}"
        )

        # Step 13: Convert to list of dictionaries
        buzzers_data = top_buzzers.to_dict("records")

        return buzzers_data

    # ======================== Retrieve Saved Results ========================

    async def get_community_detection_result(self, project_id: str) -> CommunityData:
        """
        Retrieve saved community detection results.

        Args:
            project_id: Project identifier

        Returns:
            CommunityDetectionResponse with saved data

        Raises:
            HTTPException: If results not found
        """
        try:
            result = await self.repository.get_community_detection(project_id)

            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"No community detection results found for project {project_id}",
                )

            # Convert to inner data model
            nodes = [CommunityNode(**node) for node in result.nodes]
            links = [CommunityLink(**link) for link in result.links]

            community_data = CommunityData(
                projectId=result.projectId,
                nodes=nodes,
                links=links,
                total_communities=result.total_communities,
                total_nodes=result.total_nodes,
                total_links=result.total_links,
            )

            return community_data

        except HTTPException:
            raise
        except Exception as e:
            log.exception(f"❌ Error fetching community detection results: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_buzzer_detection_result(
        self, project_id: str, limit: int = 10
    ) -> List[Buzzer]:
        """
        Retrieve saved buzzer detection results.

        Args:
            project_id: Project identifier
            limit: Maximum number of buzzers to return

        Returns:
            BuzzerDetectionResponse with saved buzzers

        Raises:
            HTTPException: If results not found
        """
        try:
            buzzers = await self.repository.get_buzzer_detection(project_id, limit)

            if not buzzers:
                raise HTTPException(
                    status_code=404,
                    detail=f"No buzzer detection results found for project {project_id}",
                )

            # Convert to Pydantic models
            buzzer_list = [
                Buzzer(
                    node=b.node,
                    projectId=b.projectId,
                    BEC=b.BEC,
                    EVC=b.EVC,
                    BEC_Norm=b.BEC_Norm,
                    EVC_Norm=b.EVC_Norm,
                    final_measure=b.final_measure,
                    tweet_url=b.tweet_url,
                )
                for b in buzzers
            ]

            return buzzer_list

        except HTTPException:
            raise
        except Exception as e:
            log.exception(f"❌ Error fetching buzzer detection results: {e}")
            raise HTTPException(status_code=500, detail=str(e))


def get_sna_service(
    repository: SNARepository = Depends(get_sna_repository),
) -> SNAService:
    """
    Dependency injection factory for SNAService.

    Args:
        repository: SNARepository instance from dependency

    Returns:
        SNAService instance
    """
    return SNAService(repository)
