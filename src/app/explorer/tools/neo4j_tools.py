import os
from typing import Any, Optional

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

VPS_IP = os.getenv("VPS_IP")
URI = f"bolt://{VPS_IP}:7687"
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver: Optional[Driver] = None


def _get_driver() -> GraphDatabase.driver:
    global driver
    if driver is None:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    return driver


def close_connections():
    global driver
    if driver is not None:
        driver.close()
        driver = None


def bfs_from_papers(
    start_paper_ids: list[str], max_level: int, direction: str = "both"
) -> list[dict[str, Any]]:
    """
    BFS traversal from starting papers using APOC path expandConfig.

    Args:
        start_paper_ids: List of paper IDs to start from
        max_level: Maximum depth of traversal (0 = only starting papers)
        direction: "outgoing", "incoming", or "both" (default)

    Returns:
        List of dicts ordered by level with keys: id, title, abstract, summary, level, source_id
    """
    if not start_paper_ids:
        return []

    if direction not in ("outgoing", "incoming", "both"):
        raise ValueError(
            f"Invalid direction: {direction}. Must be 'outgoing', 'incoming', or 'both'"
        )

    neo4j_driver = _get_driver()

    results: list[dict[str, Any]] = []

    if direction == "outgoing":
        rel_filter = "REFERENCES>"
    elif direction == "incoming":
        rel_filter = "<REFERENCES"
    else:
        rel_filter = "REFERENCES"

    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (p:Paper) WHERE p.paperId IN $paperIds
            RETURN p.paperId AS id, p.title AS title,
                   coalesce(p.abstract, '') AS abstract,
                   coalesce(p.summary, '') AS summary
            """,
            paperIds=list(start_paper_ids),
        )
        for record in result:
            results.append(
                {
                    "id": record["id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "summary": record["summary"],
                    "level": 0,
                    "source_id": record["id"],
                }
            )

        if max_level > 0:
            result = session.run(
                """
                MATCH (start:Paper) WHERE start.paperId IN $paperIds
                CALL apoc.path.expandConfig(start, {
                    relationshipFilter: $relFilter,
                    minLevel: 1,
                    maxLevel: $maxLevel,
                    bfs: true,
                    uniqueness: "node_global"
                })
                YIELD path
                RETURN path,
                       [node IN nodes(path) | node.paperId][-1] AS id,
                       [node IN nodes(path) | node.title][-1] AS title,
                       [node IN nodes(path) | coalesce(node.abstract, '')][-1] AS abstract,
                       [node IN nodes(path) | coalesce(node.summary, '')][-1] AS summary,
                       size(nodes(path)) - 1 AS level,
                       [node IN nodes(path) | node.paperId][0] AS source_id
                ORDER BY level
                """,
                paperIds=list(start_paper_ids),
                relFilter=rel_filter,
                maxLevel=max_level,
            )
            for record in result:
                results.append(
                    {
                        "id": record["id"],
                        "title": record["title"],
                        "abstract": record["abstract"],
                        "summary": record["summary"],
                        "level": record["level"],
                        "source_id": record["source_id"],
                    }
                )

    results.sort(key=lambda x: x["level"])

    return results


def random_walk(
    start_paper_ids: list[str],
    steps: int,
    direction: str = "both"
) -> list[dict[str, Any]]:
    """
    Random walk traversal that collects N nodes per start paper through iterative walks.

    For each start paper:
    - Each walk has length 4, and the last node becomes the next walk's start.
    - Continues until visited >= steps for that start paper.
    - Results from each start paper are combined.

    Args:
        start_paper_ids: List of paper IDs to start from
        steps: Target number of nodes to collect per start paper (0 returns just start papers)
        direction: "outgoing", "incoming", or "both" (default)

    Returns:
        List of dicts with keys: id, title, abstract, summary, level, source_id
    """
    if not start_paper_ids:
        return []

    if direction not in ("outgoing", "incoming", "both"):
        raise ValueError(f"Invalid direction: {direction}. Must be 'outgoing', 'incoming', or 'both'")

    WALK_LENGTH = 4
    target_nodes = steps

    neo4j_driver = _get_driver()

    all_results: list[dict[str, Any]] = []

    if direction == "outgoing":
        rel_filter = "REFERENCES>"
    elif direction == "incoming":
        rel_filter = "<REFERENCES"
    else:
        rel_filter = "REFERENCES"

    with neo4j_driver.session() as session:
        for start_id in start_paper_ids:
            visited_ids: set[str] = {start_id}
            current_start = start_id
            results: list[dict[str, Any]] = []

            # Always add starting paper at level 0 (consistent with BFS)
            result = session.run(
                """
                MATCH (p:Paper {paperId: $startId})
                RETURN p.paperId AS id, p.title AS title,
                       coalesce(p.abstract, '') AS abstract,
                       coalesce(p.summary, '') AS summary
                """,
                startId=start_id
            )
            record = result.single()
            if record:
                results.append({
                    "id": record["id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "summary": record["summary"],
                    "level": 0,
                    "source_id": start_id
                })

            if target_nodes == 0:
                all_results.extend(results)
                continue

            while len(visited_ids) < target_nodes:
                result = session.run(
                    """
                    MATCH (start:Paper {paperId: $startId})
                    CALL apoc.path.expandConfig(start, {
                        relationshipFilter: $relFilter,
                        minLevel: $walkLength,
                        maxLevel: $walkLength,
                        limit: 20,
                        bfs: true
                    }) YIELD path
                    WITH collect(path) AS paths
                    RETURN apoc.coll.randomItem(paths) AS randomWalk
                    """,
                    startId=current_start,
                    relFilter=rel_filter,
                    walkLength=WALK_LENGTH
                )
                record = result.single()
                if not record or not record["randomWalk"]:
                    break

                path = record["randomWalk"]
                nodes = list(path.nodes)
                if not nodes:
                    break

                for level, node in enumerate(nodes):
                    node_id = node.get("paperId")
                    if node_id and node_id not in visited_ids:
                        visited_ids.add(node_id)
                        results.append({
                            "id": node_id,
                            "title": node.get("title", ""),
                            "abstract": node.get("abstract", ""),
                            "summary": node.get("summary", ""),
                            "level": len(visited_ids) - 1,
                            "source_id": start_id,
                        })

                        if len(visited_ids) >= target_nodes:
                            break

                if len(visited_ids) >= target_nodes:
                    break

                next_node = nodes[-1].get("paperId")
                if not next_node or next_node in visited_ids:
                    break
                current_start = next_node

            all_results.extend(results)

    return all_results


def personalized_pagerank(
    start_paper_ids: list[str],
    top_n: int = 15,
    direction: str = "both",
    max_iterations: int = 20,
    damping_factor: float = 0.85
) -> list[dict[str, Any]]:
    """
    Personalized PageRank using GDS.

    Args:
        start_paper_ids: List of source paper IDs for personalization
        top_n: Number of top results to return (default 15)
        direction: "outgoing", "incoming", or "both" (default)
        max_iterations: Maximum iterations for PageRank (default 20)
        damping_factor: Damping factor for PageRank (default 0.85)

    Returns:
        List of dicts with keys: id, title, abstract, summary, score
    """
    if not start_paper_ids:
        return []

    if direction not in ("outgoing", "incoming", "both"):
        raise ValueError(
            f"Invalid direction: {direction}. Must be 'outgoing', 'incoming', or 'both'"
        )

    if direction == "outgoing":
        orientation = "REVERSE"
    elif direction == "incoming":
        orientation = "NATURAL"
    else:
        orientation = "UNDIRECTED"

    neo4j_driver = _get_driver()

    results: list[dict[str, Any]] = []

    with neo4j_driver.session() as session:
        session.run(
            """
            CALL gds.graph.project(
                'ppr-papers-graph',
                'Paper',
                {REFERENCES: {orientation: $orientation}}
            )
            """,
            orientation=orientation
        )

        try:
            result = session.run(
                """
                MATCH (p:Paper) WHERE p.paperId IN $paperIds
                WITH collect(p) AS sourceNodes
                CALL gds.pageRank.stream('ppr-papers-graph', {
                    maxIterations: $maxIterations,
                    dampingFactor: $dampingFactor,
                    sourceNodes: sourceNodes
                })
                YIELD nodeId, score
                ORDER BY score DESC
                LIMIT $topN
                MATCH (paper:Paper) WHERE id(paper) = nodeId
                RETURN paper.paperId AS id, paper.title AS title,
                       coalesce(paper.abstract, '') AS abstract,
                       coalesce(paper.summary, '') AS summary,
                       score
                """,
                paperIds=list(start_paper_ids),
                maxIterations=max_iterations,
                dampingFactor=damping_factor,
                topN=top_n
            )

            for record in result:
                score = record["score"]
                if score <= 0:
                    continue
                results.append({
                    "id": record["id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "summary": record["summary"],
                    "score": score
                })
        finally:
            session.run("CALL gds.graph.drop('ppr-papers-graph')")

    return results

