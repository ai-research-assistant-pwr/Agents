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
    Optimized Cypher query using last() and length() functions.
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
        # 1. Pobranie węzłów startowych
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

        # 2. Zoptymalizowana ekspansja BFS
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
                WITH start, path, last(nodes(path)) AS endNode
                RETURN endNode.paperId AS id,
                       endNode.title AS title,
                       coalesce(endNode.abstract, '') AS abstract,
                       coalesce(endNode.summary, '') AS summary,
                       length(path) AS level,
                       start.paperId AS source_id
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
    start_paper_ids: list[str], steps: int, direction: str = "both"
) -> list[dict[str, Any]]:
    """
    True Random Walk traversal with Restart (Teleportation).
    Iteratively takes one random outgoing/incoming edge. If a dead end or
    already visited node is reached, it randomly restarts from the seed node.
    """
    if not start_paper_ids:
        return []

    if direction not in ("outgoing", "incoming", "both"):
        raise ValueError(f"Invalid direction: {direction}")

    neo4j_driver = _get_driver()
    all_results: list[dict[str, Any]] = []

    # Mapowanie kierunków na notację Cypher (1-hop)
    if direction == "outgoing":
        match_pattern = "(curr)-[:REFERENCES]->(next:Paper)"
    elif direction == "incoming":
        match_pattern = "(curr)<-[:REFERENCES]-(next:Paper)"
    else:
        match_pattern = "(curr)-[:REFERENCES]-(next:Paper)"

    with neo4j_driver.session() as session:
        for start_id in start_paper_ids:
            visited_ids: set[str] = {start_id}
            current_node = start_id
            results: list[dict[str, Any]] = []

            # 1. Pobierz dane węzła startowego
            result = session.run(
                """
                MATCH (p:Paper {paperId: $startId})
                RETURN p.paperId AS id, p.title AS title,
                       coalesce(p.abstract, '') AS abstract,
                       coalesce(p.summary, '') AS summary
                """,
                startId=start_id,
            )
            record = result.single()
            if record:
                results.append(
                    {
                        "id": record["id"],
                        "title": record["title"],
                        "abstract": record["abstract"],
                        "summary": record["summary"],
                        "level": 0,
                        "source_id": start_id,
                    }
                )

            if steps == 0:
                all_results.extend(results)
                continue

            # 2. Prawdziwy Random Walk (węzeł po węźle)
            consecutive_failures = 0

            while len(visited_ids) - 1 < steps:
                # Losujemy DOKŁADNIE JEDNEGO sąsiada z obecnego węzła
                query = f"""
                MATCH {match_pattern}
                WHERE curr.paperId = $currId 
                  AND NOT next.paperId IN $visited
                RETURN next.paperId AS id, next.title AS title, 
                       coalesce(next.abstract, '') AS abstract, 
                       coalesce(next.summary, '') AS summary
                ORDER BY rand()
                LIMIT 1
                """

                step_result = session.run(
                    query, currId=current_node, visited=list(visited_ids)
                )
                next_record = step_result.single()

                if next_record:
                    # Udany skok do nowego, nieodwiedzonego węzła
                    next_id = next_record["id"]
                    visited_ids.add(next_id)
                    results.append(
                        {
                            "id": next_id,
                            "title": next_record["title"],
                            "abstract": next_record["abstract"],
                            "summary": next_record["summary"],
                            "level": len(visited_ids) - 1,
                            "source_id": start_id,
                        }
                    )
                    current_node = next_id  # Idziemy dalej z tego węzła
                    consecutive_failures = 0
                else:
                    # Ślepy zaułek (brak nieodwiedzonych sąsiadów) -> TELEPORTACJA (Restart)
                    consecutive_failures += 1
                    current_node = (
                        start_id  # Powrót do korzenia i szukanie innej ścieżki
                    )

                    # Zabezpieczenie przed nieskończoną pętlą, jeśli graf wokół start_id jest w 100% wyczerpany
                    if consecutive_failures > 5:
                        break

            all_results.extend(results)

    return all_results


def personalized_pagerank(
    start_paper_ids: list[str],
    top_n: int = 15,
    direction: str = "both",
    max_iterations: int = 20,
    damping_factor: float = 0.85,
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
            orientation=orientation,
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
                topN=top_n,
            )

            for record in result:
                score = record["score"]
                if score <= 0:
                    continue
                results.append(
                    {
                        "id": record["id"],
                        "title": record["title"],
                        "abstract": record["abstract"],
                        "summary": record["summary"],
                        "score": score,
                    }
                )
        finally:
            session.run("CALL gds.graph.drop('ppr-papers-graph')")

    return results
