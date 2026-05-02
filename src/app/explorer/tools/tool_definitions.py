from typing import Any

from .neo4j_tools import (
    bfs_from_papers,
    personalized_pagerank,
    random_walk,
)


def execute_bfs(start_paper_ids: list[str], max_level: int = 1, direction: str = "both") -> list[dict[str, Any]]:
    """Execute BFS traversal from starting papers."""
    return bfs_from_papers(start_paper_ids=start_paper_ids, max_level=max_level, direction=direction)


def execute_random_walk(start_paper_ids: list[str], steps: int = 2, direction: str = "both") -> list[dict[str, Any]]:
    """Execute random walk from starting papers."""
    return random_walk(start_paper_ids=start_paper_ids, steps=steps, direction=direction)


def execute_ppr(start_paper_ids: list[str], top_n: int = 15, direction: str = "both") -> list[dict[str, Any]]:
    """Execute personalized PageRank from starting papers."""
    return personalized_pagerank(start_paper_ids=start_paper_ids, top_n=top_n, direction=direction)


TOOL_DEFINITIONS = {
    "bfs_from_papers": {
        "description": "Explore papers using breadth-first search. Returns papers at each level from starting papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of paper IDs to start from"
                },
                "max_level": {
                    "type": "integer",
                    "description": "Maximum depth of traversal (0 = only starting papers, 1 = direct citations, etc.)"
                },
                "direction": {
                    "type": "string",
                    "enum": ["outgoing", "incoming", "both"],
                    "description": "Direction of citation edges"
                }
            },
            "required": ["start_paper_ids", "max_level"]
        },
        "execute": execute_bfs
    },
    "random_walk": {
        "description": "Perform random walk traversal from starting papers. Returns a random path of papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of paper IDs to start from"
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of hops (edges) to traverse"
                },
                "direction": {
                    "type": "string",
                    "enum": ["outgoing", "incoming", "both"],
                    "description": "Direction of citation edges"
                }
            },
            "required": ["start_paper_ids", "steps"]
        },
        "execute": execute_random_walk
    },
    "ppr": {
        "description": "Compute personalized PageRank scores from starting papers. Returns top ranked papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of source paper IDs for personalization"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top results to return"
                },
                "direction": {
                    "type": "string",
                    "enum": ["outgoing", "incoming", "both"],
                    "description": "Direction of citation edges"
                }
            },
            "required": ["start_paper_ids"]
        },
        "execute": execute_ppr
    }
}


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Get OpenAI-style tool schema for a specific tool."""
    if tool_name not in TOOL_DEFINITIONS:
        return None
    
    tool = TOOL_DEFINITIONS[tool_name]
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool["description"],
            "parameters": tool["parameters"]
        }
    }


def get_tool_executor(tool_name: str):
    """Get the execution function for a specific tool."""
    if tool_name not in TOOL_DEFINITIONS:
        return None
    return TOOL_DEFINITIONS[tool_name]["execute"]
