"""Development server entry point.

Run from the project root:

    python run_api.py

Or with auto-reload during development:

    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Environment variables
---------------------
PIPELINE_USE_MOCK   "true" (default) — no external services needed.
                    "false" — requires API keys + Neo4j/Weaviate connections.
PORT                Override the listen port (default 8000).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
