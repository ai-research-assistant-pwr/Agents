"""Global pytest configuration.

Sets environment variables BEFORE any api.* modules are imported so that
SQLAlchemy uses an in-memory SQLite database and the pipeline stays mocked.
Using conftest.py guarantees these run before test-file imports.
"""

import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["PIPELINE_USE_MOCK"] = "true"
os.environ["JWT_SECRET"] = "test-secret-key-at-least-32-chars-long"
