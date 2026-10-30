import os

# Tests always run fully offline: deterministic model, hashing embeddings, in-memory Qdrant.
os.environ.update(
    {
        "LLM_PROVIDER": "offline",
        "EMBEDDING_PROVIDER": "hashing",
        "QDRANT_PATH": ":memory:",
        "QDRANT_URL": "",
        "LANGFUSE_HOST": "",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.deps import reset_singletons  # noqa: E402


@pytest.fixture(scope="session")
def client():
    reset_singletons()
    from app.main import app

    with TestClient(app) as c:  # lifespan ingests data/sample_docs
        yield c
