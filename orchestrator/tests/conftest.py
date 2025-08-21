from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orchestrator.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


