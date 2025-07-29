
"""
Test configuration and fixtures
"""
import pytest
import httpx
import asyncio
from typing import AsyncGenerator

# Base URL for API tests
BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client():
    """Create HTTP client for API testing"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        yield client

@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing"""
    return {
        "name": "Test Agent",
        "description": "A test agent for API testing",
        "agent_type": "custom",
        "configuration": {
            "test_mode": True,
            "max_iterations": 5
        }
    }

@pytest.fixture
def sample_skill_data():
    """Sample skill data for testing"""
    return {
        "name": "test_skill",
        "skill_type": "technical",
        "description": "A test skill for API testing",
        "parameters": {
            "complexity": "medium",
            "timeout": 30
        }
    }

@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for testing"""
    return {
        "name": "Test Workflow",
        "description": "A test workflow for API testing",
        "steps": [
            {
                "name": "step1",
                "agent_type": "custom",
                "task": "Analyze input data"
            },
            {
                "name": "step2", 
                "agent_type": "custom",
                "task": "Generate output"
            }
        ]
    }
