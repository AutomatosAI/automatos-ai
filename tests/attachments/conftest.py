"""
Pytest fixtures for attachment tests (PRD-127)
"""

import pytest
from uuid import uuid4


@pytest.fixture
def workspace_id():
    """Generate a random workspace ID."""
    return uuid4()


@pytest.fixture
def user_id():
    """Generate a test user ID."""
    return f"user_{uuid4().hex[:8]}"


@pytest.fixture
def sample_png():
    """Minimal valid 1x1 transparent PNG."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def sample_jpeg():
    """Minimal JPEG header (not a valid full image, but enough for MIME detection)."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 100


@pytest.fixture
def sample_text():
    """Sample text content."""
    return b"This is sample text content for testing."


@pytest.fixture
def sample_python():
    """Sample Python source code."""
    return b'''"""Sample module."""

def hello(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(hello("World"))
'''
