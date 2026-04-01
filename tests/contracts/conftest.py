"""Shared fixtures for contract validation tests."""

import os

import httpx
import pytest


@pytest.fixture(scope="session")
def api_url() -> str:
    return os.environ["API_URL"]


@pytest.fixture(scope="session")
def workspace_id() -> str:
    return os.environ["WORKSPACE_ID"]


@pytest.fixture(scope="session")
def auth_headers() -> dict:
    return {
        "X-Api-Key": os.environ["API_KEY"],
        "X-Workspace-ID": os.environ["WORKSPACE_ID"],
        "Content-Type": "application/json",
    }


@pytest.fixture(scope="session")
def client(api_url, auth_headers) -> httpx.Client:
    with httpx.Client(
        base_url=api_url,
        headers=auth_headers,
        timeout=30.0,
        follow_redirects=True,
    ) as c:
        yield c
