"""
Authentication helpers for the Orchestrator API.

This package provides:
- Clerk JWT verification utilities (`clerk.py`)
- Shared request/user context types (`dependencies.py`)
- A "hybrid" FastAPI dependency that supports Clerk + API key + dev fallback (`hybrid.py`)
"""

