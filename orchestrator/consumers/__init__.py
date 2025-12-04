"""
Consumers - Platform Integration Layer
======================================

Thin wrappers that combine modules for specific use cases.
NO business logic here - just glue code.

Components:
- chatbot/    - Chat interface (uses rag, memory, tools)
- workflows/  - Workflow execution (uses agents, tools)
- external/   - Third-party API (exposes modules)

Usage:
    from consumers.chatbot import ChatService
    from consumers.workflows import WorkflowIntegrator
"""

__all__ = []

