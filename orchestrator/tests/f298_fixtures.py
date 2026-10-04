"""What the F298 tests share: generate_document as an agent calls it, with the
document service recorded instead of rendered (the PDF tests render for real)."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import modules.documents.generation_service as generation_service
import services.knowledge_flywheel as knowledge_flywheel
from core.models import Agent

WS = uuid.UUID("00000000-0000-0000-0000-0000000298a1")
AGENT = SimpleNamespace(id=298, workspace_id=WS, user_id=None, name="Content Creator")
FILENAME = "20261004_045759_Wholesale_Price_List.pdf"
DELIVERABLE_ID = "5f0c2a8e-2b7d-4c55-9a51-0d6f1e2b2980"


class _Query:
    def __init__(self, rows: List[Any]):
        self._rows = rows

    def filter(self, *args: Any, **kwargs: Any) -> "_Query":
        return self

    def first(self) -> Any:
        return self._rows[0] if self._rows else None


class Session:
    """The one row the tool reads: the calling agent."""

    def query(self, model: Any) -> _Query:
        return _Query([AGENT] if model is Agent else [])


class RecordedService:
    """DocumentGenerationService with the render recorded: what ``generate`` was asked for."""

    asked: List[Dict[str, Any]] = []
    fails_with: Exception | None = None

    def __init__(self, db: Any, workspace_id: Any):
        self.workspace_id = workspace_id

    async def generate(self, **kwargs: Any) -> Any:
        RecordedService.asked.append(kwargs)
        if RecordedService.fails_with is not None:
            raise RecordedService.fails_with
        return SimpleNamespace(
            filename=FILENAME, format="pdf", download_url=f"/api/documents/generated/{FILENAME}",
            size=8671, content="# Price list", template_id=None, template_name=None, s3_key="k",
        )

    def register_as_deliverable(self, result: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"success": True, "deliverable_id": DELIVERABLE_ID}

    def share_link(self, result: Any) -> str:
        return "http://localhost:9000/automatos-ai/k?X-Amz-Signature=" + "a" * 64


def recorded(monkeypatch: Any, fails_with: Exception | None = None) -> type:
    """Swap the document service for :class:`RecordedService`; the knowledge ingest does nothing."""
    RecordedService.asked = []
    RecordedService.fails_with = fails_with

    async def ingest(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(generation_service, "DocumentGenerationService", RecordedService)
    monkeypatch.setattr(knowledge_flywheel, "ingest_agent_output", ingest)
    return RecordedService


def call_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """generate_document through AgentPlatformTools.execute_tool, the entry every lane uses."""
    from modules.agents.services.agent_platform_tools import AgentPlatformTools

    with patch("modules.agents.services.agent_platform_tools.RAGService"), patch(
        "modules.agents.services.agent_platform_tools.CodeGraphService"
    ):
        tools = AgentPlatformTools(db_session=Session())
    return asyncio.run(tools.execute_tool("generate_document", parameters, agent_id=AGENT.id))
