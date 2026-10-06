"""PRD-255 US-013: agents create and edit document templates through the studio's own checks.

``create_template`` / ``update_template`` (``platform_create_template`` /
``platform_update_template``) save a block template the way ``POST`` / ``PUT
/api/documents/templates`` do: one shared validator (``template_validation``),
field-level errors handed back, document formats only (Decision Q3), a starter
never changed (copy it first), the maker tagged ``made-by:<agent>`` from the
server-minted caller, the caller's workspace only, and no delete tool.

Boundaries faked: the template store (``DocumentTemplateService``, which still runs
the real ``checked_blocks``) and the session. The block validator is the real one.
"""
from __future__ import annotations

import asyncio
import copy
import os
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from modules.documents import template_service  # noqa: E402
from modules.documents.blocks import validate_blocks  # noqa: E402
from modules.documents.presets import INVOICE, LETTER  # noqa: E402
from modules.documents.template_summary import STARTER_CREATOR, summarize_template  # noqa: E402
from modules.tools.discovery import handlers_template_writes as tw  # noqa: E402

WS = UUID("7a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d")
OTHER_WS = UUID("8b2c3d4e-5f6a-4b7c-9d8e-0f1a2b3c4d5e")
STARTER_ID = UUID("1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d")
OWN_ID = UUID("2b3c4d5e-6f7a-4b8c-9d0e-1f2a3b4c5d6e")
THEIRS_ID = UUID("3c4d5e6f-7a8b-4c9d-8e1f-2a3b4c5d6e7f")
SOCIAL_ID = UUID("4d5e6f7a-8b9c-4d0e-9f2a-3b4c5d6e7f8a")
AGENT_ID = 268
SOCIAL_BLOCKS = {"html": "<div>{{headline}}</div>", "css": "", "variables_schema": {"headline": {"type": "text"}},
                 "sizes": ["1080x1350"]}
# A heading level the schema refuses: the studio answers "blocks.0.level".
BAD_BLOCKS = {"version": 1, "blocks": [{"type": "heading", "id": "h1", "level": 9, "content": []}]}

real_checked_blocks = template_service.checked_blocks


def _row(template_id: UUID, workspace_id: UUID, name: str, fmt: str = "pdf", blocks: Any = None,
         created_by: Optional[str] = None, tags: Optional[List[str]] = None) -> NS:
    return NS(id=template_id, workspace_id=workspace_id, name=name, format=fmt, description=None,
              blocks=copy.deepcopy(LETTER["blocks"]) if blocks is None else blocks,
              sample_data=copy.deepcopy(LETTER["sample_data"]), data_schema={}, category="letter",
              tags=list(tags or []), created_by=created_by, version=1, is_active=True,
              template_content=None, template_file_path=None, thumbnail_url=None,
              created_at=None, updated_at=None)


class _Store:
    """DocumentTemplateService, faked over rows; workspace-scoped like the real one, and its
    save runs the real ``checked_blocks`` (the social contract, the known formats)."""

    rows: List[NS] = []
    saves = 0
    # The unique name per workspace and version counts removed templates too, which
    # get_template_by_name (active only) does not see: the commit refuses it.
    taken = False

    def __init__(self, db: Any) -> None:
        self.db = db

    def get_template(self, template_id: UUID, workspace_id: UUID) -> Optional[NS]:
        return next((t for t in self.rows if t.id == template_id and t.workspace_id == workspace_id), None)

    def get_template_by_name(self, workspace_id: UUID, name: str) -> Optional[NS]:
        return next((t for t in self.rows if t.workspace_id == workspace_id and t.name == name), None)

    def list_templates(self, workspace_id: UUID, format: Any = None, category: Any = None) -> List[NS]:
        return [t for t in self.rows if t.workspace_id == workspace_id]

    def create_template(self, workspace_id: UUID, name: str, format: str, description: Optional[str] = None,
                        template_content: Optional[str] = None, template_file_path: Optional[str] = None,
                        data_schema: Optional[dict] = None, sample_data: Optional[dict] = None,
                        category: str = "general", tags: Optional[list] = None, created_by: Optional[str] = None,
                        blocks: Optional[dict] = None) -> NS:
        # The real signature: an unknown keyword from the handler fails here as it would there.
        if self.taken:
            raise IntegrityError("INSERT", {}, Exception("uq_template_workspace_name_version"))
        row = _row(uuid4(), workspace_id, name, fmt=format, blocks=real_checked_blocks(format, blocks))
        fields = {"description": description, "sample_data": sample_data, "category": category,
                  "tags": list(tags or []), "created_by": created_by}
        for key, value in fields.items():
            setattr(row, key, value)
        type(self).rows = [*self.rows, row]
        type(self).saves += 1
        return row

    def update_template(self, template_id: UUID, workspace_id: UUID, **updates: Any) -> Optional[NS]:
        row = self.get_template(template_id, workspace_id)
        if row is None:
            return None
        if self.taken:
            raise IntegrityError("UPDATE", {}, Exception("uq_template_workspace_name_version"))
        if "blocks" in updates:
            updates = {**updates, "blocks": real_checked_blocks(row.format, updates["blocks"])}
        for key, value in updates.items():
            setattr(row, key, value)
        type(self).saves += 1
        return row


class _Query:
    def __init__(self, rows: List[Any]) -> None:
        self.rows = rows

    def filter(self, *conditions: Any) -> "_Query":
        return self

    def first(self) -> Any:
        return self.rows[0] if self.rows else None


class _Db:
    """The session: the calling agent's row in the caller's workspace, and the rollbacks."""

    def __init__(self, agent: Any = None) -> None:
        self.agent = agent
        self.rollbacks = 0

    def query(self, model: Any) -> _Query:
        return _Query([self.agent] if self.agent is not None else [])

    def rollback(self) -> None:
        self.rollbacks += 1


@pytest.fixture
def store(monkeypatch):
    _Store.rows = [
        _row(STARTER_ID, WS, "Branded Letter", created_by=STARTER_CREATOR),
        _row(OWN_ID, WS, "Our Quote", created_by="agent:268", tags=["made-by:brand-designer", "sales"]),
        _row(THEIRS_ID, OTHER_WS, "Their Quote", created_by="user-9"),
        _row(SOCIAL_ID, WS, "Launch card", fmt="social_image", blocks=SOCIAL_BLOCKS),
    ]
    _Store.saves = 0
    _Store.taken = False
    monkeypatch.setattr(template_service, "DocumentTemplateService", _Store)
    return _Store


@pytest.fixture
def db():
    return _Db(agent=NS(id=AGENT_ID, workspace_id=WS, slug="brand-designer", name="Brand Designer"))


def _run(coro: Any) -> Dict[str, Any]:
    return asyncio.run(coro)


def _create(db: Any, **params: Any) -> Dict[str, Any]:
    return _run(tw.create_template(db, WS, {"_agent_id": AGENT_ID, **params}))


def _update(db: Any, **params: Any) -> Dict[str, Any]:
    return _run(tw.update_template(db, WS, {"_agent_id": AGENT_ID, **params}))


def _by_id(template_id: Any) -> NS:
    return next(t for t in _Store.rows if str(t.id) == str(template_id))


# ── create ──────────────────────────────────────────────────────────────────

def test_a_template_is_made_through_the_studios_check_and_tagged_with_its_maker(store, db):
    answer = _create(db, name="Price List", format="docx", category="pricing", blocks=INVOICE["blocks"],
                     sample_data=INVOICE["sample_data"], tags=["sales", "made-by:auto"])

    assert answer["success"] is True, answer
    made = _by_id(answer["template_id"])
    assert made.workspace_id == WS and made.format == "docx" and made.category == "pricing"
    assert made.blocks == validate_blocks(INVOICE["blocks"]).model_dump()     # normalised as the studio saves
    # The maker comes from the server-minted agent, never from a tag the call sends.
    assert made.tags == ["sales", "made-by:brand-designer"]
    assert made.created_by == f"agent:{AGENT_ID}" and made.created_by != STARTER_CREATOR
    assert "render_preview" in answer["note"]


def test_a_made_template_is_listed_in_the_studio_like_any_template(store, db):
    from modules.tools.discovery.template_tools import list_templates_answer

    answer = _create(db, name="Quote", format="pdf", blocks=LETTER["blocks"])

    listed = list_templates_answer(db, WS, {"name": "Quote"})
    assert any(answer["template_id"] in row for row in listed["templates"])
    summary = summarize_template(_by_id(answer["template_id"]))
    assert summary["has_blocks"] is True and summary["is_starter"] is False


def test_the_maker_tag_reads_the_agent_of_this_workspace_only():
    assert tw.maker_tag(_Db(NS(slug="brand-designer", name="Brand Designer")), WS, {"_agent_id": 268}) == \
        "made-by:brand-designer"
    assert tw.maker_tag(_Db(NS(slug=None, name="Brand Designer")), WS, {"_agent_id": "268"}) == \
        "made-by:Brand Designer"
    # No agent row in this workspace (or no server-minted id): no name is borrowed.
    assert tw.maker_tag(_Db(None), WS, {"_agent_id": 268}) == "made-by:agent"
    assert tw.maker_tag(_Db(NS(slug="x", name="x")), WS, {"made_by": "auto"}) == "made-by:agent"


def test_a_starter_is_customised_by_copying_it_and_the_starter_stays(store, db):
    starter_before = copy.deepcopy(vars(_by_id(STARTER_ID)))

    answer = _create(db, copy_of=str(STARTER_ID))

    assert answer["success"] is True, answer
    made = _by_id(answer["template_id"])
    assert made.name == "Branded Letter (copy)" and made.format == "pdf" and made.category == "letter"
    assert made.blocks == validate_blocks(LETTER["blocks"]).model_dump()
    assert made.created_by == f"agent:{AGENT_ID}" and "made-by:brand-designer" in made.tags
    assert vars(_by_id(STARTER_ID)) == starter_before


def test_a_name_already_in_use_is_refused(store, db):
    answer = _create(db, name="Branded Letter", format="pdf", blocks=LETTER["blocks"])

    assert answer["success"] is False and "already exists" in answer["error"]
    assert store.saves == 0


def test_a_name_a_removed_template_still_holds_is_refused_and_the_session_rolled_back(store, db):
    store.taken = True

    created = _create(db, name="Old Quote", format="pdf", blocks=LETTER["blocks"])
    edited = _update(db, template_id=str(OWN_ID), name="Old Quote")

    for answer in (created, edited):
        assert answer["success"] is False and "already exists" in answer["error"]
    assert db.rollbacks == 2 and _by_id(OWN_ID).name == "Our Quote"


def test_a_call_without_a_name_blocks_or_document_format_is_refused(store, db):
    assert "needs a name" in _create(db, format="pdf", blocks=LETTER["blocks"])["error"]
    assert "needs blocks" in _create(db, name="X", format="pdf")["error"]
    assert "document templates only" in _create(db, name="X", blocks=LETTER["blocks"])["error"]
    assert store.saves == 0


# ── edit ────────────────────────────────────────────────────────────────────

def test_an_edit_saves_through_the_studios_check_and_keeps_the_maker(store, db):
    answer = _update(db, template_id=str(OWN_ID), blocks=INVOICE["blocks"], name="Our Quote v2", tags=["quotes"])

    assert answer["success"] is True, answer
    edited = _by_id(OWN_ID)
    assert edited.name == "Our Quote v2"
    assert edited.blocks == validate_blocks(INVOICE["blocks"]).model_dump()
    assert edited.tags == ["quotes", "made-by:brand-designer"]
    assert store.saves == 1


def test_an_edit_with_nothing_to_change_is_refused(store, db):
    assert "at least one of" in _update(db, template_id=str(OWN_ID))["error"]
    assert "needs the template_id" in _update(db, blocks=LETTER["blocks"])["error"]
    assert store.saves == 0


def test_a_starter_is_never_changed_the_tool_says_copy_it_first(store, db):
    before = copy.deepcopy(vars(_by_id(STARTER_ID)))

    answer = _update(db, template_id=str(STARTER_ID), name="Mine now", blocks=INVOICE["blocks"])

    assert answer["success"] is False
    assert "never changed" in answer["error"] and f"copy_of='{STARTER_ID}'" in answer["error"]
    assert vars(_by_id(STARTER_ID)) == before and store.saves == 0


def test_there_is_no_template_delete_tool():
    from modules.tools.discovery import get_action_registry

    names = [a.name for a in get_action_registry().get_all()]
    assert "platform_create_template" in names and "platform_update_template" in names
    assert not [n for n in names if "template" in n and n.startswith("platform_delete")]


# ── validation errors come back ────────────────────────────────────────────

def test_a_malformed_block_tree_is_refused_with_the_studios_field_level_errors(store, db):
    created = _create(db, name="Broken", format="pdf", blocks=BAD_BLOCKS)
    edited = _update(db, template_id=str(OWN_ID), blocks=BAD_BLOCKS)

    for answer in (created, edited):
        assert answer["success"] is False and "Invalid blocks" in answer["error"]
        assert "blocks.0.level" in [e["loc"] for e in answer["errors"]]
    assert store.saves == 0 and _by_id(OWN_ID).name == "Our Quote"


def test_the_routes_and_the_tools_share_one_validator(store, db):
    from fastapi import HTTPException

    import api.document_generation as routes

    with pytest.raises(HTTPException) as refused:
        routes._validate_blocks_or_422("pdf", BAD_BLOCKS)
    tool = _create(db, name="Broken", format="pdf", blocks=BAD_BLOCKS)
    assert refused.value.status_code == 422
    assert refused.value.detail == {"message": "Invalid blocks", "errors": tool["errors"]}
    # A social composition is the service's to check, on both paths.
    assert routes._validate_blocks_or_422("social_image", SOCIAL_BLOCKS) == SOCIAL_BLOCKS
    assert routes._validate_blocks_or_422("pdf", None) is None


def test_wrongly_typed_fields_are_refused_by_name(store, db):
    assert "sample_data must be" in _create(db, name="X", format="pdf", blocks=LETTER["blocks"], sample_data=[1])["error"]
    assert "tags must be" in _update(db, template_id=str(OWN_ID), tags="sales")["error"]
    assert "name must be" in _update(db, template_id=str(OWN_ID), name="   ")["error"]
    assert store.saves == 0


# ── scope: document formats, this workspace ────────────────────────────────

def test_social_formats_are_refused_by_both_tools(store, db):
    made = _create(db, name="Card", format="social_image", blocks=SOCIAL_BLOCKS)
    edited = _update(db, template_id=str(SOCIAL_ID), name="Card 2")
    copied = _create(db, copy_of=str(SOCIAL_ID))

    assert "document templates only" in made["error"]
    assert "social_image template" in edited["error"] and "social_image template" in copied["error"]
    assert store.saves == 0 and _by_id(SOCIAL_ID).name == "Launch card"


def test_another_workspaces_template_is_not_found(store, db):
    edited = _update(db, template_id=str(THEIRS_ID), name="Taken")
    copied = _create(db, copy_of=str(THEIRS_ID))

    for answer in (edited, copied):
        assert answer["success"] is False and answer["error"] == f"No template {THEIRS_ID} in this workspace."
    assert _by_id(THEIRS_ID).name == "Their Quote" and store.saves == 0


# ── wiring: the 3-file pattern, the gate, the session tools ────────────────

def test_the_two_actions_are_registered_writes_routed_and_on_the_gates_allow_list():
    import importlib.util
    from pathlib import Path

    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    gate_path = Path(__file__).resolve().parents[1] / "scripts" / "check_hierarchy_gate.py"
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    for name, handler in (("platform_create_template", tw.create_template),
                          ("platform_update_template", tw.update_template)):
        action = get_action_registry().get(name)
        assert action is not None and action.permission_level == "write", name
        assert PLATFORM_HANDLERS[name] is handler
        assert name in gate.ALLOW_LIST and gate.collect_registrations()[name].permission_level == "write"


def test_the_session_tools_are_writes_in_the_documents_group_and_forward_no_maker():
    from services import session_tool_groups as groups
    from services import session_tools as st

    ctx = st.SessionContext(task_id=2101, agent_id=AGENT_ID, agent_name="Brand Designer", workspace_id=str(WS))
    documents = next(g for g in groups.SESSION_TOOL_GROUPS if g.id == "documents")
    for name, action in (("create_template", "platform_create_template"),
                         ("update_template", "platform_update_template")):
        tool = st.get_tool(name)
        assert tool.action == action and tool.reads_only is False and name in documents.tools

    forwarded = st.resolve_parameters(st.get_tool("create_template"), {
        "name": "Quote", "format": "pdf", "blocks": LETTER["blocks"], "_agent_id": 1, "made_by": "auto",
    }, ctx)
    assert forwarded == {"name": "Quote", "format": "pdf", "blocks": LETTER["blocks"]}
    with pytest.raises(st.SessionToolRefused):
        st.resolve_parameters(st.get_tool("update_template"), {"name": "x"}, ctx)
