"""F344 (night 10b): a Branded Letter an agent generates is signed with the owner's name.

Night 10b: the owner generated the Branded Letter from the Template Studio and it was
signed "Gerard". An agent (a session's tools, an API agent) or Auto generating from the
same template got "template variables did not resolve: user.name": no person makes an
agent's call, the tool route passed no user, and the finalisation gate blocked the
letter. The five Branded starters that sign with {{user.name}} were unusable by any
agent. On the local edition, {{user.email}} printed "local@automatos.local".

These run the tool route for real (generate_document's handler, the generation service,
the variable resolver, WeasyPrint) over a session that holds two workspaces, and read
the letter's text back out of the PDF.
"""
from __future__ import annotations

import asyncio
import base64
import io
import uuid
from types import SimpleNamespace as NS
from typing import Any, Dict, List

import pytest
from sqlalchemy.sql.elements import BindParameter, False_, True_

from config import config
from core.models import Agent
from core.models.business_profiles import BusinessProfile
from core.models.core import User
from core.models.workspaces import Workspace
from core.workspaces.models import WorkspaceMember
from modules.documents.presets import LETTER

WS_OURS = uuid.UUID("3f4e5d6c-7b8a-4934-a1b2-c3d4e5f60344")
WS_THEIRS = uuid.UUID("9a8b7c6d-5e4f-4321-b0a1-f2e3d4c50344")
OUR_AGENT, THEIR_AGENT = 3441, 3442
OWNER = NS(id=11, name="Gerard Kavanagh", email="gerard@harbourline.ie", username="gerard")
THEIR_OWNER = NS(id=22, name="Someone Else", email="someone@elsewhere.example", username="someone")
LOCAL_PLACEHOLDER = "local@automatos.local"
COMPANY_EMAIL = "hello@harbourline.ie"


def _logo_data_uri() -> str:
    """A real 8x8 PNG as a data: URI. The Branded Letter carries a brand_logo block, and
    a kit with no logo blocks it at finalisation (brand.logo_url), so the kit needs one."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "#1E3A5F").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


KIT = {"name": "Harbourline Coffee Roasters", "logo_url": _logo_data_uri(),
       "company": {"name": "Harbourline Coffee Roasters", "email": COMPANY_EMAIL}}
LETTER_DATA = dict(LETTER["sample_data"]["data"])
CALL = {"title": "Spring campaign letter", "format": "pdf", "template_name": "Branded Letter", "data": LETTER_DATA}


def _value(element: Any) -> Any:
    if isinstance(element, BindParameter):
        return element.value
    if isinstance(element, (True_, False_)):
        return isinstance(element, True_)
    raise AssertionError(f"the fake session cannot read {element!r}")


class _Query:
    """``query(Model).filter(Model.col == value, ...).order_by(...).first()`` over in-memory rows."""

    def __init__(self, rows: List[Any]):
        self.rows = rows

    def filter(self, *criteria: Any) -> "_Query":
        return _Query([r for r in self.rows
                       if all(getattr(r, c.left.key, None) == _value(c.right) for c in criteria)])

    def order_by(self, *_args: Any) -> "_Query":
        return self

    def first(self) -> Any:
        return self.rows[0] if self.rows else None


class _Session:
    """Two workspaces' rows; every read goes through the same filters the platform writes."""

    def __init__(self, tables: Dict[Any, List[Any]]):
        self.tables = tables

    def query(self, model: Any, *_rest: Any) -> _Query:
        return _Query(list(self.tables.get(model, [])))

    def get(self, model: Any, key: Any) -> Any:
        return next((r for r in self.tables.get(model, []) if r.id == key), None)


def _session(our_owner_id: Any = OWNER.id, members: tuple = (), users: tuple = (OWNER, THEIR_OWNER),
             kit: Dict[str, Any] = KIT) -> _Session:
    return _Session({
        Agent: [NS(id=OUR_AGENT, workspace_id=WS_OURS, name="Ops Manager"),
                NS(id=THEIR_AGENT, workspace_id=WS_THEIRS, name="Their Ops Manager")],
        Workspace: [NS(id=WS_OURS, owner_id=our_owner_id, settings={"brand_kit": kit}),
                    NS(id=WS_THEIRS, owner_id=THEIR_OWNER.id, settings={"brand_kit": KIT})],
        User: list(users),
        WorkspaceMember: list(members),
        BusinessProfile: [],
    })


class _Templates:
    """The workspace's Branded Letter starter, as DocumentTemplateService finds it by name."""

    def __init__(self, db: Any):
        self.letter = NS(id=uuid.uuid4(), name=LETTER["name"], format="pdf", blocks=LETTER["blocks"],
                         template_content=None, template_file_path=None, data_schema=None)

    def get_template_by_name(self, workspace_id: Any, name: str) -> Any:
        return self.letter if name == LETTER["name"] else None

    def get_template(self, template_id: Any, workspace_id: Any) -> Any:
        return None


@pytest.fixture
def generate(monkeypatch, tmp_path):
    """generate_document on the tool route, for real down to the PDF: (answer, letter text, registrations)."""
    import modules.documents.generation_service as gs
    from modules.tools.execution import generate_document_tool as gdt
    from services.brand_rules import forget_cached_kits

    registered: List[Dict[str, Any]] = []

    def register(self, result, **kwargs):
        registered.append(kwargs)
        return {"success": True, "deliverable_id": "f344-letter"}

    async def no_ingest(*_args, **_kwargs):
        return None

    monkeypatch.setattr(gs, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(gs, "is_storage_configured", lambda: False)
    monkeypatch.setattr(gs, "DocumentTemplateService", _Templates)
    monkeypatch.setattr(gs.DocumentGenerationService, "register_as_deliverable", register)
    monkeypatch.setattr(gdt, "_ingest", no_ingest)
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")

    def go(db: _Session, agent_id: int = OUR_AGENT, call: Dict[str, Any] = CALL):
        forget_cached_kits()
        answer = asyncio.run(gdt.run_generate_document(db, dict(call), agent_id))
        return answer, _letter_text(tmp_path), registered

    yield go
    forget_cached_kits()


def _letter_text(root: Any) -> str:
    import pdfplumber

    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        return ""
    with pdfplumber.open(str(pdfs[-1])) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return " ".join(text.split())


def test_an_agents_branded_letter_is_signed_with_the_owners_name_and_delivered(generate):
    answer, text, registered = generate(_session())

    assert answer["success"] is True, answer
    assert answer["results"][0]["deliverable_id"] == "f344-letter"
    assert registered and registered[0]["agent_id"] == OUR_AGENT
    assert "Kind regards, Gerard Kavanagh" in text, text
    assert OWNER.email in text


def test_a_workspace_owned_through_its_owner_member_signs_with_that_owner(generate):
    member = NS(workspace_id=WS_OURS, user_id=OWNER.id, role="owner", is_active=True)
    answer, text, _ = generate(_session(our_owner_id=None, members=(member,)))

    assert answer["success"] is True, answer
    assert "Gerard Kavanagh" in text, text


def test_another_workspaces_owner_never_signs_our_letter(generate):
    # Our workspace has no owner on file; their workspace's owner, and their owner
    # membership, are in the same session. A tool argument naming a user is no owner either.
    theirs = NS(workspace_id=WS_THEIRS, user_id=THEIR_OWNER.id, role="owner", is_active=True)
    call = {**CALL, "user_id": THEIR_OWNER.id, "user": {"name": THEIR_OWNER.name}}
    answer, text, registered = generate(_session(our_owner_id=None, members=(theirs,)), call=call)

    assert answer["success"] is False
    assert "user.name" in answer["error"]
    assert THEIR_OWNER.name not in text and not registered


def test_their_agent_signs_with_their_owner_not_ours(generate):
    answer, text, _ = generate(_session(), agent_id=THEIR_AGENT)

    assert answer["success"] is True, answer
    assert THEIR_OWNER.name in text and OWNER.name not in text, text


def test_on_the_local_edition_the_operator_signs_and_the_placeholder_email_is_never_printed(generate, monkeypatch):
    operator = NS(id=1, name="Gerard", email=LOCAL_PLACEHOLDER, username="local")
    monkeypatch.setattr(config, "LOCAL_OPERATOR_EMAIL", LOCAL_PLACEHOLDER)
    db = _session(our_owner_id=None, users=(operator,))
    monkeypatch.setattr(config, "AUTH_EDITION", "local")

    answer, text, _ = generate(db)

    assert answer["success"] is True, answer
    assert f"Kind regards, Gerard {COMPANY_EMAIL}" in text, text  # the signature's email line
    assert LOCAL_PLACEHOLDER not in text


def test_with_no_company_email_the_placeholder_is_left_out_not_printed(generate, monkeypatch):
    operator = NS(id=1, name="Gerard", email=LOCAL_PLACEHOLDER, username="local")
    monkeypatch.setattr(config, "LOCAL_OPERATOR_EMAIL", LOCAL_PLACEHOLDER)
    kit = {"name": "Harbourline Coffee Roasters", "logo_url": KIT["logo_url"]}
    db = _session(our_owner_id=None, users=(operator,), kit=kit)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")

    answer, text, _ = generate(db)

    assert answer["success"] is True, answer  # the Letter's email chip falls back to nothing
    assert "Gerard" in text and LOCAL_PLACEHOLDER not in text and "@" not in text, text


def test_the_owners_own_studio_render_never_prints_the_placeholder_email_either(monkeypatch):
    from modules.documents.variables.resolver import VariableResolver

    operator = NS(id=1, name="Gerard", email=LOCAL_PLACEHOLDER, username="local")
    db = _session(our_owner_id=None, users=(operator,), kit={"name": "Harbourline Coffee Roasters"})

    resolved = VariableResolver(db).resolve(WS_OURS, operator.id, ["user.name", "user.email"])

    assert resolved.values == {"user.name": "Gerard"}
    assert resolved.unresolved == ["user.email"]


def test_a_real_address_is_kept_and_only_an_undeliverable_one_gives_way():
    from modules.documents.variables.document_user import deliverable_email

    assert deliverable_email("gerard@harbourline.ie", KIT) == "gerard@harbourline.ie"
    assert deliverable_email(LOCAL_PLACEHOLDER, KIT) == COMPANY_EMAIL
    assert deliverable_email("Someone@Box.LOCAL", {}) == ""
    assert deliverable_email(None, {"company": None}) == ""
