"""Brand kit at generation (B, night 9b → night 10): "[Your name]" becomes the kit's sign-off.

Night 9b: the Tidewater reorder draft (#1971) took three rounds to lose "[Your name]";
#0095 and Auto's reply 1b39361c ended "[Your name]" and "Best, [Your Name]". The
platform now fills a placeholder signature with the brand kit's sign-off: on a card's
answer before the completion writer's notes read it (so the leftover-placeholder note
never fires on a filled one), on a mission step's answer, and in a document's data
before it renders. A kit with no one to sign leaves the placeholder, for that note to
warn; a workspace without a kit is left as it was. "Dear [Name]," is the reader's.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

import api.board_tasks as bt
from core.models.workspaces import Workspace

WS = "6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e5f"
SIGN_OFF = "Gerard, Harbourline Coffee Roasters"
KIT = {"name": "Harbourline Coffee Roasters", "voice": {"tone": ["warm", "plain", "local"], "sign_off": SIGN_OFF}}
DRAFT_1971 = ("Hi Maya,\n\nPlease send two 60 kg sacks of the Kirinyaga at £9.75/kg, on our usual 60 days.\n\n"
              "Best,\n[Your name]")


@pytest.fixture(autouse=True)
def fresh_kits():
    from services.brand_rules import forget_cached_kits

    forget_cached_kits()
    yield
    forget_cached_kits()


class _Task:
    def __init__(self):
        self.id, self.status, self.result, self.error_message = 1971, "in_progress", None, None
        self.completed_at = self.lease_until = None
        self.runtime_ref = None
        self.title, self.description = "Tidewater reorder draft", "Draft the reorder email to Maya."


class _Session:
    """The workspace (with ``settings``) and the ticket, as the completion writer reads them."""

    def __init__(self, task, settings):
        self.task, self.workspace = task, NS(settings=settings)

    def query(self, *_a, **_k):
        return self

    def get(self, model, *_a, **_k):
        return self.workspace if model is Workspace else self.task

    def commit(self):
        pass


@pytest.fixture
def finalize(monkeypatch):
    async def _noop(*_a, **_k):
        return None

    for name in ("_dispatch_task_complete", "_dispatch_task_failed", "_auto_create_task_report"):
        monkeypatch.setattr(bt, name, _noop)
    monkeypatch.setattr("services.result_files.check_named_files", _noop)

    def run(text, settings):
        task = _Task()
        asyncio.run(bt.finalize_board_task_run(_Session(task, settings), task_id=task.id, workspace_id=WS,
                                               agent_id=7, exec_result={"status": "success", "result": text}))
        return task.result
    return run


def test_the_cards_draft_is_signed_with_the_kits_sign_off(finalize):
    result = finalize(DRAFT_1971, {"brand_kit": KIT})
    assert "[Your name]" not in result
    assert f"Best,\n{SIGN_OFF}" in result and result.startswith("Hi Maya,")


def test_with_no_one_to_sign_the_placeholder_stays_for_the_note_to_warn(finalize):
    unsigned = {"brand_kit": {"voice": {"tone": ["warm", "plain", "local"]}}}
    assert "Best,\n[Your name]" in finalize(DRAFT_1971, unsigned)


def test_a_workspace_without_a_kit_is_left_as_it_was(finalize):
    assert finalize(DRAFT_1971, {}) == DRAFT_1971


def test_the_company_name_signs_when_the_voice_names_no_one(finalize):
    kit = {"company": {"name": "Harbourline Coffee Roasters"}}
    assert finalize(DRAFT_1971, {"brand_kit": kit}).endswith("Best,\nHarbourline Coffee Roasters")


def test_every_placeholder_signature_is_filled_and_the_readers_name_is_not():
    from services.brand_rules import fill_sign_off

    assert fill_sign_off("Best, [Your Name]", SIGN_OFF) == f"Best, {SIGN_OFF}"
    assert fill_sign_off("Thanks,\n[Your Name/Company]", SIGN_OFF) == f"Thanks,\n{SIGN_OFF}"
    assert fill_sign_off("Cheers,\n[Name]\n", SIGN_OFF) == f"Cheers,\n{SIGN_OFF}\n"
    assert fill_sign_off("Kind regards, [Name]", SIGN_OFF) == f"Kind regards, {SIGN_OFF}"
    assert fill_sign_off("Dear [Name],\n\nThanks.", SIGN_OFF) == "Dear [Name],\n\nThanks."
    assert fill_sign_off("Best,\n[Your name]", None) == "Best,\n[Your name]"


def test_a_mission_steps_answer_is_signed_before_it_is_stored():
    from modules.coordination.dispatcher import MissionDispatcher
    from services.brand_hooks import a_mission_steps_answer_is_on_brand

    stored = {}

    def record(db, task, result):
        stored.update(result)

    run = NS(workspace_id=WS)
    db = NS(get=lambda model, key: NS(settings={"brand_kit": KIT}) if model is Workspace else run)
    a_mission_steps_answer_is_on_brand(record)(db, NS(run_id=88), {"status": "success", "result": DRAFT_1971})
    assert stored["result"].endswith(f"Best,\n{SIGN_OFF}")
    assert hasattr(MissionDispatcher.record_task_completion, "__wrapped__")


def test_a_documents_data_is_signed_before_it_renders(monkeypatch):
    import modules.documents.generation_service as gs
    from modules.documents.models import GeneratedDocument

    rendered = {}

    async def generate_pdf(template, data, workspace_id, title, user_id=None):
        rendered.update(data)
        return GeneratedDocument(path="/tmp/x.pdf", format="pdf", filename="x.pdf", size=1)

    db = NS(get=lambda model, key: NS(settings={"brand_kit": KIT}), query=lambda *a, **k: None)
    service = gs.DocumentGenerationService(db, WS)
    service.template_service = NS(get_template_by_name=lambda ws, name: None)
    monkeypatch.setattr(service, "generate_pdf", generate_pdf)
    note = "Hello club,\n\nYour box ships Monday.\n\nWarmly,\n[Your Name]"
    data = {"sections": [{"title": "October", "content": note}]}
    result = asyncio.run(service.generate(title="Club note", format="pdf", data=data, workspace_id=WS))
    assert rendered["sections"][0]["content"].endswith(f"Warmly,\n{SIGN_OFF}")
    assert SIGN_OFF in result.content and "[Your Name]" not in result.content
    assert data["sections"][0]["content"] == note              # the caller's own data is not changed
