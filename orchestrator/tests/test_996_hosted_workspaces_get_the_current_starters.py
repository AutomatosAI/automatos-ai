"""#996: a hosted workspace made before a starter changed gets the current starter at boot.

``seed_starter_templates`` ran at provisioning on hosted (and at every local boot), so a
hosted workspace kept the starter rows it was provisioned with: F356's letterhead and
layouts never reached it. ``seed_starters_everywhere`` re-runs the seeder for every
workspace from the boot leader. Its rules hold: a platform-owned row that drifted is
refreshed, a missing starter is created, a person's own row and a starter they deleted
are left as they are, and one workspace that fails does not stop the others.
"""
from __future__ import annotations

import inspect
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List

from core.models.core import DocumentTemplate
from modules.documents import seed_templates
from modules.documents.presets import PRESETS
from modules.documents.template_summary import STARTER_CREATOR

LETTER = next(preset for preset in PRESETS if preset["category"] == "letter")
INVOICE = next(preset for preset in PRESETS if preset["category"] == "invoice")
OLD_BLOCKS = {"version": 1, "blocks": [{"type": "image", "id": "logo", "source": "brand_logo", "width_mm": 50}]}


def _filters(criteria) -> Dict[str, Any]:
    return {criterion.left.key: criterion.right.value for criterion in criteria}


class _Session:
    """Workspaces and their template rows, as the seeder reads and writes them."""

    def __init__(self, workspaces: List[uuid.UUID], rows: List[Any], broken: uuid.UUID = None):
        self.workspaces, self.rows, self.broken = workspaces, rows, broken
        self.added: List[Any] = []
        self.commits = self.rollbacks = 0
        self._criteria: Dict[str, Any] = {}

    def query(self, *columns: Any) -> "_Session":
        self._criteria = {}
        return self

    def all(self) -> List[SimpleNamespace]:
        return [SimpleNamespace(id=workspace_id) for workspace_id in self.workspaces]

    def filter(self, *criteria: Any) -> "_Session":
        self._criteria = {**self._criteria, **_filters(criteria)}
        return self

    def first(self) -> Any:
        if self._criteria.get("workspace_id") == self.broken:
            raise RuntimeError("the row lock timed out")
        return next((row for row in self.rows + self.added
                     if all(getattr(row, key, None) == value for key, value in self._criteria.items())), None)

    def add(self, row: Any) -> None:
        self.added.append(row)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _row(workspace_id: uuid.UUID, preset: Dict[str, Any], **over: Any) -> SimpleNamespace:
    columns = seed_templates.starter_columns(preset)
    base = {**columns, "workspace_id": workspace_id, "created_by": STARTER_CREATOR, "is_active": True,
            "blocks": OLD_BLOCKS, "updated_at": None, "template_content": None, "template_file_path": None}
    return SimpleNamespace(**{**base, **over})


def test_an_old_platform_starter_in_an_existing_workspace_takes_the_current_preset():
    old = uuid.uuid4()
    stale = _row(old, LETTER)
    db = _Session([old], [stale])

    totals = seed_templates.seed_starters_everywhere(db)

    assert stale.blocks == LETTER["blocks"] and stale.updated_at is not None
    assert totals["workspaces"] == 1 and totals["failed"] == 0
    assert db.commits == 1


def test_a_persons_own_row_and_a_deleted_starter_are_left_as_they_are():
    ws = uuid.uuid4()
    theirs = _row(ws, LETTER, created_by="user_7")
    deleted = _row(ws, INVOICE, is_active=False)
    db = _Session([ws], [theirs, deleted])

    seed_templates.seed_starters_everywhere(db)

    assert theirs.blocks == OLD_BLOCKS and theirs.updated_at is None
    assert deleted.blocks == OLD_BLOCKS and deleted.is_active is False
    assert not [row for row in db.added if row.name in (LETTER["name"], INVOICE["name"])]


def test_a_workspace_with_no_starters_gets_them():
    ws = uuid.uuid4()
    db = _Session([ws], [])

    totals = seed_templates.seed_starters_everywhere(db)

    names = {row.name for row in db.added}
    assert {preset["name"] for preset in PRESETS} <= names
    assert all(isinstance(row, DocumentTemplate) and row.workspace_id == ws for row in db.added)
    assert totals["created"] == len(db.added)


def test_one_workspace_that_fails_does_not_stop_the_others():
    broken, fine = uuid.uuid4(), uuid.uuid4()
    stale = _row(fine, LETTER)
    db = _Session([broken, fine], [stale], broken=broken)

    totals = seed_templates.seed_starters_everywhere(db)

    assert stale.blocks == LETTER["blocks"]
    assert totals["workspaces"] == 1 and totals["failed"] == 1
    assert db.rollbacks == 1


def test_the_boot_runs_it_after_the_core_phase_from_the_boot_leader():
    import main

    assert "_seed_document_starters_everywhere" in inspect.getsource(main._then_starters)
    assert "boot_leader_lock" in inspect.getsource(main._seed_document_starters_everywhere)
    assert main._boot_phase_1_core.__wrapped__  # the core phase is wrapped by _then_starters
