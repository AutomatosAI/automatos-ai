"""A brand kit save locks the workspace row from its stamp check to its write (Gerard, 7 Oct).

F372 made the Brand kit page send the stamp it loaded (``if_updated_at``) and the PUT
refuse a stale one, but the check read the row unlocked: two saves loaded at the same
stamp could both pass it, and the second wrote over the first. Now the check, the merge
and the write run on the row taken ``FOR UPDATE`` (``brand_kit.lock_brand_kit``:
``Session.refresh(workspace, with_for_update=True)``), held to the save's commit, and
taken after the caller's last await. Pins (a recording session stands in for Postgres):

* a save takes the lock before it writes, and commits after;
* the stamp is checked on the row as the lock reads it: a save that committed while this
  one waited makes this one a ``BrandKitChanged`` (a 409 on the PUT), and nothing is written;
* the write keeps what the locked read found in the other settings;
* a refusal, or a patch that changes nothing, takes no lock;
* the PUT and the uploads read the kit under the lock.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

from modules.documents import brand_fonts, brand_kit
from tests import test_prd251w1_brand_kit_tools as t251

api = t251.api  # the documents router over one workspace, its session a MagicMock
KIT_ROUTE = t251.KIT_ROUTE
LOADED, NEWER = "2026-10-07T09:00:00+00:00", "2026-10-07T09:00:05+00:00"
LOCK = ("lock",)
COMMIT = ("commit",)


class _Session:
    """Records each lock and commit; ``meanwhile`` is the settings another save committed while this one waited."""

    def __init__(self, meanwhile: Optional[Dict[str, Any]] = None) -> None:
        self.calls: List[tuple] = []
        self.meanwhile = meanwhile

    def refresh(self, obj: Any, with_for_update: Any = None) -> None:
        assert with_for_update is True
        self.calls.append(LOCK)
        if self.meanwhile is not None:
            obj.settings, self.meanwhile = self.meanwhile, None

    def commit(self) -> None:
        self.calls.append(COMMIT)


def _workspace(stamp: str = LOADED, **settings: Any) -> SimpleNamespace:
    return SimpleNamespace(settings={"brand_kit": {"name": "Acme", "updated_at": stamp}, **settings})


def test_a_save_takes_the_lock_before_it_writes():
    db, workspace = _Session(), _workspace()
    saved = brand_kit.update_brand_kit(db, workspace, {"name": "Harbourline"}, if_updated_at=LOADED)
    assert db.calls[0] == LOCK and db.calls[-1] == COMMIT and COMMIT not in db.calls[:-1]
    assert workspace.settings["brand_kit"]["name"] == "Harbourline" == saved["name"]


def test_a_save_that_committed_while_this_one_waited_makes_this_one_stale():
    meanwhile = _workspace(NEWER).settings
    meanwhile["brand_kit"]["name"] = "Changed by Auto"
    db, workspace = _Session(meanwhile), _workspace()
    with pytest.raises(brand_kit.BrandKitChanged):
        brand_kit.update_brand_kit(db, workspace, {"name": "Old page"}, if_updated_at=LOADED)
    assert db.calls == [LOCK]  # nothing written
    assert workspace.settings["brand_kit"]["name"] == "Changed by Auto"


def test_the_write_keeps_the_other_settings_as_the_locked_read_found_them():
    meanwhile = {**_workspace().settings, "socials": {"enabled": True}}
    db, workspace = _Session(meanwhile), _workspace(socials={"enabled": False})
    brand_kit.update_brand_kit(db, workspace, {"tagline": "Roasted on the quay"})
    assert workspace.settings["socials"] == {"enabled": True}
    assert workspace.settings["brand_kit"]["tagline"] == "Roasted on the quay"


def test_a_refusal_or_a_change_of_nothing_takes_no_lock():
    db, workspace = _Session(), _workspace()
    with pytest.raises(ValidationError):
        brand_kit.update_brand_kit(db, workspace, {"primary_color": "orange"})
    with pytest.raises(brand_kit.BrandKitChanged):
        brand_kit.update_brand_kit(db, workspace, {"name": "Old page"}, if_updated_at=NEWER)
    assert brand_kit.update_brand_kit(db, workspace, {"name": "Acme"})["updated_at"] == LOADED
    assert db.calls == []


def test_every_write_through_the_writer_is_locked():
    db, workspace = _Session(), _workspace()
    brand_kit.save_brand_kit(db, workspace, brand_kit.get_brand_kit(workspace.settings))
    assert db.calls == [LOCK, COMMIT]


def _lock_then_commit(db) -> bool:
    names = [name for name, _args, _kwargs in db.mock_calls]
    return "refresh" in names and "commit" in names and names.index("refresh") < names.index("commit")


def test_the_put_and_a_font_removal_read_the_kit_under_the_lock(api, monkeypatch):
    monkeypatch.setattr(brand_fonts, "delete_brand_file", lambda path: None)  # no storage here
    saved = api.client.put(KIT_ROUTE, json={"tagline": "Made better"})
    assert saved.status_code == 200, saved.text
    api.db.refresh.assert_any_call(api.workspace, with_for_update=True)
    assert _lock_then_commit(api.db)
    api.db.reset_mock()
    font_id = "0" * 32
    api.workspace.settings["brand_kit"]["font_files"] = [{
        "id": font_id, "family": "Brand Sans", "weight": 400, "style": "normal",
        "path": f"{t251.WS}/brand/fonts/{font_id}.woff2",
    }]
    removed = api.client.delete(f"{KIT_ROUTE}/fonts/{font_id}")
    assert removed.status_code == 200, removed.text
    api.db.refresh.assert_any_call(api.workspace, with_for_update=True)
    assert _lock_then_commit(api.db) and removed.json()["font_files"] == []
