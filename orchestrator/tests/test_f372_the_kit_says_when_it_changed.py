"""F372 (night 10c) — the kit carries when it last changed, so the page's brand board never goes stale.

Night 10c restored the kit through the API (accent orange, every role derived) and the
Brand kit page still showed the navy board: the page drew the board again only after a
save made on the page. The board route was never the cause: it answers ``no-store``.
Now the kit's one writer (``save_brand_kit``) stamps ``updated_at`` on every save,
by any route, and the page keys its board on it (``frontend/hooks/use-brand-kit-stamp.ts``).
Pins:

* a change by the PUT stamps ``updated_at``, and GET answers it;
* GET's answer PUT back unchanged saves nothing and keeps the stamp (F366);
* every save through the writer stamps (the uploads use it too); a client cannot set the stamp;
* the board route still answers ``Cache-Control: private, no-store``.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import api.document_brand_kit as brand_kit_routes
from modules.documents import brand_kit
from tests import test_prd255w1_palette_roles as roles_tests

api = roles_tests.api  # the documents router over one workspace holding a v1 kit never saved since
KIT_ROUTE = roles_tests.KIT_ROUTE
TEAL = "#0f5c5c"


def _stamp(answer):
    stamp = answer["updated_at"]
    datetime.fromisoformat(stamp)  # ISO 8601
    return stamp


def test_a_kit_never_saved_since_has_no_stamp(api):
    assert api.client.get(KIT_ROUTE).json()["updated_at"] == ""


def test_a_change_by_the_put_stamps_the_kit_and_get_answers_it(api):
    saved = api.client.put(KIT_ROUTE, json={"palette": {"accent": TEAL}})
    assert saved.status_code == 200, saved.text
    stamp = _stamp(saved.json())
    assert api.client.get(KIT_ROUTE).json()["updated_at"] == stamp
    assert api.workspace.settings["brand_kit"]["updated_at"] == stamp


def test_the_get_body_put_back_unchanged_keeps_the_stamp_and_saves_nothing(api):
    stamp = _stamp(api.client.put(KIT_ROUTE, json={"name": "Harbourline"}).json())
    stored = api.workspace.settings
    body = api.client.get(KIT_ROUTE).json()
    assert api.client.put(KIT_ROUTE, json=body).json()["updated_at"] == stamp
    assert api.workspace.settings is stored  # not written again


def test_a_client_cannot_set_the_stamp(api):
    api.client.put(KIT_ROUTE, json={"updated_at": "2020-01-01T00:00:00+00:00"})
    assert api.client.get(KIT_ROUTE).json()["updated_at"] == ""
    assert brand_kit.UPDATED_AT_FIELD in brand_kit.SERVER_MANAGED_FIELDS


def test_every_save_through_the_writer_stamps_the_kit():
    workspace = SimpleNamespace(settings={"brand_kit": {"name": "Acme"}})
    kit = brand_kit.get_brand_kit(workspace.settings)
    saved = brand_kit.save_brand_kit(MagicMock(), workspace, kit)  # as a logo upload saves it
    assert _stamp(saved) and workspace.settings["brand_kit"]["updated_at"] == saved["updated_at"]
    assert kit["updated_at"] == ""  # the kit passed in is not changed


def test_the_board_route_still_answers_no_store():
    assert brand_kit_routes.BOARD_CACHE_CONTROL == "private, no-store"
