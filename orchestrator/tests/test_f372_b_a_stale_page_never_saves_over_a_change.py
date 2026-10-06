"""F372 (night 10c) — a Brand kit page left open never saves over a change made elsewhere.

The page reads the kit's fields once. With Auto, the designer or the API changing the
kit meanwhile, a Save from that page wrote the old values back. The page now sends the
stamp it loaded (``if_updated_at``) with every save. Pins (``api/document_brand_kit.py``):

* a matching stamp saves;
* a stale one is a 409 with a plain message, and nothing is written;
* no stamp saves as before (the agent tool, the designer's save, API callers);
* ``if_updated_at`` is not a kit field: the agent tool's schema does not take it.
"""
from __future__ import annotations

import json

import api.document_brand_kit as brand_kit_routes
from modules.documents import brand_kit
from tests import test_prd255w1_palette_roles as roles_tests

api = roles_tests.api  # the documents router over one workspace holding a v1 kit
KIT_ROUTE = roles_tests.KIT_ROUTE


def _changed_elsewhere(api):
    """The kit as another route left it: saved, so stamped."""
    saved = api.client.put(KIT_ROUTE, json={"name": "Changed by Auto"})
    assert saved.status_code == 200, saved.text
    return saved.json()["updated_at"]


def test_a_save_with_the_stamp_the_page_loaded_saves(api):
    loaded = _changed_elsewhere(api)
    saved = api.client.put(KIT_ROUTE, json={"name": "Harbourline", "if_updated_at": loaded})
    assert saved.status_code == 200, saved.text
    assert saved.json()["name"] == "Harbourline" and saved.json()["updated_at"] != loaded


def test_a_save_with_a_stale_stamp_is_a_409_and_writes_nothing(api):
    loaded = api.client.get(KIT_ROUTE).json()["updated_at"]  # the page loaded the kit
    _changed_elsewhere(api)  # then Auto changed it
    before = json.loads(json.dumps(api.workspace.settings))
    refused = api.client.put(KIT_ROUTE, json={"name": "Old page", "if_updated_at": loaded})
    assert refused.status_code == 409
    assert refused.json()["detail"] == "The brand kit changed since this page loaded; reload to see it"
    assert api.workspace.settings == before


def test_a_save_without_a_stamp_saves_as_before(api):
    _changed_elsewhere(api)
    saved = api.client.put(KIT_ROUTE, json={"name": "From the API"})
    assert saved.status_code == 200, saved.text
    assert api.workspace.settings["brand_kit"]["name"] == "From the API"


def test_the_stamp_to_match_is_no_kit_field():
    assert "if_updated_at" not in brand_kit.PATCH_FIELDS
    assert "if_updated_at" in brand_kit_routes.BrandKitPut.model_fields
