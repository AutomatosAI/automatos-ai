"""PRD-261 M-04 / PRD-266 US-004b — a paired phone on PRD-261's theme routes, end to end.

The phone of ``test_prd266_device_integration.py``, on its fixtures and helpers: a token minted
as pairing mints it, through the REAL ``get_request_context_hybrid`` and ``core/auth/device.py``
against Postgres; its person an ``admin`` member of the bound workspace (not the operator, so no
``super_admin`` bypass). The theme is the hello-clip package with a ``mobile`` block (light
tokens, a sans face, clips), uploaded to a workspace as private code, installed there and opted
in to by the person.

* ``GET /api/themes/mobile/{install_id}`` (``api/themes_mobile.py``; ``services/themes/membership.py``
  binds a device to its row's workspace): the install as data, ``mobile`` the block's tokens, its
  font and its mascot's clip and still, each URL loading through ``GET /api/themes/files/...`` on
  the same bearer; the same person's install in another workspace answers the route's one 404,
  the answer an unknown id gets;
* ``GET /api/themes/active`` (``members:read``): the bound workspace's answer, never the person's
  choice in another workspace; after the phone's ``PUT /api/themes/choice`` of the install, its
  ``source`` is ``choice`` and ``theme.installId`` that install, opted in;
* ``GET /api/themes/looks/{theme_id}`` (``members:read``): the person's looks in the bound
  workspace, never another member's there nor the person's own in another workspace.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from sqlalchemy import text

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd266_device_integration as harness  # noqa: E402
from core.models.themes import ThemeChoice, ThemeLook  # noqa: E402
from services.themes.serve import module_url  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

# The harness's fixtures, bound here so pytest finds them (tests/test_prd251w3_x.py's pattern).
db_session = harness.db_session
local = harness.local
ws = harness.ws
theme_files = harness.theme_files
phone = harness.phone

LOOK_FORMAT = "automatos-e2e-hello-look/v1"  # the hello theme's looks/schema.json "look.schema"
DEFAULT_LOOK = "default"  # its theme.json "look"


def _mobile(phone, install_id, headers):
    return phone.get(f"/api/themes/mobile/{install_id}", headers=headers)


# --------------------------------------------------------------------------- the install as data


def test_a_paired_phone_reads_its_workspaces_install_as_data_and_loads_every_file_it_names(
        phone, db_session, ws, theme_files):
    user_pk, headers = harness._phone_of(db_session, ws, "admin")
    install_id = harness._private_theme(db_session, theme_files, ws, user_pk)
    package = theme_files.package
    files = (harness.MOBILE_FONT, build.CLIP, build.STILL)
    font, clip, still = (module_url(package.digest, path) for path in files)

    response = _mobile(phone, install_id, headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["installId"], body["themeId"], body["version"]) == (
        str(install_id), package.manifest["id"], package.manifest["version"])
    assert body["mobile"] == {
        "source": "tokens", "tone": None, "tokens": {"light": harness.MOBILE_LIGHT, "dark": None},
        "fonts": {"sans": font},
        "mascot": {"mode": "clips", "states": [{"name": build.CLIP_STATE, "clip": clip, "still": still}]},
    }
    for url, path in zip((font, clip, still), files):
        served = phone.get(url, headers=headers)
        assert served.status_code == 200, url
        assert served.content == package.files[path].data


def test_a_paired_phone_gets_the_one_404_for_its_persons_install_in_another_workspace(
        phone, db_session, ws, theme_files, seed_workspace):
    from api.themes_mobile import MOBILE_NOT_FOUND

    user_pk, headers = harness._phone_of(db_session, ws, "admin")
    elsewhere = uuid.UUID(seed_workspace())
    harness._join(db_session, elsewhere, user_pk, "owner")
    theirs = harness._private_theme(db_session, theme_files, elsewhere, user_pk)

    foreign, unknown = _mobile(phone, theirs, headers), _mobile(phone, uuid.uuid4(), headers)

    assert (foreign.status_code, unknown.status_code) == (404, 404)
    assert foreign.json() == unknown.json() == {"detail": MOBILE_NOT_FOUND}


# --------------------------------------------------------------------------- the active theme


def _choose(db, workspace_id, user_pk, install_id: str) -> None:
    """The person's stored choice of ``install_id`` in ``workspace_id`` (one theme_choices row)."""
    db.add(ThemeChoice(workspace_id=workspace_id, user_id=user_pk, choice=install_id))
    harness._settle(db)


def _choices(db, user_pk) -> set:
    rows = db.execute(text("SELECT workspace_id, choice FROM theme_choices WHERE user_id = :uid"), {"uid": user_pk})
    return {(str(row.workspace_id), row.choice) for row in rows}


def test_a_paired_phone_reads_its_workspaces_active_theme_and_chooses_the_install(
        phone, db_session, ws, theme_files, seed_workspace):
    user_pk, headers = harness._phone_of(db_session, ws, "admin")
    install_id = str(harness._private_theme(db_session, theme_files, ws, user_pk))
    elsewhere = uuid.UUID(seed_workspace())
    harness._join(db_session, elsewhere, user_pk, "owner")
    theirs = str(harness._private_theme(db_session, theme_files, elsewhere, user_pk))
    _choose(db_session, elsewhere, user_pk, theirs)

    before = phone.get("/api/themes/active", headers=headers)
    chosen = phone.put("/api/themes/choice", json={"choice": install_id}, headers=headers)
    after = phone.get("/api/themes/active", headers=headers)

    assert before.status_code == 200, before.text
    assert before.json() == {"choice": "studio", "source": "fallback", "theme": None}
    assert chosen.status_code == 200, chosen.text
    assert after.status_code == 200, after.text
    active = after.json()
    assert (active["choice"], active["source"], active["theme"]["installId"]) == (install_id, "choice", install_id)
    assert active["theme"]["optIn"]["required"] is False
    assert chosen.json() == active
    assert _choices(db_session, user_pk) == {(str(ws), install_id), (str(elsewhere), theirs)}


# --------------------------------------------------------------------------- the person's looks


def _look(look_id: str, ground: str) -> dict:
    """A look of the hello theme's format (its knobs: accent, ground, radius)."""
    return {"schema": LOOK_FORMAT, "id": look_id, "name": look_id.title(), "sub": "",
            "accent": "green", "ground": ground, "radius": 4}


def _keep_looks(db, workspace_id, user_pk, theme_id: str, look: dict) -> None:
    """``look`` saved and active for ``user_pk`` in ``workspace_id`` (one theme_looks row)."""
    db.add(ThemeLook(workspace_id=workspace_id, user_id=user_pk, theme_id=theme_id, looks=[look], active=look))
    harness._settle(db)


def test_a_paired_phone_reads_its_persons_looks_in_its_workspace_only(
        phone, db_session, ws, theme_files, seed_workspace):
    user_pk, headers = harness._phone_of(db_session, ws, "admin")
    harness._private_theme(db_session, theme_files, ws, user_pk)
    theme_id = theme_files.package.manifest["id"]
    colleague = harness._person(db_session, ws, "admin")
    elsewhere = uuid.UUID(seed_workspace())
    harness._join(db_session, elsewhere, user_pk, "owner")
    harness._private_theme(db_session, theme_files, elsewhere, user_pk)
    ours = _look("dusk", "ink")
    _keep_looks(db_session, ws, user_pk, theme_id, ours)
    _keep_looks(db_session, ws, colleague, theme_id, _look("noon", "paper"))
    _keep_looks(db_session, elsewhere, user_pk, theme_id, _look("away", "paper"))

    response = phone.get(f"/api/themes/looks/{theme_id}", headers=headers)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "themeId": theme_id, "looks": [ours], "active": ours, "defaultLook": DEFAULT_LOOK, "dropped": 0,
    }
