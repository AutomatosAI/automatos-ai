"""Installed Marketplace themes update themselves on approval and step back on deactivation
(publish contract D1, 30 Oct 2026: every unpinned listing install of an older version updates itself
on approval; pinning opts out). ``services/themes/auto_update.py``.

Pins:

* approving a version moves each unpinned listing install of an older version to it through
  ``updates.apply_update``: the same install row and id (Automatos Go keys the theme it paints on
  it), the old version as its rollback target, the offer cleared; the answer and the ``approved``
  event count the offers and the moves;
* a pinned install only gets the offer, as before; an install that already waits on a newer listed
  version is not moved to a lower one;
* an install whose move is refused, or fails in the database after writing, keeps the offer as it
  was; the others still move and the approval stands;
* the local edition (no Clerk user, no actor, no isolated-job note) moves installs the same way;
* deactivating a version moves the unpinned installs that run it back to the newest listed version
  below it, skipping one under a ``disable`` or one this host can't run; the version left is never
  the rollback target; a pinned install on it, and every install when no listed version is below
  it, stays on it (today's behaviour: the ``disable`` sends members to the base);
* ``updates.move_back`` only goes to an older version and keeps the pin's refusal.
"""
from __future__ import annotations

import hashlib
import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.models.themes import (  # noqa: E402
    ThemeInstallation,
    ThemePackage,
    ThemeReviewEvent,
    ThemeRevocation,
    ThemeRevocationSync,
)
from services.themes import review, updates  # noqa: E402
from services.themes.choices import CONFLICT, ThemeInstallError  # noqa: E402
from services.themes.digest import package_digest  # noqa: E402
from services.themes.revocations import is_disabled  # noqa: E402

PLAY = "automatos/play"
ENGINES = {"automatos": "^1.0.0", "sdk": "^1.0.0"}
FUTURE_ENGINES = {"automatos": "^2.0.0", "sdk": "^1.0.0"}
TABLES = [m.__table__ for m in (ThemePackage, ThemeInstallation, ThemeReviewEvent, ThemeRevocation, ThemeRevocationSync)]


@pytest.fixture
def db(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _version(db, version, status="approved", engines=None):
    """A Marketplace version of Play whose digest adds up to its files (``signing_payload`` checks it)."""
    files = {"dist/entry.js": {"type": "text/javascript", "size": 1,
                               "sha256": hashlib.sha256(version.encode()).hexdigest()}}
    row = ThemePackage(
        theme_id=PLAY, version=version, owner_type="marketplace", status=status, reviewed=status == "approved",
        manifest={"id": PLAY, "name": "Play", "version": version, "engines": engines or ENGINES},
        file_manifest=files, storage_prefix=f"themes/{PLAY}/{version}/", scan_verdict="review_required",
        digest=package_digest({path: entry["sha256"] for path, entry in files.items()}),
    )
    db.add(row)
    db.commit()
    return row


def _install(db, package, **extra):
    """A listing install of ``package`` in a workspace of its own."""
    row = ThemeInstallation(workspace_id=uuid.uuid4(), theme_id=PLAY, package_id=package.id, **extra)
    db.add(row)
    db.commit()
    return row


def _approve(db, package):
    """Approve on the hosted edition: the kit's passing summary for this version first, then approve."""
    note = f"Outcome: pass\nPackage digest:  {package.digest}\nisolated job: reproducible build, digests equal"
    review.add_report_note(db, package.id, note, "pass", None)
    answer = review.approve(db, package.id, None)
    db.commit()
    return answer


def _state(db, install):
    db.refresh(install)
    return install.package_id, install.previous_package_id, install.pending_package_id


def _event(db, package, event):
    [row] = db.query(ThemeReviewEvent).filter(
        ThemeReviewEvent.package_id == package.id, ThemeReviewEvent.event == event,
    ).all()
    return row.detail


# --- approval ------------------------------------------------------------------------------------


def test_approval_moves_an_unpinned_older_listing_install_to_the_new_version_keeping_its_id(db):
    old = _version(db, "1.0.0")
    install = _install(db, old)
    install_id = install.id
    new = _version(db, "1.1.0", "pending")

    answer = _approve(db, new)

    assert (answer["status"], answer["updatesOffered"], answer["updatesApplied"]) == ("approved", 1, 1)
    assert _state(db, install) == (new.id, old.id, None)  # moved; the old version is the rollback target
    assert install.id == install_id and db.query(ThemeInstallation).count() == 1  # Go keys on this id
    detail = _event(db, new, "approved")
    assert (detail["updatesOffered"], detail["updatesApplied"]) == (1, 1)


def test_a_pinned_install_only_gets_the_offer_and_an_unpinned_one_beside_it_moves(db):
    old = _version(db, "1.0.0")
    pinned = _install(db, old, pinned_version="1.0.0")
    free = _install(db, old)
    new = _version(db, "1.1.0", "pending")

    answer = _approve(db, new)

    assert (answer["updatesOffered"], answer["updatesApplied"]) == (2, 1)
    assert _state(db, pinned) == (old.id, None, new.id)  # offered, never switched
    assert _state(db, free) == (new.id, old.id, None)


def test_an_install_waiting_on_a_newer_listed_version_is_not_moved_to_a_lower_one(db):
    old = _version(db, "1.0.0")
    newest = _version(db, "1.2.0")
    waiting = _install(db, old, pending_package_id=newest.id)
    middle = _version(db, "1.1.0", "pending")

    answer = _approve(db, middle)

    assert (answer["updatesOffered"], answer["updatesApplied"]) == (0, 0)
    assert _state(db, waiting) == (old.id, None, newest.id)


@pytest.mark.parametrize("failure", [
    ThemeInstallError(CONFLICT, "refused for the test"),
    OperationalError("UPDATE theme_installations", {}, Exception("connection reset")),
], ids=["refused", "database-error"])
def test_a_refused_or_failed_move_keeps_the_offer_and_the_others_still_move(db, monkeypatch, failure):
    old = _version(db, "1.0.0")
    first, refused, last = _install(db, old), _install(db, old), _install(db, old)
    refused_id = refused.id
    real_apply = updates.apply_update

    def apply_then_fail_for_one(session, workspace_id, install_id, package_id):
        moved = real_apply(session, workspace_id, install_id, package_id)  # it writes first: the savepoint undoes it
        if install_id == refused_id:
            raise failure
        return moved

    monkeypatch.setattr(updates, "apply_update", apply_then_fail_for_one)
    new = _version(db, "1.1.0", "pending")

    answer = _approve(db, new)

    assert (answer["status"], answer["updatesOffered"], answer["updatesApplied"]) == ("approved", 3, 2)
    assert _state(db, refused) == (old.id, None, new.id)  # exactly the offer it had before this change
    assert _state(db, first) == _state(db, last) == (new.id, old.id, None)
    db.refresh(new)
    assert new.status == "approved"


def test_the_local_edition_moves_installs_with_no_clerk_user_and_no_note(db, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    old = _version(db, "1.0.0")
    install = _install(db, old)
    new = _version(db, "1.1.0", "pending")

    answer = review.approve(db, new.id, None)  # the local operator: no isolated job, no actor row
    db.commit()

    assert answer["updatesApplied"] == 1
    assert _state(db, install) == (new.id, old.id, None)


# --- deactivation --------------------------------------------------------------------------------


def test_deactivation_moves_the_unpinned_installs_on_it_back_to_the_previous_listed_version(db):
    previous = _version(db, "1.0.0")
    bad = _version(db, "1.1.0")
    came_from = _install(db, bad, previous_package_id=previous.id)
    fresh = _install(db, bad)
    pinned = _install(db, bad, pinned_version="1.1.0")
    elsewhere = _install(db, previous)
    ids = [came_from.id, fresh.id]

    answer = review.deactivate(db, bad.id, "sends tokens off-site", None)
    db.commit()

    assert answer["installsRolledBack"] == 2 and _event(db, bad, "deactivated")["installsRolledBack"] == 2
    # Back on the previous version; the withdrawn one never becomes the rollback target.
    assert _state(db, came_from) == _state(db, fresh) == (previous.id, None, None)
    assert [came_from.id, fresh.id] == ids and db.query(ThemeInstallation).count() == 4
    assert _state(db, pinned) == (bad.id, None, None)  # pinned: it holds the version, now disabled
    assert _state(db, elsewhere) == (previous.id, None, None)
    assert is_disabled(db, bad)


@pytest.mark.parametrize("unusable", ["disabled", "incompatible"])
def test_deactivation_skips_a_version_below_that_is_disabled_or_this_host_cannot_run(db, unusable):
    oldest = _version(db, "0.9.0")
    skipped = _version(db, "1.0.0", engines=FUTURE_ENGINES if unusable == "incompatible" else None)
    if unusable == "disabled":
        db.add(ThemeRevocation(digest=skipped.digest, level="disable", reason="broken"))
        db.commit()
    bad = _version(db, "1.1.0")
    install = _install(db, bad)

    review.deactivate(db, bad.id, "broken", None)
    db.commit()

    assert _state(db, install)[0] == oldest.id


def test_deactivation_with_no_listed_version_below_keeps_todays_behaviour(db):
    only = _version(db, "1.0.0")
    install = _install(db, only)

    answer = review.deactivate(db, only.id, "sends tokens off-site", None)
    db.commit()

    assert answer["installsRolledBack"] == 0
    assert _state(db, install) == (only.id, None, None)  # stays; its disable sends members to the base
    assert is_disabled(db, only)


# --- updates.move_back ---------------------------------------------------------------------------


def test_move_back_refuses_a_version_that_is_not_older_and_a_pinned_install(db):
    old, new = _version(db, "1.0.0"), _version(db, "1.1.0")
    install = _install(db, old)
    with pytest.raises(ThemeInstallError) as forward:
        updates.move_back(db, install.workspace_id, install.id, new.id)
    assert (forward.value.status_code, forward.value.message) == (CONFLICT, updates.NOT_OLDER)

    pinned = _install(db, new, pinned_version="1.1.0")
    with pytest.raises(ThemeInstallError) as held:
        updates.move_back(db, pinned.workspace_id, pinned.id, old.id)
    assert held.value.status_code == CONFLICT and "pinned to version 1.1.0" in held.value.message
