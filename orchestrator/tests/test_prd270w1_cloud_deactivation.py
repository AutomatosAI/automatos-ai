"""PRD-270 W1 review fix P270W1-RVW-1 — deactivating a version never moves an install onto a Cloud version
whose files this install doesn't hold. ``services/themes/auto_update.py`` (``previous_listed``).

A Sync writes every catalogue version as an approved, reviewed Cloud row with no stored byte (K15): only
the version a workspace installs or updates to is downloaded (US-270-04a/b). Before this fix the roll-back
on deactivation picked the newest listed version below whatever its storage held, so an install landed on a
version whose files route serves nothing. ``review.deactivate`` runs on SQLite as the local operator (no
actor), the pattern of ``test_prd261_theme_auto_update.py``; storage is a fake S3 that answers NoSuchKey for
a missing object, records every key it is asked for, and can be down. Pinned:

* a Cloud version below whose files aren't stored is skipped: with nothing else below, the installs stay on
  the version taken down (counted 0 in the answer and the ``deactivated`` event), logged with the theme, the
  version skipped and why; with a stored version further below, they move there;
* a stored Cloud version below is moved to as before, and nothing below it is read; a pinned install never moves;
* a local-origin version below is moved to as before, and storage is never read for it;
* storage down during the check skips the candidate (logged); the deactivation still succeeds; after one
  failed read every later Cloud version is skipped unread (one read under the row lock), a local one still taken.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
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
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.models.themes import (  # noqa: E402
    THEME_SOURCE_CLOUD,
    ThemeInstallation,
    ThemePackage,
    ThemeReviewEvent,
    ThemeRevocation,
    ThemeRevocationSync,
)
from services.themes import review, serve, storage  # noqa: E402
from services.themes.contract import MANIFEST_PATH  # noqa: E402
from services.themes.digest import package_digest  # noqa: E402
from services.themes.revocations import is_disabled  # noqa: E402

PLAY = "automatos/play"
ENGINES = {"automatos": "^1.0.0", "sdk": "^1.0.0"}
TABLES = [m.__table__ for m in (ThemePackage, ThemeInstallation, ThemeReviewEvent, ThemeRevocation, ThemeRevocationSync)]
AUTO_UPDATE_LOG = "services.themes.auto_update"
LOCAL = None


class _Missing(Exception):
    """What boto3 raises for a key that isn't there (``serve.MISSING_OBJECT_CODES``)."""

    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    """Objects by key; a missing one is NoSuchKey; every key read is recorded; ``down`` makes every read fail."""

    def __init__(self) -> None:
        self.objects, self.reads, self.down = {}, [], False

    def get_object(self, Bucket, Key):  # noqa: N803 (boto3's names)
        self.reads.append(Key)
        if self.down:
            raise ConnectionError("storage is down")
        if Key not in self.objects:
            raise _Missing(Key)
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(serve, "get_s3_client", lambda: fake)
    return fake


@pytest.fixture
def db(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")  # Cloud rows exist on the local edition only
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _theme_json(version):
    return json.dumps({"id": PLAY, "name": "Play", "version": version, "engines": ENGINES}).encode()


def _version(db, version, source=THEME_SOURCE_CLOUD):
    """An approved, reviewed Marketplace version of Play (a Sync's Cloud row by default) with its theme.json listed."""
    bodies = {MANIFEST_PATH: _theme_json(version), "dist/entry.js": version.encode()}
    files = {path: {"type": "application/json" if path == MANIFEST_PATH else "text/javascript",
                    "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
             for path, body in bodies.items()}
    row = ThemePackage(
        theme_id=PLAY, version=version, owner_type="marketplace", status="approved", reviewed=True,
        manifest={"id": PLAY, "name": "Play", "version": version, "engines": ENGINES},
        file_manifest=files, storage_prefix=storage.theme_prefix(PLAY, version), scan_verdict="review_required",
        digest=package_digest({path: entry["sha256"] for path, entry in files.items()}), source=source,
    )
    db.add(row)
    db.commit()
    return row


def _store(s3, package):
    """Write the version's files as an install's download does (theme.json last)."""
    s3.objects[storage.theme_key(package.storage_prefix, "dist/entry.js")] = package.version.encode()
    s3.objects[storage.theme_key(package.storage_prefix, MANIFEST_PATH)] = _theme_json(package.version)


def _install(db, package, **extra):
    """A listing install of ``package`` in a workspace of its own."""
    row = ThemeInstallation(workspace_id=uuid.uuid4(), theme_id=PLAY, package_id=package.id, **extra)
    db.add(row)
    db.commit()
    return row


def _state(db, install):
    db.refresh(install)
    return install.package_id, install.previous_package_id, install.pending_package_id


def _deactivated_detail(db, package):
    [row] = db.query(ThemeReviewEvent).filter(
        ThemeReviewEvent.package_id == package.id, ThemeReviewEvent.event == "deactivated",
    ).all()
    return row.detail


def _deactivate(db, package):
    answer = review.deactivate(db, package.id, "breaks the composer", None)  # the local operator: no actor
    db.commit()
    return answer


def _skips(caplog, version, reason):
    return [record for record in caplog.records
            if record.name == AUTO_UPDATE_LOG and "roll-back skips" in record.getMessage()
            and f"{PLAY} {version} " in record.getMessage() and reason in record.getMessage()]


def test_a_cloud_version_below_whose_files_are_not_stored_is_skipped_and_the_install_stays(db, s3, caplog):
    below = _version(db, "1.1.0")  # synced, never installed: no byte stored
    bad = _version(db, "1.2.0")
    _store(s3, bad)
    install = _install(db, bad)

    with caplog.at_level(logging.INFO, logger=AUTO_UPDATE_LOG):
        answer = _deactivate(db, bad)

    assert answer["status"] == "deactivated" and is_disabled(db, bad)
    assert answer["installsRolledBack"] == 0 and _deactivated_detail(db, bad)["installsRolledBack"] == 0
    assert _state(db, install) == (bad.id, None, None)  # stays; its disable sends members to the base
    assert _skips(caplog, below.version, "not stored")
    assert storage.theme_key(below.storage_prefix, MANIFEST_PATH) in s3.reads


def test_a_stored_cloud_version_below_is_moved_to_and_a_pinned_install_stays(db, s3):
    oldest = _version(db, "1.0.0")
    below = _version(db, "1.1.0")
    _store(s3, below)
    bad = _version(db, "1.2.0")
    _store(s3, bad)
    install = _install(db, bad)
    pinned = _install(db, bad, pinned_version="1.2.0")

    answer = _deactivate(db, bad)

    assert answer["installsRolledBack"] == 1 and _deactivated_detail(db, bad)["installsRolledBack"] == 1
    assert _state(db, install) == (below.id, None, None)
    assert _state(db, pinned) == (bad.id, None, None)
    assert storage.theme_key(oldest.storage_prefix, MANIFEST_PATH) not in s3.reads  # the search stops at the held one


def test_the_roll_back_passes_an_unstored_cloud_version_for_a_stored_one_further_below(db, s3, caplog):
    oldest = _version(db, "1.0.0")
    _store(s3, oldest)
    skipped = _version(db, "1.1.0")
    bad = _version(db, "1.2.0")
    _store(s3, bad)
    install = _install(db, bad)

    with caplog.at_level(logging.WARNING, logger=AUTO_UPDATE_LOG):
        answer = _deactivate(db, bad)

    assert answer["installsRolledBack"] == 1
    assert _state(db, install) == (oldest.id, None, None)
    assert _skips(caplog, skipped.version, "not stored")


def test_a_local_origin_version_below_is_moved_to_without_a_storage_read(db, s3):
    below = _version(db, "1.1.0", source=LOCAL)  # nothing stored in the fake: storage must not be asked
    bad = _version(db, "1.2.0")
    install = _install(db, bad)

    answer = _deactivate(db, bad)

    assert answer["installsRolledBack"] == 1
    assert _state(db, install) == (below.id, None, None)
    assert s3.reads == []


def test_storage_down_during_the_check_skips_the_candidate_and_the_deactivation_stands(db, s3, caplog):
    below = _version(db, "1.1.0")
    _store(s3, below)
    bad = _version(db, "1.2.0")
    install = _install(db, bad)
    s3.down = True

    with caplog.at_level(logging.WARNING, logger=AUTO_UPDATE_LOG):
        answer = _deactivate(db, bad)

    assert answer["status"] == "deactivated" and is_disabled(db, bad)
    assert answer["installsRolledBack"] == 0 and _deactivated_detail(db, bad)["installsRolledBack"] == 0
    assert _state(db, install) == (bad.id, None, None)
    assert _skips(caplog, below.version, "storage could not be read")


def test_an_outage_reads_storage_once_then_skips_later_cloud_versions_unread_for_a_local_one(db, s3, caplog):
    local = _version(db, "0.9.0", source=LOCAL)
    older, newer = _version(db, "1.0.0"), _version(db, "1.1.0")
    _store(s3, older)
    _store(s3, newer)
    bad = _version(db, "1.2.0")
    install = _install(db, bad)
    s3.down = True

    with caplog.at_level(logging.WARNING, logger=AUTO_UPDATE_LOG):
        answer = _deactivate(db, bad)

    assert answer["installsRolledBack"] == 1
    assert _state(db, install) == (local.id, None, None)
    assert s3.reads == [storage.theme_key(newer.storage_prefix, MANIFEST_PATH)]  # one read under the row lock
    assert _skips(caplog, newer.version, "storage could not be read")
    assert _skips(caplog, older.version, "storage could not be read")
