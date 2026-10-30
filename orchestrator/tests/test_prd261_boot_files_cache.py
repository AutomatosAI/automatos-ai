"""PRD-261 — the theme services' in-memory caches of checked, content-addressed files.

``services/themes/bounded_cache.py`` is the one bounded cache they share: at most ``size`` entries,
the least recently used dropped first, safe across FastAPI's threadpool workers.

``services/themes/serve.py`` ``boot_files``: the boot route (``GET /api/themes/boot/{install_id}``)
read ``boot/prepaint.json``, ``looks/schema.json`` and every ``looks/presets/*.look.json`` from
object storage on every request, each re-hashed, which took the hosted boot past the frontend's
300 ms server-render budget. A stored package's files never change, so their checked contents are
kept by (digest, storage prefix). Object storage is stood in for at the S3 client
(``serve.get_s3_client``), so the real reader runs and every object read is counted. Pinned:

* a second boot of the same package reads no look file and answers the same;
* another digest reads its own files, under its own storage prefix or the same one, and the same
  digest under another prefix (another workspace's copy) reads that copy;
* a file that fails its hash check is never kept: the next boot reads it and refuses it again, and
  once the stored bytes match again the boot is kept;
* a storage failure keeps nothing (503) and the next boot reads again;
* the bound drops the least recently booted package; boots from many threads stay correct;
* a cold boot reads its look files at once, and a read the checks never reach changes no answer;
* with a warm cache, a non-member, a member who withdrew their opt-in, a disabled version and an
  uninstalled install are refused exactly as on a cold one.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Set

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers the tables the theme FKs name)
from api import themes_files  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import (  # noqa: E402
    ThemeInstallation, ThemeOptIn, ThemePackage, ThemeRevocation, ThemeRevocationSync,
)
from services.themes import looks as looks_service  # noqa: E402
from services.themes import serve  # noqa: E402
from services.themes.bounded_cache import BoundedCache  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

THREADS = 8
# How long the read-ahead test waits for every look file's read to be in flight at once.
READ_WAIT_SECONDS = 10
PREPAINT_PATH, SCHEMA_PATH = "boot/prepaint.json", "looks/schema.json"
PRESETS = ("looks/presets/default.look.json", "looks/presets/night.look.json")
LOOK_FILES = (PREPAINT_PATH, SCHEMA_PATH, *PRESETS)
WHITE, GREY, SILVER = "#fafafa", "#f0f0f0", "#e0e0e0"
ALICE, BOB = 1, 2
HOME = uuid.UUID(int=0xA1)
OTHER = uuid.UUID(int=0xA2)
PREFIX = f"themes/_workspace/{HOME}/automatos-e2e/hello/"
PREFIX_A, PREFIX_B, PREFIX_C = (f"{PREFIX}{version}/" for version in ("1.0.0", "1.0.1", "1.0.2"))
OTHER_COPY = f"themes/_workspace/{OTHER}/automatos-e2e/hello/1.0.0/"
RAW_TABLES = (
    "CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id VARCHAR, email VARCHAR)",
    "CREATE TABLE workspaces (id VARCHAR PRIMARY KEY, owner_id INTEGER, paused_at TIMESTAMP, deleted_at TIMESTAMP)",
    "CREATE TABLE workspace_members (workspace_id VARCHAR, user_id INTEGER, role VARCHAR, is_active BOOLEAN)",
)
THEME_TABLES = [model.__table__ for model in (
    ThemePackage, ThemeInstallation, ThemeOptIn, ThemeRevocation, ThemeRevocationSync,
)]
# The hello theme's boot when it may not run: its base's class, its own class taken off, nothing else.
BASE_BOOT = {
    "addClasses": ["studio"], "removeClasses": ["hello"], "tone": {"policy": "user", "value": None},
    "prepaint": None, "lookStorage": None, "lookGuard": None, "looks": None, "integrity": None, "entry": None,
}


# --- the bounded cache ---------------------------------------------------------------------------


def test_a_full_cache_drops_the_least_recently_used_entry():
    cache = BoundedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # a is now the most recently used
    cache.put("c", 3)
    assert (cache.get("a"), cache.get("b"), cache.get("c"), len(cache)) == (1, None, 3, 2)
    cache.put("a", 4)  # a new value under a kept key replaces it and drops nothing
    assert (cache.get("a"), cache.get("c"), len(cache)) == (4, 3, 2)
    cache.clear()
    assert (cache.get("a"), len(cache)) == (None, 0)


def test_the_cache_holds_its_bound_under_concurrent_use():
    cache = BoundedCache(4)

    def work(worker: int) -> None:
        for step in range(200):
            key = (worker * step) % 16
            cache.put(key, key)
            found = cache.get(key)
            assert found is None or found == key

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(work, range(THREADS)))
    assert len(cache) <= 4


def test_the_look_schema_cache_is_the_shared_bounded_cache():
    assert isinstance(looks_service._schema_cache, BoundedCache)


# --- stored packages -----------------------------------------------------------------------------


class _ClientError(Exception):
    """What boto3 raises: the error code under ``response``."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _Storage:
    """Object storage as the S3 client answers it: stored bytes by key, every read counted, from any thread."""

    def __init__(self) -> None:
        self.objects: Dict[str, bytes] = {}
        self.down = False
        self.failing: Set[str] = set()
        self._reads: List[str] = []
        self._lock = threading.Lock()

    def get_object(self, Bucket: str, Key: str) -> Dict[str, Any]:  # noqa: N803 — boto3's argument names
        with self._lock:
            self._reads.append(Key)
        if self.down or Key in self.failing:
            raise _ClientError("InternalError")
        if Key not in self.objects:
            raise _ClientError("NoSuchKey")
        data = self.objects[Key]
        return {"Body": SimpleNamespace(read=lambda: data)}

    def store(self, prefix: str, built: Any) -> None:
        self.objects.update({f"{prefix}{path}": item.data for path, item in built.files.items()})

    def take_reads(self) -> List[str]:
        """The keys read since the last call, sorted."""
        with self._lock:
            taken, self._reads = sorted(self._reads), []
        return taken


def _prepaint(colour: str) -> Dict[str, Any]:
    return build.hello_prepaint({"ground": {"paper": {"--hello-bg": colour}, "ink": {"--hello-bg": "#101010"}}})


def _built(colour: str = WHITE) -> Any:
    """The hello test package with a prepaint table painting ``colour``: each colour is its own digest."""
    replace = {PREPAINT_PATH: json.dumps(_prepaint(colour)).encode("utf-8")}
    return read_theme_package(build.build_package("hello", replace=replace))


def _row(built: Any, prefix: str, workspace: uuid.UUID = HOME) -> ThemePackage:
    """A workspace's private package row for ``built``, its files stored under ``prefix``."""
    manifest = built.manifest
    return ThemePackage(
        theme_id=manifest["id"], version=manifest["version"], owner_type="workspace", workspace_id=workspace,
        digest=built.digest, manifest=manifest, file_manifest=built.file_manifest, storage_prefix=prefix,
        status="pending", reviewed=False,
    )


def _keys(prefix: str, paths: Sequence[str] = LOOK_FILES) -> List[str]:
    return sorted(f"{prefix}{path}" for path in paths)


def _prefixes_read(storage: _Storage) -> Set[str]:
    reads = storage.take_reads()
    return {prefix for prefix in (PREFIX_A, PREFIX_B, PREFIX_C, OTHER_COPY) if any(key.startswith(prefix) for key in reads)}


def _boot_files(row: ThemePackage) -> serve.BootFiles:
    return serve.boot_files(row, serve.read_stored_file)


def _preset_ids(boot: Dict[str, Any]) -> List[str]:
    return [preset["id"] for preset in (boot.get("looks") or {}).get("presets", [])]


@pytest.fixture
def storage(monkeypatch):
    stand_in = _Storage()
    monkeypatch.setattr(serve, "get_s3_client", lambda: stand_in)
    return stand_in


def test_another_digest_reads_its_own_files_and_another_copy_reads_that_copy(storage):
    white, grey, silver = _built(WHITE), _built(GREY), _built(SILVER)
    assert len({white.digest, grey.digest, silver.digest}) == 3
    storage.store(PREFIX_A, white)
    storage.store(PREFIX_B, grey)
    storage.store(OTHER_COPY, white)
    assert _boot_files(_row(white, PREFIX_A)).prepaint == _prepaint(WHITE)
    assert _boot_files(_row(grey, PREFIX_B)).prepaint == _prepaint(GREY)
    assert storage.take_reads() == sorted(_keys(PREFIX_A) + _keys(PREFIX_B))
    assert _boot_files(_row(white, PREFIX_A)).prepaint == _prepaint(WHITE)
    assert storage.take_reads() == []  # white under PREFIX_A is kept
    # The same digest stored by another workspace is that workspace's copy: it is read and checked itself.
    assert _boot_files(_row(white, OTHER_COPY, OTHER)).prepaint == _prepaint(WHITE)
    assert storage.take_reads() == _keys(OTHER_COPY)
    # A re-upload of the same version: new bytes under the same prefix, so a new digest.
    storage.store(PREFIX_A, silver)
    assert _boot_files(_row(silver, PREFIX_A)).prepaint == _prepaint(SILVER)
    assert storage.take_reads() == _keys(PREFIX_A)


def test_the_bound_drops_the_least_recently_booted_package(storage, monkeypatch):
    monkeypatch.setattr(serve, "_boot_files", BoundedCache(2))
    rows = {}
    for colour, prefix in ((WHITE, PREFIX_A), (GREY, PREFIX_B), (SILVER, PREFIX_C)):
        built = _built(colour)
        storage.store(prefix, built)
        rows[prefix] = _row(built, prefix)
    _boot_files(rows[PREFIX_A])
    _boot_files(rows[PREFIX_B])
    assert _prefixes_read(storage) == {PREFIX_A, PREFIX_B}
    _boot_files(rows[PREFIX_A])  # A is now the most recently booted
    _boot_files(rows[PREFIX_C])  # the cache is full: B goes
    assert _prefixes_read(storage) == {PREFIX_C}
    _boot_files(rows[PREFIX_A])
    assert _prefixes_read(storage) == set()
    _boot_files(rows[PREFIX_B])
    assert _prefixes_read(storage) == {PREFIX_B}


def test_boots_from_many_threads_each_get_their_own_package(storage, monkeypatch):
    monkeypatch.setattr(serve, "_boot_files", BoundedCache(2))
    rows = []
    for colour, prefix in ((WHITE, PREFIX_A), (GREY, PREFIX_B), (SILVER, PREFIX_C)):
        built = _built(colour)
        storage.store(prefix, built)
        rows.append((_row(built, prefix), _prepaint(colour)))

    def boot(step: int) -> bool:
        row, expected = rows[step % len(rows)]
        return _boot_files(row).prepaint == expected

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        assert all(pool.map(boot, range(THREADS * 25)))
    assert len(serve._boot_files) <= 2


def test_a_cold_boot_reads_its_look_files_at_once(storage):
    built = _built()
    storage.store(PREFIX_A, built)
    together = threading.Barrier(len(LOOK_FILES), timeout=READ_WAIT_SECONDS)

    def read(prefix: str, path: str):
        together.wait()  # breaks, failing the boot, unless every look file's read is in flight at once
        return serve.read_stored_file(prefix, path)

    files = serve.boot_files(_row(built, PREFIX_A), read)
    assert files.prepaint == _prepaint(WHITE) and _preset_ids(files.looks) == ["default", "night"]
    assert storage.take_reads() == _keys(PREFIX_A)


# --- the boot route ------------------------------------------------------------------------------


@pytest.fixture
def world(storage):
    """Alice owns HOME, where a private hello package is installed and she opted in; Bob belongs nowhere."""
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        for ddl in RAW_TABLES:
            conn.execute(sa.text(ddl))
    ThemePackage.metadata.create_all(engine, tables=THEME_TABLES)
    db = sessionmaker(bind=engine)()
    for pk, name in ((ALICE, "alice"), (BOB, "bob")):
        db.execute(sa.text("INSERT INTO users VALUES (:id, :cid, :email)"),
                   {"id": pk, "cid": f"user_{name}", "email": f"{name}@example.com"})
    db.execute(sa.text("INSERT INTO workspaces VALUES (:id, :owner, NULL, NULL)"), {"id": str(HOME), "owner": ALICE})
    built = _built()
    row = _row(built, PREFIX_A)
    db.add(row)
    db.flush()
    install = ThemeInstallation(workspace_id=HOME, theme_id=row.theme_id, package_id=row.id)
    db.add_all([install, ThemeOptIn(workspace_id=HOME, user_id=ALICE, package_id=row.id)])
    db.commit()
    storage.store(PREFIX_A, built)
    try:
        yield SimpleNamespace(db=db, storage=storage, row=row, install=install)
    finally:
        db.close()
        engine.dispose()


def _boot(world, name: str):
    """``name``'s boot of the install, through the route and the real stored-file reader."""
    user = UserContext(id=f"user_{name}", email=f"{name}@example.com", clerk_user_id=f"user_{name}")
    ctx = RequestContext(workspace_id=HOME, user=user, auth_type="clerk")
    app = FastAPI()
    app.include_router(themes_files.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx

    def _db():
        yield world.db

    app.dependency_overrides[get_db] = _db
    return TestClient(app, raise_server_exceptions=False).get(f"/api/themes/boot/{world.install.id}")


def test_a_second_boot_of_the_same_package_reads_no_look_file_from_storage(world):
    first = _boot(world, "alice")
    assert first.status_code == 200
    body = first.json()
    assert body["prepaint"] == _prepaint(WHITE) and _preset_ids(body) == ["default", "night"]
    assert body["lookStorage"] is not None and body["entry"]["themeId"] == "automatos-e2e/hello"
    assert world.storage.take_reads() == _keys(PREFIX_A)
    second = _boot(world, "alice")
    assert world.storage.take_reads() == []
    assert (second.status_code, second.json()) == (200, body)


def _refused_as_today(path: str, boot: Dict[str, Any]) -> bool:
    """A boot while ``path`` is refused: the base boot without its prepaint table, the theme with no
    look fields without its schema, the theme with the other preset without one preset."""
    if path == PREPAINT_PATH:
        return boot == BASE_BOOT
    runs = boot["entry"] is not None and boot["prepaint"] == _prepaint(WHITE)
    if path == SCHEMA_PATH:
        return runs and (boot["lookStorage"], boot["looks"]) == (None, None)
    others = [look_id for look_id in ("default", "night") if path != f"looks/presets/{look_id}.look.json"]
    return runs and _preset_ids(boot) == others


@pytest.mark.parametrize("path", LOOK_FILES)
def test_a_file_failing_its_hash_check_is_never_kept_and_is_refused_again(world, path):
    key = f"{PREFIX_A}{path}"
    good = world.storage.objects[key]
    world.storage.objects[key] = good + b" "
    refused = [_boot(world, "alice").json() for _ in range(2)]
    assert _refused_as_today(path, refused[0]) and refused[1] == refused[0]
    assert world.storage.take_reads().count(key) == 2  # read and checked again by the second boot
    assert len(serve._boot_files) == 0
    world.storage.objects[key] = good
    kept = _boot(world, "alice").json()
    assert kept["prepaint"] == _prepaint(WHITE) and _preset_ids(kept) == ["default", "night"]
    assert kept["lookStorage"] is not None
    assert world.storage.take_reads() == _keys(PREFIX_A)
    assert _boot(world, "alice").json() == kept
    assert world.storage.take_reads() == []


def test_a_storage_failure_keeps_nothing_and_the_next_boot_reads_again(world):
    world.storage.down = True
    resp = _boot(world, "alice")
    assert resp.status_code == 503 and "Try again" in resp.json()["detail"]
    assert len(serve._boot_files) == 0
    world.storage.down = False
    world.storage.take_reads()
    assert _boot(world, "alice").json()["prepaint"] == _prepaint(WHITE)
    assert world.storage.take_reads() == _keys(PREFIX_A)


def test_with_a_warm_cache_a_boot_that_may_not_run_is_refused_as_before(world):
    runs = _boot(world, "alice").json()
    assert runs["entry"] is not None and len(serve._boot_files) == 1
    assert _boot(world, "bob").status_code == 404  # not a member of the install's workspace
    world.db.query(ThemeOptIn).delete()
    world.db.commit()
    assert _boot(world, "alice").json() == BASE_BOOT  # private code, and she withdrew her opt-in
    world.db.add(ThemeOptIn(workspace_id=HOME, user_id=ALICE, package_id=world.row.id))
    world.db.commit()
    world.storage.take_reads()
    assert _boot(world, "alice").json() == runs  # opted in again: answered from memory
    assert world.storage.take_reads() == []
    world.db.add(ThemeRevocation(digest=world.row.digest, level="disable", reason="bad bytes"))
    world.db.commit()
    assert _boot(world, "alice").json() == BASE_BOOT  # a disabled version
    world.install.uninstalled_at = datetime.now(timezone.utc)
    world.db.commit()
    assert _boot(world, "alice").status_code == 404  # uninstalled
    assert len(serve._boot_files) == 1  # the package's files stayed kept throughout


def test_a_read_the_checks_never_reach_changes_no_answer(world):
    # The prepaint table is refused, so the boot never checks the presets: the failed read-ahead of
    # one is no 503, the boot is the base boot it always was.
    prepaint_key = f"{PREFIX_A}{PREPAINT_PATH}"
    world.storage.objects[prepaint_key] = world.storage.objects[prepaint_key] + b" "
    world.storage.failing.add(f"{PREFIX_A}{PRESETS[1]}")
    resp = _boot(world, "alice")
    assert (resp.status_code, resp.json()) == (200, BASE_BOOT)
    assert len(serve._boot_files) == 0
