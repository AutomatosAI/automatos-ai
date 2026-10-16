"""PRD-259 S-002 — where a skin's files live: services/skins/storage.py.

* A Marketplace skin is stored under ``skins/{slug}/{version}/`` (PRD §5); a private
  import under ``skins/_workspace/{workspace_id}/{slug}/{version}/`` (decision K3), so
  two workspaces importing the same slug never share or overwrite a file.
* Every object goes to ``config.MARKETPLACE_S3_BUCKET`` through ``core.storage``'s
  client (monkeypatched here: no network), after ``ensure_bucket``.
* A slug, version or asset path that could step outside its prefix is refused.
* The module builds no client of its own.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config  # noqa: E402
from services.skins import storage  # noqa: E402
from services.skins.storage import StoredFile  # noqa: E402


class _FakeS3:
    """Records put_object calls and serves them back from get_object."""

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        self.objects[(Bucket, Key)] = (Body, ContentType)

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)][0])}


@pytest.fixture
def fake_s3(monkeypatch):
    client = _FakeS3()
    ensured = []
    monkeypatch.setattr(storage, "get_s3_client", lambda: client)
    monkeypatch.setattr(storage, "ensure_bucket", ensured.append)
    client.ensured = ensured
    return client


FILES = {
    "skin.json": StoredFile(b'{"slug": "ink"}', "application/json"),
    "assets/fonts/ink-sans.woff2": StoredFile(b"wOF2....", "font/woff2"),
    "assets/preview.png": StoredFile(b"\x89PNG....", "image/png"),
}


def test_a_marketplace_skin_lives_under_skins_slug_version(fake_s3):
    prefix = storage.skin_prefix("ink", "1.2.0")
    assert prefix == "skins/ink/1.2.0/"

    keys = storage.write_skin_files(prefix, FILES)

    bucket = config.MARKETPLACE_S3_BUCKET
    assert keys == [f"skins/ink/1.2.0/{path}" for path in FILES]
    assert set(fake_s3.objects) == {(bucket, key) for key in keys}
    assert fake_s3.objects[(bucket, "skins/ink/1.2.0/assets/fonts/ink-sans.woff2")] == (b"wOF2....", "font/woff2")
    assert fake_s3.ensured == [bucket]


def test_a_private_import_lives_under_its_workspace(fake_s3):
    workspace_a, workspace_b = uuid4(), uuid4()
    prefix_a = storage.skin_prefix("ink", "1.2.0", workspace_a)
    prefix_b = storage.skin_prefix("ink", "1.2.0", workspace_b)
    assert prefix_a == f"skins/_workspace/{workspace_a}/ink/1.2.0/"
    assert prefix_b == f"skins/_workspace/{workspace_b}/ink/1.2.0/"

    storage.write_skin_files(prefix_a, {"skin.json": StoredFile(b"A", "application/json")})
    storage.write_skin_files(prefix_b, {"skin.json": StoredFile(b"B", "application/json")})

    assert storage.read_skin_file(prefix_a, "skin.json") == b"A"
    assert storage.read_skin_file(prefix_b, "skin.json") == b"B"
    assert len(fake_s3.objects) == 2


def test_the_workspace_segment_is_a_uuid():
    with pytest.raises(ValueError):
        storage.skin_prefix("ink", "1.2.0", "../other")


@pytest.mark.parametrize("slug,version", [
    ("../ink", "1.0.0"),
    ("Ink", "1.0.0"),
    ("ink/evil", "1.0.0"),
    ("", "1.0.0"),
    ("ink", "../1.0.0"),
    ("ink", "1.0"),
    ("ink", "1.0.0/../../x"),
    ("a" * 65, "1.0.0"),
    ("ink", "1.0.0+" + "a" * 27),
])
def test_a_slug_or_version_that_is_not_one_is_refused(slug, version):
    with pytest.raises(ValueError):
        storage.skin_prefix(slug, version)


@pytest.mark.parametrize("path", [
    "", ".", "/etc/passwd", "../skin.json", "assets/../../x", "assets\\fonts\\a.woff2",
    "assets//preview.png", "./skin.json", "assets/./preview.png", "assets/fonts/a\x00.woff2", "skin.json\n",
])
def test_an_asset_path_outside_the_prefix_is_refused(fake_s3, path):
    prefix = storage.skin_prefix("ink", "1.2.0")
    with pytest.raises(ValueError):
        storage.write_skin_files(prefix, {"skin.json": StoredFile(b"{}", "application/json"), path: StoredFile(b"x", "application/octet-stream")})
    with pytest.raises(ValueError):
        storage.read_skin_file(prefix, path)
    assert fake_s3.objects == {}


@pytest.mark.parametrize("prefix", ["plugins/ink/1.0.0/", "skins/", "skins/ink/1.0.0", "skins/../x/", "skins//ink/"])
def test_a_key_needs_a_skins_prefix(prefix):
    with pytest.raises(ValueError):
        storage.skin_key(prefix, "skin.json")


def test_the_longest_slug_and_version_still_make_a_prefix():
    slug, version = "a" * 64, "1.0.0+" + "a" * 26
    assert storage.skin_prefix(slug, version) == f"skins/{slug}/{version}/"


def _imported_modules(source: str) -> set[str]:
    """Every module storage.py imports, read from its syntax tree, so a docstring that
    names MinIO (as it should: MinIO is the local store behind core.storage) can't trip it."""
    import ast

    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_no_client_of_its_own():
    source = (ORCH_ROOT / "services" / "skins" / "storage.py").read_text(encoding="utf-8")
    imported = _imported_modules(source)
    assert not any(name.split(".")[0] in {"boto3", "botocore", "minio"} for name in imported), imported
    assert "core.storage" in imported
    assert "from core.storage import ensure_bucket, get_s3_client" in source
    assert "config.MARKETPLACE_S3_BUCKET" in source
