"""GET /exports/{file_path:path} serves only the exports/ directory
(CodeQL py/path-injection #109 #110 #111).

``{file_path:path}`` can contain ``/`` and ``..``, and the old guard compared
string prefixes (``str(requested).startswith(str(safe_base))``) on the
resolved path. A sibling directory whose name happens to start with
"exports" (e.g. "exports-other") defeats that compare: its resolved path
starts with the string ``".../exports"`` without being inside it. The fix
switches to ``Path.is_relative_to``, the guard this codebase already uses for
the equivalent document routes (services/document_downloads.py).

``import main`` is DB-free at import time (confirmed by
test_route_manifest.py): the lifespan that dials Postgres only runs when the
app actually serves.
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402

import main  # noqa: E402


@pytest.fixture
def exports_cwd(tmp_path, monkeypatch):
    """``exports/`` and a sibling ``exports-other/`` under a scratch cwd.

    ``safe_base = Path("exports").resolve()`` in main.serve_export is relative
    to the process's current directory, so the fixture chdirs into it.
    """
    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "report.pdf").write_bytes(b"ours")
    (tmp_path / "exports-other").mkdir()
    (tmp_path / "exports-other" / "secret.pdf").write_bytes(b"not yours")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _serve(file_path: str):
    return asyncio.run(main.serve_export(file_path=file_path, ctx=object()))


def test_a_file_inside_exports_is_served(exports_cwd):
    resp = _serve("report.pdf")
    assert str(resp.path).endswith("exports/report.pdf")


def test_a_sibling_directory_sharing_the_exports_prefix_is_rejected(exports_cwd):
    """Regression: "../exports-other/secret.pdf" resolves to a path that
    STARTS WITH the string ".../exports" without being inside it."""
    with pytest.raises(main.HTTPException) as exc_info:
        _serve("../exports-other/secret.pdf")
    assert exc_info.value.status_code == 403


def test_traversal_above_exports_is_rejected(exports_cwd):
    with pytest.raises(main.HTTPException) as exc_info:
        _serve("../../etc/passwd")
    assert exc_info.value.status_code == 403
