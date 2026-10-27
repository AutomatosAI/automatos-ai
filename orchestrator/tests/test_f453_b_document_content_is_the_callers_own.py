"""F453 (b): the artifact viewer's text read is the caller's own workspace's, as the downloads are.

``GET /api/documents/content?path=`` took a signed-in caller but no permission and no workspace: any
member of any workspace read any file under the documents root, every tenant's local files. It now
carries ``documents:read`` and serves only ``<root>/<workspace_id>/`` (``document_downloads.workspace_file``),
the same rule as ``/download?path=``; a path anywhere else answers 404 like a missing file.
"""
from __future__ import annotations

import pytest

from tests import test_f453_a_document_download_is_the_callers_own as f453a

root = f453a.root          # a documents root with our folder, theirs, a loose file and a symlink out
CONTENT_ROUTE = "/api/documents/content"
WS, OTHER_WS, NOT_FOUND = f453a.WS, f453a.OTHER_WS, f453a.NOT_FOUND


def test_the_text_read_carries_the_hybrid_context_and_the_documents_read_gate():
    from core.auth.hybrid import get_request_context_hybrid
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR

    [route] = [r for r in f453a.router.routes if r.path == CONTENT_ROUTE and "GET" in (r.methods or ())]
    calls = f453a._flatten(route.dependant, [])
    assert get_request_context_hybrid in calls
    assert {getattr(c, PERMISSION_MARKER_ATTR) for c in calls if hasattr(c, PERMISSION_MARKER_ATTR)} == {"documents:read"}


def test_a_signed_in_stranger_to_the_workspace_is_refused(root):
    resp = f453a._client(f453a._membership_db(role=None), f453a._member("stranger")).get(
        CONTENT_ROUTE, params={"path": str(root / str(WS) / "report.md")})
    assert resp.status_code == 403 and resp.json() == {"detail": "Permission denied: documents:read"}


def test_a_file_in_the_callers_own_folder_is_read(root):
    resp = f453a._client(f453a._membership_db(role="viewer"), f453a._member("viewer")).get(
        CONTENT_ROUTE, params={"path": str(root / str(WS) / "report.md")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["content"] == "# Our report" and resp.json()["filename"] == "report.md"


@pytest.mark.parametrize("relative", [
    f"{OTHER_WS}/secret.md", "loose.md", f"{WS}/../{OTHER_WS}/secret.md", f"{WS}/theirs.md", f"{WS}/missing.md",
])
def test_another_workspaces_file_or_a_way_out_of_ours_is_not_found(root, relative):
    resp = f453a._client(f453a._membership_db(role="owner"), f453a._member("owner")).get(
        CONTENT_ROUTE, params={"path": str(root / relative)})
    assert resp.status_code == 404 and resp.json() == NOT_FOUND
    assert "Their report" not in resp.text


def test_a_path_outside_the_root_is_not_found(root):
    resp = f453a._client(f453a._membership_db(role="owner"), f453a._member("owner")).get(
        CONTENT_ROUTE, params={"path": "/etc/hosts"})
    assert resp.status_code == 404 and resp.json() == NOT_FOUND
