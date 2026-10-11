"""#848: what a session left in the ticket's deliverables folder, uploaded to a backend
that shares no folder with this host. Only that folder, never a symlink, within the
claim's limits, and a file the backend will not take is a warning, not a lost result."""
from __future__ import annotations

import os

import pytest

from automatos_cli_host import uploads
from automatos_cli_host.api import BackendError

LIMITS = {"enabled": True, "max_file_bytes": 100, "max_total_bytes": 150}


class FakeApi:
    def __init__(self, fail=None):
        self.sent = []
        self.fail = list(fail or [])     # statuses to raise, in order, before succeeding

    def upload_file(self, host_id, task_id, rel, data):
        if self.fail:
            raise BackendError(self.fail.pop(0), "nope", "url")
        self.sent.append((host_id, task_id, rel, data))
        return {"path": rel}


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "deliverables" / "sessions" / "42"
    (root / "charts").mkdir(parents=True)
    (root / "report.md").write_text("# Q3\n")
    (root / "charts" / "q3.png").write_bytes(b"\x89PNG")
    return root


def _ticket(**upload):
    return {"task_id": 42, "upload": {**LIMITS, **upload}}


def test_only_regular_files_inside_the_folder_are_candidates(folder, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "x.md").write_text("x")
    os.symlink(outside, folder / "link.txt")
    os.symlink(tmp_path / "elsewhere", folder / "linked-dir")
    files = [str(folder / "report.md"), str(folder / "charts" / "q3.png"), str(outside),
             str(folder / "link.txt"), str(folder / "linked-dir" / "x.md"), str(folder / "gone.md"),
             str(folder / "charts" / ".." / "report.md"), str(folder / "report.md")]
    got = uploads.upload_candidates(files, folder)
    assert [rel for _path, rel in got] == ["report.md", "charts/q3.png"]


def test_the_folder_matches_through_its_resolved_path(folder, tmp_path):
    alias = tmp_path / "alias"
    os.symlink(folder.parent.parent, alias)      # like macOS's /var → /private/var
    got = uploads.upload_candidates([str(alias / "sessions" / "42" / "report.md")], folder)
    assert [rel for _path, rel in got] == ["report.md"]


def test_each_file_is_sent_once_with_its_path_in_the_folder(folder):
    api = FakeApi()
    taken = uploads.upload_deliverables(api, "h1", _ticket(), folder,
                                        [str(folder / "report.md"), str(folder / "charts" / "q3.png")])
    assert taken == ["report.md", "charts/q3.png"]
    assert [(rel, data) for _h, _t, rel, data in api.sent] == [("report.md", b"# Q3\n"), ("charts/q3.png", b"\x89PNG")]


def test_nothing_is_sent_unless_the_claim_asks(folder):
    api = FakeApi()
    files = [str(folder / "report.md")]
    assert uploads.upload_deliverables(api, "h1", {"task_id": 42}, folder, files) == []
    assert uploads.upload_deliverables(api, "h1", _ticket(enabled=False), folder, files) == []
    assert uploads.upload_deliverables(api, "h1", _ticket(), None, files) == []
    assert api.sent == []


def test_a_file_past_a_limit_is_skipped_and_the_rest_still_go(folder):
    (folder / "big.bin").write_bytes(b"x" * 101)
    (folder / "mid.bin").write_bytes(b"x" * 90)
    (folder / "last.bin").write_bytes(b"x" * 90)
    api = FakeApi()
    files = [str(folder / name) for name in ("big.bin", "mid.bin", "last.bin", "report.md")]
    assert uploads.upload_deliverables(api, "h1", _ticket(), folder, files) == ["mid.bin", "report.md"]


def test_a_transient_failure_is_retried_and_a_refusal_is_not(folder, monkeypatch):
    monkeypatch.setattr(uploads, "UPLOAD_RETRY_SECONDS", 0)
    files = [str(folder / "report.md")]
    api = FakeApi(fail=[503, 0])
    assert uploads.upload_deliverables(api, "h1", _ticket(), folder, files) == ["report.md"]
    refused = FakeApi(fail=[413])
    assert uploads.upload_deliverables(refused, "h1", _ticket(), folder, files) == []
    assert refused.fail == [] and refused.sent == []
    down = FakeApi(fail=[503] * uploads.UPLOAD_ATTEMPTS)
    assert uploads.upload_deliverables(down, "h1", _ticket(), folder, files) == []


def test_the_client_sends_raw_bytes_with_the_path_as_a_query(monkeypatch):
    from automatos_cli_host import api as api_mod

    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"path": "a b/c.md"}'

    def _urlopen(req, timeout):
        seen.update(url=req.full_url, method=req.get_method(), data=req.data, timeout=timeout,
                    ctype=req.get_header("Content-type"), token=req.get_header("X-cli-host-token"))
        return Resp()

    monkeypatch.setattr(api_mod.urllib.request, "urlopen", _urlopen)
    client = api_mod.BackendClient("http://api.test/", token="tok")
    assert client.upload_file("h1", 42, "a b/c.md", b"\x00\x01") == {"path": "a b/c.md"}
    assert seen == {"url": "http://api.test/api/v1/cli-hosts/h1/tasks/42/files?path=a+b%2Fc.md", "method": "PUT",
                    "data": b"\x00\x01", "timeout": api_mod.UPLOAD_TIMEOUT, "ctype": "application/octet-stream",
                    "token": "tok"}


def test_a_file_that_goes_away_after_it_was_found_is_skipped(folder, monkeypatch):
    """It was listed, then vanished or became unreadable: a warning, and the rest still go."""
    api = FakeApi()
    real_read = uploads.Path.read_bytes

    def _read(path):
        if path.name == "report.md":
            raise PermissionError("not readable any more")
        return real_read(path)

    monkeypatch.setattr(uploads.Path, "read_bytes", _read)
    files = [str(folder / "report.md"), str(folder / "charts" / "q3.png")]
    assert uploads.upload_deliverables(api, "h1", _ticket(), folder, files) == ["charts/q3.png"]


def test_the_folder_must_exist(tmp_path):
    assert uploads.upload_candidates([str(tmp_path / "x")], tmp_path / "missing") == []
