"""CodeRabbit's review of #1448 (the CodeQL sweep): the fixes it asked for, each pinned.

- an SSH command's connection is closed whether it ran, raised or never connected;
- a stored SSH credential is read through the credential store, this workspace's only;
- the credential tester trusts a host key on first use, as exec_shell does;
- a numbered-pass title with "##" is not telemetry, as before the ReDoS fix;
- leading SQL comments are skipped at a moving position (no copy per comment);
- the email preview's tag stripping ends a tag only where a browser does.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


# --- SSH: the connection is always closed ------------------------------------------------------

class _Client:
    def __init__(self, fail_connect=False, fail_command=False):
        self.closed, self._fail_connect, self._fail_command = False, fail_connect, fail_command

    def set_missing_host_key_policy(self, policy):
        pass

    def get_host_keys(self):
        return SimpleNamespace(load=lambda path: None, add=lambda *a: None)

    def connect(self, **kwargs):
        if self._fail_connect:
            raise OSError("connection refused")

    def exec_command(self, command, timeout=None):
        if self._fail_command:
            raise TimeoutError("read timed out")
        out = SimpleNamespace(channel=SimpleNamespace(recv_exit_status=lambda: 0), read=lambda: b"ok")
        return None, out, SimpleNamespace(read=lambda: b"")

    def close(self):
        self.closed = True


def test_a_command_that_raises_still_closes_the_connection():
    from modules.tools.execution.exec_shell import _run_ssh_command

    client = _Client(fail_command=True)
    with pytest.raises(TimeoutError):
        _run_ssh_command(client, "uptime", 5, "ssh_execute", "build.acme.example", "deploy", 0.0)
    assert client.closed


def test_a_command_that_runs_closes_the_connection():
    from modules.tools.execution.exec_shell import _run_ssh_command

    client = _Client()
    assert _run_ssh_command(client, "uptime", 5, "ssh_execute", "build.acme.example", "deploy", 0.0)["success"]
    assert client.closed


def test_a_failed_connect_closes_the_client(monkeypatch):
    import paramiko

    from modules.tools.execution import exec_shell

    client = _Client(fail_connect=True)
    monkeypatch.setattr(paramiko, "SSHClient", lambda: client)
    monkeypatch.setattr(exec_shell, "apply_trust_on_first_use_policy", lambda c: None)
    with pytest.raises(OSError):
        exec_shell._connect_ssh_client("build.acme.example", 22, "deploy", 10, "pw", None)
    assert client.closed


# --- SSH: a stored credential, this workspace's only ---------------------------------------------

class _Store:
    decrypted = []

    def __init__(self, db):
        pass

    def get_credential(self, credential_id):
        return {7: SimpleNamespace(id=7, workspace_id="ws-acme")}.get(credential_id)

    def get_decrypted_credential(self, credential_id, service_name=None):
        _Store.decrypted.append(credential_id)
        return {"username": "deploy", "host": "build.acme.example", "port": 22, "private_key": "-----BEGIN"}


@pytest.fixture
def store(monkeypatch):
    from core.credentials import service

    _Store.decrypted = []
    monkeypatch.setattr(service, "CredentialStore", _Store)
    return _Store


def test_a_stored_credential_is_decrypted_through_the_store(store):
    from modules.tools.execution.exec_shell import _resolve_stored_ssh_credential

    fields, error = _resolve_stored_ssh_credential(SimpleNamespace(db=None), "ws-acme", "7", "ssh_execute")
    assert error is None and fields["username"] == "deploy" and fields["host"] == "build.acme.example"
    assert store.decrypted == [7]


@pytest.mark.parametrize(("workspace", "credential"), [("ws-globex", "7"), ("ws-acme", "8"), ("ws-acme", "seven")])
def test_another_workspaces_or_a_missing_credential_is_not_found_and_never_decrypted(store, workspace, credential):
    from modules.tools.execution.exec_shell import _resolve_stored_ssh_credential

    fields, error = _resolve_stored_ssh_credential(SimpleNamespace(db=None), workspace, credential, "ssh_execute")
    assert fields is None and error["error"] == f"Credential {credential} not found"
    assert store.decrypted == []


# --- the credential tester's host keys -------------------------------------------------------------

def test_the_credential_tester_trusts_a_host_key_on_first_use():
    from core.credentials import tester

    source = inspect.getsource(tester)
    assert "apply_trust_on_first_use_policy(ssh)" in source
    assert "WarningPolicy" not in source and "AutoAddPolicy" not in source


# --- titles, SQL comments, tags ----------------------------------------------------------------------

def test_a_title_with_a_double_hash_is_not_a_numbered_pass():
    from services.knowledge_flywheel import _NUMBERED_PASS_RE

    for title in ("Weekly pass 3", "pass#3", "pass # 12 notes"):
        assert _NUMBERED_PASS_RE.search(title), title
    assert not _NUMBERED_PASS_RE.search("Pricing pass##3.pdf")


def test_leading_sql_comments_are_skipped_in_place():
    from core.database.read_release import _past_leading_comments

    assert _past_leading_comments("/* a */ -- b\n/* c */SELECT 1") == "SELECT 1"
    many = "/*x*/" * 20_000 + "SELECT 1"
    assert _past_leading_comments(many) == "SELECT 1"


def test_the_email_preview_ends_a_tag_only_where_a_browser_does():
    import re

    source = inspect.getsource(__import__("modules.tools.formatting.schema_detector", fromlist=["x"]))
    pattern = re.compile(r"<script\b[^>]*>.*?</script(?=[\t\n\f\r />])[^>]*>", re.DOTALL | re.IGNORECASE)
    assert r"</script(?=[\t\n\f\r />])[^>]*>" in source and r"</script\b[^>]*>" not in source
    stripped = pattern.sub("", '<script>var s = "</script-x>"; leak();</script><p>Hello</p>')
    assert stripped == "<p>Hello</p>"
