"""PRD-235 W3 — the host is always on and always current.

Pure units: the service command line the installer writes, and the drift
rules (backend contract change or the host's own code changing on disk →
drain → restart exit; a version mismatch alone only warns).
"""
from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from automatos_cli_host import __version__, service
from automatos_cli_host.config import HostConfig
from automatos_cli_host.host import RESTART_EXIT_CODE, Host, source_fingerprint


def _cfg(tmp_path: Path, **kw) -> HostConfig:
    return HostConfig(url="http://127.0.0.1:8000", state_dir=tmp_path / "state", allow_dirs=[tmp_path / "repo"],
                      name="mac", **kw)


def test_service_argv_reproduces_the_host_invocation(tmp_path):
    (tmp_path / "repo").mkdir()
    argv = service.service_argv(_cfg(tmp_path, max_sessions=2, use_worktrees=False), ["--verbose"])
    assert argv[1:3] == ["-m", "automatos_cli_host"]
    assert argv[argv.index("--url") + 1] == "http://127.0.0.1:8000"
    assert argv[argv.index("--allow") + 1] == str((tmp_path / "repo").resolve())
    assert "--default-root" not in argv  # none requested → the host's first root
    cfg.default_root = tmp_path / "deliverables"
    argv = service_argv(cfg)
    assert argv[argv.index("--default-root") + 1] == str((tmp_path / "deliverables").resolve())
    assert "--max-sessions" in argv and argv[argv.index("--max-sessions") + 1] == "2"
    assert "--no-worktrees" in argv and argv[-1] == "--verbose"
    assert "--pair" not in argv and "--install" not in argv  # a service never re-pairs or re-installs


def test_nudge_signals_the_pid_file_and_reports_absence(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    assert service.nudge(cfg) is False  # no pid file
    cfg.state_dir.mkdir(parents=True)
    cfg.pid_path.write_text(f"{os.getpid()}\n")
    seen = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: seen.append((pid, sig)))
    assert service.nudge(cfg) is True
    assert seen == [(os.getpid(), signal.SIGHUP)]


def _bare_host() -> Host:
    h = Host.__new__(Host)
    h.sessions = {}
    h.pending_results = {}
    h.draining = None
    h.exit_code = 0
    h._source_fingerprint = source_fingerprint()
    h._backend_contract = None
    return h


def test_first_contract_is_remembered_and_a_change_requests_a_restart():
    h = _bare_host()
    h._check_drift({"host_contract": "abc", "expected_host_version": __version__})
    assert h._backend_contract == "abc" and h.draining is None
    h._check_drift({"host_contract": "abc"})
    assert h.draining is None
    h._check_drift({"host_contract": "def"})
    assert h.draining and "contract" in h.draining


def test_version_mismatch_alone_only_warns(caplog):
    h = _bare_host()
    with caplog.at_level("WARNING"):
        h._check_drift({"host_contract": "abc", "expected_host_version": "9.9.9"})
    assert h.draining is None
    assert any("expects host v9.9.9" in r.getMessage() for r in caplog.records)


def test_own_code_change_requests_a_restart(monkeypatch):
    h = _bare_host()
    import automatos_cli_host.host as host_mod
    monkeypatch.setattr(host_mod, "source_fingerprint", lambda: "changed")
    h._check_drift({"host_contract": "abc"})
    assert h.draining and "own code" in h.draining


def test_draining_host_stops_claiming(monkeypatch):
    h = _bare_host()
    h.cfg = HostConfig()
    h.api = None  # would explode if a claim were attempted
    h.request_restart("test")
    h._claim_and_start("host-1")  # returns before touching the API


def test_source_fingerprint_is_stable_and_short():
    a, b = source_fingerprint(), source_fingerprint()
    assert a == b and len(a) == 16
    assert RESTART_EXIT_CODE == 75
