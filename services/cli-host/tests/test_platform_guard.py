"""Where the host runs, and what it says where it cannot (issue #818).

macOS and Linux always; Windows from 10 version 1809, the first with ConPTY. On an
older Windows ``python -m automatos_cli_host`` stops before importing the host and
points at WSL2, instead of failing on a missing Windows API. On Windows the session
sandbox and Codex are refused with a sentence, never attempted.
"""
from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace

import pytest

from automatos_cli_host.__main__ import CONPTY_FIRST_BUILD, UNSUPPORTED_PLATFORM_MESSAGE, platform_supported
from automatos_cli_host.sandbox import WINDOWS_HINT, SessionSandbox, unavailable_reason


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_macos_and_linux_are_supported(platform):
    assert platform_supported(platform)


def test_windows_with_conpty_is_supported_and_older_windows_is_not():
    assert platform_supported("win32", windows_build=CONPTY_FIRST_BUILD)
    assert platform_supported("win32", windows_build=26100)            # Windows 11 24H2
    assert not platform_supported("win32", windows_build=CONPTY_FIRST_BUILD - 1)


def test_an_older_windows_exits_with_the_wsl2_message_before_importing_the_host(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    # runpy runs __main__.py afresh, so the Windows version is what it reads: 10 1803.
    monkeypatch.setattr(sys, "getwindowsversion", lambda: SimpleNamespace(build=17134), raising=False)
    # A None entry makes the import raise: the guard must stop before it.
    monkeypatch.setitem(sys.modules, "automatos_cli_host.host", None)

    with pytest.raises(SystemExit) as exited:
        runpy.run_module("automatos_cli_host", run_name="__main__")

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err == UNSUPPORTED_PLATFORM_MESSAGE
    assert "1809" in err and "WSL2" in err


def test_on_windows_a_sandboxed_session_is_refused_with_the_way_out():
    assert unavailable_reason(SessionSandbox(enabled=True), system="Windows") == WINDOWS_HINT
    assert "--no-session-sandbox" in WINDOWS_HINT
    assert unavailable_reason(SessionSandbox(enabled=False), system="Windows") is None


def test_codex_is_refused_on_native_windows(monkeypatch):
    from automatos_cli_host.adapters import adapter_for
    from automatos_cli_host.adapters.codex import CODEX_ON_WINDOWS

    monkeypatch.setattr(sys, "platform", "win32")
    refusal = adapter_for("codex").preflight()
    assert refusal.code == "codex_windows" and refusal.message == CODEX_ON_WINDOWS
    assert "WSL" in refusal.message
