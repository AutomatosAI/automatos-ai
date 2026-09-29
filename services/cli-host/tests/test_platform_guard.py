"""On native Windows the host says what it needs instead of crashing (issue #818).

The host imports ``pty`` and installs as a launchd or systemd service, so on
Windows ``python -m automatos_cli_host`` used to die with an ImportError or
"no supported service manager here". It now stops before importing the host
and points at WSL2.
"""
from __future__ import annotations

import runpy
import sys

import pytest

from automatos_cli_host.__main__ import UNSUPPORTED_PLATFORM_MESSAGE, platform_supported


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_macos_and_linux_are_supported(platform):
    assert platform_supported(platform)


def test_native_windows_is_not():
    assert not platform_supported("win32")


def test_windows_exits_with_the_wsl2_message_before_importing_the_host(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    # A None entry makes the import raise, as `import pty` does on Windows.
    monkeypatch.setitem(sys.modules, "automatos_cli_host.host", None)

    with pytest.raises(SystemExit) as exited:
        runpy.run_module("automatos_cli_host", run_name="__main__")

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err == UNSUPPORTED_PLATFORM_MESSAGE
    assert "WSL2" in err
