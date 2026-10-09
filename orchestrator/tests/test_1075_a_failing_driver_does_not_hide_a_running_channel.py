"""Issue #1075: one failing channel driver must not hide a running one.

``ChannelManager.is_running`` used to wrap every driver check in a single
``try/except Exception: pass``. A driver that raised (for example a
transient Redis error) aborted the loop, returned False, and logged nothing.
"""

from __future__ import annotations

import logging

from channels.manager import ChannelManager


class _RaisingDriver:
    def is_polling_running(self, *, connection_id: str) -> bool:
        raise RuntimeError("redis unavailable")


class _RunningDriver:
    def is_polling_running(self, *, connection_id: str) -> bool:
        return connection_id == "conn-1"


class _IdleDriver:
    def is_polling_running(self, *, connection_id: str) -> bool:
        return False


def _install_drivers(monkeypatch, mapping):
    import channels.drivers as drivers

    monkeypatch.setattr(drivers, "list_platforms", lambda: list(mapping))
    monkeypatch.setattr(drivers, "get_driver", lambda platform: mapping[platform])


def test_failing_driver_does_not_hide_a_running_channel(monkeypatch, caplog):
    _install_drivers(
        monkeypatch,
        {"slack": _RaisingDriver, "telegram": _RunningDriver},
    )
    manager = ChannelManager()

    with caplog.at_level(logging.WARNING, logger="channels.manager"):
        assert manager.is_running("conn-1") is True

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("slack" in r.getMessage() for r in warnings)
    assert any(r.exc_info and r.exc_info[0] is RuntimeError for r in warnings)


def test_failing_driver_still_returns_false_when_nothing_is_running(monkeypatch, caplog):
    _install_drivers(
        monkeypatch,
        {"slack": _RaisingDriver, "discord": _IdleDriver},
    )
    manager = ChannelManager()

    with caplog.at_level(logging.WARNING, logger="channels.manager"):
        assert manager.is_running("conn-1") is False

    assert any("slack" in r.getMessage() for r in caplog.records)


def test_legacy_adapter_short_circuits_before_drivers(monkeypatch):
    called = {"n": 0}

    def _boom():
        called["n"] += 1
        raise AssertionError("drivers should not be consulted")

    import channels.drivers as drivers

    monkeypatch.setattr(drivers, "list_platforms", _boom)
    manager = ChannelManager()

    class _Legacy:
        is_running = True

    manager._adapters["conn-1"] = _Legacy()
    assert manager.is_running("conn-1") is True
    assert called["n"] == 0
