"""Tests for core.llm.workspace_keys — operator workspace key resolution."""

from unittest import mock

from config import config as app_config
from core.llm.workspace_keys import get_platform_workspace_key

WS = "ae8320bc-95e1-4de1-bbe9-396bef19cbf8"


def test_disabled_when_setting_empty(monkeypatch):
    monkeypatch.setattr(app_config, "PLATFORM_KEY_WORKSPACE_ID", "")
    assert get_platform_workspace_key("openrouter") is None


def test_resolves_active_row(monkeypatch):
    monkeypatch.setattr(app_config, "PLATFORM_KEY_WORKSPACE_ID", WS)

    row = mock.Mock(encrypted_key="enc-blob")
    session = mock.MagicMock()
    session.query.return_value.filter.return_value.order_by.return_value.first.return_value = row

    enc = mock.Mock()
    enc.decrypt.return_value = "sk-or-live"

    with mock.patch("core.database.database.SessionLocal", return_value=session), \
         mock.patch("core.credentials.encryption.get_encryption_service", return_value=enc):
        assert get_platform_workspace_key("openrouter") == "sk-or-live"

    enc.decrypt.assert_called_once_with("enc-blob")
    session.close.assert_called_once()


def test_no_active_row_returns_none(monkeypatch):
    monkeypatch.setattr(app_config, "PLATFORM_KEY_WORKSPACE_ID", WS)

    session = mock.MagicMock()
    session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    with mock.patch("core.database.database.SessionLocal", return_value=session):
        assert get_platform_workspace_key("openrouter") is None
    session.close.assert_called_once()


def test_lookup_failure_swallowed(monkeypatch):
    monkeypatch.setattr(app_config, "PLATFORM_KEY_WORKSPACE_ID", WS)

    with mock.patch(
        "core.database.database.SessionLocal", side_effect=RuntimeError("db down")
    ):
        assert get_platform_workspace_key("openrouter") is None
