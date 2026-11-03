"""PRD-270 W1 US-270-00 (K7) — the five Automatos Cloud and public Marketplace settings.

* Each is on ``config`` with its default (read from ``config.py``'s source as well, so a CI
  environment that sets one can't hide a changed default), and named in
  ``reports/config-surface.json`` in sorted position.
* ``AUTOMATOS_CLOUD_URL`` defaults to Automatos Cloud (on for every local install, §8-Q6), and
  an empty or blank value reads as off (an empty string).
* ``MARKETPLACE_PUBLIC_RATE_PER_MINUTE`` is at least one.
* ``envs/api.defaults`` names the Cloud origin and how to turn it off, and holds no secret; no
  setting names a key.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

ORCH = Path(__file__).resolve().parents[1]
SURFACE = ORCH / "reports" / "config-surface.json"
API_DEFAULTS = ORCH.parent / "envs" / "api.defaults"
CLOUD_ORIGIN = "https://api.automatos.app"
# name -> (the default as config.py's os.getenv names it, the value config holds)
SETTINGS = {
    "AUTOMATOS_CLOUD_URL": (CLOUD_ORIGIN, CLOUD_ORIGIN),
    "AUTOMATOS_CLOUD_FEED_MAX_KB": ("1024", 1024),
    "AUTOMATOS_CLOUD_TIMEOUT_SECONDS": ("15", 15),
    "MARKETPLACE_CATALOG_FEED_MAX_KB": ("1024", 1024),
    "MARKETPLACE_PUBLIC_RATE_PER_MINUTE": ("120", 120),
}
KEY_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)$|SIGN")


@pytest.fixture
def reload_config(monkeypatch):
    """Reload config.py under a changed environment; put the module and os.environ back after.

    The settings are read when the class is defined, so a new value needs a reload. The
    snapshot keeps every other suite on the config object it bound (test_config.py's pattern).
    """
    env_snapshot = dict(os.environ)
    saved = sys.modules.get("config")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)

    def _reload(**env: str | None):
        for name, value in env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        sys.modules.pop("config", None)
        import config as fresh  # noqa: WPS433: the re-import after the environment changed

        importlib.reload(fresh)
        return fresh.config

    try:
        yield _reload
    finally:
        os.environ.clear()
        os.environ.update(env_snapshot)
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)


def test_each_setting_is_on_config_and_config_py_names_its_default():
    from config import config

    source = (ORCH / "config.py").read_text(encoding="utf-8")
    for name, (default, _value) in SETTINGS.items():
        assert hasattr(config, name), name
        assert f'os.getenv("{name}", "{default}")' in source, name
        assert len(re.findall(rf"^[ \t]+{name}[ \t]*:", source, re.M)) == 1, f"{name} is declared once"


def test_unset_each_setting_holds_its_default(reload_config):
    fresh = reload_config(**{name: None for name in SETTINGS})
    assert {name: getattr(fresh, name) for name in SETTINGS} == {name: value for name, (_d, value) in SETTINGS.items()}


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_cloud_url_reads_as_off(reload_config, value):
    assert reload_config(AUTOMATOS_CLOUD_URL=value).AUTOMATOS_CLOUD_URL == ""


def test_the_cloud_url_is_read_trimmed_and_the_public_rate_is_at_least_one(reload_config):
    fresh = reload_config(AUTOMATOS_CLOUD_URL=" http://127.0.0.1:8000 ", MARKETPLACE_PUBLIC_RATE_PER_MINUTE="0")
    assert fresh.AUTOMATOS_CLOUD_URL == "http://127.0.0.1:8000"
    assert fresh.MARKETPLACE_PUBLIC_RATE_PER_MINUTE == 1


def test_each_setting_is_in_the_config_surface_in_sorted_position():
    names = json.loads(SURFACE.read_text(encoding="utf-8"))["settings"]
    assert set(SETTINGS) <= set(names)
    assert names == sorted(names)


def test_no_setting_names_a_key():
    # The Sync and the public reads are anonymous: nothing here is a credential.
    assert [name for name in SETTINGS if KEY_NAME.search(name)] == []


def test_the_local_defaults_name_the_cloud_origin_and_how_to_turn_it_off():
    text = API_DEFAULTS.read_text(encoding="utf-8")
    (url,) = re.findall(r"^AUTOMATOS_CLOUD_URL=(.*)$", text, re.M)
    assert url == CLOUD_ORIGIN
    assert "@" not in url and "?" not in url and url.count("/") == 2  # an origin: no user, query or path
    assert "AUTOMATOS_CLOUD_URL= in envs/api.local" in text  # how to turn it off
    assert "Automatos Cloud [ACTIVE]" in text
