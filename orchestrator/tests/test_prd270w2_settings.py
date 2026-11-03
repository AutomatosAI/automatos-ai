"""PRD-270 W2 US-270-11a (K7) — the three Cloud bundle settings.

* Each is on ``config`` with its default (read from ``config.py``'s source as well, so a CI
  environment that sets one can't hide a changed default), declared once with a comment above
  it, beside W1's ``AUTOMATOS_CLOUD_*`` block, and named in ``reports/config-surface.json`` in
  sorted position.
* ``AUTOMATOS_CLOUD_PLUGINS_PER_SYNC`` is never negative (a negative value would slice a list
  from its end).
* No setting names a key: the Sync and the hosted bundle routes are anonymous.
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
# name -> (the default as config.py's os.getenv names it, the value config holds)
SETTINGS = {
    "MARKETPLACE_CATALOG_BUNDLE_MAX_KB": ("256", 256),
    "AUTOMATOS_CLOUD_BUNDLE_MAX_KB": ("256", 256),
    "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC": ("5", 5),
}
# W1's block (K7): the new settings sit between its first and last line.
W1_FIRST, W1_LAST = "AUTOMATOS_CLOUD_URL", "MARKETPLACE_PUBLIC_RATE_PER_MINUTE"
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


def _declaration(lines: list[str], name: str) -> int:
    found = [i for i, line in enumerate(lines) if re.match(rf"^[ \t]+{name}[ \t]*:", line)]
    assert len(found) == 1, f"{name} is declared once"
    return found[0]


def test_each_setting_is_on_config_and_config_py_names_its_default():
    from config import config

    source = (ORCH / "config.py").read_text(encoding="utf-8")
    for name, (default, _value) in SETTINGS.items():
        assert hasattr(config, name), name
        assert f'os.getenv("{name}", "{default}")' in source, name


def test_each_setting_has_a_comment_and_sits_in_w1s_block():
    lines = (ORCH / "config.py").read_text(encoding="utf-8").splitlines()
    first, last = _declaration(lines, W1_FIRST), _declaration(lines, W1_LAST)
    for name in SETTINGS:
        at = _declaration(lines, name)
        assert first < at < last, f"{name} is beside W1's AUTOMATOS_CLOUD_* block"
        assert lines[at - 1].strip().startswith("#"), f"{name} has a comment above it"


def test_unset_each_setting_holds_its_default(reload_config):
    fresh = reload_config(**{name: None for name in SETTINGS})
    assert {name: getattr(fresh, name) for name in SETTINGS} == {name: value for name, (_d, value) in SETTINGS.items()}


def test_each_setting_reads_the_environment_and_plugins_per_sync_is_never_negative(reload_config):
    fresh = reload_config(
        MARKETPLACE_CATALOG_BUNDLE_MAX_KB="64", AUTOMATOS_CLOUD_BUNDLE_MAX_KB="128", AUTOMATOS_CLOUD_PLUGINS_PER_SYNC="0",
    )
    assert (fresh.MARKETPLACE_CATALOG_BUNDLE_MAX_KB, fresh.AUTOMATOS_CLOUD_BUNDLE_MAX_KB) == (64, 128)
    assert fresh.AUTOMATOS_CLOUD_PLUGINS_PER_SYNC == 0
    assert reload_config(AUTOMATOS_CLOUD_PLUGINS_PER_SYNC="-3").AUTOMATOS_CLOUD_PLUGINS_PER_SYNC == 0


def test_each_setting_is_in_the_config_surface_in_sorted_position():
    names = json.loads(SURFACE.read_text(encoding="utf-8"))["settings"]
    assert set(SETTINGS) <= set(names)
    assert names == sorted(names)


def test_no_setting_names_a_key():
    assert [name for name in SETTINGS if KEY_NAME.search(name)] == []
