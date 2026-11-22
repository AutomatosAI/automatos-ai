"""PRD-282 US-301: ``LOG_FORMAT`` and ``LOG_LEVEL``, read through config.py and checked when logging starts.

* Unset or empty, each is today's behaviour: text lines at INFO.
* Case and surrounding spaces are ignored; ``WARN`` and ``FATAL`` are Python's own aliases.
* Any other value stops start-up: ``setup_logging`` raises, naming the setting and what it
  may be, never the value. main.py calls it when it is imported, so the API does not start.
* ``LOG_LEVEL`` is honoured: it was passed by the chart, compose and the images, and ignored.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import config
from core.monitoring.log_settings import LogSettingRefused, log_format, log_level

ORCH = Path(__file__).resolve().parents[1]
PLANTED = "planted-282-value"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_empty_is_todays_text_at_info(raw):
    assert log_format(raw) == "text"
    assert log_level(raw) == logging.INFO


@pytest.mark.parametrize("raw, expected", [("json", "json"), (" JSON ", "json"), ("Text", "text")])
def test_a_format_is_read_without_regard_to_case_or_spaces(raw, expected):
    assert log_format(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("debug", logging.DEBUG), (" Info ", logging.INFO), ("WARNING", logging.WARNING), ("warn", logging.WARNING),
    ("error", logging.ERROR), ("CRITICAL", logging.CRITICAL), ("fatal", logging.CRITICAL),
])
def test_a_level_is_a_level_name_in_any_case(raw, expected):
    assert log_level(raw) == expected


@pytest.mark.parametrize("name, read, raw", [
    ("LOG_FORMAT", log_format, PLANTED), ("LOG_FORMAT", log_format, "logfmt"),
    ("LOG_LEVEL", log_level, PLANTED), ("LOG_LEVEL", log_level, "NOTSET"), ("LOG_LEVEL", log_level, "10"),
])
def test_any_other_value_is_refused_naming_the_setting_never_the_value(name, read, raw):
    with pytest.raises(LogSettingRefused) as refused:
        read(raw)
    assert str(refused.value).startswith(f"{name} cannot be used")
    assert raw not in str(refused.value)


def test_both_settings_are_in_config_and_its_surface():
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert {"LOG_FORMAT", "LOG_LEVEL"} <= set(surface)
    assert isinstance(config.LOG_FORMAT, str) and isinstance(config.LOG_LEVEL, str)


def _setup_in_a_fresh_process(**settings) -> subprocess.CompletedProcess:
    """setup_logging as main.py calls it at import, in a fresh interpreter with ``settings`` in its environment."""
    script = (
        "import logging\n"
        "from core.monitoring.automatos_logging import setup_logging\n"
        "root = setup_logging(service='prd282-test', enable_relay=False)\n"
        "print(logging.getLevelName(root.level))\n"
    )
    env = {**{k: v for k, v in os.environ.items() if k not in ("LOG_FORMAT", "LOG_LEVEL")}, **settings}
    return subprocess.run([sys.executable, "-c", script], cwd=ORCH, env=env, capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize("name", ["LOG_FORMAT", "LOG_LEVEL"])
def test_start_up_stops_on_a_value_it_cannot_use(name):
    done = _setup_in_a_fresh_process(**{name: PLANTED})
    assert done.returncode != 0
    assert f"LogSettingRefused: {name} cannot be used" in done.stderr
    assert PLANTED not in done.stderr


@pytest.mark.parametrize("settings, level", [({}, "INFO"), ({"LOG_LEVEL": ""}, "INFO"),
                                             ({"LOG_LEVEL": "warning"}, "WARNING"), ({"LOG_LEVEL": "DEBUG"}, "DEBUG")])
def test_log_level_sets_the_level_and_unset_is_info(settings, level):
    done = _setup_in_a_fresh_process(**settings)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == level
