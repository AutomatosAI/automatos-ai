"""PRD-282 US-301: ``LOG_FORMAT`` and ``LOG_LEVEL``, read through config.py and checked when logging starts.

* Unset or empty, each is today's behaviour: text lines at INFO.
* Case and surrounding spaces are ignored; ``WARN`` and ``FATAL`` are Python's own aliases.
* Any other value never stops start-up: the default is used (INFO, text) and one warning
  names the setting and what it may be, never the value.
* ``LOG_LEVEL`` is honoured: it was passed by the chart, compose and the images, and ignored.
  uvicorn's own three loggers take it too, so WARNING drops its access and start-up lines.
* At DEBUG, the AWS SDK's loggers stay at WARNING, as the HTTP libraries do, and loggers under
  ``core.credentials`` stay at INFO: credential handling is not debug-logged.
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
from core.monitoring.log_settings import log_format, log_level, unusable

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
def test_any_other_value_is_the_default_with_a_warning_naming_the_setting_never_the_value(name, read, raw):
    assert read(raw) == read("")
    settings = {"LOG_FORMAT": (raw, None), "LOG_LEVEL": (None, raw)}[name]
    (warning,) = unusable(*settings)
    assert warning.startswith(f"{name} is not one of ") and raw not in warning
    assert unusable("json", "debug") == () and unusable("", "") == ()


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
        "print(','.join(logging.getLevelName(logging.getLogger(n).level) for n in ('uvicorn', 'uvicorn.error', 'uvicorn.access')))\n"
    )
    env = {**{k: v for k, v in os.environ.items() if k not in ("LOG_FORMAT", "LOG_LEVEL")}, **settings}
    return subprocess.run([sys.executable, "-c", script], cwd=ORCH, env=env, capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize("name", ["LOG_FORMAT", "LOG_LEVEL"])
def test_start_up_goes_on_with_one_warning_on_a_value_it_cannot_use(name):
    done = _setup_in_a_fresh_process(**{name: PLANTED})
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines()[0] == "INFO"
    warnings = [line for line in done.stderr.splitlines() if f"{name} is not one of" in line]
    assert len(warnings) == 1 and "WARNING" in warnings[0], done.stderr[-2000:]
    assert PLANTED not in done.stderr + done.stdout


@pytest.mark.parametrize("settings, level", [({}, "INFO"), ({"LOG_LEVEL": ""}, "INFO"),
                                             ({"LOG_LEVEL": "warning"}, "WARNING"), ({"LOG_LEVEL": "DEBUG"}, "DEBUG")])
def test_log_level_sets_the_level_and_unset_is_info(settings, level):
    done = _setup_in_a_fresh_process(**settings)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == [level, ",".join([level] * 3)], "uvicorn's loggers follow it"


HELD = """
import logging
from core.monitoring.automatos_logging import setup_logging
root = setup_logging(service='prd282-test', enable_relay=True)
relay = next(h for h in root.handlers if type(h).__name__ == 'LogRelayHandler')
shipped = []
relay.emit = shipped.append
logging.getLogger('prd282.app').debug('APP-DEBUG-282')
for name in ('botocore.endpoint', 'boto3.resources', 's3transfer.tasks', 'urllib3.connectionpool'):
    logging.getLogger(name).debug('HELD-DEBUG-282 ' + name)
logging.getLogger('botocore.endpoint').warning('SDK-WARNING-282')
logging.getLogger('core.credentials.service').debug('CREDENTIAL-DEBUG-282')
logging.getLogger('core.credentials.service').info('CREDENTIAL-INFO-282')
print('|'.join(r.getMessage() for r in shipped))
"""


@pytest.mark.parametrize("log_format_value", ["text", "json"])
def test_at_debug_the_sdks_and_credential_handling_write_no_debug_line(log_format_value):
    env = {**{k: v for k, v in os.environ.items() if k not in ("LOG_FORMAT", "LOG_LEVEL")},
           "LOG_LEVEL": "DEBUG", "LOG_FORMAT": log_format_value}
    done = subprocess.run([sys.executable, "-W", "ignore", "-c", HELD], cwd=ORCH, env=env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    console, shipped = done.stderr, done.stdout
    for written in (console, shipped):
        assert "APP-DEBUG-282" in written, "DEBUG is on"
        assert "SDK-WARNING-282" in written and "CREDENTIAL-INFO-282" in written
        assert "HELD-DEBUG-282" not in written and "CREDENTIAL-DEBUG-282" not in written
